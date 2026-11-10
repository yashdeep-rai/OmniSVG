"""
refinement/pipeline.py
======================
Recursive refinement loop for SVG quality improvement.

Pipeline stages per iteration:
  1. Render the current SVG → PNG.
  2. Run the multimodal critic → CriticReport.
  3. Check stopping conditions.
  4. Localize detected errors → LocalizedComponent list.
  5. Generate constrained edit proposals.
  6. Apply edits to a candidate SVG (deep copy).
  7. Render and validate the candidate.
  8. Regression check: accept only if the candidate is not worse.
  9. Record iteration history and continue.

Stopping conditions:
  - critic.satisfied is True (no errors)
  - No actionable errors remain
  - Localization confidence too low for all errors
  - SVG becomes invalid after editing
  - Edit introduces regression (pixel diff worsens by > threshold)
  - No meaningful progress (same error fingerprint across 2 iterations)
  - Maximum iteration count reached

The original SVG is preserved at all times.  The best-known valid SVG is
never overwritten by a failed candidate.
"""

from __future__ import annotations

import copy
import json
import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

from .svg_parser import (
    SVGDocument,
    parse_svg,
    save_svg,
    svg_to_string,
    reset_id_counter,
)
from .renderer import render_svg, render_svg_iteration, pixel_diff_score, RenderResult
from .critic import CriticReport, analyze as critic_analyze, QWEN_MODEL_PATH
from .localizer import (
    LocalizedComponent,
    localize,
    generate_edit_proposals,
    LOCALIZATION_THRESHOLD,
)
from .editor import SVGEditor, EditProposal, apply_edit_proposals, EditLog


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class RefinementConfig:
    """Configuration for the refinement pipeline."""

    max_iterations: int = 3
    min_localization_confidence: float = LOCALIZATION_THRESHOLD
    regression_threshold: float = 5.0       # max acceptable pixel diff increase
    render_size: int = 512
    verbose: bool = False
    critic_model_path: str = QWEN_MODEL_PATH
    auto_unload_critic: bool = False         # unload after each call (slow but saves VRAM)
    skip_critic: bool = False                # for testing without the VLM
    min_progress_improvement: float = 0.5   # min reduction in high-conf errors to continue


# ---------------------------------------------------------------------------
# Iteration record
# ---------------------------------------------------------------------------

@dataclass
class IterationRecord:
    """Complete record of one refinement iteration."""
    iteration: int
    svg_path: str
    render_path: Optional[str]
    critic_report: Optional[Dict[str, Any]]       # serialised CriticReport
    localized_components: List[Dict[str, Any]]    # serialised LocalizedComponent list
    proposed_edits: List[Dict[str, Any]]          # serialised EditProposal list
    edit_results: List[Dict[str, Any]]
    candidate_svg_path: Optional[str]
    candidate_render_path: Optional[str]
    candidate_accepted: bool
    rejection_reason: Optional[str]
    stopping_reason: Optional[str]
    runtime_s: float


# ---------------------------------------------------------------------------
# Pipeline result
# ---------------------------------------------------------------------------

@dataclass
class RefinementResult:
    """Final result of the full refinement run."""
    prompt: str
    original_svg_path: str
    final_svg_path: str
    initial_render_path: Optional[str]
    final_render_path: Optional[str]
    iterations: List[IterationRecord]
    stopping_reason: str
    total_runtime_s: float
    total_elements_modified: int
    initial_error_count: int
    final_error_count: int
    output_dir: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt": self.prompt,
            "original_svg_path": self.original_svg_path,
            "final_svg_path": self.final_svg_path,
            "initial_render_path": self.initial_render_path,
            "final_render_path": self.final_render_path,
            "stopping_reason": self.stopping_reason,
            "total_runtime_s": self.total_runtime_s,
            "total_elements_modified": self.total_elements_modified,
            "initial_error_count": self.initial_error_count,
            "final_error_count": self.final_error_count,
            "iteration_count": len(self.iterations),
            "iterations": [_iter_to_dict(r) for r in self.iterations],
        }

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        logger.info("Saved refinement result to %s", path)


def _iter_to_dict(r: IterationRecord) -> Dict[str, Any]:
    return {
        "iteration": r.iteration,
        "svg_path": r.svg_path,
        "render_path": r.render_path,
        "critic_report": r.critic_report,
        "localized_components": r.localized_components,
        "proposed_edits": r.proposed_edits,
        "edit_results": r.edit_results,
        "candidate_svg_path": r.candidate_svg_path,
        "candidate_render_path": r.candidate_render_path,
        "candidate_accepted": r.candidate_accepted,
        "rejection_reason": r.rejection_reason,
        "stopping_reason": r.stopping_reason,
        "runtime_s": r.runtime_s,
    }


# ---------------------------------------------------------------------------
# Stopping condition helpers
# ---------------------------------------------------------------------------

def _error_fingerprint(report: CriticReport) -> frozenset:
    """Create a hashable fingerprint of a critic report for progress detection."""
    return frozenset(
        (e.type, e.component, e.expected, e.observed)
        for e in report.errors
    )


def _count_high_conf_errors(report: CriticReport, threshold: float = 0.6) -> int:
    return len(report.high_confidence_errors(threshold))


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_refinement(
    prompt: str,
    svg_path: Path | str,
    output_dir: Path | str,
    config: Optional[RefinementConfig] = None,
) -> RefinementResult:
    """
    Run the recursive refinement pipeline on an existing SVG.

    Parameters
    ----------
    prompt:
        The original text prompt used (or intended) for the SVG.
    svg_path:
        Path to the initial SVG file.
    output_dir:
        Directory where all outputs (renders, edited SVGs, logs) are saved.
    config:
        RefinementConfig; uses defaults if None.

    Returns
    -------
    RefinementResult
    """
    if config is None:
        config = RefinementConfig()

    pipeline_start = time.perf_counter()
    svg_path = Path(svg_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log = logging.getLogger(__name__)
    if config.verbose:
        logging.basicConfig(level=logging.DEBUG)

    log.info("=== Refinement Pipeline START ===")
    log.info("Prompt: %s", prompt)
    log.info("SVG:    %s", svg_path)
    log.info("Output: %s", output_dir)
    log.info("Max iterations: %d", config.max_iterations)

    # --- Preserve the original SVG ---
    original_svg_path = output_dir / "original.svg"
    shutil.copy2(str(svg_path), str(original_svg_path))
    log.info("Original SVG copied to %s", original_svg_path)

    # --- Current best SVG (starts as original) ---
    current_svg_path = output_dir / "current.svg"
    shutil.copy2(str(original_svg_path), str(current_svg_path))

    # --- State tracking ---
    iteration_records: List[IterationRecord] = []
    total_elements_modified = 0
    initial_error_count = 0
    final_error_count = 0
    stopping_reason = "max_iterations_reached"

    # For progress detection
    previous_fingerprint: Optional[frozenset] = None
    previous_high_conf_count: Optional[int] = None

    # --- Render initial SVG ---
    initial_render_path = output_dir / "iter_000_initial.png"
    init_render = render_svg(
        current_svg_path,
        output_size=config.render_size,
        save_to=initial_render_path,
    )

    if not init_render.success or init_render.is_blank:
        log.error("Initial SVG failed to render: %s", init_render.error_message)
        return RefinementResult(
            prompt=prompt,
            original_svg_path=str(original_svg_path),
            final_svg_path=str(current_svg_path),
            initial_render_path=None,
            final_render_path=None,
            iterations=[],
            stopping_reason="initial_render_failed",
            total_runtime_s=time.perf_counter() - pipeline_start,
            total_elements_modified=0,
            initial_error_count=0,
            final_error_count=0,
            output_dir=str(output_dir),
        )

    current_render = init_render
    current_render_path = initial_render_path

    # -----------------------------------------------------------------------
    # Main refinement loop
    # -----------------------------------------------------------------------

    for iteration in range(1, config.max_iterations + 1):
        iter_start = time.perf_counter()
        log.info("\n--- Iteration %d / %d ---", iteration, config.max_iterations)

        rec = IterationRecord(
            iteration=iteration,
            svg_path=str(current_svg_path),
            render_path=str(current_render_path),
            critic_report=None,
            localized_components=[],
            proposed_edits=[],
            edit_results=[],
            candidate_svg_path=None,
            candidate_render_path=None,
            candidate_accepted=False,
            rejection_reason=None,
            stopping_reason=None,
            runtime_s=0.0,
        )

        # --- Stage 1: Critic ---
        if config.skip_critic:
            # Dummy satisfied report for testing without VLM
            log.info("[Critic] Skipped (config.skip_critic=True)")
            report = CriticReport(
                satisfied=True,
                overall_confidence=1.0,
                errors=[],
                raw_response="[skipped]",
                model_failed=False,
                error_message=None,
                runtime_s=0.0,
            )
        else:
            log.info("[Critic] Analyzing image vs prompt …")
            try:
                report = critic_analyze(
                    prompt=prompt,
                    image=current_render.image,
                    model_path=config.critic_model_path,
                    auto_load=True,
                    auto_unload=config.auto_unload_critic,
                )
            except Exception as e:
                log.exception("Critic call failed: %s", e)
                report = CriticReport.failure(f"Critic exception: {e}")

        rec.critic_report = report.to_dict()

        if iteration == 1:
            initial_error_count = len(report.errors)
        final_error_count = len(report.errors)

        log.info(
            "[Critic] satisfied=%s  errors=%d  confidence=%.2f",
            report.satisfied,
            len(report.errors),
            report.overall_confidence,
        )

        # --- Stopping: satisfied ---
        if report.satisfied and not report.model_failed:
            rec.stopping_reason = "satisfied"
            stopping_reason = "satisfied"
            rec.runtime_s = time.perf_counter() - iter_start
            iteration_records.append(rec)
            log.info("[Stop] Critic satisfied — done.")
            break

        # --- Stopping: no actionable errors ---
        high_conf_errors = report.high_confidence_errors(0.5)
        if not high_conf_errors and not report.model_failed:
            rec.stopping_reason = "no_actionable_errors"
            stopping_reason = "no_actionable_errors"
            rec.runtime_s = time.perf_counter() - iter_start
            iteration_records.append(rec)
            log.info("[Stop] No actionable errors remain.")
            break

        # --- Stopping: no meaningful progress ---
        current_fingerprint = _error_fingerprint(report)
        current_high_conf_count = _count_high_conf_errors(report)

        if previous_fingerprint is not None:
            if current_fingerprint == previous_fingerprint:
                rec.stopping_reason = "no_progress"
                stopping_reason = "no_progress"
                rec.runtime_s = time.perf_counter() - iter_start
                iteration_records.append(rec)
                log.info("[Stop] Error fingerprint unchanged — no progress.")
                break

        previous_fingerprint = current_fingerprint
        previous_high_conf_count = current_high_conf_count

        # --- Stage 2: Localize ---
        log.info("[Localize] Mapping errors to SVG elements …")
        try:
            doc = parse_svg(current_svg_path)
            localized = localize(
                doc,
                report,
                min_confidence=config.min_localization_confidence,
            )
        except Exception as e:
            log.exception("Localization failed: %s", e)
            localized = []

        rec.localized_components = [lc.to_dict() for lc in localized]
        log.info("[Localize] Found %d localizable components", len(localized))

        # --- Stopping: localization confidence too low ---
        if not localized:
            rec.stopping_reason = "localization_failed"
            stopping_reason = "localization_failed"
            rec.runtime_s = time.perf_counter() - iter_start
            iteration_records.append(rec)
            log.info("[Stop] No components could be localized with sufficient confidence.")
            break

        # --- Stage 3: Generate edit proposals ---
        log.info("[Edit] Generating proposals …")
        proposals = generate_edit_proposals(localized, doc)

        rec.proposed_edits = [
            {
                "operation": p.operation,
                "element_ids": p.element_ids,
                "params": p.params,
                "rationale": p.rationale,
                "confidence": p.confidence,
            }
            for p in proposals
        ]
        log.info("[Edit] %d proposals generated", len(proposals))

        if not proposals:
            rec.stopping_reason = "no_edit_proposals"
            stopping_reason = "no_edit_proposals"
            rec.runtime_s = time.perf_counter() - iter_start
            iteration_records.append(rec)
            log.info("[Stop] No actionable edit proposals generated.")
            break

        # --- Stage 4: Apply edits to a candidate SVG ---
        log.info("[Edit] Applying proposals …")
        try:
            editor, edit_results = apply_edit_proposals(doc, proposals, min_confidence=0.4)
            candidate_svg_str = editor.to_svg_string()
        except Exception as e:
            log.exception("Edit application failed: %s", e)
            rec.rejection_reason = f"Edit exception: {e}"
            rec.runtime_s = time.perf_counter() - iter_start
            iteration_records.append(rec)
            continue

        rec.edit_results = [
            {
                "success": r.success,
                "modified_count": r.modified_count,
                "error": r.error,
            }
            for r in edit_results
        ]

        elements_modified = sum(r.modified_count for r in edit_results if r.success)
        total_elements_modified += elements_modified
        log.info("[Edit] %d elements modified", elements_modified)

        if elements_modified == 0:
            rec.rejection_reason = "No elements were actually modified"
            rec.runtime_s = time.perf_counter() - iter_start
            iteration_records.append(rec)
            log.info("[Reject] No elements were modified — skipping.")
            continue

        # Save the candidate SVG
        candidate_svg_path = output_dir / f"candidate_iter_{iteration:03d}.svg"
        candidate_svg_path.write_text(candidate_svg_str, encoding="utf-8")
        rec.candidate_svg_path = str(candidate_svg_path)

        # --- Stage 5: Render candidate ---
        candidate_render_path = output_dir / f"iter_{iteration:03d}_candidate.png"
        candidate_render = render_svg(
            candidate_svg_path,
            output_size=config.render_size,
            save_to=candidate_render_path,
        )
        rec.candidate_render_path = str(candidate_render_path)

        if not candidate_render.success or candidate_render.is_blank:
            rec.rejection_reason = (
                f"Candidate render failed: {candidate_render.error_message}"
            )
            rec.runtime_s = time.perf_counter() - iter_start
            iteration_records.append(rec)
            log.info("[Reject] Candidate SVG failed to render.")
            continue

        # --- Stage 6: Regression check ---
        if current_render.image is not None and candidate_render.image is not None:
            prev_diff = pixel_diff_score(
                current_render.image,
                init_render.image,  # compare against original
            )
            cand_diff = pixel_diff_score(
                candidate_render.image,
                init_render.image,
            )
            log.info(
                "[Regression] prev_diff=%.2f  cand_diff=%.2f  threshold=%.2f",
                prev_diff,
                cand_diff,
                config.regression_threshold,
            )
            # Only reject if significantly worse (large increase in diff from original)
            # AND we're moving further from original without critic improvement
            if cand_diff > prev_diff + config.regression_threshold:
                rec.rejection_reason = (
                    f"Regression: candidate diff {cand_diff:.1f} > "
                    f"prev diff {prev_diff:.1f} + threshold {config.regression_threshold}"
                )
                rec.runtime_s = time.perf_counter() - iter_start
                iteration_records.append(rec)
                log.info("[Reject] Regression detected — keeping current SVG.")
                continue

        # --- Accept the candidate ---
        shutil.copy2(str(candidate_svg_path), str(current_svg_path))
        current_render = candidate_render
        current_render_path = candidate_render_path
        rec.candidate_accepted = True

        log.info(
            "[Accept] Iteration %d candidate accepted (%d elements modified)",
            iteration,
            elements_modified,
        )

        rec.runtime_s = time.perf_counter() - iter_start
        iteration_records.append(rec)

    # -----------------------------------------------------------------------
    # Final render
    # -----------------------------------------------------------------------

    final_render_path = output_dir / "final.png"
    final_render = render_svg(
        current_svg_path,
        output_size=config.render_size,
        save_to=final_render_path,
    )

    # Copy final SVG to named output
    final_svg_path = output_dir / "final.svg"
    shutil.copy2(str(current_svg_path), str(final_svg_path))

    total_runtime = time.perf_counter() - pipeline_start

    log.info("=== Refinement Pipeline END ===")
    log.info("Stopping reason:  %s", stopping_reason)
    log.info("Total runtime:    %.1fs", total_runtime)
    log.info("Iterations done:  %d", len(iteration_records))
    log.info("Elements modified: %d", total_elements_modified)
    log.info("Initial errors:   %d", initial_error_count)
    log.info("Final errors:     %d", final_error_count)

    return RefinementResult(
        prompt=prompt,
        original_svg_path=str(original_svg_path),
        final_svg_path=str(final_svg_path),
        initial_render_path=str(initial_render_path) if init_render.success else None,
        final_render_path=str(final_render_path) if final_render.success else None,
        iterations=iteration_records,
        stopping_reason=stopping_reason,
        total_runtime_s=total_runtime,
        total_elements_modified=total_elements_modified,
        initial_error_count=initial_error_count,
        final_error_count=final_error_count,
        output_dir=str(output_dir),
    )
