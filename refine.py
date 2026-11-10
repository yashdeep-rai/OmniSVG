"""
refine.py — Semantic-Aware Localized SVG Refinement
=====================================================
Entry point for the refinement system.

Two workflows:

  Workflow A: Refine an existing SVG
  -----------------------------------
    python refine.py refine \\
        --prompt "A red apple next to a green leaf" \\
        --svg path/to/input.svg \\
        --output ./refined_output

  Workflow B: Generate and refine
  --------------------------------
    python refine.py generate \\
        --prompt "A red apple next to a green leaf" \\
        --output ./gen_refine_output \\
        --model-path  "C:\\...\\models\\Qwen2.5-VL-7B-Instruct" \\
        --weight-path "C:\\...\\models\\OmniSVG1.1_8B\\pytorch_model.bin"

Common options
--------------
    --max-iterations  N    (default 3)
    --render-size     N    (default 512)
    --verbose
    --skip-critic          (run without loading VLM; for testing)
    --critic-model-path    (default: ../models/Qwen2.5-VL-7B-Instruct)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# Ensure project root is on the Python path when running from any cwd
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from refinement.pipeline import run_refinement, RefinementConfig
from refinement.renderer import render_svg, validate_svg
from refinement.critic import QWEN_MODEL_PATH

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------

DEFAULT_QWEN_PATH = os.environ.get(
    "QWEN_VL_MODEL_PATH",
    r"C:\Users\CSE_SDPL\Documents\123cs0077_Project\models\Qwen2.5-VL-7B-Instruct",
)
DEFAULT_WEIGHT_PATH = os.environ.get(
    "OMNISVG_WEIGHT_PATH",
    r"C:\Users\CSE_SDPL\Documents\123cs0077_Project\models\OmniSVG1.1_8B\pytorch_model.bin",
)

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="refine.py",
        description="Semantic-Aware Localized SVG Refinement",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # ---- Workflow A: refine ----
    refine_p = subparsers.add_parser(
        "refine",
        help="Refine an existing SVG file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    refine_p.add_argument(
        "--prompt", required=True,
        help="The text prompt that describes what the SVG should depict.",
    )
    refine_p.add_argument(
        "--svg", required=True,
        help="Path to the input SVG file.",
    )
    refine_p.add_argument(
        "--output", default="./refinement_output",
        help="Output directory for all results.",
    )
    _add_common_args(refine_p)

    # ---- Workflow B: generate ----
    gen_p = subparsers.add_parser(
        "generate",
        help="Generate an SVG from a prompt, then refine it.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    gen_p.add_argument(
        "--prompt", required=True,
        help="The text prompt for SVG generation.",
    )
    gen_p.add_argument(
        "--output", default="./gen_refinement_output",
        help="Output directory for all results.",
    )
    gen_p.add_argument(
        "--model-path", default=DEFAULT_QWEN_PATH,
        help="Path to Qwen2.5-VL model (for OmniSVG generation).",
    )
    gen_p.add_argument(
        "--weight-path", default=DEFAULT_WEIGHT_PATH,
        help="Path to OmniSVG weight file (pytorch_model.bin).",
    )
    gen_p.add_argument(
        "--num-candidates", type=int, default=1,
        help="Number of SVG candidates to generate.",
    )
    gen_p.add_argument(
        "--max-length", type=int, default=512,
        help="Maximum token length for SVG generation.",
    )
    _add_common_args(gen_p)

    return parser


def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--max-iterations", type=int, default=3,
        help="Maximum refinement iterations.",
    )
    p.add_argument(
        "--render-size", type=int, default=512,
        help="Render size in pixels (width and height).",
    )
    p.add_argument(
        "--critic-model-path", default=DEFAULT_QWEN_PATH,
        help="Path to Qwen2.5-VL-7B-Instruct model for the critic.",
    )
    p.add_argument(
        "--skip-critic", action="store_true", default=False,
        help="Skip the VLM critic (useful for testing without GPU).",
    )
    p.add_argument(
        "--verbose", action="store_true", default=False,
        help="Enable verbose logging.",
    )
    p.add_argument(
        "--regression-threshold", type=float, default=5.0,
        help="Maximum allowed pixel-diff increase for candidate acceptance.",
    )


# ---------------------------------------------------------------------------
# Workflow A: refine
# ---------------------------------------------------------------------------


def workflow_refine(args: argparse.Namespace) -> None:
    svg_path = Path(args.svg)
    if not svg_path.exists():
        print(f"[ERROR] SVG file not found: {svg_path}", file=sys.stderr)
        sys.exit(1)

    ok, msg = validate_svg(svg_path)
    if not ok:
        print(f"[ERROR] SVG validation failed: {msg}", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output)

    config = RefinementConfig(
        max_iterations=args.max_iterations,
        render_size=args.render_size,
        verbose=args.verbose,
        critic_model_path=args.critic_model_path,
        skip_critic=args.skip_critic,
        regression_threshold=args.regression_threshold,
    )

    print(f"\n{'='*60}")
    print("Semantic-Aware SVG Refinement — Workflow A")
    print(f"{'='*60}")
    print(f"Prompt:         {args.prompt}")
    print(f"Input SVG:      {svg_path}")
    print(f"Output dir:     {output_dir}")
    print(f"Max iterations: {config.max_iterations}")
    print(f"Skip critic:    {config.skip_critic}")
    print(f"{'='*60}\n")

    result = run_refinement(
        prompt=args.prompt,
        svg_path=svg_path,
        output_dir=output_dir,
        config=config,
    )

    _print_result_summary(result)

    # Save the full result JSON
    result_path = output_dir / "refinement_result.json"
    result.save(result_path)
    print(f"\nFull result saved to: {result_path}")


# ---------------------------------------------------------------------------
# Workflow B: generate + refine
# ---------------------------------------------------------------------------


def workflow_generate(args: argparse.Namespace) -> None:
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Generate SVG using inference.py ---
    print(f"\n{'='*60}")
    print("Semantic-Aware SVG Refinement — Workflow B (Generate + Refine)")
    print(f"{'='*60}")
    print(f"Prompt: {args.prompt}")
    print(f"Output: {output_dir}")
    print(f"\n[1/2] Generating initial SVG with OmniSVG …")

    # Write prompt to temp file
    prompt_file = output_dir / "prompt.txt"
    prompt_file.write_text(args.prompt, encoding="utf-8")

    gen_output_dir = output_dir / "generation"
    gen_output_dir.mkdir(parents=True, exist_ok=True)

    inference_script = _HERE / "inference.py"
    if not inference_script.exists():
        print(f"[ERROR] inference.py not found at {inference_script}", file=sys.stderr)
        sys.exit(1)

    cmd = [
        sys.executable,
        str(inference_script),
        "--task", "text-to-svg",
        "--input", str(prompt_file),
        "--output", str(gen_output_dir),
        "--model-path", args.model_path,
        "--weight-path", args.weight_path,
        "--num-candidates", str(args.num_candidates),
        "--max-length", str(args.max_length),
        "--save-png",
    ]

    if args.verbose:
        cmd.append("--verbose")

    print(f"Command: {' '.join(cmd)}\n")

    try:
        subprocess.run(cmd, check=True, cwd=str(_HERE))
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] OmniSVG generation failed: {e}", file=sys.stderr)
        sys.exit(1)

    # Find the generated SVG
    svg_files = list(gen_output_dir.glob("*.svg"))
    if not svg_files:
        print("[ERROR] No SVG files generated.", file=sys.stderr)
        sys.exit(1)

    # Take the first generated SVG (best candidate)
    generated_svg = sorted(svg_files)[0]
    print(f"\n[1/2] Generated: {generated_svg}")

    # --- Step 2: Refine ---
    print(f"\n[2/2] Refining SVG …\n")

    config = RefinementConfig(
        max_iterations=args.max_iterations,
        render_size=args.render_size,
        verbose=args.verbose,
        critic_model_path=args.critic_model_path,
        skip_critic=args.skip_critic,
        regression_threshold=args.regression_threshold,
    )

    refine_output_dir = output_dir / "refinement"
    result = run_refinement(
        prompt=args.prompt,
        svg_path=generated_svg,
        output_dir=refine_output_dir,
        config=config,
    )

    _print_result_summary(result)

    # Save result JSON
    result_path = output_dir / "refinement_result.json"
    result.save(result_path)
    print(f"\nFull result saved to: {result_path}")


# ---------------------------------------------------------------------------
# Output summary
# ---------------------------------------------------------------------------


def _print_result_summary(result) -> None:
    print(f"\n{'='*60}")
    print("Refinement Complete")
    print(f"{'='*60}")
    print(f"Stopping reason:   {result.stopping_reason}")
    print(f"Iterations done:   {len(result.iterations)}")
    print(f"Elements modified: {result.total_elements_modified}")
    print(f"Initial errors:    {result.initial_error_count}")
    print(f"Final errors:      {result.final_error_count}")
    print(f"Total runtime:     {result.total_runtime_s:.1f}s")
    print(f"\nOutputs:")
    print(f"  Original SVG:    {result.original_svg_path}")
    print(f"  Final SVG:       {result.final_svg_path}")
    if result.initial_render_path:
        print(f"  Initial render:  {result.initial_render_path}")
    if result.final_render_path:
        print(f"  Final render:    {result.final_render_path}")
    print(f"  Output dir:      {result.output_dir}")

    if result.iterations:
        print(f"\nIteration summary:")
        for rec in result.iterations:
            accepted = "✓" if rec.candidate_accepted else "✗"
            errors = (rec.critic_report or {}).get("errors", [])
            print(
                f"  [{accepted}] Iter {rec.iteration}: "
                f"{len(errors)} critic errors, "
                f"{len(rec.localized_components)} localized, "
                f"{len(rec.proposed_edits)} edits proposed, "
                f"{rec.runtime_s:.1f}s"
            )
            if rec.rejection_reason:
                print(f"       Rejected: {rec.rejection_reason}")
            if rec.stopping_reason:
                print(f"       Stop: {rec.stopping_reason}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = build_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.command == "refine":
        workflow_refine(args)
    elif args.command == "generate":
        workflow_generate(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
