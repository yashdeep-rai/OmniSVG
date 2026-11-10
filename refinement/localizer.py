"""
refinement/localizer.py
========================
Map critic diagnoses (CriticReport) to actual SVG elements (SVGDocument).

Localization is hard because OmniSVG generates flat, unlabelled <path>
lists.  This module uses multiple evidence sources and returns candidate
components with explicit confidence scores.

Evidence sources (in priority order):
  1. Existing SVG groups (<g>) — highest confidence.
  2. Explicit color matches between error description and fill colors.
  3. Spatial region matching (spatial_error / position hints in description).
  4. Candidate groups from extract_candidate_groups().
  5. Semantic similarity between error component name and color clusters.

If no evidence is strong enough, the localizer returns a low-confidence
result rather than making a destructive guess.

Output schema:
  [
    {
      "component":   "banana",
      "element_ids": ["path_0003", "path_0004"],
      "confidence":  0.82,
      "reason":      "Fill color #ffdd00 matches yellow; center is right-of-midline"
    }
  ]
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Imports (local)
# ---------------------------------------------------------------------------

from .svg_parser import (
    SVGDocument,
    SVGElement,
    ElementGroup,
    extract_candidate_groups,
    _bbox_center,
    _union_bbox,
    _hex_to_rgb,
)
from .critic import CriticReport, ErrorFinding


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class LocalizedComponent:
    """
    A semantic component with associated SVG element candidates.

    Confidence reflects how certain we are that these elements belong to
    this component.  Below LOCALIZATION_THRESHOLD, edits should not be
    applied automatically.
    """
    component: str                       # e.g. "banana"
    error: ErrorFinding                  # the critic error that triggered this
    element_ids: List[str]               # internal SVG element IDs
    confidence: float                    # 0–1
    reason: str                          # human-readable explanation
    bbox: Optional[Tuple[float, float, float, float]]  # union bbox of candidates

    def to_dict(self) -> dict:
        return {
            "component": self.component,
            "element_ids": self.element_ids,
            "confidence": self.confidence,
            "reason": self.reason,
            "bbox": list(self.bbox) if self.bbox else None,
            "error": self.error.to_dict(),
        }


LOCALIZATION_THRESHOLD = 0.40  # below this, don't apply destructive edits


# ---------------------------------------------------------------------------
# Color matching utilities
# ---------------------------------------------------------------------------

# Maps common color names to approximate hex prefixes or RGB ranges
_COLOR_NAME_MAP: Dict[str, List[str]] = {
    "red":     ["#ff0000", "#cc0000", "#dd0000", "#ee1100", "#ff3333", "#dd3344", "#cc3333",
                "#aa1122", "#bb2233"],
    "green":   ["#00ff00", "#00cc00", "#009900", "#008800", "#006600", "#33aa33", "#228833",
                "#44aa44", "#00bb00"],
    "blue":    ["#0000ff", "#0000cc", "#0033ff", "#3366ff", "#336699", "#556688", "#224488",
                "#3355aa", "#5577bb"],
    "yellow":  ["#ffff00", "#ffee00", "#ffdd00", "#ffe000", "#ffcc00"],
    "orange":  ["#ff8800", "#ff6600", "#ff7700", "#ff9900", "#ee7700"],
    "purple":  ["#800080", "#aa00aa", "#9900cc", "#770077"],
    "pink":    ["#ff69b4", "#ff88aa", "#ffaacc", "#ffbbcc", "#ee88bb"],
    "black":   ["#000000", "#111111", "#222222"],
    "white":   ["#ffffff", "#eeeeee", "#f0f0f0"],
    "gray":    ["#808080", "#888888", "#999999", "#aaaaaa", "#555555", "#667777", "#6688aa"],
    "grey":    ["#808080", "#888888", "#999999"],
    "brown":   ["#8b4513", "#a0522d", "#7b3f00", "#8b5000"],
    "cyan":    ["#00ffff", "#00cccc", "#00aaaa"],
    "magenta": ["#ff00ff", "#cc00cc"],
    "gold":    ["#ffd700", "#ffcc00", "#ffdd00"],
    "silver":  ["#c0c0c0", "#aaaaaa", "#bbbbbb"],
    "navy":    ["#000080", "#000066", "#112244"],
    "teal":    ["#008080", "#006666"],
}


def _color_name_to_hex_prefixes(name: str) -> List[str]:
    """Return a list of hex strings for a color name."""
    name = name.strip().lower()
    # Direct lookup
    if name in _COLOR_NAME_MAP:
        return _COLOR_NAME_MAP[name]
    # Partial match
    for key, vals in _COLOR_NAME_MAP.items():
        if key in name or name in key:
            return vals
    return []


def _fill_matches_color(fill: str, color_name: str) -> float:
    """
    Return a confidence score (0–1) that `fill` hex matches `color_name`.
    """
    if not fill or not fill.startswith("#"):
        return 0.0
    hex_candidates = _color_name_to_hex_prefixes(color_name)
    if not hex_candidates:
        return 0.0

    fill_lower = fill.strip().lower()

    # Exact match
    if fill_lower in [h.lower() for h in hex_candidates]:
        return 1.0

    # Prefix match (first 4 chars = #rgb)
    fill_prefix = fill_lower[:4]
    for h in hex_candidates:
        if h.lower()[:4] == fill_prefix:
            return 0.85

    # RGB distance
    try:
        fill_rgb = _hex_to_rgb(fill)
    except ValueError:
        return 0.0

    best = 0.0
    for h in hex_candidates:
        try:
            ref_rgb = _hex_to_rgb(h)
            dist = sum(abs(fill_rgb[i] - ref_rgb[i]) for i in range(3))
            # dist in [0, 765]; threshold at 100 for "close"
            score = max(0.0, 1.0 - dist / 200.0)
            best = max(best, score)
        except ValueError:
            continue

    return best


def _extract_color_from_text(text: str) -> Optional[str]:
    """
    Extract a color name or hex code from free text.

    Returns the first recognized color name or '#...' string.
    """
    # Hex code
    m = re.search(r"#[0-9a-fA-F]{3,6}", text)
    if m:
        return m.group(0)
    # Color name
    text_lower = text.lower()
    for name in _COLOR_NAME_MAP:
        if name in text_lower:
            return name
    return None


# ---------------------------------------------------------------------------
# Spatial matching
# ---------------------------------------------------------------------------

def _spatial_confidence(
    elem: SVGElement,
    spatial_hint: str,
    viewbox_w: float = 200.0,
    viewbox_h: float = 200.0,
) -> float:
    """
    Score how well an element's position matches a spatial description.
    Returns 0–1.
    """
    if elem.bbox is None:
        return 0.3  # unknown position

    cx, cy = _bbox_center(elem.bbox) or (viewbox_w / 2, viewbox_h / 2)
    hint = spatial_hint.lower()
    score = 0.5  # base: no information

    if "left" in hint:
        score = 1.0 - (cx / viewbox_w)
    elif "right" in hint:
        score = cx / viewbox_w
    if "top" in hint or "upper" in hint:
        score *= 1.0 - (cy / viewbox_h)
    elif "bottom" in hint or "lower" in hint:
        score *= cy / viewbox_h
    if "center" in hint:
        # near center of image
        dist_from_center = ((cx - viewbox_w / 2) ** 2 + (cy - viewbox_h / 2) ** 2) ** 0.5
        max_dist = (viewbox_w ** 2 + viewbox_h ** 2) ** 0.5 / 2
        score = 1.0 - dist_from_center / max_dist

    return max(0.0, min(1.0, score))


# ---------------------------------------------------------------------------
# Main localization logic
# ---------------------------------------------------------------------------

def localize(
    doc: SVGDocument,
    report: CriticReport,
    min_confidence: float = LOCALIZATION_THRESHOLD,
) -> List[LocalizedComponent]:
    """
    Map critic findings to SVG elements.

    Parameters
    ----------
    doc:
        Parsed SVGDocument (with assigned IDs).
    report:
        CriticReport from the critic module.
    min_confidence:
        Minimum confidence to include in the result.

    Returns
    -------
    List of LocalizedComponent objects, in descending confidence order.
    """
    if not report.errors:
        return []

    # Pre-compute candidate groups once
    candidate_groups = extract_candidate_groups(doc)

    results: List[LocalizedComponent] = []

    for error in report.errors:
        lc = _localize_one(doc, error, candidate_groups)
        if lc is not None and lc.confidence >= min_confidence:
            results.append(lc)
        elif lc is not None:
            logger.info(
                "Dropping low-confidence localization for '%s' (%.2f < %.2f)",
                error.component,
                lc.confidence,
                min_confidence,
            )

    # Sort by confidence desc
    results.sort(key=lambda x: x.confidence, reverse=True)
    return results


def _localize_one(
    doc: SVGDocument,
    error: ErrorFinding,
    candidate_groups: List[ElementGroup],
) -> Optional[LocalizedComponent]:
    """
    Attempt to localize one ErrorFinding.  Returns None if no evidence.
    """
    component = error.component.strip()
    description = error.description

    # --- 1. Explicit color match from error fields ---
    color_to_find: Optional[str] = None

    if error.type == "color_mismatch":
        # "expected" tells us the correct color; "observed" the wrong one
        # We want to find elements with the WRONG color (what was observed)
        # so we can change them to the expected color.
        color_to_find = _extract_color_from_text(error.observed)
        if color_to_find is None:
            color_to_find = _extract_color_from_text(description)
    else:
        # For other error types, try to find by component name → color
        color_to_find = _extract_color_from_text(description)
        if color_to_find is None:
            # Try expected color of the component
            color_to_find = _extract_color_from_text(error.expected)

    scored_elements: List[Tuple[SVGElement, float, str]] = []

    if color_to_find:
        for elem in doc.elements:
            if elem.fill:
                score = _fill_matches_color(elem.fill, color_to_find)
                if score > 0.3:
                    scored_elements.append((elem, score, f"fill {elem.fill} matches '{color_to_find}'"))

    # --- 2. Candidate group matching ---
    # If a candidate group's fill color matches expected/observed, boost it.
    for grp in candidate_groups:
        if grp.fill_color:
            if color_to_find:
                score = _fill_matches_color(grp.fill_color, color_to_find)
                if score > 0.3:
                    for eid in grp.element_ids:
                        if eid in doc.element_by_id:
                            elem = doc.element_by_id[eid]
                            # Add with boosted score
                            boost = score * grp.confidence
                            scored_elements.append(
                                (elem, min(0.95, boost + 0.1),
                                 f"Group '{grp.label}' fill matches '{color_to_find}'")
                            )

    # --- 3. Spatial filtering ---
    spatial_hint = ""
    for word in ["left", "right", "top", "bottom", "center", "upper", "lower"]:
        if word in description.lower() or word in component.lower():
            spatial_hint = word
            break

    if spatial_hint:
        filtered: List[Tuple[SVGElement, float, str]] = []
        for elem, score, reason in scored_elements:
            spatial_s = _spatial_confidence(elem, spatial_hint)
            combined = score * 0.6 + spatial_s * 0.4
            filtered.append((elem, combined, reason + f" + spatial({spatial_hint})"))
        scored_elements = filtered

    # --- 4. Deduplicate and pick best ---
    if not scored_elements:
        # No color or spatial evidence — return uncertain fallback
        if doc.elements:
            # Return all elements with very low confidence
            return LocalizedComponent(
                component=component,
                error=error,
                element_ids=[e.internal_id for e in doc.elements[:5]],
                confidence=0.15,
                reason="No color or spatial evidence found; returning first 5 elements",
                bbox=_union_bbox([e.bbox for e in doc.elements[:5]]),
            )
        return None

    # Deduplicate by element ID, keeping highest score per element
    best_by_id: Dict[str, Tuple[SVGElement, float, str]] = {}
    for elem, score, reason in scored_elements:
        if elem.internal_id not in best_by_id or score > best_by_id[elem.internal_id][1]:
            best_by_id[elem.internal_id] = (elem, score, reason)

    sorted_items = sorted(best_by_id.values(), key=lambda x: x[1], reverse=True)

    # Take elements within 70% of top score
    top_score = sorted_items[0][1]
    selected = [(e, s, r) for e, s, r in sorted_items if s >= top_score * 0.7]

    # Cap at 10 elements to avoid selecting everything
    selected = selected[:10]

    final_elements = [e for e, s, r in selected]
    final_confidence = min(0.95, top_score)

    # Reduce confidence if we have too many elements (uncertain assignment)
    if len(final_elements) > 5:
        final_confidence *= 0.7

    reasons = list(set(r for e, s, r in selected[:3]))
    reason_str = "; ".join(reasons)

    return LocalizedComponent(
        component=component,
        error=error,
        element_ids=[e.internal_id for e in final_elements],
        confidence=final_confidence,
        reason=reason_str,
        bbox=_union_bbox([e.bbox for e in final_elements]),
    )


# ---------------------------------------------------------------------------
# Edit proposal generation
# ---------------------------------------------------------------------------

def generate_edit_proposals(
    localized: List[LocalizedComponent],
    doc: SVGDocument,
) -> List["EditProposal"]:
    """
    Convert localized components into EditProposal objects.

    Only generates proposals for error types that have clear, safe edits.
    """
    from .editor import EditProposal

    proposals: List[EditProposal] = []

    for lc in localized:
        error = lc.error

        if error.type == "color_mismatch":
            # Determine the target color
            target_color = _extract_color_from_text(error.expected)
            if target_color is None:
                logger.info("Cannot determine target color for '%s'; skipping", lc.component)
                continue

            # Convert color name to hex if needed
            hex_color = _color_name_to_best_hex(target_color)
            if hex_color is None:
                logger.info("Cannot convert '%s' to hex; skipping", target_color)
                continue

            proposals.append(
                EditProposal(
                    operation="change_fill_color",
                    element_ids=lc.element_ids,
                    params={"color": hex_color},
                    rationale=(
                        f"Change {lc.component} fill to {hex_color} "
                        f"(expected '{error.expected}', observed '{error.observed}')"
                    ),
                    confidence=lc.confidence * 0.9,  # slight discount for indirect inference
                )
            )

        elif error.type == "attribute_error":
            # Check for opacity-related issues
            if "transparent" in error.expected.lower() or "opacity" in error.description.lower():
                target_opacity = 1.0 if "opaque" in error.expected.lower() else 0.5
                proposals.append(
                    EditProposal(
                        operation="change_opacity",
                        element_ids=lc.element_ids,
                        params={"opacity": target_opacity},
                        rationale=f"Adjust opacity for {lc.component}",
                        confidence=lc.confidence * 0.7,
                    )
                )

        # Other error types (missing_object, spatial_error, layout_error)
        # are too complex for automatic safe edits at this stage.
        # We log them and skip.
        elif error.type in ("missing_object", "spatial_error", "layout_error"):
            logger.info(
                "Error type '%s' for '%s' is not auto-correctable at this confidence level",
                error.type,
                lc.component,
            )

    return proposals


def _color_name_to_best_hex(name: str) -> Optional[str]:
    """Return the best representative hex for a color name."""
    # If already a hex, return as-is
    if name.startswith("#"):
        return name.lower()

    prefixes = _color_name_to_hex_prefixes(name)
    if not prefixes:
        return None

    # Return the most saturated/pure one (first in the list is typically the canonical)
    return prefixes[0]
