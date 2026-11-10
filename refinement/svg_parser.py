"""
refinement/svg_parser.py
========================
Parse OmniSVG-generated SVG files, assign stable internal identifiers,
extract geometric information, and support round-trip save after edits.

OmniSVG generates flat lists of <path> elements with no IDs, no groups,
and no semantic labels.  This module:

  1. Parses the SVG with xml.etree.ElementTree (no lxml required).
  2. Assigns stable IDs (path_0001, path_0002, ...) to every element that
     lacks one — without touching the visual output.
  3. Extracts per-element attributes (fill, stroke, opacity, transform,
     path data) and computes approximate axis-aligned bounding boxes from
     SVG path 'd' attributes.
  4. Identifies existing groups (<g>) and relationships.
  5. Returns a structured SVGDocument dataclass.
  6. Supports saving the (possibly modified) SVG back to disk.

Design constraints:
  - Preserve the original SVG structure as much as possible.
  - Do NOT assume one element = one semantic object.
  - When semantic ownership is uncertain, represent that uncertainty.
  - Support candidate component grouping by color, bbox overlap, and
    spatial proximity.
"""

from __future__ import annotations

import copy
import json
import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SVG namespace helpers
# ---------------------------------------------------------------------------

SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"

# Register namespaces so ET preserves them on write.
ET.register_namespace("", SVG_NS)
ET.register_namespace("xlink", XLINK_NS)


def _strip_ns(tag: str) -> str:
    """Return local name without namespace, e.g. '{http://...}path' → 'path'."""
    if tag.startswith("{"):
        return tag.split("}", 1)[1]
    return tag


# ---------------------------------------------------------------------------
# Bounding-box extraction from SVG 'd' attribute
# ---------------------------------------------------------------------------

_COORD_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def _extract_numbers(s: str) -> List[float]:
    """Extract all numbers from a string (handles 'M10.0 20.0 C ...')."""
    return [float(x) for x in _COORD_RE.findall(s)]


def _path_bbox(d: str) -> Optional[Tuple[float, float, float, float]]:
    """
    Compute an approximate axis-aligned bounding box from an SVG path 'd'.

    Returns (x_min, y_min, x_max, y_max) or None if 'd' is empty / invalid.

    This is intentionally approximate: it operates on raw coordinate values
    rather than evaluating Bézier curves, which is sufficient for localization
    heuristics.  The true bbox of a curve lies within the convex hull of its
    control points, so this gives a conservative outer bound.
    """
    if not d:
        return None
    nums = _extract_numbers(d)
    if len(nums) < 2:
        return None
    xs = nums[0::2]
    ys = nums[1::2]
    # guard against mismatched lengths
    n = min(len(xs), len(ys))
    if n == 0:
        return None
    xs = xs[:n]
    ys = ys[:n]
    return min(xs), min(ys), max(xs), max(ys)


def _bbox_area(bbox: Optional[Tuple[float, float, float, float]]) -> float:
    if bbox is None:
        return 0.0
    x0, y0, x1, y1 = bbox
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def _bbox_iou(
    a: Optional[Tuple[float, float, float, float]],
    b: Optional[Tuple[float, float, float, float]],
) -> float:
    """Intersection-over-union for two bboxes; returns 0 if either is None."""
    if a is None or b is None:
        return 0.0
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    if inter == 0.0:
        return 0.0
    union = _bbox_area(a) + _bbox_area(b) - inter
    return inter / union if union > 0 else 0.0


def _bbox_center(
    bbox: Optional[Tuple[float, float, float, float]]
) -> Optional[Tuple[float, float]]:
    if bbox is None:
        return None
    x0, y0, x1, y1 = bbox
    return (x0 + x1) / 2, (y0 + y1) / 2


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

SHAPE_TAGS = {"path", "rect", "circle", "ellipse", "line", "polyline", "polygon", "text", "use"}


@dataclass
class SVGElement:
    """Structured representation of one SVG element node."""

    internal_id: str              # Stable ID assigned by this module (path_0001 …)
    tag: str                      # Local tag name (path, rect, circle …)
    original_id: Optional[str]    # Original id="" attribute, if any
    fill: Optional[str]           # fill attribute value
    stroke: Optional[str]         # stroke attribute value
    stroke_width: Optional[str]   # stroke-width attribute value
    opacity: Optional[str]        # opacity attribute value
    fill_opacity: Optional[str]   # fill-opacity attribute value
    transform: Optional[str]      # transform attribute value
    path_data: Optional[str]      # 'd' attribute for path elements
    bbox: Optional[Tuple[float, float, float, float]]  # approx bbox (x0,y0,x1,y1)
    group_id: Optional[str]       # ID of enclosing <g> element, if any
    raw_attribs: Dict[str, str]   # All original attributes
    index: int                    # Position in document order


@dataclass
class SVGGroup:
    """Represents a <g> element with its child element IDs."""

    group_id: str
    original_id: Optional[str]
    children: List[str]           # internal_ids of child elements
    fill: Optional[str]
    opacity: Optional[str]
    transform: Optional[str]
    raw_attribs: Dict[str, str]


@dataclass
class SVGDocument:
    """
    Structured representation of an SVG document.

    Elements are stored in document order.  The `tree` field contains the
    live ElementTree so that edits applied to individual ET.Element objects
    are reflected in the tree before save.
    """

    source_path: Optional[Path]
    width: Optional[str]
    height: Optional[str]
    viewbox: Optional[str]
    elements: List[SVGElement]            # All shape elements in document order
    groups: List[SVGGroup]               # All <g> elements
    element_by_id: Dict[str, SVGElement] # internal_id → SVGElement
    node_by_id: Dict[str, ET.Element]    # internal_id → ET.Element (live)
    tree: ET.ElementTree                 # The underlying ElementTree
    root: ET.Element                     # SVG root element
    id_assigned: bool                    # True if we wrote any new id attributes

    # -----------------------------------------------------------------------
    # Convenience queries
    # -----------------------------------------------------------------------

    def elements_by_fill(self, fill: str) -> List[SVGElement]:
        """Return all elements whose fill matches (case-insensitive)."""
        f = fill.strip().lower()
        return [e for e in self.elements if e.fill and e.fill.strip().lower() == f]

    def elements_in_region(
        self,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        threshold: float = 0.1,
    ) -> List[SVGElement]:
        """
        Return elements whose bbox overlaps the given region by at least
        `threshold` IoU.
        """
        region = (x0, y0, x1, y1)
        return [
            e for e in self.elements
            if e.bbox is not None and _bbox_iou(e.bbox, region) >= threshold
        ]

    def spatial_region(self, description: str) -> Tuple[float, float, float, float]:
        """
        Convert a coarse spatial description to a bbox region within the
        SVG coordinate space.  Assumes a 200×200 viewBox (OmniSVG default).

        Supported tokens: left, right, top, bottom, center, upper, lower.
        Returns (x0, y0, x1, y1).
        """
        W, H = 200.0, 200.0
        # parse viewbox if available
        if self.viewbox:
            parts = self.viewbox.replace(",", " ").split()
            if len(parts) == 4:
                try:
                    W = float(parts[2])
                    H = float(parts[3])
                except ValueError:
                    pass

        desc = description.lower()
        x0, y0, x1, y1 = 0.0, 0.0, W, H

        if "left" in desc:
            x1 = W * 0.5
        if "right" in desc:
            x0 = W * 0.5
        if "top" in desc or "upper" in desc:
            y1 = H * 0.5
        if "bottom" in desc or "lower" in desc:
            y0 = H * 0.5
        if "center" in desc:
            x0 = W * 0.25
            x1 = W * 0.75
            y0 = H * 0.25
            y1 = H * 0.75

        return x0, y0, x1, y1

    def to_dict(self) -> dict:
        """Serialise document metadata (without the live tree) for logging."""
        return {
            "source_path": str(self.source_path) if self.source_path else None,
            "width": self.width,
            "height": self.height,
            "viewbox": self.viewbox,
            "element_count": len(self.elements),
            "group_count": len(self.groups),
            "elements": [
                {
                    "internal_id": e.internal_id,
                    "tag": e.tag,
                    "original_id": e.original_id,
                    "fill": e.fill,
                    "stroke": e.stroke,
                    "opacity": e.opacity,
                    "bbox": list(e.bbox) if e.bbox else None,
                    "group_id": e.group_id,
                    "index": e.index,
                }
                for e in self.elements
            ],
        }


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

_ID_COUNTER: Dict[str, int] = {}


def _next_id(prefix: str = "path") -> str:
    """Generate a stable sequential ID string."""
    _ID_COUNTER[prefix] = _ID_COUNTER.get(prefix, 0) + 1
    return f"{prefix}_{_ID_COUNTER[prefix]:04d}"


def reset_id_counter() -> None:
    """Reset the global ID counter (useful in tests)."""
    _ID_COUNTER.clear()


def parse_svg(
    source: Path | str,
    assign_ids: bool = True,
) -> SVGDocument:
    """
    Parse an SVG file and return a structured SVGDocument.

    Parameters
    ----------
    source:
        Path to the SVG file, or an SVG string.
    assign_ids:
        If True, assign stable `id` attributes to elements that lack them.
        The assigned IDs are stored in element.internal_id and also written
        into the live ET.Element so they are preserved on save.

    Returns
    -------
    SVGDocument
    """
    source_path: Optional[Path] = None

    if isinstance(source, (str, Path)) and Path(str(source)).exists():
        source_path = Path(source)
        tree = ET.parse(str(source_path))
        root = tree.getroot()
    else:
        # Treat as SVG string
        svg_str = str(source)
        root = ET.fromstring(svg_str)
        tree = ET.ElementTree(root)

    # Extract root attributes
    width = root.get("width")
    height = root.get("height")
    viewbox = root.get("viewBox") or root.get("viewbox")

    elements: List[SVGElement] = []
    groups: List[SVGGroup] = []
    element_by_id: Dict[str, SVGElement] = {}
    node_by_id: Dict[str, ET.Element] = {}
    id_assigned = False

    # Map ET.Element → internal_id for group bookkeeping
    _node_to_iid: Dict[int, str] = {}

    index = 0

    def _parse_node(node: ET.Element, parent_group_id: Optional[str] = None) -> None:
        nonlocal index, id_assigned
        local = _strip_ns(node.tag)

        if local == "g":
            # --- Group element ---
            orig_gid = node.get("id")
            if orig_gid is None and assign_ids:
                orig_gid = _next_id("g")
                node.set("id", orig_gid)
                id_assigned = True
            g_internal = orig_gid or _next_id("g")
            child_ids: List[str] = []
            grp = SVGGroup(
                group_id=g_internal,
                original_id=node.get("id"),
                children=child_ids,
                fill=node.get("fill"),
                opacity=node.get("opacity"),
                transform=node.get("transform"),
                raw_attribs=dict(node.attrib),
            )
            groups.append(grp)
            for child in node:
                _parse_node(child, g_internal)
            # Collect child internal IDs after recursion
            for child in node:
                cid = _node_to_iid.get(id(child))
                if cid:
                    child_ids.append(cid)
            return

        if local not in SHAPE_TAGS:
            # Recurse into unknown containers (defs, symbol, etc.)
            for child in node:
                _parse_node(child, parent_group_id)
            return

        # --- Shape element ---
        orig_id = node.get("id")
        if orig_id is None and assign_ids:
            new_id = _next_id(local)
            node.set("id", new_id)
            internal_id = new_id
            id_assigned = True
        else:
            internal_id = orig_id or _next_id(local)

        _node_to_iid[id(node)] = internal_id

        d = node.get("d", "")
        bbox = _path_bbox(d) if local == "path" else _element_bbox(node, local)

        elem = SVGElement(
            internal_id=internal_id,
            tag=local,
            original_id=orig_id,
            fill=node.get("fill"),
            stroke=node.get("stroke"),
            stroke_width=node.get("stroke-width"),
            opacity=node.get("opacity"),
            fill_opacity=node.get("fill-opacity"),
            transform=node.get("transform"),
            path_data=d if local == "path" else None,
            bbox=bbox,
            group_id=parent_group_id,
            raw_attribs=dict(node.attrib),
            index=index,
        )
        index += 1
        elements.append(elem)
        element_by_id[internal_id] = elem
        node_by_id[internal_id] = node

    for child in root:
        _parse_node(child)

    doc = SVGDocument(
        source_path=source_path,
        width=width,
        height=height,
        viewbox=viewbox,
        elements=elements,
        groups=groups,
        element_by_id=element_by_id,
        node_by_id=node_by_id,
        tree=tree,
        root=root,
        id_assigned=id_assigned,
    )

    logger.info(
        "Parsed SVG: %d elements, %d groups, id_assigned=%s",
        len(elements),
        len(groups),
        id_assigned,
    )
    return doc


def _element_bbox(
    node: ET.Element, tag: str
) -> Optional[Tuple[float, float, float, float]]:
    """Compute bbox for non-path shape elements."""
    try:
        if tag == "rect":
            x = float(node.get("x", 0))
            y = float(node.get("y", 0))
            w = float(node.get("width", 0))
            h = float(node.get("height", 0))
            return x, y, x + w, y + h
        if tag == "circle":
            cx = float(node.get("cx", 0))
            cy = float(node.get("cy", 0))
            r = float(node.get("r", 0))
            return cx - r, cy - r, cx + r, cy + r
        if tag == "ellipse":
            cx = float(node.get("cx", 0))
            cy = float(node.get("cy", 0))
            rx = float(node.get("rx", 0))
            ry = float(node.get("ry", 0))
            return cx - rx, cy - ry, cx + rx, cy + ry
        if tag in ("line",):
            x1 = float(node.get("x1", 0))
            y1 = float(node.get("y1", 0))
            x2 = float(node.get("x2", 0))
            y2 = float(node.get("y2", 0))
            return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)
        if tag in ("polyline", "polygon"):
            pts = node.get("points", "")
            nums = _extract_numbers(pts)
            if len(nums) < 2:
                return None
            xs = nums[0::2]
            ys = nums[1::2]
            return min(xs), min(ys), max(xs), max(ys)
    except (ValueError, TypeError):
        pass
    return None


# ---------------------------------------------------------------------------
# Group-candidate extraction (spatial/color clustering)
# ---------------------------------------------------------------------------


@dataclass
class ElementGroup:
    """
    A candidate grouping of SVG elements that may correspond to one
    semantic object.  Confidence reflects how certain we are.
    """

    label: str                    # Human-readable label (e.g. "yellow_cluster_1")
    element_ids: List[str]        # internal_ids
    fill_color: Optional[str]     # Dominant fill color, if uniform
    bbox: Optional[Tuple[float, float, float, float]]
    confidence: float             # 0–1; lower = more uncertain
    reason: str                   # Why these elements were grouped


def _union_bbox(
    bboxes: List[Optional[Tuple[float, float, float, float]]]
) -> Optional[Tuple[float, float, float, float]]:
    valid = [b for b in bboxes if b is not None]
    if not valid:
        return None
    x0 = min(b[0] for b in valid)
    y0 = min(b[1] for b in valid)
    x1 = max(b[2] for b in valid)
    y1 = max(b[3] for b in valid)
    return x0, y0, x1, y1


def extract_candidate_groups(
    doc: SVGDocument,
    color_proximity_threshold: float = 30.0,
    spatial_proximity_threshold: float = 20.0,
) -> List[ElementGroup]:
    """
    Extract candidate element groups from a parsed SVGDocument.

    Strategy (in priority order):
    1. Existing <g> groups → high confidence.
    2. Elements sharing the same fill color → medium confidence.
    3. Elements with similar colors (within threshold) that are spatially
       adjacent → lower confidence.

    Returns a list of ElementGroup objects.  A single element may appear in
    multiple groups (overlap is allowed because the semantic assignment is
    uncertain).
    """
    groups: List[ElementGroup] = []

    # --- Strategy 1: existing <g> groups ---
    for g in doc.groups:
        if not g.children:
            continue
        child_elems = [doc.element_by_id[c] for c in g.children if c in doc.element_by_id]
        bbox = _union_bbox([e.bbox for e in child_elems])
        fills = set(e.fill for e in child_elems if e.fill)
        fill = next(iter(fills)) if len(fills) == 1 else None
        groups.append(
            ElementGroup(
                label=f"group_{g.group_id}",
                element_ids=g.children,
                fill_color=fill,
                bbox=bbox,
                confidence=0.85,
                reason=f"Existing <g id='{g.group_id}'>",
            )
        )

    # --- Strategy 2: same fill color ---
    by_fill: Dict[str, List[SVGElement]] = {}
    for e in doc.elements:
        f = (e.fill or "none").strip().lower()
        by_fill.setdefault(f, []).append(e)

    for fill_color, elems in by_fill.items():
        if fill_color in ("none", "", "transparent"):
            continue
        if len(elems) < 1:
            continue
        ids = [e.internal_id for e in elems]
        bbox = _union_bbox([e.bbox for e in elems])
        groups.append(
            ElementGroup(
                label=f"fill_{fill_color.replace('#', 'color_')}",
                element_ids=ids,
                fill_color=fill_color,
                bbox=bbox,
                confidence=0.60 if len(elems) > 1 else 0.50,
                reason=f"Same fill color: {fill_color}",
            )
        )

    # --- Strategy 3: spatially adjacent elements ---
    # For every pair, if bboxes are close, add a low-confidence group.
    # We skip this if there are too many elements to avoid O(n²) explosion.
    if len(doc.elements) <= 100:
        for i, a in enumerate(doc.elements):
            for b in doc.elements[i + 1:]:
                if a.bbox is None or b.bbox is None:
                    continue
                # Distance between bbox centers
                ca = _bbox_center(a.bbox)
                cb = _bbox_center(b.bbox)
                if ca is None or cb is None:
                    continue
                dist = ((ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2) ** 0.5
                if dist < spatial_proximity_threshold:
                    # Check they share similar (but not necessarily identical) color
                    if a.fill and b.fill:
                        try:
                            ca_rgb = _hex_to_rgb(a.fill)
                            cb_rgb = _hex_to_rgb(b.fill)
                            cdist = sum(abs(ca_rgb[k] - cb_rgb[k]) for k in range(3))
                            if cdist < color_proximity_threshold * 3:
                                groups.append(
                                    ElementGroup(
                                        label=f"spatial_{a.internal_id}_{b.internal_id}",
                                        element_ids=[a.internal_id, b.internal_id],
                                        fill_color=None,
                                        bbox=_union_bbox([a.bbox, b.bbox]),
                                        confidence=0.35,
                                        reason=(
                                            f"Spatial proximity ({dist:.1f}px) "
                                            f"+ similar color"
                                        ),
                                    )
                                )
                        except ValueError:
                            pass

    return groups


def _hex_to_rgb(color: str) -> Tuple[int, int, int]:
    """Parse '#rrggbb' or '#rgb' to (r, g, b)."""
    c = color.strip().lstrip("#")
    if len(c) == 3:
        c = "".join(ch * 2 for ch in c)
    if len(c) != 6:
        raise ValueError(f"Cannot parse color: {color!r}")
    r = int(c[0:2], 16)
    g = int(c[2:4], 16)
    b = int(c[4:6], 16)
    return r, g, b


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------


def save_svg(doc: SVGDocument, dest: Path | str) -> None:
    """
    Write the (possibly modified) SVGDocument back to disk.

    The ElementTree in `doc.tree` reflects all edits made to its nodes.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    doc.tree.write(
        str(dest),
        encoding="unicode",
        xml_declaration=False,
    )
    logger.info("Saved SVG to %s", dest)


def svg_to_string(doc: SVGDocument) -> str:
    """Serialise the document to an SVG string."""
    return ET.tostring(doc.root, encoding="unicode")


# ---------------------------------------------------------------------------
# Quick CLI check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.DEBUG)
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if path:
        d = parse_svg(path)
        print(json.dumps(d.to_dict(), indent=2))
        groups = extract_candidate_groups(d)
        print(f"\nCandidate groups: {len(groups)}")
        for g in groups[:10]:
            print(f"  {g.label}: ids={g.element_ids[:5]} conf={g.confidence:.2f} reason={g.reason}")
    else:
        print("Usage: python svg_parser.py <path.svg>")
