"""
refinement/editor.py
====================
Constrained, audited SVG element editing.

This module implements a safe editing interface for SVGDocument objects.
Every operation:
  1. Verifies that target element IDs exist in the document.
  2. Applies the edit to the live ET.Element.
  3. Updates the corresponding SVGElement dataclass.
  4. Records the modification in an append-only edit log.
  5. Preserves ALL unaffected elements.
  6. Validates the modified SVG before returning.

Supported operations:
  - change_fill_color     : change the fill of specified element(s)
  - change_stroke         : change stroke color and/or width
  - change_opacity        : change opacity / fill-opacity
  - change_transform      : set or replace the transform attribute
  - replace_element       : replace path data ('d') of a path element
  - insert_element        : add a new path element to the SVG root
  - remove_element        : remove a path element (use with caution)

Design constraints:
  - Before any edit, verify element IDs exist.
  - Preserve the original SVG as a separate file (done at pipeline level).
  - Reject malformed SVGs and invalid operations.
  - If an operation is ambiguous or unsafe, report the problem instead of
    silently modifying unrelated elements.
  - Do NOT modify elements merely because they share a similar color.
"""

from __future__ import annotations

import copy
import json
import logging
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .svg_parser import SVGDocument, save_svg, svg_to_string, parse_svg
from .renderer import validate_svg

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Edit log
# ---------------------------------------------------------------------------

@dataclass
class EditRecord:
    """One entry in the edit log."""
    timestamp: float
    operation: str
    element_ids: List[str]
    before_values: Dict[str, Any]
    after_values: Dict[str, Any]
    success: bool
    error: Optional[str]


@dataclass
class EditLog:
    """Append-only log of all edits applied to an SVGDocument."""
    records: List[EditRecord] = field(default_factory=list)

    def add(self, record: EditRecord) -> None:
        self.records.append(record)

    def to_dict(self) -> list:
        return [
            {
                "timestamp": r.timestamp,
                "operation": r.operation,
                "element_ids": r.element_ids,
                "before": r.before_values,
                "after": r.after_values,
                "success": r.success,
                "error": r.error,
            }
            for r in self.records
        ]

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Edit result
# ---------------------------------------------------------------------------

@dataclass
class EditResult:
    """Result of an editing operation."""
    success: bool
    modified_count: int
    error: Optional[str]
    records: List[EditRecord] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Core editor
# ---------------------------------------------------------------------------

class SVGEditor:
    """
    Safe, audited editor for SVGDocument objects.

    Usage
    -----
    doc = parse_svg("input.svg")
    editor = SVGEditor(doc)
    result = editor.change_fill_color(["path_0003", "path_0004"], "#ffff00")
    editor.save("output.svg")
    editor.edit_log.save("edits.json")
    """

    def __init__(self, doc: SVGDocument) -> None:
        self.doc = doc
        self.edit_log = EditLog()

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _resolve_ids(self, element_ids: List[str]) -> tuple[List[str], Optional[str]]:
        """
        Verify that all provided IDs exist in the document.

        Returns (valid_ids, error_message).  If any ID is missing,
        error_message is set and valid_ids is empty.
        """
        missing = [eid for eid in element_ids if eid not in self.doc.element_by_id]
        if missing:
            return [], f"Element IDs not found in document: {missing}"
        return list(element_ids), None

    def _record(
        self,
        operation: str,
        element_ids: List[str],
        before: Dict[str, Any],
        after: Dict[str, Any],
        success: bool,
        error: Optional[str] = None,
    ) -> EditRecord:
        rec = EditRecord(
            timestamp=time.time(),
            operation=operation,
            element_ids=element_ids,
            before_values=before,
            after_values=after,
            success=success,
            error=error,
        )
        self.edit_log.add(rec)
        return rec

    def _validate_current(self) -> tuple[bool, str]:
        """Validate the current state of the document."""
        svg_str = svg_to_string(self.doc)
        return validate_svg(svg_str)

    # -----------------------------------------------------------------------
    # Operations
    # -----------------------------------------------------------------------

    def change_fill_color(
        self,
        element_ids: List[str],
        new_color: str,
        also_change_opacity: Optional[float] = None,
    ) -> EditResult:
        """
        Change the fill color of the specified elements.

        Parameters
        ----------
        element_ids:
            Internal IDs of elements to modify.
        new_color:
            CSS color string (e.g. "#ffff00", "red", "none").
        also_change_opacity:
            If provided, also set fill-opacity to this value.
        """
        valid_ids, err = self._resolve_ids(element_ids)
        if err:
            rec = self._record("change_fill_color", element_ids, {}, {}, False, err)
            return EditResult(success=False, modified_count=0, error=err, records=[rec])

        before = {eid: self.doc.element_by_id[eid].fill for eid in valid_ids}
        modified = 0

        for eid in valid_ids:
            node = self.doc.node_by_id[eid]
            elem = self.doc.element_by_id[eid]
            node.set("fill", new_color)
            elem.fill = new_color
            if also_change_opacity is not None:
                node.set("fill-opacity", str(also_change_opacity))
                elem.fill_opacity = str(also_change_opacity)
            modified += 1

        after = {eid: new_color for eid in valid_ids}
        rec = self._record("change_fill_color", valid_ids, before, after, True)
        logger.info("change_fill_color: %d elements → %s", modified, new_color)

        ok, msg = self._validate_current()
        if not ok:
            logger.error("Post-edit validation failed: %s", msg)
            return EditResult(success=False, modified_count=modified,
                              error=f"Post-edit validation: {msg}", records=[rec])

        return EditResult(success=True, modified_count=modified, error=None, records=[rec])

    def change_stroke(
        self,
        element_ids: List[str],
        stroke_color: Optional[str] = None,
        stroke_width: Optional[str] = None,
    ) -> EditResult:
        """Change stroke color and/or width of specified elements."""
        valid_ids, err = self._resolve_ids(element_ids)
        if err:
            rec = self._record("change_stroke", element_ids, {}, {}, False, err)
            return EditResult(success=False, modified_count=0, error=err, records=[rec])

        if stroke_color is None and stroke_width is None:
            err = "change_stroke: at least one of stroke_color or stroke_width must be set"
            rec = self._record("change_stroke", element_ids, {}, {}, False, err)
            return EditResult(success=False, modified_count=0, error=err, records=[rec])

        before: Dict[str, Any] = {}
        for eid in valid_ids:
            elem = self.doc.element_by_id[eid]
            before[eid] = {"stroke": elem.stroke, "stroke_width": elem.stroke_width}

        modified = 0
        for eid in valid_ids:
            node = self.doc.node_by_id[eid]
            elem = self.doc.element_by_id[eid]
            if stroke_color is not None:
                node.set("stroke", stroke_color)
                elem.stroke = stroke_color
            if stroke_width is not None:
                node.set("stroke-width", stroke_width)
                elem.stroke_width = stroke_width
            modified += 1

        after: Dict[str, Any] = {}
        for eid in valid_ids:
            elem = self.doc.element_by_id[eid]
            after[eid] = {"stroke": elem.stroke, "stroke_width": elem.stroke_width}

        rec = self._record("change_stroke", valid_ids, before, after, True)
        logger.info("change_stroke: %d elements", modified)

        ok, msg = self._validate_current()
        if not ok:
            return EditResult(success=False, modified_count=modified,
                              error=f"Post-edit validation: {msg}", records=[rec])

        return EditResult(success=True, modified_count=modified, error=None, records=[rec])

    def change_opacity(
        self,
        element_ids: List[str],
        opacity: Optional[float] = None,
        fill_opacity: Optional[float] = None,
    ) -> EditResult:
        """Change opacity and/or fill-opacity of specified elements."""
        valid_ids, err = self._resolve_ids(element_ids)
        if err:
            rec = self._record("change_opacity", element_ids, {}, {}, False, err)
            return EditResult(success=False, modified_count=0, error=err, records=[rec])

        if opacity is None and fill_opacity is None:
            err = "change_opacity: at least one of opacity or fill_opacity must be set"
            rec = self._record("change_opacity", element_ids, {}, {}, False, err)
            return EditResult(success=False, modified_count=0, error=err, records=[rec])

        before: Dict[str, Any] = {
            eid: {
                "opacity": self.doc.element_by_id[eid].opacity,
                "fill_opacity": self.doc.element_by_id[eid].fill_opacity,
            }
            for eid in valid_ids
        }
        modified = 0

        for eid in valid_ids:
            node = self.doc.node_by_id[eid]
            elem = self.doc.element_by_id[eid]
            if opacity is not None:
                val = f"{max(0.0, min(1.0, opacity)):.4f}"
                node.set("opacity", val)
                elem.opacity = val
            if fill_opacity is not None:
                val = f"{max(0.0, min(1.0, fill_opacity)):.4f}"
                node.set("fill-opacity", val)
                elem.fill_opacity = val
            modified += 1

        after: Dict[str, Any] = {
            eid: {
                "opacity": self.doc.element_by_id[eid].opacity,
                "fill_opacity": self.doc.element_by_id[eid].fill_opacity,
            }
            for eid in valid_ids
        }

        rec = self._record("change_opacity", valid_ids, before, after, True)
        ok, msg = self._validate_current()
        if not ok:
            return EditResult(success=False, modified_count=modified,
                              error=f"Post-edit validation: {msg}", records=[rec])

        return EditResult(success=True, modified_count=modified, error=None, records=[rec])

    def change_transform(
        self,
        element_ids: List[str],
        transform: str,
    ) -> EditResult:
        """Set or replace the transform attribute of specified elements."""
        valid_ids, err = self._resolve_ids(element_ids)
        if err:
            rec = self._record("change_transform", element_ids, {}, {}, False, err)
            return EditResult(success=False, modified_count=0, error=err, records=[rec])

        before = {eid: self.doc.element_by_id[eid].transform for eid in valid_ids}
        modified = 0

        for eid in valid_ids:
            node = self.doc.node_by_id[eid]
            elem = self.doc.element_by_id[eid]
            node.set("transform", transform)
            elem.transform = transform
            modified += 1

        rec = self._record("change_transform", valid_ids, before, {eid: transform for eid in valid_ids}, True)
        ok, msg = self._validate_current()
        if not ok:
            return EditResult(success=False, modified_count=modified,
                              error=f"Post-edit validation: {msg}", records=[rec])

        return EditResult(success=True, modified_count=modified, error=None, records=[rec])

    def replace_path_data(
        self,
        element_id: str,
        new_d: str,
    ) -> EditResult:
        """
        Replace the 'd' attribute (path data) of a single path element.

        This is a high-risk operation — only use when you have a valid
        replacement path.
        """
        valid_ids, err = self._resolve_ids([element_id])
        if err:
            rec = self._record("replace_path_data", [element_id], {}, {}, False, err)
            return EditResult(success=False, modified_count=0, error=err, records=[rec])

        elem = self.doc.element_by_id[element_id]
        if elem.tag != "path":
            err = f"replace_path_data: element {element_id!r} is a <{elem.tag}>, not <path>"
            rec = self._record("replace_path_data", [element_id], {}, {}, False, err)
            return EditResult(success=False, modified_count=0, error=err, records=[rec])

        before = {"d": elem.path_data}
        node = self.doc.node_by_id[element_id]
        node.set("d", new_d)
        elem.path_data = new_d

        rec = self._record("replace_path_data", [element_id], before, {"d": new_d}, True)
        ok, msg = self._validate_current()
        if not ok:
            return EditResult(success=False, modified_count=1,
                              error=f"Post-edit validation: {msg}", records=[rec])

        return EditResult(success=True, modified_count=1, error=None, records=[rec])

    def insert_element(
        self,
        tag: str,
        attribs: Dict[str, str],
        position: int = -1,
    ) -> EditResult:
        """
        Insert a new SVG element into the root of the document.

        Parameters
        ----------
        tag:
            Local tag name, e.g. "path", "circle".
        attribs:
            Attribute dictionary (should include 'd', 'fill', etc.).
        position:
            Insertion index in the root's child list (-1 = append).
        """
        from .svg_parser import _next_id, _path_bbox, _element_bbox, SVGElement

        new_id = attribs.get("id") or _next_id(tag)
        attribs["id"] = new_id

        ns_tag = f"{{{ET.QName(self.doc.root.tag).namespace}}}{tag}" \
            if self.doc.root.tag.startswith("{") else tag

        node = ET.Element(ns_tag, attribs)

        children = list(self.doc.root)
        if position == -1 or position >= len(children):
            self.doc.root.append(node)
        else:
            self.doc.root.insert(position, node)

        d = attribs.get("d", "")
        bbox = _path_bbox(d) if tag == "path" else _element_bbox(node, tag)

        new_elem = SVGElement(
            internal_id=new_id,
            tag=tag,
            original_id=None,
            fill=attribs.get("fill"),
            stroke=attribs.get("stroke"),
            stroke_width=attribs.get("stroke-width"),
            opacity=attribs.get("opacity"),
            fill_opacity=attribs.get("fill-opacity"),
            transform=attribs.get("transform"),
            path_data=d if tag == "path" else None,
            bbox=bbox,
            group_id=None,
            raw_attribs=dict(attribs),
            index=len(self.doc.elements),
        )

        self.doc.elements.append(new_elem)
        self.doc.element_by_id[new_id] = new_elem
        self.doc.node_by_id[new_id] = node

        rec = self._record("insert_element", [new_id], {}, {"tag": tag, "attribs": attribs}, True)
        ok, msg = self._validate_current()
        if not ok:
            return EditResult(success=False, modified_count=1,
                              error=f"Post-edit validation: {msg}", records=[rec])

        logger.info("Inserted <%s id='%s'>", tag, new_id)
        return EditResult(success=True, modified_count=1, error=None, records=[rec])

    def remove_element(
        self,
        element_id: str,
        require_confirmation: bool = True,
    ) -> EditResult:
        """
        Remove an element from the SVG.

        This is a destructive operation.  Set require_confirmation=False only
        when you are certain the element should be deleted.
        """
        if require_confirmation:
            logger.warning(
                "remove_element called for %s; set require_confirmation=False to proceed",
                element_id,
            )
            err = (
                f"remove_element for {element_id!r} requires explicit confirmation. "
                "Set require_confirmation=False."
            )
            rec = self._record("remove_element", [element_id], {}, {}, False, err)
            return EditResult(success=False, modified_count=0, error=err, records=[rec])

        valid_ids, err = self._resolve_ids([element_id])
        if err:
            rec = self._record("remove_element", [element_id], {}, {}, False, err)
            return EditResult(success=False, modified_count=0, error=err, records=[rec])

        node = self.doc.node_by_id[element_id]
        elem = self.doc.element_by_id[element_id]
        before = {"tag": elem.tag, "fill": elem.fill, "d": elem.path_data}

        # Remove from tree
        parent = None
        for p in self.doc.root.iter():
            if node in list(p):
                parent = p
                break
        if parent is None:
            parent = self.doc.root
        try:
            parent.remove(node)
        except ValueError as e:
            err = f"Could not remove element from tree: {e}"
            rec = self._record("remove_element", [element_id], before, {}, False, err)
            return EditResult(success=False, modified_count=0, error=err, records=[rec])

        # Remove from document
        del self.doc.element_by_id[element_id]
        del self.doc.node_by_id[element_id]
        self.doc.elements = [e for e in self.doc.elements if e.internal_id != element_id]

        rec = self._record("remove_element", [element_id], before, {}, True)
        ok, msg = self._validate_current()
        if not ok:
            return EditResult(success=False, modified_count=1,
                              error=f"Post-edit validation: {msg}", records=[rec])

        logger.info("Removed element %s", element_id)
        return EditResult(success=True, modified_count=1, error=None, records=[rec])

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------

    def save(self, dest: Path | str) -> None:
        """Save the current state of the document."""
        save_svg(self.doc, Path(dest))

    def to_svg_string(self) -> str:
        """Return the current document as an SVG string."""
        return svg_to_string(self.doc)


# ---------------------------------------------------------------------------
# High-level edit proposal application
# ---------------------------------------------------------------------------

@dataclass
class EditProposal:
    """
    A structured proposal for one editing action.

    Fields mirror the SVGEditor method signatures so the pipeline can apply
    them programmatically.
    """
    operation: str
    element_ids: List[str]
    params: Dict[str, Any]
    rationale: str
    confidence: float


def apply_edit_proposals(
    doc: SVGDocument,
    proposals: List[EditProposal],
    min_confidence: float = 0.4,
) -> tuple[SVGEditor, List[EditResult]]:
    """
    Apply a list of EditProposal objects to a deep copy of `doc`.

    Only proposals with confidence >= min_confidence are applied.
    Returns the editor (with the modified doc copy) and result list.

    The original `doc` is NOT modified.
    """
    from .svg_parser import parse_svg

    # Deep-copy the document so we don't touch the original
    svg_str = svg_to_string(doc)
    doc_copy = parse_svg(svg_str)

    editor = SVGEditor(doc_copy)
    results: List[EditResult] = []

    for proposal in proposals:
        if proposal.confidence < min_confidence:
            logger.info(
                "Skipping proposal %s (confidence %.2f < %.2f)",
                proposal.operation,
                proposal.confidence,
                min_confidence,
            )
            continue

        op = proposal.operation
        eids = proposal.element_ids
        p = proposal.params

        try:
            if op == "change_fill_color":
                res = editor.change_fill_color(eids, p["color"],
                                               p.get("fill_opacity"))
            elif op == "change_stroke":
                res = editor.change_stroke(eids, p.get("stroke_color"),
                                           p.get("stroke_width"))
            elif op == "change_opacity":
                res = editor.change_opacity(eids, p.get("opacity"),
                                            p.get("fill_opacity"))
            elif op == "change_transform":
                res = editor.change_transform(eids, p["transform"])
            elif op == "replace_path_data":
                res = editor.replace_path_data(eids[0], p["d"])
            elif op == "insert_element":
                res = editor.insert_element(p["tag"], p["attribs"], p.get("position", -1))
            elif op == "remove_element":
                res = editor.remove_element(eids[0], require_confirmation=False)
            else:
                err = f"Unknown operation: {op!r}"
                res = EditResult(success=False, modified_count=0, error=err)
                logger.warning(err)
        except Exception as e:
            err = f"Exception in {op}: {e}"
            res = EditResult(success=False, modified_count=0, error=err)
            logger.exception("Edit proposal failed: %s", err)

        results.append(res)

    return editor, results
