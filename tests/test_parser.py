"""
tests/test_parser.py — Unit tests for refinement/svg_parser.py
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from refinement.svg_parser import (
    parse_svg,
    save_svg,
    svg_to_string,
    extract_candidate_groups,
    reset_id_counter,
    _path_bbox,
    _hex_to_rgb,
    SVGDocument,
)
from tests.conftest import (
    SIMPLE_RED_HEART_SVG,
    TWO_COLORED_SVG,
    MULTI_PATH_SVG,
    MALFORMED_SVG,
    EMPTY_SVG,
    GROUP_SVG,
)


@pytest.fixture(autouse=True)
def reset_ids():
    reset_id_counter()


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

class TestParsing:
    def test_parse_simple_svg(self):
        doc = parse_svg(SIMPLE_RED_HEART_SVG)
        assert isinstance(doc, SVGDocument)
        assert len(doc.elements) == 1

    def test_parse_two_colored_svg(self):
        doc = parse_svg(TWO_COLORED_SVG)
        assert len(doc.elements) == 2

    def test_parse_multi_path_svg(self):
        doc = parse_svg(MULTI_PATH_SVG)
        assert len(doc.elements) == 4

    def test_parse_empty_svg(self):
        doc = parse_svg(EMPTY_SVG)
        assert len(doc.elements) == 0
        assert len(doc.groups) == 0

    def test_parse_group_svg(self):
        doc = parse_svg(GROUP_SVG)
        assert len(doc.groups) >= 2
        assert len(doc.elements) == 3

    def test_parse_viewbox(self):
        doc = parse_svg(SIMPLE_RED_HEART_SVG)
        assert doc.viewbox is not None
        assert "200" in doc.viewbox

    def test_parse_width_height(self):
        doc = parse_svg(SIMPLE_RED_HEART_SVG)
        assert doc.width == "200px"
        assert doc.height == "200px"

    def test_fill_extracted(self):
        doc = parse_svg(SIMPLE_RED_HEART_SVG)
        assert doc.elements[0].fill == "#dd3344"

    def test_path_tag(self):
        doc = parse_svg(SIMPLE_RED_HEART_SVG)
        assert doc.elements[0].tag == "path"


# ---------------------------------------------------------------------------
# ID assignment
# ---------------------------------------------------------------------------

class TestIDAssignment:
    def test_ids_assigned_when_missing(self):
        doc = parse_svg(SIMPLE_RED_HEART_SVG, assign_ids=True)
        elem = doc.elements[0]
        assert elem.internal_id is not None
        assert elem.internal_id.startswith("path_")

    def test_ids_stable_sequential(self):
        reset_id_counter()
        doc = parse_svg(TWO_COLORED_SVG, assign_ids=True)
        ids = [e.internal_id for e in doc.elements]
        assert ids[0] < ids[1]  # lexicographic; path_0001 < path_0002

    def test_original_id_preserved(self):
        svg = '<svg xmlns="http://www.w3.org/2000/svg"><path id="my_path" fill="red" d="M0 0 L10 0"/></svg>'
        doc = parse_svg(svg)
        assert doc.elements[0].original_id == "my_path"
        assert doc.elements[0].internal_id == "my_path"

    def test_no_id_change_when_disabled(self):
        doc = parse_svg(SIMPLE_RED_HEART_SVG, assign_ids=False)
        elem = doc.elements[0]
        # internal_id is generated but NOT written to the XML node
        assert elem.original_id is None

    def test_element_by_id_populated(self):
        doc = parse_svg(TWO_COLORED_SVG)
        for elem in doc.elements:
            assert elem.internal_id in doc.element_by_id

    def test_node_by_id_populated(self):
        doc = parse_svg(TWO_COLORED_SVG)
        for elem in doc.elements:
            assert elem.internal_id in doc.node_by_id
            assert isinstance(doc.node_by_id[elem.internal_id], ET.Element)

    def test_id_assigned_flag_set(self):
        doc = parse_svg(SIMPLE_RED_HEART_SVG, assign_ids=True)
        assert doc.id_assigned is True

    def test_id_assigned_flag_not_set_when_ids_exist(self):
        svg = '<svg xmlns="http://www.w3.org/2000/svg"><path id="existing" fill="red" d="M0 0"/></svg>'
        doc = parse_svg(svg, assign_ids=True)
        assert doc.id_assigned is False  # no new IDs needed


# ---------------------------------------------------------------------------
# SVG structure preservation
# ---------------------------------------------------------------------------

class TestStructurePreservation:
    def test_svg_to_string_contains_svg_tag(self):
        doc = parse_svg(SIMPLE_RED_HEART_SVG)
        s = svg_to_string(doc)
        assert "<svg" in s

    def test_svg_to_string_preserves_paths(self):
        doc = parse_svg(TWO_COLORED_SVG)
        s = svg_to_string(doc)
        assert "ff0000" in s
        assert "ffdd00" in s

    def test_save_and_reload(self, tmp_path):
        doc = parse_svg(SIMPLE_RED_HEART_SVG)
        out = tmp_path / "out.svg"
        save_svg(doc, out)
        assert out.exists()
        doc2 = parse_svg(out)
        assert len(doc2.elements) == 1

    def test_parse_from_file(self, tmp_path):
        p = tmp_path / "test.svg"
        p.write_text(SIMPLE_RED_HEART_SVG, encoding="utf-8")
        doc = parse_svg(p)
        assert len(doc.elements) == 1
        assert doc.source_path == p


# ---------------------------------------------------------------------------
# Geometric information
# ---------------------------------------------------------------------------

class TestBboxExtraction:
    def test_bbox_extracted_for_paths(self):
        doc = parse_svg(MULTI_PATH_SVG)
        for elem in doc.elements:
            if elem.path_data:
                # bbox may be None for empty/trivial paths
                if elem.path_data.strip():
                    assert elem.bbox is not None, f"No bbox for {elem.internal_id}"

    def test_bbox_values_sensible(self):
        doc = parse_svg(TWO_COLORED_SVG)
        for elem in doc.elements:
            if elem.bbox:
                x0, y0, x1, y1 = elem.bbox
                assert x0 <= x1
                assert y0 <= y1

    def test_path_bbox_helper(self):
        d = "M10 20 L90 80 L50 100 Z"
        bbox = _path_bbox(d)
        assert bbox is not None
        x0, y0, x1, y1 = bbox
        assert x0 == pytest.approx(10.0)
        assert y0 == pytest.approx(20.0)
        assert x1 == pytest.approx(90.0)
        assert y1 == pytest.approx(100.0)

    def test_path_bbox_none_for_empty(self):
        assert _path_bbox("") is None
        assert _path_bbox("M") is None


# ---------------------------------------------------------------------------
# Group candidate extraction
# ---------------------------------------------------------------------------

class TestCandidateGroups:
    def test_groups_extracted_from_g_elements(self):
        doc = parse_svg(GROUP_SVG)
        groups = extract_candidate_groups(doc)
        g_groups = [g for g in groups if g.label.startswith("group_")]
        assert len(g_groups) >= 2

    def test_color_groups_extracted(self):
        doc = parse_svg(TWO_COLORED_SVG)
        groups = extract_candidate_groups(doc)
        color_groups = [g for g in groups if "fill_" in g.label or "color_" in g.label]
        assert len(color_groups) >= 2

    def test_group_confidence_g_higher(self):
        doc = parse_svg(GROUP_SVG)
        groups = extract_candidate_groups(doc)
        g_groups = [g for g in groups if g.label.startswith("group_")]
        for g in g_groups:
            assert g.confidence >= 0.7


# ---------------------------------------------------------------------------
# to_dict serialization
# ---------------------------------------------------------------------------

class TestSerialization:
    def test_to_dict_returns_dict(self):
        doc = parse_svg(SIMPLE_RED_HEART_SVG)
        d = doc.to_dict()
        assert isinstance(d, dict)
        assert "elements" in d
        assert "element_count" in d

    def test_to_dict_element_count_correct(self):
        doc = parse_svg(TWO_COLORED_SVG)
        d = doc.to_dict()
        assert d["element_count"] == 2


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

class TestUtils:
    def test_hex_to_rgb(self):
        assert _hex_to_rgb("#ff0000") == (255, 0, 0)
        assert _hex_to_rgb("#00ff00") == (0, 255, 0)
        assert _hex_to_rgb("#fff") == (255, 255, 255)

    def test_hex_to_rgb_invalid(self):
        with pytest.raises(ValueError):
            _hex_to_rgb("notacolor")
