"""
tests/conftest.py — Shared fixtures for the refinement test suite.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root is on the path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ---------------------------------------------------------------------------
# Minimal SVG fixtures (hand-authored; do NOT require any model)
# ---------------------------------------------------------------------------

SIMPLE_RED_HEART_SVG = """\
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0.0 0.0 200.0 200.0" height="200px" width="200px">
<path fill="#dd3344" fill-opacity="1.0" filling="0" d="M120.0 60.0 C114.0 60.0 108.0 66.0 105.0 71.0 C101.0 66.0 95.0 60.0 89.0 60.0 C78.0 60.0 69.0 69.0 69.0 80.0 C69.0 86.0 72.0 90.0 75.0 94.0 C82.0 104.0 101.0 117.0 102.0 118.0 C103.0 118.0 104.0 119.0 105.0 119.0 C106.0 119.0 106.0 118.0 107.0 118.0 C108.0 117.0 127.0 104.0 134.0 94.0 C137.0 90.0 140.0 86.0 140.0 80.0 C140.0 69.0 131.0 60.0 120.0 60.0 L120.0 60.0 Z"></path>
</svg>"""

TWO_COLORED_SVG = """\
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0.0 0.0 200.0 200.0" height="200px" width="200px">
<path fill="#ff0000" fill-opacity="1.0" d="M10.0 10.0 L90.0 10.0 L90.0 90.0 L10.0 90.0 Z"/>
<path fill="#ffdd00" fill-opacity="1.0" d="M110.0 10.0 L190.0 10.0 L190.0 90.0 L110.0 90.0 Z"/>
</svg>"""

MULTI_PATH_SVG = """\
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0.0 0.0 200.0 200.0" height="200px" width="200px">
<path fill="#ffffff" fill-opacity="1.0" d="M0.0 0.0 L199.0 0.0 L199.0 199.0 L0.0 199.0 L0.0 0.0 Z"/>
<path fill="#ffdd00" fill-opacity="1.0" d="M152.0 104.0 C152.0 123.0 141.0 139.0 124.0 139.0 C108.0 139.0 95.0 123.0 95.0 104.0 C95.0 85.0 108.0 69.0 124.0 69.0 C141.0 69.0 152.0 85.0 152.0 104.0 Z"/>
<path fill="#ff3333" fill-opacity="1.0" d="M80.0 60.0 C70.0 60.0 62.0 68.0 62.0 80.0 C62.0 92.0 70.0 100.0 80.0 100.0 C90.0 100.0 98.0 92.0 98.0 80.0 C98.0 68.0 90.0 60.0 80.0 60.0 Z"/>
<path fill="#33aa33" fill-opacity="1.0" d="M100.0 130.0 L120.0 110.0 L140.0 130.0 L130.0 130.0 L130.0 160.0 L110.0 160.0 L110.0 130.0 Z"/>
</svg>"""

MALFORMED_SVG = "<svg><path d='INVALID XML"

EMPTY_SVG = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200"></svg>'

GROUP_SVG = """\
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0.0 0.0 200.0 200.0" height="200px" width="200px">
<g id="apple">
  <path fill="#ff0000" d="M80 60 C70 60 62 68 62 80 C62 92 70 100 80 100 Z"/>
  <path fill="#ff3333" d="M85 55 L90 50 L88 48 Z"/>
</g>
<g id="leaf">
  <path fill="#00bb00" d="M100 70 L120 60 L115 80 Z"/>
</g>
</svg>"""
