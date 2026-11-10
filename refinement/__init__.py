"""
Semantic-Aware Localized Refinement of SVGs Using Multimodal LLMs.

Package structure:
  svg_parser  — Parse SVG XML, assign stable IDs, extract geometry.
  renderer    — Render SVG → PNG via CairoSVG.
  editor      — Constrained, audited SVG element editing.
  critic      — Multimodal VLM-based prompt-vs-image comparison.
  localizer   — Map critic diagnoses → SVG element candidates.
  pipeline    — Recursive refinement loop.
"""

__version__ = "0.1.0"
