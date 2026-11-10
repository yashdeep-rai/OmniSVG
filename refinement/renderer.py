"""
refinement/renderer.py
======================
Render SVG files to PNG images using CairoSVG.

Design goals:
  - Render SVG → PIL Image or PNG bytes.
  - Handle rendering failures cleanly (never silently succeed on failure).
  - Validate SVG well-formedness before rendering.
  - Record output dimensions and basic image statistics.
  - Support saving intermediate renderings for each refinement iteration.
  - Provide a lightweight fallback check if CairoSVG raises an exception.
"""

from __future__ import annotations

import io
import logging
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Import CairoSVG (required)
# ---------------------------------------------------------------------------

try:
    import cairosvg
    _CAIROSVG_AVAILABLE = True
except ImportError:
    _CAIROSVG_AVAILABLE = False
    logger.warning("cairosvg not installed — rendering will fail.")

try:
    from PIL import Image
    import numpy as np
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False
    logger.warning("Pillow not installed — image stats will be unavailable.")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RenderResult:
    """Outcome of a render attempt."""

    success: bool
    image: Optional["Image.Image"]     # PIL Image, or None on failure
    width: Optional[int]
    height: Optional[int]
    mean_pixel_value: Optional[float]  # Mean pixel brightness 0–255
    is_blank: bool                     # True if image is nearly all white
    error_message: Optional[str]
    render_time_s: float
    png_path: Optional[Path]           # Path where PNG was saved, if any


BLANK_THRESHOLD = 250.0  # Mean pixel value above this → blank


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def is_well_formed(svg_source: str | Path) -> tuple[bool, str]:
    """
    Check that an SVG is syntactically well-formed XML.

    Returns (ok: bool, message: str).
    """
    try:
        if isinstance(svg_source, Path):
            ET.parse(str(svg_source))
        else:
            ET.fromstring(str(svg_source))
        return True, "ok"
    except ET.ParseError as e:
        return False, f"XML ParseError: {e}"
    except Exception as e:
        return False, f"Unexpected error: {e}"


def has_svg_tag(svg_string: str) -> bool:
    """Quickly check that the string contains an <svg ...> opening tag."""
    return "<svg" in svg_string.lower()


def validate_svg(svg_source: str | Path) -> tuple[bool, str]:
    """
    Combined SVG validation: well-formed XML + has <svg> root.

    Returns (ok: bool, reason: str).
    """
    if isinstance(svg_source, Path):
        try:
            text = svg_source.read_text(encoding="utf-8")
        except OSError as e:
            return False, f"Cannot read file: {e}"
    else:
        text = str(svg_source)

    if len(text.strip()) < 20:
        return False, "SVG content too short"

    if not has_svg_tag(text):
        return False, "No <svg> tag found"

    return is_well_formed(svg_source if isinstance(svg_source, Path) else text)


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------


def render_svg(
    svg_source: str | Path,
    output_size: int = 512,
    save_to: Optional[Path] = None,
) -> RenderResult:
    """
    Render an SVG to a PIL Image using CairoSVG.

    Parameters
    ----------
    svg_source:
        Path to an SVG file, or SVG string content.
    output_size:
        Width and height of the output PNG in pixels.
    save_to:
        If provided, save the rendered PNG to this path.

    Returns
    -------
    RenderResult
    """
    start = time.perf_counter()

    if not _CAIROSVG_AVAILABLE:
        return RenderResult(
            success=False,
            image=None,
            width=None,
            height=None,
            mean_pixel_value=None,
            is_blank=True,
            error_message="cairosvg not installed",
            render_time_s=time.perf_counter() - start,
            png_path=None,
        )

    # --- Load SVG content ---
    if isinstance(svg_source, Path):
        ok, msg = validate_svg(svg_source)
        if not ok:
            return RenderResult(
                success=False,
                image=None,
                width=None,
                height=None,
                mean_pixel_value=None,
                is_blank=True,
                error_message=f"Validation failed: {msg}",
                render_time_s=time.perf_counter() - start,
                png_path=None,
            )
        svg_bytes = svg_source.read_bytes()
        svg_str = svg_bytes.decode("utf-8", errors="replace")
    else:
        svg_str = str(svg_source)
        ok, msg = validate_svg(svg_str)
        if not ok:
            return RenderResult(
                success=False,
                image=None,
                width=None,
                height=None,
                mean_pixel_value=None,
                is_blank=True,
                error_message=f"Validation failed: {msg}",
                render_time_s=time.perf_counter() - start,
                png_path=None,
            )
        svg_bytes = svg_str.encode("utf-8")

    # --- Render ---
    try:
        png_data = cairosvg.svg2png(
            bytestring=svg_bytes,
            output_width=output_size,
            output_height=output_size,
        )
    except Exception as e:
        return RenderResult(
            success=False,
            image=None,
            width=None,
            height=None,
            mean_pixel_value=None,
            is_blank=True,
            error_message=f"CairoSVG render error: {e}",
            render_time_s=time.perf_counter() - start,
            png_path=None,
        )

    if not png_data:
        return RenderResult(
            success=False,
            image=None,
            width=None,
            height=None,
            mean_pixel_value=None,
            is_blank=True,
            error_message="CairoSVG returned empty bytes",
            render_time_s=time.perf_counter() - start,
            png_path=None,
        )

    # --- Convert to PIL ---
    try:
        image_rgba = Image.open(io.BytesIO(png_data)).convert("RGBA")
        bg = Image.new("RGB", image_rgba.size, (255, 255, 255))
        bg.paste(image_rgba, mask=image_rgba.split()[3])
        image = bg
    except Exception as e:
        return RenderResult(
            success=False,
            image=None,
            width=None,
            height=None,
            mean_pixel_value=None,
            is_blank=True,
            error_message=f"PIL conversion error: {e}",
            render_time_s=time.perf_counter() - start,
            png_path=None,
        )

    # --- Statistics ---
    w, h = image.size
    arr = None
    mean_val = None
    is_blank = False

    if _PIL_AVAILABLE:
        try:
            arr = np.array(image)
            mean_val = float(arr.mean())
            is_blank = mean_val > BLANK_THRESHOLD
        except Exception:
            pass

    # --- Save ---
    saved_path: Optional[Path] = None
    if save_to is not None:
        try:
            save_to = Path(save_to)
            save_to.parent.mkdir(parents=True, exist_ok=True)
            image.save(str(save_to), format="PNG")
            saved_path = save_to
            logger.info("Saved PNG to %s", save_to)
        except Exception as e:
            logger.warning("Failed to save PNG to %s: %s", save_to, e)

    elapsed = time.perf_counter() - start
    logger.info("Rendered SVG in %.2fs → %dx%d, mean=%.1f, blank=%s",
                elapsed, w, h, mean_val or 0, is_blank)

    return RenderResult(
        success=True,
        image=image,
        width=w,
        height=h,
        mean_pixel_value=mean_val,
        is_blank=is_blank,
        error_message=None,
        render_time_s=elapsed,
        png_path=saved_path,
    )


def render_svg_iteration(
    svg_source: str | Path,
    output_dir: Path,
    iteration: int,
    label: str = "",
    output_size: int = 512,
) -> RenderResult:
    """
    Render an SVG and save it as an iteration-labelled PNG.

    File name: iter_{iteration:03d}_{label}.png
    """
    safe_label = label.replace(" ", "_").replace("/", "_")[:40]
    fname = f"iter_{iteration:03d}_{safe_label}.png" if safe_label else f"iter_{iteration:03d}.png"
    save_path = Path(output_dir) / fname
    return render_svg(svg_source, output_size=output_size, save_to=save_path)


# ---------------------------------------------------------------------------
# Pixel-level comparison
# ---------------------------------------------------------------------------

def pixel_diff_score(
    img_a: "Image.Image",
    img_b: "Image.Image",
) -> float:
    """
    Compute mean absolute pixel difference between two PIL RGB images.
    Returns a float in [0, 255]; lower = more similar.
    Returns -1.0 if comparison is not possible.
    """
    if not _PIL_AVAILABLE:
        return -1.0
    try:
        a = np.array(img_a.convert("RGB")).astype(float)
        b = np.array(img_b.convert("RGB")).astype(float)
        if a.shape != b.shape:
            # Resize b to match a
            b_img = img_b.resize(img_a.size, Image.LANCZOS)
            b = np.array(b_img.convert("RGB")).astype(float)
        return float(np.mean(np.abs(a - b)))
    except Exception as e:
        logger.warning("pixel_diff_score error: %s", e)
        return -1.0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.DEBUG)
    if len(sys.argv) < 2:
        print("Usage: python renderer.py <input.svg> [output.png]")
        sys.exit(1)
    src = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else src.with_suffix(".png")
    result = render_svg(src, save_to=out)
    if result.success:
        print(f"OK: {result.width}x{result.height}  mean={result.mean_pixel_value:.1f}  blank={result.is_blank}")
    else:
        print(f"FAILED: {result.error_message}")
        sys.exit(1)
