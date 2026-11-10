"""
refinement/critic.py
====================
Multimodal critic that compares a rendered SVG image against the original
text prompt using the locally available Qwen2.5-VL-7B-Instruct model.

Key design decisions
--------------------
* The VL model is loaded SEPARATELY from the OmniSVG inference pipeline to
  avoid VRAM conflicts.  It is loaded lazily (on first call) and can be
  explicitly unloaded to free memory between pipeline stages.
* We load in 4-bit quantization (bitsandbytes) if available; otherwise we
  use bfloat16 with CPU offload to stay within 16 GB VRAM.
* The critic returns structured JSON describing detected visual errors.
* All model responses are validated before parsing.  Malformed JSON is
  handled gracefully with a partial-parse fallback.
* Deterministic checks (explicit SVG attributes) take priority over model
  judgments.

Output schema (CriticReport)
-----------------------------
{
  "satisfied": false,
  "overall_confidence": 0.75,
  "errors": [
    {
      "type": "color_mismatch",        # see ERROR_TYPES below
      "component": "banana",
      "expected": "yellow",
      "observed": "green",
      "confidence": 0.90,
      "description": "The banana appears green instead of yellow."
    }
  ],
  "raw_response": "..."                # full model text (for debugging)
}

Valid error types
-----------------
  missing_object    — an object expected from the prompt is absent
  extra_object      — an unexpected object is present
  color_mismatch    — an object has the wrong color
  attribute_error   — wrong shape, size, count, or other attribute
  spatial_error     — wrong position or spatial relationship
  layout_error      — overlapping or poorly composed elements
  rendering_failure — SVG failed to render (no image available)
  unknown           — catch-all
"""

from __future__ import annotations

import base64
import gc
import io
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

QWEN_MODEL_PATH = os.environ.get(
    "QWEN_VL_MODEL_PATH",
    r"C:\Users\CSE_SDPL\Documents\123cs0077_Project\models\Qwen2.5-VL-7B-Instruct",
)

ERROR_TYPES = {
    "missing_object",
    "extra_object",
    "color_mismatch",
    "attribute_error",
    "spatial_error",
    "layout_error",
    "rendering_failure",
    "unknown",
}

# System prompt directing the VLM to act as a visual critic
CRITIC_SYSTEM_PROMPT = """You are a precise visual quality critic for SVG illustrations.

Your task:
Given a text prompt and a rendered SVG image, identify ALL visual errors where the image does NOT match the prompt.

Output ONLY a single valid JSON object (no extra text) with this exact schema:
{
  "satisfied": <true|false>,
  "overall_confidence": <0.0-1.0>,
  "errors": [
    {
      "type": "<missing_object|extra_object|color_mismatch|attribute_error|spatial_error|layout_error|unknown>",
      "component": "<what object or part is affected>",
      "expected": "<what the prompt says it should be>",
      "observed": "<what you actually see in the image>",
      "confidence": <0.0-1.0>,
      "description": "<one concise sentence>"
    }
  ]
}

Rules:
- If the image fully satisfies the prompt, set satisfied=true and errors=[].
- List EVERY discrepancy, no matter how small.
- Be specific about colors (use color names or hex values where possible).
- Be specific about spatial positions (left, right, top, bottom, center).
- confidence is YOUR confidence in this specific error finding (0=uncertain, 1=certain).
- overall_confidence is your confidence in the overall satisfied judgment.
- Output ONLY the JSON. No preamble, no explanation, no markdown code fences."""

# Fallback user prompt template
CRITIC_USER_TEMPLATE = """Text prompt: {prompt}

Please analyze the SVG image above and identify all visual errors compared to the prompt.
Output ONLY a JSON object following the schema in the system prompt."""


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ErrorFinding:
    """One detected visual error."""
    type: str
    component: str
    expected: str
    observed: str
    confidence: float
    description: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "component": self.component,
            "expected": self.expected,
            "observed": self.observed,
            "confidence": self.confidence,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ErrorFinding":
        return cls(
            type=d.get("type", "unknown"),
            component=d.get("component", ""),
            expected=d.get("expected", ""),
            observed=d.get("observed", ""),
            confidence=float(d.get("confidence", 0.5)),
            description=d.get("description", ""),
        )

    def validate(self) -> "ErrorFinding":
        """Normalize field values to ensure they're valid."""
        if self.type not in ERROR_TYPES:
            self.type = "unknown"
        self.confidence = max(0.0, min(1.0, float(self.confidence)))
        return self


@dataclass
class CriticReport:
    """Full output from the critic for one (prompt, image) pair."""
    satisfied: bool
    overall_confidence: float
    errors: List[ErrorFinding]
    raw_response: str
    model_failed: bool            # True if the model call failed entirely
    error_message: Optional[str]  # Set if model_failed=True
    runtime_s: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "satisfied": self.satisfied,
            "overall_confidence": self.overall_confidence,
            "errors": [e.to_dict() for e in self.errors],
            "raw_response": self.raw_response,
            "model_failed": self.model_failed,
            "error_message": self.error_message,
            "runtime_s": self.runtime_s,
        }

    def high_confidence_errors(self, threshold: float = 0.6) -> List[ErrorFinding]:
        """Return errors with confidence >= threshold."""
        return [e for e in self.errors if e.confidence >= threshold]

    def errors_by_type(self, error_type: str) -> List[ErrorFinding]:
        return [e for e in self.errors if e.type == error_type]

    @classmethod
    def failure(cls, message: str, runtime_s: float = 0.0) -> "CriticReport":
        """Create a failure report when the model cannot be called."""
        return cls(
            satisfied=False,
            overall_confidence=0.0,
            errors=[
                ErrorFinding(
                    type="rendering_failure",
                    component="",
                    expected="",
                    observed="",
                    confidence=1.0,
                    description=message,
                )
            ],
            raw_response="",
            model_failed=True,
            error_message=message,
            runtime_s=runtime_s,
        )


# ---------------------------------------------------------------------------
# JSON parsing helpers
# ---------------------------------------------------------------------------

def _extract_json_from_text(text: str) -> Optional[str]:
    """
    Try to extract a JSON object from model output that may contain
    extra text or markdown code fences.
    """
    # Try direct parse first
    text = text.strip()
    if text.startswith("{"):
        return text

    # Remove markdown code fences
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    text = text.strip()

    # Find first { ... } block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return match.group(0)

    return None


def _parse_critic_response(raw: str) -> Optional[Dict[str, Any]]:
    """Parse a raw model response into a dict.  Returns None on failure."""
    extracted = _extract_json_from_text(raw)
    if not extracted:
        return None
    try:
        return json.loads(extracted)
    except json.JSONDecodeError:
        # Try to fix common issues: trailing commas, single quotes
        fixed = re.sub(r",\s*([}\]])", r"\1", extracted)
        fixed = fixed.replace("'", '"')
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            return None


def _build_critic_report(parsed: Optional[Dict[str, Any]], raw: str, runtime_s: float) -> CriticReport:
    """Convert a parsed dict (or None) to a CriticReport."""
    if parsed is None:
        logger.warning("Could not parse critic response as JSON")
        return CriticReport(
            satisfied=False,
            overall_confidence=0.3,
            errors=[
                ErrorFinding(
                    type="unknown",
                    component="",
                    expected="",
                    observed="",
                    confidence=0.3,
                    description="Could not parse model response as JSON",
                )
            ],
            raw_response=raw,
            model_failed=False,
            error_message="JSON parse failure",
            runtime_s=runtime_s,
        )

    satisfied = bool(parsed.get("satisfied", False))
    overall_confidence = float(parsed.get("overall_confidence", 0.5))
    overall_confidence = max(0.0, min(1.0, overall_confidence))

    errors: List[ErrorFinding] = []
    for item in parsed.get("errors", []):
        try:
            ef = ErrorFinding.from_dict(item).validate()
            errors.append(ef)
        except Exception as e:
            logger.warning("Skipping malformed error entry: %s — %s", item, e)

    return CriticReport(
        satisfied=satisfied,
        overall_confidence=overall_confidence,
        errors=errors,
        raw_response=raw,
        model_failed=False,
        error_message=None,
        runtime_s=runtime_s,
    )


# ---------------------------------------------------------------------------
# Image encoding
# ---------------------------------------------------------------------------

def _image_to_base64(image: "Image.Image") -> str:
    """Encode a PIL image to a base64 PNG string."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _pil_to_temp_path(image: "Image.Image") -> str:
    """Save a PIL image to a temporary file and return its path."""
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        image.save(f.name, format="PNG")
        return f.name


# ---------------------------------------------------------------------------
# Critic model (lazy-loaded singleton)
# ---------------------------------------------------------------------------

_critic_model = None
_critic_processor = None
_critic_model_path: Optional[str] = None


def load_critic_model(
    model_path: str = QWEN_MODEL_PATH,
    use_4bit: bool = False,
    max_gpu_memory: str = "13GiB",
) -> None:
    """
    Load the Qwen2.5-VL critic model into memory.

    Call this explicitly before running batch evaluations to avoid repeated
    load/unload overhead.  If already loaded with the same path, no-op.

    Parameters
    ----------
    model_path:
        Local path or HF repo ID for Qwen2.5-VL-7B-Instruct.
    use_4bit:
        If True, attempt to load in 4-bit (requires bitsandbytes).
    max_gpu_memory:
        Maximum GPU memory to allocate (e.g. "13GiB").
    """
    global _critic_model, _critic_processor, _critic_model_path

    if _critic_model is not None and _critic_model_path == model_path:
        logger.info("Critic model already loaded from %s", model_path)
        return

    unload_critic_model()  # free any existing model first

    logger.info("Loading critic model from %s …", model_path)
    start = time.perf_counter()

    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    dtype = torch.bfloat16

    load_kwargs: Dict[str, Any] = {
        "torch_dtype": dtype,
        "trust_remote_code": True,
    }

    if use_4bit:
        try:
            from transformers import BitsAndBytesConfig
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            load_kwargs["quantization_config"] = bnb_cfg
            load_kwargs["device_map"] = "auto"
            logger.info("Using 4-bit quantization")
        except ImportError:
            logger.warning("bitsandbytes not available; falling back to bfloat16 with offload")
            use_4bit = False

    if not use_4bit:
        from accelerate import infer_auto_device_map, dispatch_model
        # Load on CPU first
        load_kwargs["device_map"] = "cpu"

    _critic_processor = AutoProcessor.from_pretrained(
        model_path,
        padding_side="left",
        trust_remote_code=True,
    )

    _critic_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        **load_kwargs,
    )

    if not use_4bit:
        # Dispatch with limited GPU memory
        import torch
        device_map = infer_auto_device_map(
            _critic_model,
            max_memory={0: max_gpu_memory, "cpu": "50GiB"},
        )
        _critic_model = dispatch_model(_critic_model, device_map=device_map)

    _critic_model = _critic_model.eval()
    _critic_model_path = model_path
    elapsed = time.perf_counter() - start
    logger.info("Critic model loaded in %.1fs", elapsed)


def unload_critic_model() -> None:
    """Unload the critic model to free GPU/CPU memory."""
    global _critic_model, _critic_processor, _critic_model_path
    import torch

    if _critic_model is not None:
        del _critic_model
        _critic_model = None
    if _critic_processor is not None:
        del _critic_processor
        _critic_processor = None
    _critic_model_path = None

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Critic model unloaded")


def is_critic_loaded() -> bool:
    return _critic_model is not None


# ---------------------------------------------------------------------------
# Core critic function
# ---------------------------------------------------------------------------

def analyze(
    prompt: str,
    image: "Image.Image",
    model_path: str = QWEN_MODEL_PATH,
    max_new_tokens: int = 1024,
    auto_load: bool = True,
    auto_unload: bool = False,
) -> CriticReport:
    """
    Compare a rendered SVG image against the original text prompt.

    Parameters
    ----------
    prompt:
        The original text prompt used to generate the SVG.
    image:
        PIL Image of the rendered SVG.
    model_path:
        Local path or HF repo ID for Qwen2.5-VL-7B-Instruct.
    max_new_tokens:
        Maximum tokens to generate in the critic response.
    auto_load:
        If True, load the model automatically if not already loaded.
    auto_unload:
        If True, unload the model after each call (saves VRAM but slow).

    Returns
    -------
    CriticReport
    """
    import torch

    start = time.perf_counter()

    if image is None:
        return CriticReport.failure("No image provided for critic", runtime_s=0.0)

    if auto_load and not is_critic_loaded():
        try:
            load_critic_model(model_path)
        except Exception as e:
            return CriticReport.failure(f"Failed to load critic model: {e}",
                                        runtime_s=time.perf_counter() - start)

    if not is_critic_loaded():
        return CriticReport.failure("Critic model not loaded",
                                    runtime_s=time.perf_counter() - start)

    # Save image to temp file (Qwen processor requires file path or URL)
    tmp_path = _pil_to_temp_path(image)

    try:
        messages = [
            {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": tmp_path},
                    {"type": "text", "text": CRITIC_USER_TEMPLATE.format(prompt=prompt)},
                ],
            },
        ]

        # Apply chat template
        text_input = _critic_processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Process inputs
        try:
            from qwen_vl_utils import process_vision_info
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = _critic_processor(
                text=[text_input],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
        except Exception as e:
            logger.warning("qwen_vl_utils failed: %s; trying basic processor", e)
            inputs = _critic_processor(
                text=[text_input],
                images=[image],
                padding=True,
                truncation=True,
                return_tensors="pt",
            )

        # Move inputs to model device
        try:
            first_param = next(_critic_model.parameters())
            device = first_param.device
        except StopIteration:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model_inputs = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                try:
                    if k in ("pixel_values",) and v.dtype == torch.float32:
                        model_inputs[k] = v.to(device, dtype=torch.bfloat16)
                    else:
                        model_inputs[k] = v.to(device)
                except Exception:
                    model_inputs[k] = v
            else:
                model_inputs[k] = v

        # Generate
        with torch.no_grad():
            output_ids = _critic_model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
            )

        # Decode only the generated tokens
        input_len = inputs["input_ids"].shape[1]
        generated = output_ids[:, input_len:]
        raw_response = _critic_processor.tokenizer.decode(
            generated[0].cpu().tolist(),
            skip_special_tokens=True,
        )

        logger.debug("Critic raw response: %s", raw_response[:500])

    except Exception as e:
        logger.exception("Critic inference error: %s", e)
        return CriticReport.failure(
            f"Critic inference error: {e}",
            runtime_s=time.perf_counter() - start,
        )
    finally:
        # Clean up temp file
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        if auto_unload:
            unload_critic_model()

    runtime_s = time.perf_counter() - start
    parsed = _parse_critic_response(raw_response)
    report = _build_critic_report(parsed, raw_response, runtime_s)

    logger.info(
        "Critic: satisfied=%s, errors=%d, confidence=%.2f, time=%.1fs",
        report.satisfied,
        len(report.errors),
        report.overall_confidence,
        runtime_s,
    )
    return report


# ---------------------------------------------------------------------------
# Deterministic pre-checks (no model required)
# ---------------------------------------------------------------------------

def deterministic_color_check(
    prompt: str,
    doc: "SVGDocument",
) -> List[ErrorFinding]:
    """
    Check for explicit color mentions in the prompt against SVG fill colors.

    This is a heuristic, not a ground-truth check.  It catches simple cases
    like "yellow banana" when all fills are green.

    Returns a list of ErrorFinding objects (may be empty).
    """
    from .svg_parser import SVGDocument

    COLOR_KEYWORDS = {
        "red": ["#ff0000", "#cc0000", "#dd0000", "#ee1100", "#ff3333", "#dd3344", "#cc3333"],
        "green": ["#00ff00", "#00cc00", "#009900", "#008800", "#006600", "#33aa33"],
        "blue": ["#0000ff", "#0000cc", "#0033ff", "#3366ff", "#336699", "#556688"],
        "yellow": ["#ffff00", "#ffee00", "#ffdd00", "#ffe000"],
        "orange": ["#ff8800", "#ff6600", "#ff7700", "#ff9900"],
        "purple": ["#800080", "#aa00aa", "#9900cc"],
        "pink": ["#ff69b4", "#ff88aa", "#ffaacc", "#ffbbcc"],
        "black": ["#000000", "#111111"],
        "white": ["#ffffff", "#eeeeee"],
        "gray": ["#808080", "#888888", "#999999", "#aaaaaa", "#555555", "#667777"],
        "grey": ["#808080", "#888888", "#999999"],
        "brown": ["#8b4513", "#a0522d", "#7b3f00"],
    }

    all_fills = set()
    for e in doc.elements:
        if e.fill and e.fill.startswith("#"):
            all_fills.add(e.fill.lower())

    errors: List[ErrorFinding] = []
    prompt_lower = prompt.lower()

    for color_name, hex_variants in COLOR_KEYWORDS.items():
        if color_name in prompt_lower:
            # Check if any fill matches this color family
            found = any(
                any(f.startswith(h[:4]) for h in hex_variants)
                for f in all_fills
            )
            if not found and all_fills:
                errors.append(ErrorFinding(
                    type="color_mismatch",
                    component=color_name,
                    expected=color_name,
                    observed=f"fills are {sorted(all_fills)[:5]}",
                    confidence=0.55,
                    description=(
                        f"Prompt mentions '{color_name}' but no fill color "
                        f"closely matches this color family."
                    ),
                ).validate())

    return errors


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import argparse
    from PIL import Image

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Run SVG critic")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--image", required=True, help="Path to rendered PNG")
    parser.add_argument("--model-path", default=QWEN_MODEL_PATH)
    args = parser.parse_args()

    img = Image.open(args.image).convert("RGB")
    report = analyze(args.prompt, img, model_path=args.model_path)
    print(json.dumps(report.to_dict(), indent=2))
