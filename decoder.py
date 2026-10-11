import os
import torch
import torch.nn as nn
import yaml

from transformers import Qwen2_5_VLForConditionalGeneration


def load_config(config_path=None):
    """Load configuration from config.yaml."""

    if config_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))

        possible_paths = [
            os.path.join(current_dir, "config.yaml"),
            os.path.join(current_dir, "..", "config.yaml"),
            "config.yaml",
        ]

        for path in possible_paths:
            if os.path.exists(path):
                config_path = path
                break

        if config_path is None:
            raise FileNotFoundError("config.yaml not found")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


class SketchDecoder(nn.Module):
    """
    OmniSVG autoregressive decoder.

    Loading order:

        1. Load Qwen2.5-VL completely on CPU
        2. Resize vocabulary to OmniSVG's 197000 tokens
        3. Return the model without Accelerate dispatch
        4. inference.py loads the OmniSVG checkpoint
        5. Only after the checkpoint is loaded, inference.py dispatches
           the model across GPU and CPU

    This avoids stale Accelerate hooks and meta tensors while loading
    the OmniSVG checkpoint.
    """

    def __init__(self, config_path=None, model_path=None, **kwargs):
        super().__init__()

        config_data = load_config(config_path)

        model_config = config_data.get("model", {})
        huggingface_config = config_data.get("huggingface", {})

        self.bos_token_id = model_config["bos_token_id"]
        self.eos_token_id = model_config["eos_token_id"]
        self.pad_token_id = model_config["pad_token_id"]

        # OmniSVG 1.1 8B vocabulary.
        #
        # config.yaml in this repository does not expose vocab_size
        # at the top-level model section, so use the known 8B value.
        self.vocab_size = model_config.get("vocab_size", 197000)

        if model_path is None:
            model_path = huggingface_config.get("qwen_model")

        if model_path is None:
            raise ValueError(
                "Qwen model path was not provided and could not "
                "be found in config.yaml"
            )

        print(f"Loading Qwen2.5-VL from: {model_path}")
        print("Loading base model completely on CPU...")

        self.transformer = (
            Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="sdpa",
                device_map="cpu",
            )
        )

        print("Base Qwen model loaded on CPU.")

        # --------------------------------------------------------
        # Resize vocabulary BEFORE any Accelerate dispatch.
        # --------------------------------------------------------

        old_vocab_size = self.transformer.config.vocab_size

        print(
            f"Resizing vocabulary from "
            f"{old_vocab_size} to {self.vocab_size}..."
        )

        self.transformer.resize_token_embeddings(
            self.vocab_size,
            mean_resizing=False,
        )

        self.transformer.config.vocab_size = self.vocab_size

        print("Vocabulary resized successfully.")

        print(
            "Input embedding shape:",
            tuple(self.transformer.get_input_embeddings().weight.shape),
        )

        print(
            "LM head shape:",
            tuple(self.transformer.lm_head.weight.shape),
        )

        # IMPORTANT:
        # Do NOT dispatch the model here.
        #
        # The OmniSVG checkpoint must be loaded first.
        print(
            "\nSketchDecoder initialized on CPU."
            "\nOmniSVG weights should be loaded before dispatch."
        )

        # Set generation-related IDs.
        self.transformer.config.bos_token_id = self.bos_token_id
        self.transformer.config.eos_token_id = self.eos_token_id
        self.transformer.config.pad_token_id = self.pad_token_id

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "Forward pass not included in open-source version"
        )