import os
import sys
import torch

# Make sure we can import decoder.py from the current directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from decoder import SketchDecoder


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = r"C:\Users\CSE_SDPL\Documents\123cs0077_Project\models\Qwen2.5-VL-7B-Instruct"

WEIGHT_PATH = r"C:\Users\CSE_SDPL\Documents\123cs0077_Project\models\OmniSVG1.1_8B\pytorch_model.bin"


# ============================================================
# HELPER
# ============================================================

def print_tensor_info(name, tensor):
    print(f"\n{name}")
    print("-" * 70)
    print("Shape :", tuple(tensor.shape))
    print("Dtype :", tensor.dtype)

    try:
        x = tensor.float()
        print("Mean  :", x.mean().item())
        print("Std   :", x.std().item())
        print("Min   :", x.min().item())
        print("Max   :", x.max().item())
    except Exception as e:
        print("Statistics unavailable:", e)


# ============================================================
# MAIN
# ============================================================

print("=" * 70)
print("OmniSVG 1.1 8B WEIGHT SANITY CHECK")
print("=" * 70)

print("\nQwen model:")
print(MODEL_PATH)

print("\nOmniSVG weights:")
print(WEIGHT_PATH)

if not os.path.exists(MODEL_PATH):
    print("\nERROR: Qwen model directory does not exist.")
    sys.exit(1)

if not os.path.exists(WEIGHT_PATH):
    print("\nERROR: OmniSVG pytorch_model.bin does not exist.")
    sys.exit(1)


# ============================================================
# STEP 1 — LOAD QWEN + RESIZE
# ============================================================

print("\n")
print("=" * 70)
print("STEP 1: Loading Qwen2.5-VL")
print("=" * 70)

print("\nLoading base model on CPU...")
print("This may take a while.\n")

decoder = SketchDecoder(
    model_path=MODEL_PATH
)

model = decoder.transformer

print("\nBase + resized model loaded.")

print("\nModel vocabulary size:")
print("config.vocab_size =", model.config.vocab_size)

print("\nEmbedding shape:")
print(tuple(model.get_input_embeddings().weight.shape))

print("\nLM head shape:")
print(tuple(model.lm_head.weight.shape))


# ============================================================
# STEP 2 — LOAD OMNISVG WEIGHTS
# ============================================================

print("\n")
print("=" * 70)
print("STEP 2: Loading OmniSVG state_dict")
print("=" * 70)

print("\nLoading:")
print(WEIGHT_PATH)

print("\nThis will temporarily require a large amount of RAM.")

state_dict = torch.load(
    WEIGHT_PATH,
    map_location="cpu"
)

print("\nState dict loaded successfully.")

print("Number of tensors:", len(state_dict))


# ============================================================
# STEP 3 — INSPECT IMPORTANT WEIGHTS
# ============================================================

print("\n")
print("=" * 70)
print("STEP 3: Inspecting important tensors")
print("=" * 70)


important_keys = [
    "model.embed_tokens.weight",
    "lm_head.weight",

    "transformer.model.embed_tokens.weight",
    "transformer.lm_head.weight",

    "transformer.model.layers.0.self_attn.q_proj.weight",
    "transformer.model.layers.0.mlp.gate_proj.weight",
]


for key in important_keys:

    if key in state_dict:
        print_tensor_info(key, state_dict[key])

    else:
        print(f"\n{key}")
        print("-" * 70)
        print("NOT FOUND")


# ============================================================
# STEP 4 — FIND VOCAB-SIZED TENSORS
# ============================================================

print("\n")
print("=" * 70)
print("STEP 4: Searching for 197000-vocabulary tensors")
print("=" * 70)

found_vocab_tensors = []

for key, tensor in state_dict.items():

    if not torch.is_tensor(tensor):
        continue

    shape = tuple(tensor.shape)

    if len(shape) >= 2 and 197000 in shape:
        found_vocab_tensors.append((key, shape))


if found_vocab_tensors:

    print("\nFound tensors containing dimension 197000:\n")

    for key, shape in found_vocab_tensors:
        print(f"{key}")
        print(f"    shape = {shape}")

else:

    print("\nWARNING:")
    print("No tensor containing dimension 197000 was found.")


# ============================================================
# STEP 5 — LOAD STATE DICT
# ============================================================

print("\n")
print("=" * 70)
print("STEP 5: Loading OmniSVG weights into model")
print("=" * 70)

print("\nCalling load_state_dict(strict=False)...")

missing, unexpected = model.load_state_dict(
    state_dict,
    strict=False
)

print("\nState dict loading finished.")

print("\nMissing keys:", len(missing))
print("Unexpected keys:", len(unexpected))


if missing:

    print("\nFirst missing keys:")

    for key in missing[:30]:
        print("  ", key)


if unexpected:

    print("\nFirst unexpected keys:")

    for key in unexpected[:30]:
        print("  ", key)


# ============================================================
# STEP 6 — INSPECT ACTUAL LOADED MODEL
# ============================================================

print("\n")
print("=" * 70)
print("STEP 6: Inspecting ACTUAL loaded model")
print("=" * 70)


embedding = model.get_input_embeddings().weight
lm_head = model.lm_head.weight

print_tensor_info(
    "ACTUAL MODEL INPUT EMBEDDING",
    embedding
)

print_tensor_info(
    "ACTUAL MODEL LM HEAD",
    lm_head
)


# ============================================================
# STEP 7 — CHECK VOCABULARY SIZE
# ============================================================

print("\n")
print("=" * 70)
print("STEP 7: Vocabulary consistency")
print("=" * 70)

print("\nExpected OmniSVG vocabulary:")
print("197000")

print("\nModel config vocabulary:")
print(model.config.vocab_size)

print("\nInput embedding:")
print(tuple(embedding.shape))

print("\nLM head:")
print(tuple(lm_head.shape))


if embedding.shape[0] == 197000:
    print("\n[OK] Input embedding has 197000 tokens.")
else:
    print("\n[ERROR] Input embedding does NOT have 197000 tokens.")


if lm_head.shape[0] == 197000:
    print("[OK] LM head has 197000 tokens.")
else:
    print("[ERROR] LM head does NOT have 197000 tokens.")


# ============================================================
# STEP 8 — CHECK OMNISVG SPECIAL TOKEN RANGES
# ============================================================

print("\n")
print("=" * 70)
print("STEP 8: OmniSVG token range")
print("=" * 70)

print("""
Expected OmniSVG 8B token ranges:

Base Qwen vocabulary:
    0 - 152063

SVG structural tokens:
    152064+
    
Known OmniSVG special tokens:
    mask = 152065
    eom  = 152066

Coordinate / SVG command region:
    around 152067 - 192074

Color region:
    192075 - 196172

Model vocabulary:
    197000
""")


# ============================================================
# STEP 9 — COMPARE STATE_DICT AND MODEL LM HEAD
# ============================================================

print("\n")
print("=" * 70)
print("STEP 9: Comparing OmniSVG LM head")
print("=" * 70)


possible_lm_keys = [
    "lm_head.weight",
    "transformer.lm_head.weight",
]


for key in possible_lm_keys:

    if key in state_dict:

        source = state_dict[key]

        print("\nFound:", key)

        print("Checkpoint shape:")
        print(tuple(source.shape))

        print("Model shape:")
        print(tuple(lm_head.shape))

        if source.shape == lm_head.shape:
            print("[OK] Shapes match.")

            # Compare a few values
            try:

                source_sample = source.float().flatten()[:1000]
                model_sample = lm_head.detach().cpu().float().flatten()[:1000]

                difference = (
                    source_sample - model_sample
                ).abs().mean().item()

                print(
                    "Mean absolute difference:",
                    difference
                )

            except Exception as e:
                print("Could not compare values:", e)

        else:
            print("[ERROR] Shapes DO NOT MATCH.")


# ============================================================
# FINAL
# ============================================================

print("\n")
print("=" * 70)
print("FINAL RESULT")
print("=" * 70)

if (
    embedding.shape[0] == 197000
    and lm_head.shape[0] == 197000
    and len(missing) == 0
):
    print("""
[PASS]

The OmniSVG checkpoint appears to match the resized
197000-token model structure.

The next debugging step should therefore focus on
generation/tokenization rather than model loading.
""")

else:

    print("""
[PROBLEM DETECTED]

The OmniSVG checkpoint does not appear to have been
loaded cleanly into the resized model.

We need to inspect the missing/unexpected keys and
tensor shapes before running inference again.
""")


print("\nDone.")