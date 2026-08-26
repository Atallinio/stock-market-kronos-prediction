"""PyTorch -> Flax NNX weight transfer for the Kronos models.

Loads the official Kronos checkpoints from HuggingFace (PyTorch), then maps
every state-dict entry onto the matching NNX parameter path and transfers it.
Requires the upstream Kronos repository to be cloned next to this one for its
PyTorch model definitions:

    git clone https://github.com/shiyu-coder/Kronos

Extracted from kronos_rag.ipynb (kaggle sys.path hacks removed; pass the
repo location via kronos_path instead).
"""

import logging
import sys

import jax.numpy as jnp
import numpy as np
from flax import nnx
from flax.traverse_util import flatten_dict, unflatten_dict

logger = logging.getLogger(__name__)

# HuggingFace checkpoint ids per Kronos variant (predictor, tokenizer).
MODEL_HF_IDS = {
    "base":  ("NeoQuasar/Kronos-base", "NeoQuasar/Kronos-Tokenizer-base"),
    "small": ("NeoQuasar/Kronos-small", "NeoQuasar/Kronos-Tokenizer-base"),
    "mini":  ("NeoQuasar/Kronos-mini", "NeoQuasar/Kronos-Tokenizer-2k"),
}


def print_model_structures(pt_state_dict, nnx_model):
    """Print PyTorch state dict keys and Flax NNX model parameter paths side by side."""
    print("=" * 90)
    print(" 1. PYTORCH STATE DICT KEYS & SHAPES")
    print("=" * 90)
    for k, v in pt_state_dict.items():
        shape = tuple(v.shape) if hasattr(v, "shape") else "N/A"
        print(f"  {k:<55} | Shape: {shape}")

    print("\n" + "=" * 90)
    print(" 2. FLAX NNX MODEL STATE KEYS & SHAPES")
    print("=" * 90)
    nnx_flat = flatten_dict(nnx.state(nnx_model).to_pure_dict())
    for path_tuple, val in nnx_flat.items():
        dot_path = ".".join(str(p) for p in path_tuple)
        shape = getattr(val, "shape", "N/A")
        print(f"  {dot_path:<55} | Tuple: {str(path_tuple):<20} | Shape: {shape}")
    print("=" * 90 + "\n")


def load_pt_state_dict_into_nnx(pt_state_dict, nnx_model, inspect_first=False, ignore_keys=("num_batches_tracked")):
    """Map every key of a PyTorch state dict onto the NNX model and load it.

    Handles the linear-convention flip (PT uses out-features-major, NNX uses
    in-features-major) and the RMSNorm scale parameter naming.
    """
    if inspect_first:
        print_model_structures(pt_state_dict, nnx_model)

    pt_numpy = {
        k: (v.detach().cpu().numpy() if hasattr(v, "detach") else np.array(v))
        for k, v in pt_state_dict.items()
    }
    current_nnx_state = nnx.state(nnx_model)
    flat_nnx = flatten_dict(current_nnx_state.to_pure_dict())

    new_flat_nnx = {}
    matched_nnx_keys = set()

    for pt_key, arr in pt_numpy.items():
        if any(pt_key.split('.')[-1] == ignored for ignored in ignore_keys):
            continue
        parts = [int(p) if p.isdigit() else p for p in pt_key.split(".")]
        prefix = parts[:-1]
        suffix = parts[-1]
        if suffix == "weight":
            kernel_path = tuple(prefix + ["kernel"])
            scale_path = tuple(prefix + ["scale"])
            emb_path = tuple(prefix + ["embedding"])

            if kernel_path in flat_nnx:
                target_path = kernel_path
                if arr.ndim == 2:
                    arr = arr.T
                elif arr.ndim == 4:
                    arr = np.transpose(arr, (2, 3, 1, 0))
            elif scale_path in flat_nnx:
                target_path = scale_path
            elif emb_path in flat_nnx:
                target_path = emb_path
            else:
                target_path = tuple(parts)
        elif suffix == "bias":
            target_path = tuple(parts)
        else:
            target_path = tuple(parts)

        if target_path in flat_nnx:
            expected_shape = flat_nnx[target_path].shape
            if arr.shape == expected_shape:
                new_flat_nnx[target_path] = jnp.array(arr)
                matched_nnx_keys.add(target_path)
            else:
                logger.warning("Shape mismatch at %s -> %s: PT %s vs NNX %s", pt_key, target_path, arr.shape, expected_shape)
        else:
            logger.warning("Target path %s not found in NNX model.", target_path)

    unmapped_nnx = set(flat_nnx.keys()) - matched_nnx_keys
    unmapped_weights = [p for p in unmapped_nnx if "rngs" not in p]
    if unmapped_weights:
        missing_paths = [".".join(str(p) for p in p) for p in unmapped_weights]
        logger.warning("Unmapped NNX parameters (e.g. LoRA params): %s... (%d total)", missing_paths, len(missing_paths))
    else:
        logger.info("All base weight parameters successfully mapped!")

    new_nested_state = unflatten_dict(new_flat_nnx)
    nnx.update(nnx_model, new_nested_state)
    logger.info("Weight loading completed successfully.")


def load_models(model_name="base", kronos_path="Kronos"):
    """Instantiate the Flax tokenizer + predictor and transfer HF weights.

    Args:
        model_name: one of "base", "small", "mini".
        kronos_path: path to the cloned upstream Kronos repo (added to sys.path).

    Returns:
        (tokenizer, model) as nnx.Module, both in eval mode.
    """
    from model import (
        Kronos, KronosTokenizer,
        TOKENIZER_BASE_CONFIG, KRONOS_BASE_CONFIG, KRONOS_SMALL_CONFIG, KRONOS_MINI_CONFIG,
    )

    if model_name not in MODEL_HF_IDS:
        raise ValueError(f"Unknown model {model_name!r}; choose from {sorted(MODEL_HF_IDS)}")

    kronos_hf, tokenizer_hf = MODEL_HF_IDS[model_name]
    kronos_config = {
        "base": KRONOS_BASE_CONFIG,
        "small": KRONOS_SMALL_CONFIG,
        "mini": KRONOS_MINI_CONFIG,
    }[model_name]

    if kronos_path not in sys.path:
        sys.path.insert(0, kronos_path)
    from Kronos.model.kronos import Kronos as PyTorchKronos, KronosTokenizer as PyTorchKronosTokenizer

    rngs = nnx.Rngs(0)
    tokenizer = KronosTokenizer(TOKENIZER_BASE_CONFIG, rngs=rngs)
    model = Kronos(kronos_config, rngs=rngs)

    logger.info("Loading %s from HuggingFace (tokenizer: %s)...", kronos_hf, tokenizer_hf)
    pt_tokenizer = PyTorchKronosTokenizer.from_pretrained(tokenizer_hf)
    pt_model = PyTorchKronos.from_pretrained(kronos_hf)
    pt_tokenizer.eval()
    pt_model.eval()

    logger.info("Transferring weights to Flax Tokenizer...")
    load_pt_state_dict_into_nnx(pt_tokenizer.state_dict(), tokenizer)
    logger.info("Transferring weights to Flax Predictor...")
    load_pt_state_dict_into_nnx(pt_model.state_dict(), model)

    tokenizer.eval()
    model.eval()
    logger.info("Model ready (%s variant).", model_name)
    return tokenizer, model
