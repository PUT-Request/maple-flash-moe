"""Compare file-backed expert activation with a git revision on Apple Silicon.

Run: python benchmarks/flash_moe_benchmark.py --baseline-ref 210e6d6
Synthetic weights use Maple's real projection dimensions. Timings include disk
row copies, slot maps, and GPU evaluation; they are NOT model token throughput.
The OS page cache is warm, as with a model that fits the filesystem cache.
"""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import statistics
import subprocess
import tempfile
import time
import types
from unittest.mock import patch

import mlx.core as mx
import numpy as np

from mlx_lm.models import switch_layers


def baseline_module(ref, filename="switch_layers"):

    source = subprocess.check_output(
        ["git", "show", f"{ref}:mlx_lm/models/{filename}.py"], text=True
    )
    module = types.ModuleType("mlx_lm.models._benchmark_baseline")
    module.__package__ = "mlx_lm.models"
    exec(compile(source, f"{ref}:switch_layers.py", "exec"), module.__dict__)
    return module


def make_projection(module, index):
    projection = types.SimpleNamespace()
    module._flash_init_resident(projection, 8, index, "mlp.up_gate_proj")
    mx.eval(projection.weight, projection.scales, projection.biases)
    return projection


def evaluate(projection):
    mx.eval(projection.weight, projection.scales, projection.biases, projection.slot_of)


def measure(module, index, requests, repeats, host_ids):
    projection = make_projection(module, index)
    timings = []
    for i in range(repeats + 5):
        ids = requests[i % len(requests)]
        start = time.perf_counter()
        module._flash_set_active(
            projection, ids if host_ids else mx.array(ids, dtype=mx.int32)
        )
        evaluate(projection)
        if i >= 5:
            timings.append(time.perf_counter() - start)
    return statistics.median(timings) * 1000


def measure_forward(ref, directory, repeats):
    """Whole forward passes on a small synthetic model; no checkpoint download."""
    from mlx_lm.models import maple

    baseline = baseline_module(ref)
    old_maple = baseline_module(ref, "maple")
    args = maple.ModelArgs(
        hidden_size=256,
        moe_intermediate_size=128,
        num_experts=64,
        num_experts_per_tok=8,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        vocab_size=128,
        layer_types=["sliding_attention", "full_attention"],
        quantization={"group_size": 128, "bits": 2},
    )
    mx.random.seed(42)
    model = maple.Model(args)
    tensors = {}
    for i, layer in enumerate(model.layers):
        for name in ("up_gate_proj", "down_proj"):
            linear = getattr(layer.mlp.switch_mlp, name).to_quantized(
                group_size=128, bits=2
            )
            setattr(layer.mlp.switch_mlp, name, linear)
            parts = (
                ("up_proj", "gate_proj") if name == "up_gate_proj" else ("down_proj",)
            )
            for part, weight, alpha in zip(
                parts,
                mx.split(linear.weight, len(parts), axis=1),
                mx.split(linear.scales[..., 0].astype(mx.bfloat16), len(parts), axis=1),
            ):
                prefix = f"model.layers.{i}.mlp.switch_mlp.{part}"
                tensors[prefix + ".weight"] = weight
                tensors[prefix + ".row_alpha"] = alpha
    mx.save_safetensors(str(Path(directory) / "model.safetensors"), tensors)
    model.set_dtype(mx.bfloat16)
    mx.eval(model.parameters())
    timings, outputs = [], []
    for original in (True, False):
        with ExitStack() as stack:
            if original:
                for name in (
                    "_flash_init_resident",
                    "_flash_set_active",
                    "_flash_resolve",
                ):
                    stack.enter_context(
                        patch.object(switch_layers, name, getattr(baseline, name))
                    )
                stack.enter_context(
                    patch.object(
                        maple.MapleGate, "set_active", old_maple.MapleGate.set_active
                    )
                )
                stack.enter_context(
                    patch.object(
                        maple.MapleSparseMoeBlock,
                        "_select",
                        old_maple.MapleSparseMoeBlock._select,
                    )
                )
            prefill, decode, logits = [], [], []
            for trial in range(repeats + 2):
                maple.FlashMoE(active=8, reselect_every=1).attach(model, directory)
                cache = model.make_cache()
                start = time.perf_counter()
                result = model(mx.array([list(range(32))]), cache=cache)
                mx.eval(result)
                if trial >= 2:
                    prefill.append(time.perf_counter() - start)
                if trial == repeats + 1:
                    logits.append(result)
                for token in range(32, 44):
                    start = time.perf_counter()
                    result = model(mx.array([[token]]), cache=cache)
                    mx.eval(result)
                    if trial >= 2:
                        decode.append(time.perf_counter() - start)
                    if trial == repeats + 1:
                        logits.append(result)
            timings.append(
                (statistics.median(prefill) * 1000, statistics.median(decode) * 1000)
            )
            outputs.append(logits)
    assert all(mx.array_equal(a, b) for a, b in zip(*outputs)), "forward logits differ"
    for i, phase in enumerate(
        ("synthetic_forward_prefill_32", "synthetic_forward_decode")
    ):
        before, after = timings[0][i], timings[1][i]
        print(
            json.dumps(
                dict(
                    scenario=phase,
                    baseline_ms=round(before, 3),
                    optimized_ms=round(after, 3),
                    speedup=round(before / after, 2),
                    exact=True,
                )
            ),
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ref", default="210e6d6")
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument(
        "--forward",
        action="store_true",
        help="Also measure a two-layer synthetic Maple model",
    )
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    baseline = baseline_module(args.baseline_ref)
    rng = np.random.default_rng(42)
    with tempfile.TemporaryDirectory() as directory:
        # Raw file index has the same layout used by the safetensors reader.
        index = {}
        for part in ("up_proj", "gate_proj"):
            for name, shape, dtype in (
                ("weight", (256, 512, 128), "U32"),
                ("row_alpha", (256, 512), "BF16"),
            ):
                key = f"mlp.{part}.{name}"
                path = Path(directory) / key
                if dtype == "U32":
                    data = rng.integers(0, 2**32, size=shape, dtype=np.uint32)
                else:
                    data = (
                        rng.uniform(0.001, 0.1, shape)
                        .astype(np.float32)
                        .view(np.uint32)
                        >> 16
                    ).astype(np.uint16)
                data.tofile(path)
                index[key] = (str(path), 0, dtype, shape)
        scenarios = {
            "decode_changing_8": [list(range(i, i + 8)) for i in range(0, 248, 8)],
            "decode_repeated_8": [list(range(8))],
            "prefill_reordered_256": [list(range(256)), list(reversed(range(256)))],
            "prefill_repeated_256": [list(range(256))],
        }
        for name, requests in scenarios.items():
            old, new = make_projection(baseline, index), make_projection(
                switch_layers, index
            )
            for ids in requests[:2]:
                baseline._flash_set_active(old, mx.array(ids, dtype=mx.int32))
                switch_layers._flash_set_active(new, ids)
                evaluate(old)
                evaluate(new)
                for field in ("weight", "scales", "biases", "slot_of"):
                    assert mx.array_equal(
                        getattr(old, field), getattr(new, field)
                    ), field
            before = measure(baseline, index, requests, args.repeats, False)
            after = measure(switch_layers, index, requests, args.repeats, True)
            print(
                json.dumps(
                    dict(
                        scenario=name,
                        baseline_ms=round(before, 3),
                        optimized_ms=round(after, 3),
                        speedup=round(before / after, 2),
                        exact=True,
                    )
                ),
                flush=True,
            )

    if args.forward:
        with tempfile.TemporaryDirectory() as directory:
            measure_forward(args.baseline_ref, directory, args.repeats)


if __name__ == "__main__":
    main()
