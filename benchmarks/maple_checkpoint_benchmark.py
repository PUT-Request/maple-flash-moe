"""Short, memory-monitored A/B benchmark of an on-disk Maple checkpoint.

Expert tensors are replaced by lazy placeholders during construction and dropped
before evaluation. Flash-MoE then reads only eight selected experts per layer.
Prompt chunks default to one token; --prefill-step-size tests small batches.
Run each variant in a separate process; compare the reported logit hashes.
"""

import argparse
import cProfile
from contextlib import ExitStack
import gc
import hashlib
import json
import os
from pathlib import Path
import resource
import statistics
import sys
import threading
import time
from unittest.mock import patch

import mlx.core as mx
import numpy as np

from flash_moe_benchmark import baseline_module
from mlx_lm.models import maple, switch_layers
from mlx_lm.utils import load


def memory_watchdog(limit):
    while True:
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform != "darwin":
            peak *= 1024
        if peak > limit:
            print(
                json.dumps(
                    {"error": "process memory budget exceeded", "peak_bytes": peak}
                ),
                flush=True,
            )
            os._exit(2)
        time.sleep(0.05)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--variant", choices=("baseline", "optimized"), required=True)
    parser.add_argument("--baseline-ref", default="210e6d6")
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--prefill-step-size", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument(
        "--server-prefill",
        action="store_true",
        help="Use the server's PromptProcessingBatch, evaluating KV caches rather than discarded logits",
    )
    parser.add_argument(
        "--compare-prefill-hook",
        action="store_true",
        help="Compare the cache-only hook on/off in one process (requires --server-prefill)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Repeat in-process; with 2-3 runs, the first is warmup",
    )
    parser.add_argument("--rss-limit-gib", type=float, default=2.0)
    parser.add_argument(
        "--profile", type=Path, help="Write a decode-only cProfile trace"
    )
    parser.add_argument(
        "--continuation",
        help="Feed known continuation tokens to avoid greedy repetition",
    )
    args = parser.parse_args()
    if not 1 <= args.tokens <= 16:
        parser.error("--tokens must be between 1 and 16 for this short test")
    if not 1 <= args.runs <= 3:
        parser.error("--runs must be between 1 and 3")
    if args.compare_prefill_hook and not args.server_prefill:
        parser.error("--compare-prefill-hook requires --server-prefill")
    if args.compare_prefill_hook and args.variant != "optimized":
        parser.error("--compare-prefill-hook requires --variant optimized")
    threading.Thread(
        target=memory_watchdog, args=(args.rss_limit_gib * 2**30,), daemon=True
    ).start()
    mx.set_cache_limit(32 * 2**20)
    mx.set_memory_limit(1024 * 2**20)  # MLX guideline; the watchdog also monitors RSS.
    index = switch_layers._build_file_index(args.model)
    disk = switch_layers._DiskHolder()
    disk.index = index
    expert_bytes = 0
    dense_bytes = 0

    def selective_load(filename, *unused_args, **unused_kwargs):
        nonlocal expert_bytes, dense_bytes
        arrays = {}
        for key, (path, offset, dtype, shape) in index.items():
            if Path(path).resolve() != Path(filename).resolve():
                continue
            np_dtype, mx_dtype = switch_layers._SAFETENSORS_DTYPE[dtype]
            size = int(np.prod(shape)) * np.dtype(np_dtype).itemsize
            if ".mlp.switch_mlp." in key:
                # No checkpoint expert bytes are read here. These placeholders
                # only provide shapes for sanitize/quantization/strict loading.
                arrays[key] = mx.zeros(shape, dtype=mx_dtype)
                expert_bytes += size
            else:
                arrays[key] = disk.read(key, list(range(shape[0])))
                mx.eval(arrays[key])
                dense_bytes += size
        return arrays

    with ExitStack() as stack:
        if args.variant == "baseline":
            old_switch = baseline_module(args.baseline_ref)
            old_maple = baseline_module(args.baseline_ref, "maple")
            for name in ("_flash_init_resident", "_flash_set_active", "_flash_resolve"):
                stack.enter_context(
                    patch.object(switch_layers, name, getattr(old_switch, name))
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
        with patch.object(mx, "load", selective_load):
            model, tokenizer = load(
                args.model, lazy=True, model_config={"use_flash_head": False}
            )
        maple.prepare_flash_moe(
            model, active=8, shared=0, reselect_every=1, model_path=str(args.model)
        )
        for layer in model.layers:
            for name in ("up_gate_proj", "down_proj"):
                assert getattr(layer.mlp.switch_mlp, name).weight.shape[0] == 8
        # No full expert tensor remains in the graph when evaluation begins.
        disk._maps.clear()
        gc.collect()
        mx.eval(model.parameters())
        print(
            json.dumps(
                {
                    "phase": "loaded",
                    "variant": args.variant,
                    "expert_bytes_skipped": expert_bytes,
                    "nonexpert_bytes_read": dense_bytes,
                    "mlx_active_gb": mx.get_active_memory() / 1e9,
                }
            ),
            flush=True,
        )
        measured, hashes, prefill_times = [], [], []
        hook_times = {False: [], True: []}
        prefill_hook = getattr(model, "prefill_cache", None)
        run_modes = (
            [False, True, True, False, False, True]
            if args.compare_prefill_hook
            else [args.variant == "optimized"] * args.runs
        )
        for run, hook_enabled in enumerate(run_modes):
            model.prefill_cache = prefill_hook if hook_enabled else None
            warmup = (
                run < 2 if args.compare_prefill_hook else args.runs > 1 and run == 0
            )
            cache = model.make_cache()
            for layer in model.layers:
                layer.mlp._selected = False
                layer.mlp._tok = 0
            digest = hashlib.sha256()
            prompt = tokenizer.encode(args.prompt)

            def step(token):
                logits = model(mx.array([[token]]), cache=cache)
                mx.eval(logits)
                return logits

            if args.server_prefill:
                from mlx_lm.generate import PromptProcessingBatch

                batch = PromptProcessingBatch(
                    model, [0], [cache], prefill_step_size=args.prefill_step_size
                )
                cache = batch.prompt_cache
            start = time.perf_counter()
            if args.server_prefill:
                batch.prompt([prompt[:-1]])
                logits = step(prompt[-1])
                digest.update(np.asarray(logits.astype(mx.float32)).tobytes())
            else:
                for offset in range(0, len(prompt), args.prefill_step_size):
                    chunk = prompt[offset : offset + args.prefill_step_size]
                    logits = model(mx.array([chunk]), cache=cache)
                    mx.eval(logits)
                    digest.update(np.asarray(logits.astype(mx.float32)).tobytes())
            prefill_seconds = time.perf_counter() - start
            generated = []
            elapsed = []
            continuation = (
                tokenizer.encode(args.continuation) if args.continuation else None
            )
            if continuation is not None and len(continuation) < args.tokens:
                parser.error("continuation must contain at least --tokens tokens")
            profiler = cProfile.Profile() if args.profile else None
            for i in range(args.tokens):
                token = (
                    continuation[i]
                    if continuation is not None
                    else int(mx.argmax(logits[0, -1]).item())
                )
                generated.append(token)
                if profiler:
                    profiler.enable()
                start = time.perf_counter()
                logits = step(token)
                elapsed.append(time.perf_counter() - start)
                if profiler:
                    profiler.disable()
                digest.update(np.asarray(logits.astype(mx.float32)).tobytes())
                print(
                    json.dumps(
                        {
                            "phase": "decode",
                            "step": i + 1,
                            "milliseconds": round(elapsed[-1] * 1000, 2),
                            "mlx_active_gb": round(mx.get_active_memory() / 1e9, 3),
                        }
                    ),
                    flush=True,
                )
            if profiler:
                profiler.dump_stats(
                    str(args.profile) + (f".{run + 1}" if args.runs > 1 else "")
                )
            print(
                json.dumps(
                    {
                        "variant": args.variant,
                        "run": run + 1,
                        "warmup": warmup,
                        "prefill_hook": hook_enabled,
                        "teacher_forcing": continuation is not None,
                        "prompt_tokens": len(prompt),
                        "prefill_step_size": args.prefill_step_size,
                        "prefill_tokens_per_second": len(prompt) / prefill_seconds,
                        "decode_steps": len(elapsed),
                        "prefill_seconds": prefill_seconds,
                        "decode_tokens_per_second": len(elapsed) / sum(elapsed),
                        "median_decode_ms": statistics.median(elapsed) * 1000,
                        "mlx_peak_gb": mx.get_peak_memory() / 1e9,
                        "process_peak_gb": resource.getrusage(
                            resource.RUSAGE_SELF
                        ).ru_maxrss
                        / (1e9 if sys.platform == "darwin" else 1e9 / 1024),
                        "logit_sha256": digest.hexdigest(),
                        "token_ids": generated,
                        "text": tokenizer.decode(generated),
                    }
                ),
                flush=True,
            )

            if not warmup:
                measured.extend(elapsed)
                prefill_times.append(prefill_seconds)
                hook_times[hook_enabled].append(prefill_seconds)
            hashes.append(digest.hexdigest())
        model.prefill_cache = prefill_hook
        assert len(set(hashes)) == 1, "logits changed between repeated runs"
        print(
            json.dumps(
                {
                    "summary": True,
                    "variant": args.variant,
                    "measured_decode_steps": len(measured),
                    "decode_tokens_per_second": len(measured) / sum(measured),
                    "median_decode_ms": statistics.median(measured) * 1000,
                    "prefill_tokens_per_second": len(prompt)
                    * len(prefill_times)
                    / sum(prefill_times),
                    "server_prefill": args.server_prefill,
                    "logit_sha256": hashes[-1],
                    "mlx_peak_gb": mx.get_peak_memory() / 1e9,
                    "process_peak_gb": resource.getrusage(
                        resource.RUSAGE_SELF
                    ).ru_maxrss
                    / (1e9 if sys.platform == "darwin" else 1e9 / 1024),
                }
            ),
            flush=True,
        )
        if args.compare_prefill_hook:
            rates = {
                key: len(prompt) * len(times) / sum(times)
                for key, times in hook_times.items()
            }
            print(
                json.dumps(
                    {
                        "comparison": "cache_only_prefill",
                        "before_tokens_per_second": rates[False],
                        "after_tokens_per_second": rates[True],
                        "speedup": rates[True] / rates[False],
                        "exact_logits": True,
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
