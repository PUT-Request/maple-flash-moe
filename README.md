# Maple on mlx-lm

Maple is a 20B-A1B ternary MoE with 24 layers, 256 experts, top-8, 512-token sliding
window on 3 of every 4 layers. Weights are 2-bit packed `{-α, 0, +α}`, one α per
row. 

This fork runs on the stock MLX build for portability. We intend to release a faster custom
library in the coming days.

## Setup

Requires Apple Silicon and [uv](https://docs.astral.sh/uv/).

```sh
git clone git@github.com:eli32-vlc/maple-flash-moe.git
cd maple-flash-moe
./setup.sh
source .venv/bin/activate
hf download deepgrove/maple-2bit-mlx --local-dir maple-2bit-mlx
```

## Run

This fork ships its own `maple.py` implementation, so it takes precedence over
any `model_file` baked into a checkpoint — you always run the fixed code here,
even against older `deepgrove/maple-2bit-mlx` snapshots. `--trust-remote-code`
is therefore optional.

### Best quality (recommended)

Flash-MoE with exactly the model's top-8 routing (`--active-experts 8`) and no
expert reselect during decode (`--reselect-every 0`) gives the highest quality
at ~0.63 GB resident:

```sh
python -m mlx_lm server --model ./maple-2bit-mlx --trust-remote-code --flash-moe \
  --active-experts 8 --reselect-every 0 \
  --kv-bits 8 --kv-v-bits 4 --port 8080
```

### Fastest decode

Reselecting experts less often cuts the ~100 MB/token of expert disk reads.
`--active-experts 16 --reselect-every 32` is a good speed/quality balance
(~32× fewer decode-time reads):

```sh
python -m mlx_lm server --model ./maple-2bit-mlx --trust-remote-code --flash-moe \
  --active-experts 16 --reselect-every 32 \
  --kv-bits 8 --kv-v-bits 4 --port 8080
```

### Single-shot / interactive

```sh
python -m mlx_lm generate --model ./maple-2bit-mlx --trust-remote-code --flash-head \
  --prompt "Write a haiku about a grove." --temp 1.0 --top-p 0.95 --top-k 20

python -m mlx_lm chat --model ./maple-2bit-mlx --trust-remote-code --max-tokens -1 \
  --temp 1.0 --top-p 0.95
```

Enable flash head for extra speed.
```sh
python -m mlx_lm chat --model ./maple-2bit-mlx --trust-remote-code --max-tokens -1 \
  --temp 1.0 --top-p 0.95 --flash-head
```

| chip | head | decode tok/s | prefill tok/s | peak |
| --- | --- | --- | --- | --- |
| M4 | exact (default) | 169 | 1075 | 6.51 GB |
| M4 | `--flash-head` | **218** | 1075 | 6.69 GB |
| M5 Pro | exact (default) | 359 | 3773 | 6.73 GB |
| M5 Pro | `--flash-head` | **395** | 3857 | 6.92 GB |

### Tuning guide

The flash-MoE flags trade memory, decode speed, and quality. Maple routes each
token through its top-8 of 256 experts; flash-MoE keeps only an *active* subset
resident in DRAM and streams the rest from disk.

| knob | default | lower | higher |
| --- | --- | --- | --- |
| `--active-experts` | 8 | not below 8 (must be ≥ top-k) | more resident experts → fewer disk reads → faster decode; `256` = full model in RAM (~5.9 GB) |
| `--reselect-every` | 1 | 0 = keep the set fixed after prefill → best quality | N → reselect every N tokens → ~N× fewer disk reads, routing grows stale between reselects |
| `--shared-experts` | 0 | 0 (keep, quality is lossless while `active ≥ top_k + shared`) | fixed always-resident experts, useful once you raise `--active-experts` |
| `--kv-bits` / `--kv-v-bits` | none (fp16 KV) | 8/4 recommended; 4/4 smallest KV | 8/8 or none → better KV fidelity, more RAM |
| `--prefill-step-size` | 2048 | less RAM per prefill step | 8192 → faster prefill on long prompts |
| `--max-tokens` | 256 | — | keep high (e.g. 64000); this model's reasoning block needs room or replies come back empty |
| `--temp` | 0 | 0 = greedy, **loops/degenerates on 2-bit weights** | more variety but more drift; 0.5–0.8 is the sweet spot |
| `--top-p` | 1.0 | 0.9 tighter/coherent | closer to 1.0 more varied |
| `--top-k` | 0 (off) | 20–40 guards against garbage tokens | off/higher more freedom |
| `--min-p` | 0 | 0.05 recommended anti-repetition floor | higher (0.1–0.2) kills repetition, can over-restrict |

Quick fixes:
- **Off-topic drift / repetition** → raise `--min-p` (0.1–0.2), lower `--temp` (0.5–0.6), lower `--top-k` (20).
- **Too slow** → raise `--active-experts` or `--reselect-every`; drop `--flash-moe` if RAM allows.
- **OOM on long prompts** → lower `--kv-bits`/`--kv-v-bits`, lower `--prefill-step-size`.
- **Empty replies** → `--max-tokens` is too low for the reasoning block.

## Convert

```sh
python -m mlx_lm.ternary /path/to/maple-bf16 -o maple-2bit-mlx --flash-head
```

Streams and converts shard by shard, so the 38 GB bf16 source is never fully resident.

- `--flash-head` — ~2 min of k-means, score 4748
  vocabulary-cluster centroids, then compute exact logits only for the top 512
  clusters (special tokens always scored). Greedy is exact whenever the true
  argmax is in a probed cluster. Attach to an already-converted
  directory with `python -m mlx_lm.ternary maple-2bit-mlx --flash-head-only`
  (rewrites in place; point it at a real directory, not hardlinks).
- `--group-scales` — repeat each row's α across every group (+0.6 GB), only for
  tools that read MLX quantized checkpoints generically. Default stores the row
  scale once as `row_alpha`; `sanitize()` expands it at load.

## Flash-MoE expert offload

Maple's 256 experts × 24 layers are the bulk of the resident memory: the full
model sits at **~5.9–6.5 GB** on Apple Silicon. Flash-MoE keeps only a small
*active* set of experts in DRAM and streams the rest from disk on demand.

How it works:

- **File-backed checkpoint.** Expert weights stay on disk. A byte-range reader
  (`_DiskHolder` in `mlx_lm/models/switch_layers.py`) reads only selected rows.
  Small decode selections use cached file handles and `pread`, reusing rows
  from the previous selection. Batch selections use memory-mapped NumPy views.
  The host cache retains at most eight rows / 2 MiB per source and clears that
  source on larger selections; for Maple this adds at most about 145 MiB of
  cached raw weights and scales, independently of the MLX execution tensors.
- **IFP (Inactive-Expert-Free Policy).** The gating function is masked to the
  active set, so only resident experts are ever scored.
- **Per-token reselect.** The active set is recomputed from the *current token's*
  true top-k routing (default every token via `--reselect-every 1`). A small LRU
  buffer in `_flash_page` keeps recently-used experts warm across tokens.
  `--reselect-every 0` keeps the set fixed after prefill (best quality, slower);
  a larger interval (e.g. 32) cuts the ~100 MB/token of expert disk reads ~N×.
- **Fused projections.** The checkpoint stores unfused `up_proj`/`gate_proj` plus
  a per-row `row_alpha`; the reader concatenates the two source rows per active
  expert and expands `row_alpha` to per-group scales/biases (BF16, bit-reinterpreted).
- **Shared LRU expert cache (optional).** `--moe-cache-size N` (FreeToken-style)
  replaces the per-layer resident set with one global slot pool shared across all
  layers: routed experts are paged in on demand and evicted by LRU, so recurring
  experts stay hot across tokens. Intended for memory-tight configs; the legacy
  per-layer path (default) is faster on Apple Silicon.

### Lossless at the default

With `--active-experts 8 --shared-experts 0` and per-token decode, the active set
*is* the model's true top-8 routing, so generated tokens are **bit-exact** vs the
full model (verified: weight/scales/biases max diff = 0.0). Larger
`--active-experts` trades a little quality for headroom; `--active-experts 256`
reproduces the full model exactly.

> Prefill note: during batch prefill every token routes to its own top-k set, so
> the union of experts can far exceed `active`. The prefill pass keeps that full
> union resident (only decode caps to `--active-experts`), which is what makes
> prefill exact; resident memory at generation time is still bounded by the
> activated set.

| config | peak RAM | notes |
| --- | --- | --- |
| full (no flash) | ~5.9–6.5 GB | baseline |
| `--flash-moe --active-experts 8 --shared-experts 0` | **0.63 GB** | lossless (top-8 routing) |
| `--flash-moe --active-experts 32 --shared-experts 2` | 0.94 GB | coherent, slight quality cost |
| `--flash-moe --active-experts 256` | 5.89 GB | equals full model |

> All flash-MoE memory numbers measured on a MacBook Air (Apple M2, 16 GB
> unified memory, stock MLX build).

> Note on `--shared-experts`: quality loss is zero whenever
> `active >= top_k + shared`. Since Maple's `top_k = 8`, keep `--shared-experts 0`
> at the default `active = 8`, or raise `--active-experts` above 8 if you enable
> shared experts.

### Run with flash-MoE

```sh
python -m mlx_lm generate --model ./maple-2bit-mlx --trust-remote-code \
  --flash-moe --active-experts 8 --shared-experts 0 \
  --prompt "Write a haiku about a grove." --temp 0.7 --top-p 0.9 --top-k 40

python -m mlx_lm chat --model ./maple-2bit-mlx --trust-remote-code \
  --flash-moe --active-experts 8 --kv-bits 8 --kv-v-bits 4 --max-tokens -1
```

### Reproduce the flash-MoE optimization benchmark

For interactive serving, `--prefill-step-size 1` minimizes expert memory but is
slow. A small increase to `--prefill-step-size 4`, with prompt/decode concurrency
both kept at 1, processes four prompt tokens per chunk. In a short warmed M2
test this raised prefill from 16.7 to 22.6 tokens/s, while peak process memory
rose from 1.34 to 1.72 GB. Larger contexts add KV-cache memory. At top-8 routing,
a four-token chunk selects at most 32 experts per layer, rather than the full
256. These timings are from the checkpoint harness, not an HTTP load test;
batching can also change floating-point rounding and sampled output.

Maple now uses a cache-only prefill path in generation and server prompt
processing. It skips the final layer's MLP and expert paging when its output
is discarded: that MLP cannot affect the KV caches. Fixed or stale flash
routing and shared expert caches retain the original MLP calls to preserve
routing state. This applies automatically, without changing server flags.

A paired M2 test using the actual server prompt-processing class, an 18-token
prompt, and step size 4 measured 17.75 → 18.58 prompt tokens/s (1.047×).
The two measured runs per variant followed two warmup runs; timings varied
substantially, so this is preliminary evidence of a modest gain. All compared
logits were bit-exact. Peak process memory across both variants was 1.93 GB;
the loader skipped the full 4.87 GB expert tensors. Reproduce with:

```sh
python benchmarks/maple_checkpoint_benchmark.py \
  --model /path/to/maple-2bit-mlx --variant optimized \
  --server-prefill --compare-prefill-hook --prefill-step-size 4 --tokens 1 \
  --prompt 'Please explain in simple terms why the sky looks blue during the day and red at sunset.'
```

The low-memory checkpoint benchmark keeps eight experts per layer in the MLX
execution tensors. It skips full expert tensors during model construction and
feeds the prompt one token at a time to bound the prefill expert set. The exact
head is enabled and the short KV cache is unquantized. A watchdog exits if
observed process memory exceeds 2 GiB; this is monitoring, not a hard OS
allocation limit.

Run the variants sequentially:

```sh
python benchmarks/maple_checkpoint_benchmark.py \
  --model /path/to/maple-2bit-mlx --variant baseline --tokens 8 --runs 3 \
  --continuation ' Paris, a city known for its history and culture.'
python benchmarks/maple_checkpoint_benchmark.py \
  --model /path/to/maple-2bit-mlx --variant optimized --tokens 8 --runs 3 \
  --continuation ' Paris, a city known for its history and culture.'
```

With `--runs 3`, the first run warms up the model and the next two contribute
16 timed decode steps. `--continuation` feeds a fixed, varied token sequence,
so a repeated greedy token cannot inflate cache-hit benefits. The SHA-256
covers every prompt/decode logit tensor, and the harness also checks that
repeated runs produce identical logits. `--profile /tmp/decode.prof` optionally
writes a decode-only cProfile trace for each run.

Measured on Apple M2 (16 GB), MLX 0.32.2, with the local 5.31 GB Maple checkpoint
against revision `210e6d6`:

| measurement | original | optimized |
| --- | --- | --- |
| warmed decode throughput | 11.62 tokens/s | 19.53 tokens/s |
| peak process memory | 1.44 GB | 1.34 GB |
| peak MLX allocations | 0.65 GB | 0.65 GB |

This short comparison measured **1.68× throughput**, with bit-exact logits.
The loader skipped 4.87 GB of expert data. It does not establish 5× overall
inference speed, long-context performance, or output quality. Cold-start runs
varied substantially; compare the same workload after warmup on your hardware.

The optimized path transfers routing IDs once, builds slot maps and masks on
the CPU, preserves BF16 scale bits directly, and materializes contiguous scale
groups for efficient M2 quantized matmuls. An unchanged active set reuses its
resident tensors; partially overlapping sets reuse raw rows from the bounded
host cache. The masked router bypasses two redundant synchronous residency
checks per layer. Routing, expert counts, and quantization settings are unchanged.

A separate synthetic activation/forward benchmark and regression tests are
available without downloading a checkpoint:

```sh
python benchmarks/flash_moe_benchmark.py --baseline-ref 210e6d6 --forward --repeats 20
python -m pytest tests/test_flash_moe.py tests/test_maple_kernels.py -q
```

## KV cache quantization

The KV cache can be quantized independently for keys and values:

- `--kv-bits 8 --kv-v-bits 4` → 8-bit keys, 4-bit values (recommended default).
  Wired through `mlx_lm/models/cache.py` and `mlx_lm/models/base.py` with separate
  `k_bits`/`v_bits`.
- Sliding-window layers use `RotatingKVCache`, which does not support KV
  quantization yet, so those layers keep full-precision KV.

## Server (OpenAI-compatible)

The server exposes OpenAI-style `/v1/chat/completions` and `/v1/completions` and
supports every flash-MoE / KV-quant flag:

```sh
python -m mlx_lm server --model ./maple-2bit-mlx --trust-remote-code \
  --flash-moe --active-experts 8 --shared-experts 0 \
  --kv-bits 8 --kv-v-bits 4 \
  --port 8080 --prefill-step-size 8192 --max-tokens 8192
```

It logs live progress: prefill percentage + tok/s, and per-phase
`Prompt: N tokens, X tok/s` / `Generation: N tokens, Y tok/s`.

> Sampling matters for 2-bit weights: the default server `temp=0` (greedy) can
> loop or degenerate on a 2-bit model. Use `--temp 0.7 --top-p 0.9 --top-k 40
> --min-p 0.05` for coherent output.

## Diff vs upstream mlx-lm

| file | what |
| --- | --- |
| `mlx_lm/models/maple.py` | the model (also copied into every converted checkpoint) |
| `mlx_lm/ternary.py` | bf16 → ternary converter + FlashHead generator |
| `tests/test_maple_kernels.py` | kernel + precision self-check — `pytest tests/test_maple_kernels.py -v` |
| `generate.py`, `chat.py`, `server.py`, `benchmark.py` | support for `--flash-head` flag |
| `mlx_lm/models/switch_layers.py` | flash-MoE disk holder + paging — byte-range expert streaming, IFP, per-token reselect |
| `mlx_lm/models/cache.py`, `mlx_lm/models/base.py` | q8/q4 KV cache quantization (separate `k_bits`/`v_bits`) |
| `mlx_lm/models/maple.py` | `FlashMoE`, `_select` (per-token top-k active set), `prepare_flash_moe` |
| `generate.py`, `server.py` | `--flash-moe`, `--active-experts`, `--shared-experts`, `--reselect-every`, `--kv-bits`, `--kv-v-bits` flags + server tok/s metrics |
| `setup.sh` | uv venv + editable install |
