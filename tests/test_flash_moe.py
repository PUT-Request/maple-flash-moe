"""File-backed expert activation must preserve checkpoint bits and routing."""

import os
from contextlib import nullcontext
import hashlib
from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx
import numpy as np
import pytest

from mlx_lm.models import switch_layers as sl


@pytest.fixture
def checkpoint(tmp_path):
    rng = np.random.default_rng(12)
    tensors = {}
    for part in ("up_proj", "gate_proj", "down_proj"):
        tensors[f"mlp.{part}.weight"] = mx.array(
            rng.integers(0, 2**32, (16, 32, 8), dtype=np.uint32)
        )
        tensors[f"mlp.{part}.row_alpha"] = mx.array(
            rng.uniform(0.001, 0.1, (16, 32)).astype(np.float32)
        ).astype(mx.bfloat16)
    mx.save_safetensors(str(tmp_path / "experts.safetensors"), tensors)
    return sl._build_file_index(tmp_path), tensors


@pytest.mark.parametrize("dtype", list(sl._SAFETENSORS_DTYPE))
@pytest.mark.parametrize("row_cache_limit", [0, 4])
def test_disk_reader_preserves_bits_order_and_duplicates(
    tmp_path, dtype, row_cache_limit
):
    np_dtype, mlx_dtype = sl._SAFETENSORS_DTYPE[dtype]
    raw = np.arange(48, dtype=np_dtype).reshape(6, 8)
    if dtype == "BF16":
        # Include NaNs, infinities, subnormals, and signed zero: a view must
        # preserve every bit, even when float conversion would canonicalize it.
        raw[0] = [0x0000, 0x8000, 0x0001, 0x7F80, 0xFF80, 0x7FC1, 0x3F80, 0xFFFF]
    path = tmp_path / "data"
    with path.open("wb") as file:
        file.write(b"header!!")
        file.write(raw.tobytes())
    disk = sl._DiskHolder(row_cache_limit=row_cache_limit)
    disk.index = {"x": (str(path), 8, dtype, raw.shape)}
    for ids in ([5, 0, 3, 0], [], [2], [-1]):
        result = disk.read("x", ids)
        assert result.dtype == mlx_dtype
        if dtype == "BF16":
            result = result.view(mx.uint16)
        np.testing.assert_array_equal(np.asarray(result), raw[ids])
    assert len(disk._maps) == (0 if row_cache_limit else 1)


def _cached_disk(tmp_path):
    raw = np.arange(48, dtype=np.uint32).reshape(6, 8)
    path = tmp_path / "rows"
    raw.tofile(path)
    disk = sl._DiskHolder(row_cache_limit=2)
    disk.index = {"x": (str(path), 0, "U32", raw.shape)}
    return disk, raw, path


def test_row_cache_reuses_overlap_and_discards_old_selections(tmp_path):
    disk, raw, _ = _cached_disk(tmp_path)
    with patch.object(os, "pread", wraps=os.pread) as read:
        for ids, expected_reads in (([3, 1], 2), ([1, 4], 3), ([3, 4], 4), ([4, 3], 4)):
            np.testing.assert_array_equal(np.asarray(disk.read("x", ids)), raw[ids])
            assert read.call_count == expected_reads
            cached = disk._row_cache["x"][2]
            assert cached.base is None
            assert cached.nbytes == 2 * raw[0].nbytes
        # Prefill unions larger than the row limit must not stay cached.
        disk.read("x", [0, 1, 2])
        assert "x" not in disk._row_cache
        disk.read("x", [3])
        assert read.call_count == 5


def test_row_cache_enforces_byte_limit_and_does_not_alias_results(tmp_path):
    disk, raw, _ = _cached_disk(tmp_path)
    result = disk.read("x", [0])
    result[0, 0] = 999
    mx.eval(result)
    np.testing.assert_array_equal(np.asarray(disk.read("x", [0])), raw[[0]])
    disk._row_cache_bytes = raw[0].nbytes - 1
    disk.read("x", [1])
    assert not disk._row_cache


def test_row_cache_handles_short_reads_and_rejects_truncation(tmp_path):
    disk, raw, path = _cached_disk(tmp_path)
    pread = os.pread
    with patch.object(
        os, "pread", side_effect=lambda fd, n, off: pread(fd, min(n, 3), off)
    ):
        np.testing.assert_array_equal(np.asarray(disk.read("x", [0])), raw[[0]])
    with path.open("r+b") as file:
        file.truncate(raw[0].nbytes)
    with pytest.raises(EOFError):
        disk.read("x", [1])
    for ids in ([6], [-7]):
        with pytest.raises(IndexError):
            disk.read("x", ids)


def test_row_cache_invalidates_changed_file_index(tmp_path):
    disk, raw, path = _cached_disk(tmp_path)
    disk.read("x", [0])
    disk.index["x"] = (str(path), raw[0].nbytes, "U32", (5, 8))
    np.testing.assert_array_equal(np.asarray(disk.read("x", [0])), raw[[1]])


@pytest.mark.parametrize("projection", ["up_gate_proj", "down_proj"])
def test_activation_matches_checkpoint_and_reuses_identical_set(checkpoint, projection):
    index, tensors = checkpoint
    resident = SimpleNamespace()
    sl._flash_init_resident(resident, 8, index, f"mlp.{projection}")
    parts = ["up_proj", "gate_proj"] if projection == "up_gate_proj" else ["down_proj"]
    with patch.object(resident.disk, "read", wraps=resident.disk.read) as read:
        for ids in ([7, 1, 5], [7, 1, 5], list(range(16)), [9, 0], [0, 9]):
            previous = getattr(resident, "_active_id_list", None)
            count = read.call_count
            sl._flash_set_active(resident, mx.array(ids, dtype=mx.int32))
            if previous == ids:
                assert read.call_count == count
            want_weight = mx.concatenate(
                [tensors[f"mlp.{part}.weight"][mx.array(ids)] for part in parts], axis=1
            )
            want_alpha = mx.concatenate(
                [tensors[f"mlp.{part}.row_alpha"][mx.array(ids)] for part in parts],
                axis=1,
            )
            assert mx.array_equal(resident.weight, want_weight)
            assert mx.array_equal(resident.scales, want_alpha[..., None])
            assert mx.array_equal(resident.biases, -want_alpha[..., None])
            expected_map = np.full(16, -1, dtype=np.int32)
            expected_map[ids] = np.arange(len(ids))
            np.testing.assert_array_equal(np.asarray(resident.slot_of), expected_map)
            assert resident._age.shape == (len(ids),)
            requests = mx.array([ids[-1], ids[0], ids[-1]])
            resident._flash_ifp = False
            ordinary = sl._flash_resolve(resident, requests)
            resident._flash_ifp = True
            fast = sl._flash_resolve(resident, requests)
            assert mx.array_equal(fast, ordinary)
    # Reattaching the same object must discard the no-op activation shortcut.
    sl._flash_init_resident(resident, 8, index, f"mlp.{projection}")
    sl._flash_set_active(resident, [0, 9])
    assert mx.array_equal(resident.weight, want_weight)
    # An unrestricted paging call changes resident rows, so the original set
    # must be read again even when the caller requests the same ids as before.
    sl._flash_page(resident, 15)
    with patch.object(resident.disk, "read", wraps=resident.disk.read) as read:
        sl._flash_set_active(resident, [0, 9])
        assert read.call_count > 0
    assert mx.array_equal(resident.weight, want_weight)


@pytest.fixture
def tiny_flash_model(tmp_path):
    from mlx_lm.models import maple

    mx.random.seed(7)
    args = maple.ModelArgs(
        hidden_size=256,
        moe_intermediate_size=128,
        num_experts=16,
        num_experts_per_tok=8,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        vocab_size=128,
        layer_types=["sliding_attention", "full_attention"],
        quantization={"group_size": 128, "bits": 2},
    )
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
            weights = mx.split(linear.weight, len(parts), axis=1)
            alphas = mx.split(
                linear.scales[..., 0].astype(mx.bfloat16), len(parts), axis=1
            )
            for part, weight, alpha in zip(parts, weights, alphas):
                prefix = f"model.layers.{i}.mlp.switch_mlp.{part}"
                tensors[f"{prefix}.weight"] = weight
                tensors[f"{prefix}.row_alpha"] = alpha
    mx.save_safetensors(str(tmp_path / "experts.safetensors"), tensors)
    model.set_dtype(mx.bfloat16)
    mx.eval(model.parameters())

    return model


@pytest.mark.parametrize("reselect_every", [0, 1, 3])
def test_ifp_forward_matches_checked_routing(
    tmp_path, tiny_flash_model, reselect_every
):
    from mlx_lm.models import maple

    model = tiny_flash_model

    def run(ifp):
        maple.FlashMoE(active=8, reselect_every=reselect_every).attach(
            model, str(tmp_path)
        )
        for layer in model.layers:
            for name in ("up_gate_proj", "down_proj"):
                getattr(layer.mlp.switch_mlp, name)._flash_ifp = ifp
        cache = model.make_cache()
        outputs = []
        # Repeated prefill chunks, a prefill/decode transition, and reselection.
        for tokens in ([[1, 2, 3]], [[4, 5]], [[6]], [[7]], [[8]], [[9]]):
            result = model(mx.array(tokens), cache=cache)
            mx.eval(result)
            outputs.append(result)
        return outputs

    reference, optimized = run(False), run(True)
    for before, after in zip(reference, optimized):
        assert mx.array_equal(before, after)


@pytest.mark.parametrize("reselect_every", [0, 1, 3])
@pytest.mark.parametrize("chunk_size", [1, 3])
@pytest.mark.parametrize("batched", [False, True])
def test_cache_only_prefill_preserves_kv_and_decode(
    tmp_path, tiny_flash_model, reselect_every, chunk_size, batched
):
    from mlx_lm.generate import PromptProcessingBatch
    from mlx_lm.models import maple

    model = tiny_flash_model

    def run(hook):
        maple.FlashMoE(active=8, reselect_every=reselect_every).attach(
            model, str(tmp_path)
        )
        cache = model.make_cache()
        processor = None
        if batched:
            processor = PromptProcessingBatch(
                model, [0], [cache], prefill_step_size=chunk_size
            )
            cache = processor.prompt_cache
        digest = hashlib.sha256()
        last_mlp = model.layers[-1].mlp
        with patch.object(last_mlp, "_select", wraps=last_mlp._select) as select:
            with nullcontext() if hook else patch.object(model, "prefill_cache", None):
                prompt = [1, 2, 3, 4, 5, 6, 7]
                for start in range(0, len(prompt), chunk_size):
                    chunk = prompt[start : start + chunk_size]
                    if processor is not None:
                        processor.prompt([chunk])
                    else:
                        call = model.prefill_cache if hook else model
                        call(mx.array([chunk]), cache=cache)
                        mx.eval([c.state for c in cache])
                    for c in cache:
                        for array in c.state:
                            digest.update(
                                np.array(
                                    array.view(mx.uint16)
                                    if array.dtype == mx.bfloat16
                                    else array
                                ).tobytes()
                            )
                prefill_selections = select.call_count
                logits = []
                for token in [8, 9, 10]:
                    result = model(mx.array([[token]]), cache=cache)
                    mx.eval(result)
                    logits.append(result)
        return digest.hexdigest(), logits, prefill_selections

    reference, optimized = run(False), run(True)
    assert reference[0] == optimized[0]
    assert all(mx.array_equal(a, b) for a, b in zip(reference[1], optimized[1]))
    if reselect_every == 1:
        assert reference[2] > 0
        assert optimized[2] == 0
    else:
        assert reference[2] == optimized[2]


@pytest.mark.parametrize("reselect_every", [0, 1, 3])
def test_generate_step_uses_cache_only_prefill(
    tmp_path, tiny_flash_model, reselect_every
):
    from mlx_lm.generate import generate_step
    from mlx_lm.models import maple

    model = tiny_flash_model
    outputs = []
    for hook in (False, True):
        maple.FlashMoE(active=8, reselect_every=reselect_every).attach(
            model, str(tmp_path)
        )
        with nullcontext() if hook else patch.object(model, "prefill_cache", None):
            result = list(
                generate_step(
                    mx.array([1, 2, 3, 4, 5, 6, 7]),
                    model,
                    max_tokens=2,
                    prefill_step_size=3,
                )
            )
            mx.eval([logprobs for _, logprobs in result])
            outputs.append(result)
    assert [t for t, _ in outputs[0]] == [t for t, _ in outputs[1]]
    assert all(mx.array_equal(a[1], b[1]) for a, b in zip(*outputs))
