import hashlib
import json
from pathlib import Path

import torch

from experiments.rebuttal.audit_ema_checkpoint import audit_checkpoint
from experiments.rebuttal.make_self_forcing_seed0_manifest import build_records
from experiments.rebuttal.merge_qwen_rewrite_shards import (
    load_pair_mapping,
    merge_in_prompt_order,
)
from experiments.rebuttal.resolve_all_gpus import resolve_requested
from experiments.rebuttal.summarize_qwen_4step_comparison import audit_manifest
from scripts.export_videos import (
    historical_self_forcing_random_shapes,
    load_manifest,
    select_records_for_shard,
)


ASSET_ROOT = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "rebuttal"
    / "assets"
    / "qwen_vbench"
)


def test_bundled_qwen_vbench_assets_are_pinned():
    expected_pairs = {
        "shard00_pairs.jsonl": (
            472,
            "53e85750f9fec2ff0a1af9b1d8ac9adf3c9e6b69dbf69cf529d3b56be4017d7e",
        ),
        "shard01_pairs.jsonl": (
            472,
            "a9126faa105e2aeb976b352877576f75a97b57e6784c78cb20d3b8c1d5dbdbb6",
        ),
    }
    for name, (expected_lines, expected_sha256) in expected_pairs.items():
        path = ASSET_ROOT / name
        assert len(path.read_text(encoding="utf-8").splitlines()) == expected_lines
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_sha256

    full_info_path = ASSET_ROOT / "VBench_full_info.json"
    assert (
        hashlib.sha256(full_info_path.read_bytes()).hexdigest()
        == "12d720a3f5ec60d7640edadd2272876056da098632171fc30356be25674c4deb"
    )
    full_info = json.loads(full_info_path.read_text(encoding="utf-8"))
    assert len(full_info) == 946
    assert len({record["prompt_en"] for record in full_info}) == 944


def make_prompt_files(tmp_path, count=944):
    prompts = [f"official prompt {index:04d}" for index in range(count)]
    rewrites = [f"qwen rewrite {index:04d}" for index in range(count)]
    prompt_path = tmp_path / "prompts.txt"
    rewrite_path = tmp_path / "rewrites.txt"
    prompt_path.write_text("\n".join(prompts) + "\n", encoding="utf-8")
    rewrite_path.write_text("\n".join(rewrites) + "\n", encoding="utf-8")
    return prompts, rewrites, prompt_path, rewrite_path


def make_manifest(prompts, rewrites, prompt_path, manifest_path):
    prompt_digest = hashlib.sha256(prompt_path.read_bytes()).hexdigest()
    records = []
    for index, (prompt, rewrite) in enumerate(zip(prompts, rewrites)):
        records.append(
            {
                "prompt_index": index,
                "sample_index": 0,
                "seed": index,
                "output_name": f"{prompt}-0.mp4",
                "prompt": prompt,
                "extended_prompt": rewrite,
                "prompt_file_sha256": prompt_digest,
            }
        )
    manifest_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    return records


def test_historical_pair_shards_are_reordered_by_official_prompt(tmp_path):
    prompts = ["prompt zero", "prompt one", "prompt two", "prompt three"]
    rewrites = ["rewrite zero", "rewrite one", "rewrite two", "rewrite three"]
    shard0 = tmp_path / "shard00_pairs.jsonl"
    shard1 = tmp_path / "shard01_pairs.jsonl"
    shard0.write_text(
        "\n".join(
            json.dumps({"prompt": prompts[index], "rewrite": rewrites[index]})
            for index in (0, 2)
        )
        + "\n",
        encoding="utf-8",
    )
    shard1.write_text(
        "\n".join(
            json.dumps({"prompt": prompts[index], "rewrite": rewrites[index]})
            for index in (1, 3)
        )
        + "\n",
        encoding="utf-8",
    )
    assert merge_in_prompt_order(prompts, load_pair_mapping([shard0, shard1])) == rewrites


def test_pair_shards_reject_duplicate_prompts(tmp_path):
    shard0 = tmp_path / "shard00_pairs.jsonl"
    shard1 = tmp_path / "shard01_pairs.jsonl"
    record = json.dumps({"prompt": "duplicate", "rewrite": "rewrite"}) + "\n"
    shard0.write_text(record, encoding="utf-8")
    shard1.write_text(record, encoding="utf-8")
    try:
        load_pair_mapping([shard0, shard1])
    except ValueError as error:
        assert "Duplicate prompt" in str(error)
    else:
        raise AssertionError("Duplicate prompt was accepted")


def test_qwen_manifest_matches_historical_two_stream_seed_zero(tmp_path):
    prompts, rewrites, prompt_path, rewrite_path = make_prompt_files(tmp_path)
    manifest_path = tmp_path / "manifest.jsonl"
    records = build_records(prompt_path, rewrite_path)
    manifest_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    audit = audit_manifest(prompt_path, rewrite_path, manifest_path)
    assert audit["process_seed"] == 0
    assert audit["historical_num_rng_streams"] == 2
    assert audit["historical_prompt_sharding"] == "even_odd"
    for index, record in enumerate(records):
        assert record["seed"] == index // 2
        assert record["initial_noise_seed"] == index // 2
        assert record["rng_shard_index"] == index % 2
        assert record["rng_position_in_shard"] == index // 2

    records[1]["seed"] = 1
    manifest_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    try:
        audit_manifest(prompt_path, rewrite_path, manifest_path)
    except ValueError as error:
        assert "not protocol matched" in str(error)
    else:
        raise AssertionError("Non-historical seed was accepted")


def test_eight_gpu_partition_preserves_two_historical_streams(tmp_path):
    _, _, prompt_path, rewrite_path = make_prompt_files(tmp_path)
    records = build_records(prompt_path, rewrite_path)
    selected_indices = set()
    for shard_index in range(8):
        selected = select_records_for_shard(records, shard_index, 8)
        assert len(selected) == 118
        assert {record["rng_shard_index"] for record in selected} == {
            shard_index % 2
        }
        assert [record["rng_position_in_shard"] for record in selected] == list(
            range(shard_index // 2, 472, 4)
        )
        selected_indices.update(record["prompt_index"] for record in selected)
    assert selected_indices == set(range(944))


def test_historical_rng_schedule_matches_self_forcing_all4():
    shapes = historical_self_forcing_random_shapes()
    assert shapes == [(3, 16, 60, 104)] * 21
    frame_equivalents = sum(shape[0] for shape in shapes)
    assert frame_equivalents == 63


def test_manifest_extended_prompts_must_match_dataset(tmp_path):
    prompts = ["prompt zero"]
    rewrites = ["rewrite zero"]
    prompt_path = tmp_path / "prompts.txt"
    prompt_path.write_text("prompt zero\n", encoding="utf-8")
    manifest_path = tmp_path / "manifest.jsonl"
    records = make_manifest(prompts, rewrites, prompt_path, manifest_path)
    dataset = [{"prompts": prompts[0], "extended_prompts": rewrites[0]}]
    assert load_manifest(str(manifest_path), dataset, str(prompt_path)) == records

    records[0]["extended_prompt"] = "wrong rewrite"
    manifest_path.write_text(json.dumps(records[0]) + "\n", encoding="utf-8")
    try:
        load_manifest(str(manifest_path), dataset, str(prompt_path))
    except ValueError as error:
        assert "extended_prompt does not match" in str(error)
    else:
        raise AssertionError("Mismatched extended prompt was accepted")


def test_all_gpu_resolution_rejects_subsets():
    assert resolve_requested("all", [0, 1, 2, 3]) == [0, 1, 2, 3]
    assert resolve_requested("3,2,1,0", [0, 1, 2, 3]) == [0, 1, 2, 3]
    try:
        resolve_requested("0,1", [0, 1, 2, 3])
    except ValueError as error:
        assert "every physical GPU" in str(error)
    else:
        raise AssertionError("GPU subset was accepted")


def test_released_checkpoint_audit_selects_generator_ema(tmp_path):
    checkpoint = tmp_path / "self_forcing_dmd.pt"
    torch.save({"generator_ema": {"weight": torch.ones(2)}}, checkpoint)
    audit = audit_checkpoint(checkpoint)
    assert audit["use_ema"] is True
    assert audit["selected_weight_source"] == "generator_ema"
    assert audit["num_generator_ema_tensors"] == 1
