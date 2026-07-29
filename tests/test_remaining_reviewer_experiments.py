import json

from experiments.rebuttal.make_self_forcing_seed0_manifest import build_records
from experiments.rebuttal.merge_qwen_rewrite_shards import (
    load_pair_mapping,
    merge_in_prompt_order,
)
from experiments.rebuttal.summarize_qwen_4step_comparison import audit_manifest


def make_prompt_files(tmp_path, count=944):
    prompts = [f"official prompt {index:04d}" for index in range(count)]
    rewrites = [f"qwen rewrite {index:04d}" for index in range(count)]
    prompt_path = tmp_path / "prompts.txt"
    rewrite_path = tmp_path / "rewrites.txt"
    prompt_path.write_text("\n".join(prompts) + "\n", encoding="utf-8")
    rewrite_path.write_text("\n".join(rewrites) + "\n", encoding="utf-8")
    return prompts, rewrites, prompt_path, rewrite_path


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
    mapping = load_pair_mapping([shard0, shard1])
    assert merge_in_prompt_order(prompts, mapping) == rewrites


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


def test_self_forcing_seed_manifest_encodes_two_even_odd_processes(tmp_path):
    prompts, rewrites, prompt_path, rewrite_path = make_prompt_files(tmp_path)
    records = build_records(prompt_path, rewrite_path)
    assert len(records) == 944
    for index, record in enumerate(records):
        assert record["prompt"] == prompts[index]
        assert record["extended_prompt"] == rewrites[index]
        assert record["sample_index"] == 0
        assert record["seed"] == 0
        assert record["rng_protocol"] == "self_forcing_two_shard_seed0"
        assert record["rng_shard_index"] == index % 2
        assert record["rng_position_in_shard"] == index // 2


def test_qwen_manifest_audit_rejects_per_record_seed_sequence(tmp_path):
    _, _, prompt_path, rewrite_path = make_prompt_files(tmp_path)
    records = build_records(prompt_path, rewrite_path)
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    audit = audit_manifest(prompt_path, rewrite_path, manifest_path)
    assert audit["process_seed"] == 0
    assert audit["num_generation_shards"] == 2

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
        raise AssertionError("Per-record seed sequence was accepted")
