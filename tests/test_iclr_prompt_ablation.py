"""CPU fixtures for prompt comparisons; never measured video results."""
import hashlib
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from experiments.iclr.prompt_ablation import (
    command_for_condition, conditions, expected_intent, input_payloads, prepare_inputs, summarize,
)
from experiments.iclr.protocol import REPO
from experiments.rebuttal.consolidate_results import NORMALIZE_DIC


@pytest.mark.parametrize("samples", [1, 5])
def test_original_and_qwen_keep_identical_sampling_records(samples, tmp_path):
    payloads = input_payloads(samples)
    original = [json.loads(line) for line in payloads["original.jsonl"].splitlines()]
    rewritten = [json.loads(line) for line in payloads["qwen.jsonl"].splitlines()]
    assert len(original) == len(rewritten) == 944 * samples
    for left, right in zip(original, rewritten):
        assert "extended_prompt" not in left and right["extended_prompt"]
        assert left == {k: v for k, v in right.items()
                        if k not in ("extended_prompt", "rewrite_file_sha256")}
    prepare_inputs(tmp_path, samples)
    prepare_inputs(tmp_path, samples)
    with pytest.raises(ValueError, match="different prompt protocol"):
        prepare_inputs(tmp_path, 5 if samples == 1 else 1)


def make_args(tmp_path):
    checkpoint = tmp_path / "fixture.pt"
    checkpoint.write_bytes(b"synthetic checkpoint placeholder, not model weights")
    config = tmp_path / "fixture.yaml"
    config.write_text("num_frame_per_block: 1\n")
    return SimpleNamespace(output=str(tmp_path / "evaluation"), checkpoint=str(checkpoint),
        config=str(config), method="framewise", schedule="ffe", weight_source="generator",
        conditioning="paired", samples_per_prompt=1, gpus="0,1,2,3,4,5,6,7",
        python=sys.executable, vbench_python="missing_vbench_python")


def test_shared_checkpoint_schedule_and_weight_selection(tmp_path):
    args = make_args(tmp_path)
    original = command_for_condition(args, "original")
    rewritten = command_for_condition(args, "qwen")
    for flag in ("--checkpoint_path", "--config_path", "--schedule", "--samples_per_prompt"):
        assert original[original.index(flag) + 1] == rewritten[rewritten.index(flag) + 1]
    assert "--extended_prompt_path" not in original
    assert "--extended_prompt_path" in rewritten
    assert "--require_no_ema" in original and "--require_no_ema" in rewritten
    args.weight_source = "generator_ema"
    assert "--use_ema" in command_for_condition(args, "original")
    assert "--require_no_ema" not in command_for_condition(args, "original")


def test_prompt_summary_audits_actual_exports_and_score_protocol(tmp_path):
    args = make_args(tmp_path)
    root = tmp_path / "evaluation"
    prepare_inputs(root, 1)
    for condition, score in (("original", .4), ("qwen", .43)):
        directory = root / condition
        videos, vbench = directory / "videos", directory / "vbench"
        videos.mkdir(parents=True)
        vbench.mkdir()
        intent = expected_intent(args, condition)
        (videos / "export.intent.json").write_text(json.dumps(intent))
        done = {**intent, "status": "complete", "num_videos": 944,
                "weight_source": "generator", "latent_frames_per_video": 21,
                "rgb_frames_per_video": 81}
        (videos / "export.done").write_text(json.dumps(done))
        scores = {k.replace(" ", "_"): v["Min"] + score * (v["Max"] - v["Min"])
                  for k, v in NORMALIZE_DIC.items()}
        scoring = {"mode": "vbench_standard", "samples_per_prompt": 1,
                   "official_five_sample_protocol": False, "dimensions": list(scores)}
        (vbench / f"{condition}_protocol.json").write_text(json.dumps(scoring))
        (vbench / f"{condition}_eval_results.json").write_text(json.dumps(scores))
    result = summarize(args)
    assert result["historical_headline_replication"] is False
    gain = next(row for row in result["gains"] if row["metric"] == "total_score")
    assert gain["qwen_minus_original"] == pytest.approx(.03)
    args.conditioning = "original"
    assert len(summarize(args)["scores"]) == 1
    assert summarize(args)["gains"] == []
    args.conditioning = "qwen"
    result = summarize(args)
    assert [row["conditioning"] for row in result["scores"]] == ["qwen"]
    assert result["gains"] == []
    args.conditioning = "paired"
    scoring_path = root / "qwen/vbench/qwen_protocol.json"
    scoring["samples_per_prompt"] = 5
    scoring_path.write_text(json.dumps(scoring))
    with pytest.raises(ValueError, match="scoring protocol"):
        summarize(args)
    scoring["samples_per_prompt"] = 1
    scoring_path.write_text(json.dumps(scoring))
    intent_path = root / "qwen/videos/export.intent.json"
    intent["checkpoint_path"] = "/another/checkpoint.pt"
    intent_path.write_text(json.dumps(intent))
    with pytest.raises(ValueError, match="export intent"):
        summarize(args)


def test_prompt_cli_dry_run_does_not_need_assets_or_launch_jobs(tmp_path):
    output = tmp_path / "absent"
    result = subprocess.run([sys.executable, "-m", "experiments.iclr.prompt_ablation",
        "--checkpoint", "/missing/headline.pt", "--output", str(output), "--dry-run"],
        cwd=REPO, capture_output=True, text=True, check=True)
    assert "--name original" in result.stdout and "--name qwen" in result.stdout
    assert "evaluation only, no training/GPU access" in result.stdout
    assert not output.exists()


def test_qwen_only_preserves_paper_gan_ablation_sampling(tmp_path):
    from experiments.rebuttal.audit_paired_noema_vbench import resolved_config_sha256
    historical = json.loads((REPO / "experiments/rebuttal/results/final_metrics.json").read_text())[
        "reviewer_runbook_experiments"]["dmd_only_step200_full16"]["pairing"]
    payloads = input_payloads(1)
    assert hashlib.sha256(payloads["original.jsonl"].encode()).hexdigest() == historical["manifest_sha256"]
    assert resolved_config_sha256(REPO / "experiments/rebuttal/configs/eval_ffe.yaml") == historical["resolved_config_sha256"]
    args = make_args(tmp_path)
    args.conditioning = "qwen"
    assert conditions(args) == ("qwen",)
    output = tmp_path / "absent"
    result = subprocess.run([sys.executable, "-m", "experiments.iclr.prompt_ablation",
        "--checkpoint", "/missing/paper_step200.pt", "--output", str(output),
        "--conditioning", "qwen", "--weight-source", "generator", "--schedule", "ffe",
        "--samples-per-prompt", "1", "--dry-run"], cwd=REPO, capture_output=True, text=True, check=True)
    assert "--name qwen" in result.stdout and "--name original" not in result.stdout
    assert "--schedule ffe" in result.stdout and "--require_no_ema" in result.stdout
    assert "--samples_per_prompt 1" in result.stdout and "--extended_prompt_path" in result.stdout
    assert not output.exists()
