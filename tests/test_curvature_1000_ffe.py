"""CPU checks for the historical checkpoint FFE handoff, without model loading."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from omegaconf import OmegaConf
import pytest
import torch

from experiments.rebuttal import evaluate_curvature_1000_ffe as task
from experiments.rebuttal.audit_noema_checkpoint import audit_checkpoint
from experiments.rebuttal.paper_vbench_protocol import ASSETS, pinned_inputs


@pytest.fixture
def inputs(tmp_path):
    root = tmp_path / "historical"
    for arm in ("curved", "rectified"):
        path = root / "training" / arm / "checkpoint_model_001000/model.pt"
        path.parent.mkdir(parents=True)
        torch.save({"step": 1000, "generator": {"weight": torch.ones(1)}}, path)
    directory = root / "manifests"
    directory.mkdir()
    prompts, _ = pinned_inputs()
    prompt_file = directory / "vbench_prompts.txt"
    prompt_file.write_text("\n".join(prompts) + "\n")
    digest = hashlib.sha256(prompt_file.read_bytes()).hexdigest()
    records = [{"prompt_index": i, "sample_index": 0, "seed": i, "prompt": prompt,
                "output_name": f"{prompt}-0.mp4", "prompt_file_sha256": digest}
               for i, prompt in enumerate(prompts)]
    (directory / "vbench_single_sample_seed0.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records))
    return SimpleNamespace(experiment_root=str(root), output=None, python=sys.executable,
                           vbench_python=sys.executable, gpus="0,1,2,3,4,5,6,7",
                           full_info_path=str(ASSETS / "VBench_full_info.json"))


def test_dry_run_needs_no_assets_and_does_not_create_outputs(tmp_path):
    root = tmp_path / "missing"
    result = subprocess.run([sys.executable, "-m", task.__name__,
        "--experiment-root", str(root), "--full-info-path", "/missing/info.json",
        "--vbench-python", "/missing/vbpy", "--dry-run"],
        cwd=task.REPO, text=True, capture_output=True, check=True)
    assert "DRY RUN" in result.stdout and not root.exists()
    assert result.stdout.count("--schedule main_ffe") == 2
    assert "train_curvature" not in result.stdout and "--use_ema" not in result.stdout


def test_original_noise_manifest_is_required(inputs):
    task.validate_inputs(inputs)
    manifest = Path(inputs.experiment_root) / "manifests/vbench_single_sample_seed0.jsonl"
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    rows[100]["seed"] += 1
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="row 100"):
        task.validate_inputs(inputs)


def test_checkpoint_audit_will_not_substitute_step300_or_ema(inputs):
    _, commands = task.build_plan(inputs)
    for command in commands[:2]:
        path = Path(command[command.index("--checkpoint") + 1])
        expected = int(command[command.index("--expected_step") + 1])
        assert audit_checkpoint(path, expected)["selected_weight_source"] == "generator"
        torch.save({"step": 300, "generator": {"weight": torch.ones(1)}}, path)
        with pytest.raises(ValueError, match="step mismatch"):
            audit_checkpoint(path, expected)
        torch.save({"step": 1000, "generator_ema": {"weight": torch.ones(1)}}, path)
        with pytest.raises(KeyError, match="no raw"):
            audit_checkpoint(path, expected)


def test_actual_infer_shell_changes_only_first_frame_schedule(inputs, tmp_path):
    # Run the real shell/config-generation code; replace only the GPU exporter.
    wrapper = tmp_path / "cpu_python"
    wrapper.write_text(f"#!{sys.executable}\n" +
        "import os, sys\n"
        "if sys.argv[1] == '-':\n"
        "    os.execv(sys.executable, [sys.executable, *sys.argv[1:]])\n"
        "assert sys.argv[1] == 'scripts/export_videos.py'\n"
        "assert '--use_ema' not in sys.argv\n")
    wrapper.chmod(0o755)
    root = Path(inputs.experiment_root)
    configs = {}
    for schedule in ("all1", "main_ffe"):
        output = tmp_path / schedule
        subprocess.run(["bash", "scripts/infer.sh", "--method", "framewise",
            "--schedule", schedule, "--config_path", str(task.SCRIPTS / "configs/eval_all1.yaml"),
            "--checkpoint_path", str(root / "training/curved/checkpoint_model_001000/model.pt"),
            "--prompt_path", str(root / "manifests/vbench_prompts.txt"),
            "--output_folder", str(output), "--num_output_frames", "21"],
            cwd=task.REPO, env=dict(os.environ, PYTHON_BIN=str(wrapper), USE_EMA="0",
                                   CUDA_VISIBLE_DEVICES=""), capture_output=True, check=True)
        configs[schedule] = OmegaConf.to_container(OmegaConf.load(
            output / f"infer_config_framewise_{schedule}_shard0.yaml"), resolve=True)
    all1, ffe = configs["all1"], configs["main_ffe"]
    assert ffe.pop("first_frame_denoising_step_list") == [1000, 750, 500, 250]
    assert ffe == all1
    assert ffe["num_frame_per_block"] == 1 and ffe["rollout_schedule"] == "fixed"
    assert ffe["denoising_step_list"] == [1000]
    assert ffe["model_kwargs"]["local_attn_size"] == 21
    assert ffe["image_or_video_shape"][1] == 21
