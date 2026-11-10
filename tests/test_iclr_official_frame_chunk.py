"""Guard the official checkout boundary and the actual paired execution plan."""
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from omegaconf import OmegaConf
import pytest

from experiments.iclr import official_frame_chunk as task


@pytest.fixture
def pair(tmp_path):
    root = tmp_path / "official"
    subprocess.run(["git", "clone", "--quiet", "--shared", "--no-checkout", str(task.REPO), str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "checkout", "--quiet", "--detach", task.OFFICIAL], check=True)
    output = tmp_path / "pair"
    task.prepare(root, output)
    return root, output


def test_only_block_changes_and_official_checkout_is_untouched(pair):
    root, output = pair
    frame, chunk = [task.read_yaml(output / "configs" / f"{arm}.yaml") for arm in task.ARMS]
    assert {k for k in frame if frame[k] != chunk[k]} == {"num_frame_per_block"}
    original = task.read_yaml(root / "config.yaml")
    assert {k for k in frame if frame[k] != original[k]} == {"seed"}
    assert frame["trainer"] == "one_forcing" and frame["seed"] == 2026
    assert frame["max_steps"] == 200 and frame["gradient_num_blocks"] == 17
    assert task.git(root, "status", "--porcelain") == ""
    assert task.check(output)["weight_source"] == "generator"


def test_source_byte_audit_rejects_hidden_modification(pair):
    root, output = pair
    path = root / "trainer/one_forcing.py"
    subprocess.run(["git", "-C", str(root), "update-index", "--assume-unchanged", str(path)], check=True)
    path.write_text(path.read_text() + "\n# unexpected edit\n")
    assert task.git(root, "status", "--porcelain", "--untracked-files=no") == ""
    with pytest.raises(ValueError, match="content changed"):
        task.check(output)


@pytest.mark.parametrize("field,value", [("gradient_num_blocks", 18), ("trainer", "iclr"),
                                          ("seed", 0), ("max_steps", 750), ("ema_weight", 0)])
def test_changing_config_and_rehashing_manifest_does_not_bypass_recipe(pair, field, value):
    _, output = pair
    cfg = output / "configs/frame.yaml"
    values = task.read_yaml(cfg); values[field] = value
    OmegaConf.save(OmegaConf.create(values), cfg)
    path = output / "pair.json"
    record = json.loads(path.read_text())
    record["training_configs"]["frame"] = task.sha256_file(cfg)
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="config differs"):
        task.check(output)


def test_commands_use_native_official_entries_and_raw_only(pair):
    root, output = pair
    assets = {k: f"/assets/{k} with spaces" for k in ("generator", "teacher", "data", "wan")}
    a, b = [task.train_command(output, arm, sys.executable, assets) for arm in task.ARMS]
    assert "train.py" in a and "--nproc_per_node=8" in a
    assert not any("experiments.iclr.train" in s or "manual_generator" in s for s in a)
    assert a[a.index("--generator_ckpt")+1] == b[b.index("--generator_ckpt")+1]
    assert a[a.index("--data_path")+1] == b[b.index("--data_path")+1]
    cmds = task.export_commands(output, "frame", sys.executable)
    assert len(cmds) == 8
    assert all(c[1] == "scripts/export_videos.py" and "--use_ema" not in c for c in cmds)
    for i, c in enumerate(cmds):
        assert int(c[c.index("--seed")+1]) == i * 118 * 5
    result = subprocess.run([sys.executable, "-m", "experiments.iclr.official_frame_chunk", "train",
        "--pair", str(output), "--arm", "frame", "--generator", "/missing/init.pt",
        "--teacher", "/missing/teacher", "--data", "/missing/data", "--wan", "/missing/wan",
        "--dry-run"], cwd=task.REPO, text=True, capture_output=True, check=True)
    assert "DRY RUN" in result.stdout and str(root) in result.stdout
    assert not (output / "runs").exists()


def test_native_shards_preserve_every_initial_seed_and_prompt_pair():
    prompts, rewrites = task.pinned_inputs()
    shards = task.split_prompts(prompts, rewrites)
    records = [(s["process_seed"] + i*5+j, p, s["rewrites"][i], j)
               for s in shards for i,p in enumerate(s["prompts"]) for j in range(5)]
    assert [r[0] for r in records] == list(range(4720))
    assert records == [(i*5+j,p,rewrites[i],j) for i,p in enumerate(prompts) for j in range(5)]


def log_rows(count):
    return [{"step": i, "critic_loss": .1, **({"generator_loss": .2} if i % 5 == 1 else {})}
            for i in range(1, count+1)]


def test_monitor_requires_real_complete_logs_and_correct_update_cadence(tmp_path):
    run = tmp_path / "runs/frame"; run.mkdir(parents=True)
    log = run / "train.log"
    rows = log_rows(200)
    log.write_text("\n".join(repr(r) for r in rows))
    result = task.monitor(tmp_path, "frame", 200)
    assert result["last_step"] == 200 and result["generator_updates"] == 40
    for bad in (rows[:-1], rows[:8]+rows[9:], rows+[rows[-1]], [dict(rows[0],critic_loss=float('inf'))]+rows[1:]):
        log.write_text("\n".join(repr(r) for r in bad))
        with pytest.raises((ValueError, SyntaxError)):
            task.monitor(tmp_path, "frame", 200)


def test_raw_validation_rejects_missing_wrong_shape_and_nonfinite_without_ema_fallback(tmp_path):
    import torch
    init, final = tmp_path / "init.pt", tmp_path / "final.pt"
    good = {"model.a": torch.ones(2), "model.b": torch.zeros(1,3)}
    torch.save({"generator": good}, init)
    torch.save({"generator": good, "generator_ema": {}}, final)
    assert task.raw_checkpoint(final, init)["weight_source"] == "generator"
    for payload in ({"generator_ema":good}, {"generator":{"model.a":torch.ones(2)}},
                    {"generator":dict(good, **{"model.a":torch.ones(3)})},
                    {"generator":dict(good, **{"model.a":torch.full((2,),float('nan'))})}):
        torch.save(payload, final)
        with pytest.raises(ValueError):
            task.raw_checkpoint(final, init)


def test_prepare_refuses_existing_output(pair):
    root, output = pair
    with pytest.raises(FileExistsError):
        task.prepare(root, output)


def test_external_worker_records_step10_completion_and_rejects_mismatched_receipt(pair, monkeypatch):
    import torch
    root, output = pair
    assets = output.parent / "assets"; assets.mkdir()
    for name in ("teacher", "data", "wan"):
        (assets/name).mkdir()
    (assets/'data/data.mdb').write_bytes(b'cpu fixture')
    for name in ('models_t5_umt5-xxl-enc-bf16.pth', 'Wan2.1_VAE.pth'):
        (assets/'wan'/name).write_bytes(b'cpu fixture')
    state = {'model.weight': torch.ones(2)}
    torch.save({'generator':state}, assets/'generator.pt')
    original_popen = subprocess.Popen
    seen = []

    class TrainingProcess:
        def __init__(self, command, **kwargs):
            seen.append((command, kwargs))
            final = output/'runs/frame/checkpoint_model_000200/model.pt'
            final.parent.mkdir()
            torch.save({'generator':state}, final)
            self.stdout = iter(repr(row)+'\n' for row in log_rows(200))
        def wait(self): return 0
        def poll(self): return 0

    def popen(command, *args, **kwargs):
        if isinstance(command, list) and command[0] == sys.executable and 'train.py' in command:
            return TrainingProcess(command, **kwargs)
        return original_popen(command, *args, **kwargs)

    monkeypatch.setattr(task.subprocess, 'Popen', popen)
    monkeypatch.setattr(task, 'require_idle', lambda gpu_ids: None)
    monkeypatch.setenv('PYTHONPATH', '/must/not/leak/to/official')
    args = SimpleNamespace(pair=str(output), arm='frame', python=sys.executable,
        gpus='0,1,2,3,4,5,6,7', generator=str(assets/'generator.pt'),
        teacher=str(assets/'teacher'), data=str(assets/'data'), wan=str(assets/'wan'))
    task.worker_train(args)
    assert len(seen) == 1 and seen[0][1]['cwd'] == root
    assert 'PYTHONPATH' not in seen[0][1]['env']
    assert json.loads((output/'runs/frame/step10_health.json').read_text())['last_step'] == 10
    manifest = task.check(output)
    assert task.audit_training(output, 'frame', manifest)['generator_updates'] == 40
    done = output/'runs/frame/training.done'
    changed = json.loads(done.read_text()); changed['seed'] = 0
    done.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='completion'):
        task.audit_training(output, 'frame', manifest)
