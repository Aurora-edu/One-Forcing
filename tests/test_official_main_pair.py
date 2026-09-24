import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from omegaconf import OmegaConf

from experiments.rebuttal import launch_official_main_pair as launcher
from experiments.rebuttal import official_main_pair as pair


class OfficialMainPairTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "official"
        self.root.mkdir()
        source = subprocess.check_output(
            ["git", "show", f"{pair.OFFICIAL_COMMIT}:config.yaml"], cwd=pair.REPO_ROOT
        )
        (self.root / "config.yaml").write_bytes(source)
        self.original = OmegaConf.to_container(OmegaConf.create(source.decode()), resolve=True)
        self.pair_dir = Path(self.temporary.name) / "pair"

    def test_official_source_requires_pinned_clean_main(self):
        source = (self.root / "config.yaml").read_bytes()
        with patch.object(pair, "_git", side_effect=[pair.OFFICIAL_COMMIT, ""]), patch.object(
            pair.subprocess, "check_output", return_value=source
        ):
            self.assertEqual(pair.official_config(self.root), self.original)
        with patch.object(pair, "_git", return_value="wrong"):
            with self.assertRaisesRegex(ValueError, "main@"):
                pair.official_config(self.root)
        with patch.object(pair, "_git", side_effect=[pair.OFFICIAL_COMMIT, " M train.py"]):
            with self.assertRaisesRegex(ValueError, "tracked modifications"):
                pair.official_config(self.root)

    def test_prepare_keeps_official_recipe_and_pairs_only_gan_weights(self):
        with patch.object(pair, "official_config", return_value=self.original):
            audit = pair.prepare(self.root, self.pair_dir, seed=34567)
            self.assertEqual(audit["shared_effective_seed"], 34567)
            full = pair._config(self.pair_dir / pair.CONFIG_NAMES["full"])
            dmd = pair._config(self.pair_dir / pair.CONFIG_NAMES["dmd"])
            self.assertEqual(full, dict(self.original, seed=34567))
            self.assertEqual(dmd, dict(full, gan_g_weight=0.0, gan_d_weight=0.0))
            self.assertNotIn("rollout_schedule", full)
            with self.assertRaises(FileExistsError):
                pair.prepare(self.root, self.pair_dir, seed=34567)
            dmd["lr"] = 0.1
            (self.pair_dir / pair.CONFIG_NAMES["dmd"]).write_text(
                OmegaConf.to_yaml(OmegaConf.create(dmd)), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "DMD-only config differs"):
                pair.check(self.root, self.pair_dir)

    def test_official_command_has_no_rebuttal_training_flags(self):
        assets = {
            "generator_ckpt": "/assets/ode.pt",
            "teacher_model_path": "/assets/wan14",
            "data_path": "/assets/clean_data",
        }
        command = pair.official_command(
            "/env/bin/python", Path("/pair/full.yaml"), Path("/runs/full"), assets
        )
        self.assertEqual(command[1:6], ["-m", "torch.distributed.run", "--standalone", "--nproc_per_node=8", "train.py"])
        self.assertIn("--disable-wandb", command)
        self.assertIn("--no_visualize", command)
        self.assertNotIn("--prompt_embedding_cache_path", command)
        self.assertNotIn("--manual_generator_backward", command)
        self.assertNotIn("--seed", command)

    def test_launch_requires_eight_idle_distinct_gpus(self):
        with self.assertRaisesRegex(ValueError, "eight distinct"):
            launcher._gpu_ids("0,1,2,3,4,5,6,6")
        clean = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
        with patch.object(launcher.subprocess, "run", return_value=clean):
            self.assertEqual(launcher._gpu_ids("0,1,2,3,4,5,6,7"), list("01234567"))
        busy = subprocess.CompletedProcess(args=[], returncode=0, stdout="1234\n", stderr="")
        with patch.object(launcher.subprocess, "run", return_value=busy):
            with self.assertRaisesRegex(RuntimeError, "already has compute"):
                launcher._gpu_ids("0,1,2,3,4,5,6,7")

    def test_training_audit_requires_all_200_finite_official_steps(self):
        run_dir = Path(self.temporary.name) / "run"
        run_dir.mkdir()
        assets = {
            "generator_ckpt": "/ode.pt", "teacher_model_path": "/wan14",
            "data_path": "/clean_data", "wan_1_3b_path": "/wan13",
            "cuda_visible_devices": "0,1,2,3,4,5,6,7",
        }
        command = pair.official_command(
            "/env/bin/python", self.pair_dir / pair.CONFIG_NAMES["full"], run_dir, assets
        )
        launch = {
            "arm": "full", "official_main_commit": pair.OFFICIAL_COMMIT,
            "config_path": str((self.pair_dir / pair.CONFIG_NAMES["full"]).resolve()),
            "config_sha256": "same", "world_size": 8, "run_dir": str(run_dir.resolve()),
            "python": "/env/bin/python", "asset_paths": assets, "command": command,
        }
        (run_dir / "official_launch.json").write_text(json.dumps(launch), encoding="utf-8")
        (run_dir / "train.log").write_text(
            "".join(str({"step": step, "critic_loss": 1.0}) + "\n" for step in range(1, 201)),
            encoding="utf-8",
        )
        checkpoint = run_dir / "checkpoint_model_000200/model.pt"
        checkpoint.parent.mkdir()
        checkpoint.write_bytes(b"test")
        audit = {"checkpoint_path": str(checkpoint), "checkpoint_size_bytes": 4}
        evaluation = {"checkpoint_path": str(checkpoint), "checkpoint_size_bytes": 4}
        with patch.object(pair, "audit_checkpoint", return_value=audit), patch.object(
            pair, "audit_run", return_value=evaluation
        ):
            result = pair._audit_arm(
                "full", run_dir, Path("/eval"), self.pair_dir,
                {"configs_sha256": {"full": "same"}},
            )
            self.assertEqual(result["checkpoint"], audit)
            (run_dir / "train.log").write_text(
                "".join(str({"step": step, "critic_loss": 1.0}) + "\n" for step in range(1, 200)),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "steps 1–200"):
                pair._audit_arm("full", run_dir, Path("/eval"), self.pair_dir, {"configs_sha256": {"full": "same"}})

    def test_report_uses_only_two_newly_evaluated_arms(self):
        shared_protocol = {key: "same" for key in pair.PAIRED_FIELDS}
        arms = {}
        for arm, score in (("full", 0.835), ("dmd", 0.79)):
            arms[arm] = {
                "launch": {
                    "asset_paths": {key: "/same" for key in (
                        "generator_ckpt", "teacher_model_path", "data_path", "wan_1_3b_path", "cuda_visible_devices"
                    )},
                    "asset_stats": {"data_mdb_size": 100},
                    "rebuttal_commit": "same",
                },
                "checkpoint": {"step": 200},
                "evaluation": {
                    "protocol": dict(shared_protocol),
                    "normalized_aggregates": {key: score for key in pair.SCORE_KEYS},
                    "scores": {f"dimension_{index}": score for index in range(16)},
                },
            }
        with patch.object(pair, "check", return_value={"shared_effective_seed": 123}), patch.object(
            pair, "_audit_arm", side_effect=lambda arm, *_: arms[arm]
        ):
            result = pair.report(self.root, self.pair_dir, Path("/full"), Path("/dmd"), Path("/fe"), Path("/de"))
            self.assertAlmostEqual(result["gan_gain"]["total_score"], 4.5)
            self.assertIn("83.76", pair.render_markdown(result))
            arms["dmd"]["evaluation"]["protocol"]["manifest_sha256"] = "other"
            with self.assertRaisesRegex(ValueError, "manifest_sha256"):
                pair.report(self.root, self.pair_dir, Path("/full"), Path("/dmd"), Path("/fe"), Path("/de"))


if __name__ == "__main__":
    unittest.main()
