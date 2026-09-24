import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from omegaconf import OmegaConf

from experiments.rebuttal import published_ffe_dmd_ablation as ablation
from experiments.rebuttal import preflight
from utils.config import load_config


class PublishedFfeDmdAblationTests(unittest.TestCase):
    def test_dmd_recipe_only_zeroes_gan(self):
        audit = ablation.verify_recipe()
        self.assertEqual(audit["only_training_objective_difference"], ["gan_g_weight", "gan_d_weight"])
        self.assertEqual(audit["training_steps"], 200)

    def test_recipe_rejects_training_schedule_drift(self):
        original_load = ablation.load_config

        def altered_load(path):
            config = original_load(path)
            if str(path) == str(ablation.DMD_CONFIG):
                config.rollout_schedule = "fixed"
            return config

        with patch.object(ablation, "load_config", side_effect=altered_load):
            with self.assertRaisesRegex(ValueError, "beyond GAN weights"):
                ablation.verify_recipe()

    def test_report_labels_reference_as_unpaired(self):
        scores = dict(ablation.PUBLISHED_SCORES)
        report = {
            "published_one_forcing": scores,
            "dmd_only": {key: value - 1 for key, value in scores.items()},
            "published_minus_dmd": {key: 1 for key in scores},
            "checkpoint_audit": {"checkpoint_path": "/tmp/dmd/model.pt"},
            "note": "not paired-seed estimates",
        }
        markdown = ablation.render_markdown(report)
        self.assertIn("One-Forcing (paper reference)", markdown)
        self.assertIn("83.76 | 85.22 | 77.91", markdown)
        self.assertIn("not paired-seed estimates", markdown)

    def test_report_accepts_only_completed_eight_rank_dmd_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            run_dir.mkdir()
            config = load_config(str(ablation.DMD_CONFIG))
            config.no_save = False
            config.disable_wandb = True
            OmegaConf.save(config, run_dir / "resolved_config.yaml")
            (run_dir / "training.done").write_text(
                json.dumps({"final_step": 200, "max_steps": 200}), encoding="utf-8"
            )
            (run_dir / "run_metadata.json").write_text(
                json.dumps({"config_path": str(ablation.DMD_CONFIG), "world_size": 8}),
                encoding="utf-8",
            )
            checkpoint = run_dir / "checkpoint_model_000200/model.pt"
            checkpoint.parent.mkdir()
            checkpoint.write_bytes(b"test")
            evaluated = {
                "checkpoint_path": str(checkpoint),
                "checkpoint_size_bytes": 4,
                "method": "framewise",
                "scores": {str(index): 0.5 for index in range(16)},
                "protocol": {
                    "prompt_count": 944,
                    "samples_per_prompt": 5,
                    "conditioning": "pinned_qwen_rewrite",
                    "scoring_prompt": "original_vbench_prompt",
                },
                "normalized_aggregates": {
                    "total_score": 0.8,
                    "quality_score": 0.81,
                    "semantic_score": 0.75,
                },
            }
            audit = {"checkpoint_path": str(checkpoint), "checkpoint_size_bytes": 4}
            with patch.object(ablation, "audit_checkpoint", return_value=audit), patch.object(
                ablation, "audit_run", return_value=evaluated
            ):
                report = ablation.build_report(run_dir, Path(temporary) / "eval")
            self.assertEqual(report["dmd_only"]["total_score"], 80.0)
            self.assertAlmostEqual(report["published_minus_dmd"]["total_score"], 3.76)
            self.assertIn("not_paired", report["comparison_type"])

            (run_dir / "run_metadata.json").write_text(
                json.dumps({"config_path": str(ablation.DMD_CONFIG), "world_size": 4}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "8-rank"):
                ablation.build_report(run_dir, Path(temporary) / "eval")

    def test_b200_version_exception_is_hardware_gated(self):
        with patch.object(preflight.torch, "__version__", "2.11.0+cu128"), patch.object(
            preflight.torchvision, "__version__", "0.26.0+cu128"
        ), patch.object(preflight.torch.cuda, "get_device_capability", return_value=(10, 0)):
            preflight.validate_versions("0,1", allow_b200_torch_deviation=True)
            with self.assertRaisesRegex(RuntimeError, "audited versions"):
                preflight.validate_versions("0,1", allow_b200_torch_deviation=False)
        with patch.object(preflight.torch.cuda, "get_device_capability", return_value=(9, 0)):
            with self.assertRaisesRegex(RuntimeError, "sm_100"):
                preflight.validate_versions("0", allow_b200_torch_deviation=True)


if __name__ == "__main__":
    unittest.main()
