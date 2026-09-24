import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from omegaconf import OmegaConf

from experiments.rebuttal import preflight
from experiments.rebuttal import published_ffe_dmd_ablation as ablation
from utils.config import load_config


class PairedFfeGanAblationTests(unittest.TestCase):
    def test_recipe_inherits_one_forcing_and_only_zeroes_gan(self):
        audit = ablation.verify_recipe()
        self.assertEqual(audit["only_arm_difference"], ["gan_g_weight", "gan_d_weight"])
        self.assertEqual(audit["training_steps_per_arm"], 200)

    def test_recipe_rejects_schedule_drift_in_one_arm(self):
        original_load = ablation.load_config

        def altered_load(path):
            config = original_load(path)
            if str(path) == str(ablation.DMD_CONFIG):
                config.rollout_schedule = "fixed"
            return config

        with patch.object(ablation, "load_config", side_effect=altered_load):
            with self.assertRaisesRegex(ValueError, "DMD-only arm versus full"):
                ablation.verify_recipe()

    def test_training_audit_requires_clean_eight_rank_step200(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            config = load_config(str(ablation.FULL_CONFIG))
            config.no_save = False
            config.disable_wandb = True
            OmegaConf.save(config, run_dir / "resolved_config.yaml")
            (run_dir / "training.done").write_text(
                json.dumps({"final_step": 200, "max_steps": 200}), encoding="utf-8"
            )
            metadata = {
                "config_path": str(ablation.FULL_CONFIG),
                "world_size": 8,
                "git_commit": "abc123",
                "git_status_porcelain": [],
            }
            (run_dir / "run_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            checkpoint = run_dir / "checkpoint_model_000200/model.pt"
            checkpoint.parent.mkdir()
            checkpoint.write_bytes(b"test")
            audit = {"checkpoint_path": str(checkpoint), "checkpoint_size_bytes": 4}
            with patch.object(ablation, "audit_checkpoint", return_value=audit):
                result = ablation.audit_training("full", run_dir)
            self.assertEqual(result["metadata"]["world_size"], 8)

            metadata["world_size"] = 4
            (run_dir / "run_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "8-rank"):
                ablation.audit_training("full", run_dir)

    def test_report_gain_uses_only_newly_evaluated_pair(self):
        shared_protocol = {
            "prompt_count": 944,
            "samples_per_prompt": 5,
            "conditioning": "pinned_qwen_rewrite",
            "scoring_prompt": "original_vbench_prompt",
            "official_five_sample_protocol": True,
            **{key: "same" for key in ablation.PAIRED_FIELDS if key not in (
                "samples_per_prompt", "conditioning", "scoring_prompt"
            )},
        }
        training = {}
        evaluations = {}
        for label, score in (("full", 0.82), ("dmd", 0.79)):
            training[label] = {
                "resolved_config": OmegaConf.to_container(
                    load_config(str(ablation.CONFIG_PATHS[label])), resolve=True
                ),
                "metadata": {"git_commit": "same"},
                "checkpoint_audit": {"checkpoint_path": f"/{label}/model.pt"},
            }
            evaluations[label] = {
                "protocol": dict(shared_protocol),
                "scores": {f"dimension_{index}": score for index in range(16)},
                "normalized_aggregates": {
                    "total_score": score,
                    "quality_score": score + 0.01,
                    "semantic_score": score - 0.01,
                },
            }

        with patch.object(ablation, "audit_training", side_effect=lambda label, _: training[label]), patch.object(
            ablation, "audit_evaluation", side_effect=lambda label, *_: evaluations[label]
        ):
            report = ablation.build_report(Path("/full"), Path("/dmd"), Path("/fe"), Path("/de"))
            self.assertAlmostEqual(report["gan_gain"]["total_score"], 3.0)
            self.assertAlmostEqual(report["full_minus_paper_reference"]["total_score"], -1.76)
            markdown = ablation.render_markdown(report)
            self.assertIn("DMD+GAN (new One-Forcing run)", markdown)
            self.assertIn("GAN gain (paired full − DMD-only)", markdown)
            self.assertIn("external", ablation.__doc__)

            evaluations["dmd"]["protocol"]["manifest_sha256"] = "different"
            with self.assertRaisesRegex(ValueError, "manifest_sha256 differs"):
                ablation.build_report(Path("/full"), Path("/dmd"), Path("/fe"), Path("/de"))

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
