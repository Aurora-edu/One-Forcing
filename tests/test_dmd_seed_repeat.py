import copy
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from omegaconf import OmegaConf

from experiments.rebuttal import official_main_pair as pair
from experiments.rebuttal import report_dmd_seed_repeat as repeat


class DmdSeedRepeatTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.run = self.root / "run"
        self.run.mkdir()
        self.baseline = repeat.load_baseline(repeat.BASELINE)

    @staticmethod
    def records(count):
        records = []
        for step in range(1, count + 1):
            record = {
                "step": step, "critic_loss": 1.0, "critic_grad_norm": 2.0,
                "gan_d_loss": 0.0, "r1_loss": 0.0, "r2_loss": 0.0,
            }
            if (step - 1) % 5 == 0:
                record.update({
                    "generator_loss": 0.5, "dmd_loss": 0.5,
                    "gan_g_loss": 0.0, "generator_grad_norm": 1.0,
                })
            records.append(record)
        return records

    def write_log(self, records):
        # Match official stdout, including unrelated timing lines.
        content = "Loading pretrained generator from /assets/causal_ode.pt\n"
        content += "".join(str(record) + "\n" for record in records)
        content += "{'step': 10, 'per_iteration_time': 3.5}\n"
        (self.run / "train.log").write_text(content, encoding="utf-8")

    def mocked_arm(self):
        baseline = self.baseline
        return {
            "checkpoint": {"step": 200, "selected_weight_source": "generator"},
            "evaluation": {
                "protocol": copy.deepcopy(baseline["eval_protocol"]),
                "scores": copy.deepcopy(baseline["dimensions"]["dmd"]),
                "normalized_aggregates": {key: baseline["dmd_only"][key] / 100 for key in pair.SCORE_KEYS},
            },
        }

    def test_seed1_config_changes_only_seed_from_previous_dmd(self):
        official = self.root / "official"
        official.mkdir()
        source = subprocess.check_output(
            ["git", "show", f"{pair.OFFICIAL_COMMIT}:config.yaml"], cwd=pair.REPO_ROOT
        )
        (official / "config.yaml").write_bytes(source)
        original = OmegaConf.to_container(OmegaConf.create(source.decode()), resolve=True)
        with patch.object(pair, "official_config", return_value=original):
            pair.prepare(official, self.root / "old", seed=48491)
            pair.prepare(official, self.root / "new", seed=1)
        old = pair._config(self.root / "old" / pair.CONFIG_NAMES["dmd"])
        new = pair._config(self.root / "new" / pair.CONFIG_NAMES["dmd"])
        self.assertEqual(new, dict(old, seed=1))
        self.assertEqual(new["resume_ckpt"], "")
        self.assertEqual(new["generator_ckpt"], "checkpoints/framewise/causal_ode.pt")
        self.assertEqual(new["denoising_step_list"], [1000])
        self.assertNotIn("rollout_schedule", new)

    def test_step10_and_completion_update_counts(self):
        for count, updates in ((10, 2), (200, 40)):
            with self.subTest(count=count):
                self.write_log(self.records(count))
                result = repeat.audit_training_log(self.run, min_step=count)
                self.assertEqual(result["generator_updates_logged"], updates)
                self.assertEqual(result["critic_updates_logged"], count)

    def test_rejects_missing_nonfinite_and_nonzero_gan_records(self):
        cases = []
        missing = self.records(10)
        del missing[4]
        cases.append((missing, "contiguous"))
        nonfinite = self.records(10)
        nonfinite[3]["critic_loss"] = float("inf")
        cases.append((nonfinite, "non-finite"))
        for field in ("gan_d_loss", "gan_g_loss", "r1_loss", "r2_loss"):
            changed = self.records(10)
            changed[0][field] = 0.03
            cases.append((changed, field))
        for records, message in cases:
            with self.subTest(message=message):
                self.write_log(records)
                with self.assertRaisesRegex(ValueError, message):
                    repeat.audit_training_log(self.run, min_step=10)

    def test_rejects_wrong_update_schedule_loss_and_zero_gradients(self):
        wrong_schedule = self.records(10)
        wrong_schedule[1]["generator_loss"] = 0.5
        wrong_loss = self.records(10)
        wrong_loss[0]["generator_loss"] = 0.6
        no_gradient = self.records(10)
        for record in no_gradient:
            if "generator_grad_norm" in record:
                record["generator_grad_norm"] = 0.0
        for records, message in (
            (wrong_schedule, "Unexpected generator update"),
            (wrong_loss, "differs from DMD"),
            (no_gradient, "No nonzero generator"),
        ):
            with self.subTest(message=message):
                self.write_log(records)
                with self.assertRaisesRegex(ValueError, message):
                    repeat.audit_training_log(self.run, min_step=10)

    def test_report_audits_only_dmd_and_does_not_claim_gan_gain(self):
        self.write_log(self.records(200))
        arm = self.mocked_arm()
        with patch.object(pair, "check", return_value={"shared_effective_seed": 1}), patch.object(
            pair, "_audit_arm", return_value=arm
        ) as audit:
            result = repeat.build_report(self.root, self.root / "configs", self.run, self.root / "eval")
        audit.assert_called_once()
        self.assertEqual(audit.call_args.args[0], "dmd")
        self.assertNotIn("gan_gain", result)
        self.assertNotIn("dmd_gan", result)
        self.assertEqual(result["repeat_training_seed"], 1)
        self.assertEqual(result["training_log_audit"]["generator_updates_logged"], 40)
        for value in result["delta_repeat_minus_baseline"].values():
            self.assertAlmostEqual(value, 0.0)
        markdown = repeat.render_markdown(result)
        self.assertIn("DMD+GAN was not retrained", markdown)
        self.assertIn("| Original seed 48491 | 83.52", markdown)
        prefix = self.root / "results" / "seed1"
        repeat.write_report(result, prefix)
        self.assertEqual(json.loads(prefix.with_suffix(".json").read_text())["repeat_training_seed"], 1)
        with self.assertRaises(FileExistsError):
            repeat.write_report(result, prefix)

    def test_report_rejects_wrong_seed_or_eval_protocol(self):
        self.write_log(self.records(200))
        with patch.object(pair, "check", return_value={"shared_effective_seed": 48491}):
            with self.assertRaisesRegex(ValueError, "preselects training seed 1"):
                repeat.build_report(self.root, self.root, self.run, self.root)
        arm = self.mocked_arm()
        arm["evaluation"]["protocol"]["manifest_sha256"] = "changed"
        with patch.object(pair, "check", return_value={"shared_effective_seed": 1}), patch.object(
            pair, "_audit_arm", return_value=arm
        ):
            with self.assertRaisesRegex(ValueError, "manifest_sha256"):
                repeat.build_report(self.root, self.root, self.run, self.root)

    def test_baseline_rejects_changed_source_and_inconsistent_scores(self):
        path = self.root / "baseline.json"
        for field, value, message in (
            ("training_source", "other", "official-main"),
            ("shared_effective_seed", 0, "seed-48491"),
            ("dmd_only", dict(self.baseline["dmd_only"], total_score=80.0), "aggregate"),
        ):
            with self.subTest(field=field):
                changed = dict(self.baseline, **{field: value})
                path.write_text(json.dumps(changed), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    repeat.load_baseline(path)


if __name__ == "__main__":
    unittest.main()
