import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from experiments.rebuttal.paper_vbench_protocol import (
    EXPECTED_PROMPTS,
    audit_inputs,
    audit_run,
    prepare,
    protocol_paths,
    sha256_file,
)
from experiments.rebuttal.summarize_paper_aligned_vbench import summarize


class PaperVBenchProtocolTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_prepared_protocol_contains_exact_qwen_pairing_and_five_samples(self):
        paths = prepare(self.root, 5)
        protocol = audit_inputs(self.root, 5)
        prompts = paths["prompts"].read_text(encoding="utf-8").splitlines()
        rewrites = paths["rewrites"].read_text(encoding="utf-8").splitlines()
        records = [
            json.loads(line)
            for line in paths["manifest"].read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(len(prompts), EXPECTED_PROMPTS)
        self.assertEqual(len(rewrites), EXPECTED_PROMPTS)
        self.assertEqual(len(records), EXPECTED_PROMPTS * 5)
        self.assertTrue(protocol["official_five_sample_protocol"])
        for prompt_index in (0, 471, 943):
            for sample_index in range(5):
                record = records[prompt_index * 5 + sample_index]
                self.assertEqual(record["prompt"], prompts[prompt_index])
                self.assertEqual(record["extended_prompt"], rewrites[prompt_index])
                self.assertEqual(record["seed"], prompt_index * 5 + sample_index)
                self.assertEqual(
                    record["output_name"],
                    f"{prompts[prompt_index]}-{sample_index}.mp4",
                )

    def test_shell_entry_prepares_without_gpus(self):
        runner = Path(__file__).resolve().parents[1] / "experiments/rebuttal/run_paper_aligned_vbench.sh"
        subprocess.run(
            [
                "bash", str(runner), "--name", "preflight",
                "--output_root", str(self.root), "--prepare_only",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(audit_inputs(self.root / "manifests", 5)["prompt_count"], 944)
        subprocess.run(
            [
                "bash", str(runner), "--name", "sf_preflight",
                "--model", "self_forcing", "--schedule", "all4",
                "--output_root", str(self.root), "--prepare_only",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        rejected = subprocess.run(
            [
                "bash", str(runner), "--name", "sf_bad",
                "--model", "self_forcing", "--schedule", "ffe",
                "--output_root", str(self.root), "--prepare_only",
            ],
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn("must use --schedule all4", rejected.stderr)

    def test_protocol_rejects_unrewritten_manifest(self):
        paths = prepare(self.root, 1)
        records = [
            json.loads(line)
            for line in paths["manifest"].read_text(encoding="utf-8").splitlines()
        ]
        records[0]["extended_prompt"] = records[0]["prompt"]
        paths["manifest"].write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "pinned Qwen protocol"):
            audit_inputs(self.root, 1)
        with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
            prepare(self.root, 1)

    def test_run_audit_checks_conditioning_and_weight_provenance(self):
        manifests = self.root / "manifests"
        prepare(manifests, 1)
        protocol = audit_inputs(manifests, 1)
        paths = protocol_paths(manifests, 1)
        videos = self.root / "videos"
        vbench = self.root / "vbench"
        videos.mkdir()
        vbench.mkdir()
        config_path = Path(__file__).resolve().parents[1] / "experiments/rebuttal/configs/eval_ffe.yaml"
        checkpoint_path = self.root / "model.pt"
        checkpoint_path.write_bytes(b"synthetic-checkpoint")
        intent = {
            "prompt_sha256": protocol["prompt_sha256"],
            "extended_prompt_sha256": protocol["rewrite_sha256"],
            "manifest_sha256": protocol["manifest_sha256"],
            "selected_manifest_records": EXPECTED_PROMPTS,
            "method": "framewise",
            "schedule": "ffe",
            "num_output_frames": 21,
            "sink_size": 0,
            "fps": 16,
            "use_ema": False,
            "prompt_path": str(paths["prompts"]),
            "extended_prompt_path": str(paths["rewrites"]),
            "manifest_path": str(paths["manifest"]),
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_size_bytes": checkpoint_path.stat().st_size,
            "config_path": str(config_path),
            "config_sha256": sha256_file(config_path),
        }
        done = {
            "status": "complete",
            "manifest_sha256": protocol["manifest_sha256"],
            "num_videos": EXPECTED_PROMPTS,
            "weight_source": "generator",
            "use_ema": False,
            "latent_frames_per_video": 21,
            "rgb_frames_per_video": 81,
            "fps": 16,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_size_bytes": checkpoint_path.stat().st_size,
        }
        name = "paired_ffe"
        dimensions = [
            "subject_consistency", "background_consistency", "temporal_flickering",
            "motion_smoothness", "dynamic_degree", "aesthetic_quality",
            "imaging_quality", "object_class", "multiple_objects", "human_action",
            "color", "spatial_relationship", "scene", "temporal_style",
            "appearance_style", "overall_consistency",
        ]
        (videos / "export.intent.json").write_text(json.dumps(intent), encoding="utf-8")
        (videos / "export.done").write_text(json.dumps(done), encoding="utf-8")
        (vbench / f"{name}_protocol.json").write_text(
            json.dumps({
                "mode": "vbench_standard", "samples_per_prompt": 1,
                "official_five_sample_protocol": False, "dimensions": dimensions,
            }),
            encoding="utf-8",
        )
        (vbench / f"{name}_eval_results.json").write_text(
            json.dumps({dimension: [0.5, ["sample"]] for dimension in dimensions}),
            encoding="utf-8",
        )
        audited = audit_run(self.root, name, 1, "ffe")
        self.assertEqual(audited["status"], "pass")
        self.assertEqual(audited["checkpoint_size_bytes"], checkpoint_path.stat().st_size)
        del intent["sink_size"]
        (videos / "export.intent.json").write_text(json.dumps(intent), encoding="utf-8")
        self.assertEqual(audit_run(self.root, name, 1, "ffe")["status"], "pass")
        intent["sink_size"] = 3
        (videos / "export.intent.json").write_text(json.dumps(intent), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "sink_size=0"):
            audit_run(self.root, name, 1, "ffe")
        intent["sink_size"] = 0
        (videos / "export.intent.json").write_text(json.dumps(intent), encoding="utf-8")
        first_path = self.root / "first.json"
        second_path = self.root / "second.json"
        first_path.write_text(json.dumps(audited), encoding="utf-8")
        second_path.write_text(json.dumps(audited), encoding="utf-8")
        paired = summarize(
            [f"first={first_path}", f"second={second_path}"],
            ["delta=second,first"],
        )
        self.assertEqual(
            paired["comparisons"]["delta"]["normalized_aggregates"]["total_score"],
            0.0,
        )
        mismatched = json.loads(second_path.read_text(encoding="utf-8"))
        mismatched["protocol"]["manifest_sha256"] = "wrong"
        second_path.write_text(json.dumps(mismatched), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "manifest_sha256"):
            summarize([f"first={first_path}", f"second={second_path}"], [])

        intent["extended_prompt_sha256"] = None
        (videos / "export.intent.json").write_text(json.dumps(intent), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "extended_prompt_sha256"):
            audit_run(self.root, name, 1, "ffe")

        intent["extended_prompt_sha256"] = protocol["rewrite_sha256"]
        intent["use_ema"] = True
        intent["method"] = "chunkwise"
        intent["schedule"] = "all4"
        self_forcing_config = Path(__file__).resolve().parents[1] / "self_forcing_config.yaml"
        intent["config_path"] = str(self_forcing_config)
        intent["config_sha256"] = sha256_file(self_forcing_config)
        done["use_ema"] = True
        done["weight_source"] = "generator_ema"
        (videos / "export.intent.json").write_text(json.dumps(intent), encoding="utf-8")
        (videos / "export.done").write_text(json.dumps(done), encoding="utf-8")
        self.assertEqual(
            audit_run(self.root, name, 1, "all4", "generator_ema")["weight_source"],
            "generator_ema",
        )
        with self.assertRaisesRegex(ValueError, "config"):
            audit_run(self.root, name, 1, "all4")


if __name__ == "__main__":
    unittest.main()
