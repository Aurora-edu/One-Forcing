import hashlib
import json
import tempfile
import unittest
from unittest import mock
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from experiments.rebuttal.analyze_curvature import curvature_metrics
from experiments.rebuttal.audit_noema_checkpoint import audit_checkpoint
from experiments.rebuttal.curvature_intervention import (
    curvature_profile,
    rectify_trajectory,
)
from experiments.rebuttal.evaluate_fvd import frechet_distance, manifold_metrics
from experiments.rebuttal.estimate_training_eta import estimate
from experiments.rebuttal.estimate_evaluation_eta import estimate as estimate_evaluation
from experiments.rebuttal.build_prompt_cache import selected_indices, trim_padding
from experiments.rebuttal.prepare_vbench_prompts import unique_vbench_prompts
from experiments.rebuttal.prepare_vbench_subset import prepare_subset
from experiments.rebuttal.statistics_utils import mean_std_ci95
from experiments.rebuttal.split_long_videos import starts_for
from experiments.rebuttal.run_sharded_inference import parse_gpus
from experiments.rebuttal.run_checkpoint_sweep import select_result_dimensions
from experiments.rebuttal.validate_sharded_export import validate_export
from pipeline.causal_inference import CausalInferencePipeline
from scripts.export_videos import load_generator_state, select_records_for_shard
from scripts.run_vbench import validate_result_json, validate_standard_coverage
from trainer.one_forcing import _move_optimizer_state
from utils.config import load_config
from utils.dataset import cycle


REPO_ROOT = Path(__file__).resolve().parents[1]


class DummyGenerator:
    def __init__(self):
        self.model = SimpleNamespace(local_attn_size=21)

    def get_scheduler(self):
        return SimpleNamespace()


class RebuttalConfigTests(unittest.TestCase):
    def test_config_inheritance_and_ablation_deltas(self):
        one_forcing = load_config(
            str(
                REPO_ROOT
                / "experiments/rebuttal/configs/train_1step_one_forcing.yaml"
            )
        )
        dmd_only = load_config(
            str(REPO_ROOT / "experiments/rebuttal/configs/train_1step_dmd_only.yaml")
        )
        four_step = load_config(
            str(
                REPO_ROOT
                / "experiments/rebuttal/configs/train_4step_one_forcing.yaml"
            )
        )
        self.assertEqual(list(one_forcing.denoising_step_list), [1000])
        self.assertEqual(dmd_only.gan_g_weight, 0.0)
        self.assertEqual(dmd_only.gan_d_weight, 0.0)
        self.assertEqual(list(four_step.denoising_step_list), [1000, 750, 500, 250])
        self.assertFalse(one_forcing.randomize_seed)

    def test_config_cycle_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.yaml").write_text("_base_: b.yaml\n", encoding="utf-8")
            (root / "b.yaml").write_text("_base_: a.yaml\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Cyclic"):
                load_config(str(root / "a.yaml"))


class ScheduleTests(unittest.TestCase):
    @staticmethod
    def pipeline(schedule, first_steps=None):
        args = SimpleNamespace(
            model_kwargs={},
            denoising_step_list=[1000],
            first_frame_denoising_step_list=first_steps,
            warp_denoising_step=False,
            num_frame_per_block=1,
            independent_first_frame=False,
            rollout_schedule=schedule,
            first_rollout_num_frames=4,
        )
        if first_steps is None:
            del args.first_frame_denoising_step_list
        return CausalInferencePipeline(
            args,
            device=torch.device("cpu"),
            generator=DummyGenerator(),
            text_encoder=object(),
            vae=object(),
        )

    def test_all1_framewise_blocks(self):
        pipeline = self.pipeline("fixed")
        self.assertEqual(pipeline._build_rollout_frame_counts(7, None), [1] * 7)
        self.assertEqual(len(pipeline._block_denoising_step_list(0)), 1)

    def test_ffe_first_four_then_one(self):
        pipeline = self.pipeline("first4then1", [1000, 750, 500, 250])
        self.assertEqual(
            pipeline._build_rollout_frame_counts(7, None),
            [4, 1, 1, 1],
        )
        self.assertEqual(len(pipeline._block_denoising_step_list(0)), 4)
        self.assertEqual(len(pipeline._block_denoising_step_list(1)), 1)

    def test_all4_every_block(self):
        pipeline = self.pipeline("fixed")
        pipeline.denoising_step_list = [1000, 750, 500, 250]
        self.assertEqual(pipeline._build_rollout_frame_counts(3, None), [1, 1, 1])
        self.assertEqual(len(pipeline._block_denoising_step_list(0)), 4)
        self.assertEqual(len(pipeline._block_denoising_step_list(2)), 4)


class MetricTests(unittest.TestCase):
    def test_frechet_distance_identity(self):
        rng = np.random.default_rng(0)
        features = rng.normal(size=(32, 8))
        self.assertAlmostEqual(frechet_distance(features, features), 0.0, places=6)

    def test_frechet_distance_detects_shift(self):
        rng = np.random.default_rng(1)
        real = rng.normal(size=(32, 8))
        shifted = real + 2.0
        self.assertGreater(frechet_distance(real, shifted), 1.0)

    def test_i3d_manifold_metrics_identity_and_shift(self):
        rng = np.random.default_rng(2)
        real = rng.normal(size=(32, 8))
        identity = manifold_metrics(real, real.copy(), nearest_k=3)
        self.assertEqual(identity["precision"], 1.0)
        self.assertEqual(identity["recall"], 1.0)
        self.assertEqual(identity["coverage"], 1.0)

        shifted = manifold_metrics(real, real + 100.0, nearest_k=3)
        self.assertEqual(shifted["precision"], 0.0)
        self.assertEqual(shifted["recall"], 0.0)
        self.assertEqual(shifted["coverage"], 0.0)

    def test_three_seed_ci_uses_student_t(self):
        mean, std, ci95 = mean_std_ci95([1.0, 2.0, 3.0])
        self.assertEqual(mean, 2.0)
        self.assertEqual(std, 1.0)
        self.assertAlmostEqual(ci95, 2.4841377117, places=8)

    def test_straight_trajectory_has_zero_curvature(self):
        points = torch.arange(5, dtype=torch.float32).reshape(5, 1, 1, 1, 1)
        metrics = curvature_metrics(points)
        self.assertAlmostEqual(metrics["path_excess_ratio"], 0.0, places=7)
        self.assertAlmostEqual(metrics["mean_turning_angle_radians"], 0.0, places=7)
        self.assertAlmostEqual(metrics["normalized_second_difference"], 0.0, places=7)

    def test_paired_curvature_rectification_preserves_controlled_endpoints(self):
        timesteps = [1000.0, 750.0, 500.0, 250.0, 0.0, None]
        trajectory = torch.tensor(
            [10.0, 8.0, 8.5, 3.0, 0.0, 123.0], dtype=torch.float32
        ).reshape(6, 1, 1, 1, 1)
        rectified = rectify_trajectory(trajectory, timesteps)
        self.assertTrue(torch.equal(rectified[0], trajectory[0]))
        self.assertTrue(torch.equal(rectified[-2], trajectory[-2]))
        self.assertTrue(torch.equal(rectified[-1], trajectory[-1]))
        self.assertGreater(max(curvature_profile(trajectory, timesteps)), 0.0)
        for value in curvature_profile(rectified, timesteps):
            self.assertAlmostEqual(value, 0.0, places=12)

    def test_long_window_positions_are_exact(self):
        self.assertEqual(
            starts_for(961, 81, ["early", "middle", "late"]),
            {"early": 0, "middle": 440, "late": 880},
        )


class ResourcePathTests(unittest.TestCase):
    def test_evaluation_eta_uses_schedule_specific_video_counts(self):
        result = estimate_evaluation(10.0, 20.0, 30.0, num_gpus=8)
        expected_seconds = 5120 * 10.0 + 11622 * 20.0 + 9440 * 30.0
        self.assertAlmostEqual(
            result["total_generation_hours"],
            expected_seconds / 8 / 3600.0,
        )

    def test_existing_sweep_result_is_filtered_to_common_dimensions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.json"
            destination = root / "selected.json"
            source.write_text(
                '{"a": [0.1, ["detail"]], "b": [0.2, ["detail"]]}',
                encoding="utf-8",
            )
            select_result_dimensions(source, destination, ["b"])
            self.assertEqual(
                json.loads(destination.read_text(encoding="utf-8")),
                {"b": [0.2, ["detail"]]},
            )
            with self.assertRaisesRegex(ValueError, "missing requested"):
                select_result_dimensions(source, destination, ["c"])

    def test_training_eta_weights_generator_update_cadence(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            (run_dir / "resolved_config.yaml").write_text(
                "max_steps: 100\ndfake_gen_update_ratio: 5\n",
                encoding="utf-8",
            )
            records = [{"step": 1}]
            for step in range(2, 11):
                record = {
                    "step": step,
                    "per_iteration_time": 30.0 if step == 6 else 10.0,
                }
                if step == 6:
                    record["generator_loss"] = 1.0
                records.append(record)
            (run_dir / "metrics.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            result = estimate(run_dir, minimum_step=10)
            self.assertEqual(result["cadence_weighted_seconds_per_step"], 14.0)
            self.assertAlmostEqual(
                result["projected_compute_hours_total"],
                1400.0 / 3600.0,
            )

    def test_sharded_export_requires_all_provenance_records(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "videos"
            output.mkdir()
            checkpoint = root / "model.pt"
            checkpoint.write_bytes(b"checkpoint")
            manifest = root / "manifest.jsonl"
            records = [
                {"output_name": f"video-{index}.mp4"}
                for index in range(3)
            ]
            manifest.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            for record in records:
                (output / record["output_name"]).touch()
            for shard_index, shard_count in enumerate([2, 1]):
                payload = {
                    "checkpoint_path": str(checkpoint.resolve()),
                    "manifest_path": str(manifest.resolve()),
                    "num_videos": shard_count,
                    "num_total_videos": 3,
                    "shard_index": shard_index,
                    "num_shards": 2,
                    "latent_frames_per_video": 2,
                    "rgb_frames_per_video": 5,
                    "fps": 16,
                    "weight_source": "generator",
                    "use_ema": False,
                }
                (output / f"export.shard_{shard_index:02d}_of_02.done").write_text(
                    json.dumps(payload),
                    encoding="utf-8",
                )
            with mock.patch(
                "experiments.rebuttal.validate_sharded_export.validate_video"
            ) as validate_video_mock:
                result = validate_export(
                    output_folder=output,
                    manifest_path=manifest,
                    checkpoint_path=checkpoint,
                    num_shards=2,
                    latent_frames=2,
                    fps=16,
                    expected_weight_source="generator",
                )
            self.assertEqual(result["num_videos"], 3)
            self.assertFalse(result["use_ema"])
            self.assertEqual(validate_video_mock.call_count, 3)
            (output / "export.shard_01_of_02.done").unlink()
            with self.assertRaisesRegex(FileNotFoundError, "Missing shard"):
                validate_export(
                    output_folder=output,
                    manifest_path=manifest,
                    checkpoint_path=checkpoint,
                    num_shards=2,
                    latent_frames=2,
                    fps=16,
                )

    def test_gpu_shard_parser_rejects_duplicates(self):
        self.assertEqual(parse_gpus("0,2,7"), [0, 2, 7])
        with self.assertRaisesRegex(ValueError, "duplicates"):
            parse_gpus("0,0")

    def test_manifest_shards_are_disjoint_and_complete(self):
        records = list(range(17))
        shards = [
            select_records_for_shard(records, shard_index=index, num_shards=4)
            for index in range(4)
        ]
        self.assertEqual(sorted(item for shard in shards for item in shard), records)
        self.assertEqual(
            sum(len(shard) for shard in shards),
            len(set(item for shard in shards for item in shard)),
        )
        with self.assertRaisesRegex(ValueError, "shard_index"):
            select_records_for_shard(records, shard_index=4, num_shards=4)

    def test_dataloader_cycle_advances_sampler_epoch(self):
        class Sampler:
            def __init__(self):
                self.epochs = []

            def set_epoch(self, epoch):
                self.epochs.append(epoch)

        class Loader:
            def __init__(self):
                self.sampler = Sampler()

            def __iter__(self):
                yield self.sampler.epochs[-1]

        loader = Loader()
        iterator = cycle(loader)
        self.assertEqual([next(iterator), next(iterator), next(iterator)], [0, 1, 2])
        self.assertEqual(loader.sampler.epochs, [0, 1, 2])

    def test_distributed_cache_prefix_is_exact_union(self):
        indices = selected_indices(
            101,
            max_prompts=0,
            first_batches_per_rank=12,
            world_size=4,
            train_batch_size=1,
            seed=0,
        )
        self.assertEqual(len(indices), 48)
        self.assertEqual(len(indices), len(set(indices)))

    def test_trim_padding_and_optimizer_state_move(self):
        embedding = torch.zeros(8, 4, dtype=torch.bfloat16)
        embedding[:3] = 1
        self.assertEqual(tuple(trim_padding(embedding).shape), (3, 4))

        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = torch.optim.AdamW([parameter], lr=0.1)
        parameter.square().sum().backward()
        optimizer.step()
        _move_optimizer_state(optimizer, "cpu")
        self.assertEqual(optimizer.state[parameter]["exp_avg"].device.type, "cpu")

    def test_use_ema_does_not_silently_fall_back(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pt"
            torch.save({"generator": {"weight": torch.ones(1)}}, checkpoint)
            with self.assertRaisesRegex(KeyError, "no generator_ema"):
                load_generator_state(str(checkpoint), use_ema=True)

    def test_generator_checkpoint_normalizes_nested_fsdp_wrappers(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pt"
            torch.save(
                {
                    "generator": {
                        "model.blocks.0._fsdp_wrapped_module.weight": torch.ones(1)
                    }
                },
                checkpoint,
            )
            state = load_generator_state(str(checkpoint), use_ema=False)
            self.assertEqual(list(state), ["model.blocks.0.weight"])

    def test_raw_checkpoint_audit_checks_internal_step(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pt"
            torch.save(
                {
                    "step": 200,
                    "generator": {"weight": torch.ones(1)},
                    "generator_ema": {"weight": torch.zeros(1)},
                },
                checkpoint,
            )
            result = audit_checkpoint(checkpoint, expected_step=200)
            self.assertFalse(result["use_ema"])
            self.assertEqual(result["selected_weight_source"], "generator")
            with self.assertRaisesRegex(ValueError, "step mismatch"):
                audit_checkpoint(checkpoint, expected_step=400)

    def test_vbench_standard_requires_all_five_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            full_info = root / "full_info.json"
            full_info.write_text(
                '[{"prompt_en": "test prompt", "dimension": ["temporal_flickering"]}]',
                encoding="utf-8",
            )
            videos = []
            for index in range(5):
                path = root / f"test prompt-{index}.mp4"
                path.touch()
                videos.append(path)
            validate_standard_coverage(
                videos, str(full_info), ["temporal_flickering"]
            )
            with self.assertRaisesRegex(ValueError, "five samples"):
                validate_standard_coverage(
                    videos[:-1], str(full_info), ["temporal_flickering"]
                )

    def test_vbench_standard_accepts_explicit_single_sample_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            full_info = root / "full_info.json"
            full_info.write_text(
                '[{"prompt_en": "test prompt", "dimension": ["scene"]}]',
                encoding="utf-8",
            )
            video = root / "test prompt-0.mp4"
            video.touch()
            validate_standard_coverage(
                [video], str(full_info), ["scene"], samples_per_prompt=1
            )
            with self.assertRaisesRegex(ValueError, "samples_per_prompt"):
                validate_standard_coverage(
                    [video], str(full_info), ["scene"], samples_per_prompt=0
                )

    def test_vbench_nan_or_empty_result_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            result = Path(directory) / "result.json"
            result.write_text(
                '{"temporal_flickering": [NaN, []]}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "non-finite"):
                validate_result_json(result, ["temporal_flickering"])

    def test_vbench_prompt_extraction_deduplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            full_info = Path(directory) / "full_info.json"
            full_info.write_text(
                "["
                '{"prompt_en": "first", "dimension": ["a"]},'
                '{"prompt_en": "first", "dimension": ["b"]},'
                '{"prompt_en": "second", "dimension": ["a"]}'
                "]",
                encoding="utf-8",
            )
            self.assertEqual(
                unique_vbench_prompts(full_info),
                ["first", "second"],
            )
            self.assertEqual(
                unique_vbench_prompts(full_info, ["b"]),
                ["first"],
            )
            with self.assertRaisesRegex(ValueError, "None of the requested"):
                unique_vbench_prompts(full_info, ["missing"])

    def test_vbench_subset_preserves_full_manifest_seeds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            full_info = root / "full_info.json"
            full_info.write_text(
                "["
                '{"prompt_en": "first", "dimension": ["a"]},'
                '{"prompt_en": "second", "dimension": ["b"]}'
                "]",
                encoding="utf-8",
            )
            prompts = root / "prompts.txt"
            prompts.write_text("first\nsecond\n", encoding="utf-8")
            prompt_hash = hashlib.sha256(prompts.read_bytes()).hexdigest()
            manifest = root / "manifest.jsonl"
            records = []
            for prompt_index, prompt in enumerate(["first", "second"]):
                for sample_index in range(2):
                    records.append(
                        {
                            "prompt_index": prompt_index,
                            "sample_index": sample_index,
                            "seed": 100 + prompt_index * 10 + sample_index,
                            "output_name": f"{prompt}-{sample_index}.mp4",
                            "prompt": prompt,
                            "prompt_file_sha256": prompt_hash,
                        }
                    )
            manifest.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            subset_prompts, subset_records = prepare_subset(
                full_info,
                prompts,
                manifest,
                root / "subset.txt",
                root / "subset.jsonl",
                ["b"],
            )
            self.assertEqual(subset_prompts, ["second"])
            self.assertEqual(
                [record["seed"] for record in subset_records],
                [110, 111],
            )
            self.assertEqual(
                [record["prompt_index"] for record in subset_records],
                [0, 0],
            )


if __name__ == "__main__":
    unittest.main()
