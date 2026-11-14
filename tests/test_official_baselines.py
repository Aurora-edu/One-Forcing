"""CPU checks for the actual pinned native loops, using tiny injected modules.

OFFICIAL_BASELINE_SOURCE_ROOT points at the `source` directory created by prepare.
No Wan weights, CUDA allocations, model downloads, or video inference are needed.
"""
import hashlib
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1] / "experiments/rebuttal"
sys.path.insert(0, str(HERE))
from official_baseline_common import (
    LOCK, adapted_pipeline_source, check_completed_video, check_source,
    immutable_json, json_write, load_config, select_weights, selected_records,
    validate_output_name,
)
from paper_vbench_protocol import prepare


class ProtocolTests(unittest.TestCase):
    def test_same_rewrites_seeds_and_full_coverage_with_any_shard_count(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = prepare(Path(directory), 5)
            rows = selected_records(paths["manifest"], "full")
            smoke = selected_records(paths["manifest"], "smoke")
            self.assertEqual(len(rows), 4720)
            self.assertEqual(len(smoke), 10)
            self.assertEqual(len({r["prompt_index"] for r in smoke}), 10)
            for row in rows:
                self.assertEqual(row["seed"], row["prompt_index"] * 5 + row["sample_index"])
                self.assertTrue(row["extended_prompt"].strip())
                self.assertEqual(row["output_name"], f"{row['prompt']}-{row['sample_index']}.mp4")
                validate_output_name(row["output_name"])
            for shards in [1, 3, 8]:
                union = [r for rank in range(shards) for r in rows[rank::shards]]
                self.assertEqual(sorted(r["seed"] for r in union), list(range(4720)))

    def test_weight_keys_never_fall_back_and_normalization_cannot_collide(self):
        with self.assertRaises(ValueError):
            select_weights({"generator": {}}, "generator_ema")
        self.assertEqual(select_weights({"generator": {"model._fsdp_wrapped_module.w": 3}}, "generator"), {"model.w": 3})
        with self.assertRaises(ValueError):
            select_weights({"generator": {"model.w": 3, "model._fsdp_wrapped_module.w": 4}}, "generator")

    def test_resume_rejects_modified_video_or_experiment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video, metadata = root / "x.mp4", root / "x.json"
            row = {"seed": 0, "extended_prompt": "paired rewrite"}
            self.assertFalse(check_completed_video(video, metadata, row, "intent"))
            video.write_bytes(b"fixture")
            json_write(metadata, {"record": row, "intent_sha256": "intent",
                                  "video_sha256": hashlib.sha256(b"fixture").hexdigest()})
            self.assertTrue(check_completed_video(video, metadata, row, "intent"))
            with self.assertRaises(ValueError):
                check_completed_video(video, metadata, row, "different")
            video.write_bytes(b"corrupt")
            with self.assertRaises(ValueError):
                check_completed_video(video, metadata, row, "intent")
            immutable_json(root / "intent.json", {"a": 1})
            with self.assertRaises(ValueError):
                immutable_json(root / "intent.json", {"a": 2})


@unittest.skipUnless(os.environ.get("OFFICIAL_BASELINE_SOURCE_ROOT"), "prepare official sources and set OFFICIAL_BASELINE_SOURCE_ROOT")
class NativeLoopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        cls.sources = Path(os.environ["OFFICIAL_BASELINE_SOURCE_ROOT"])

    def pipeline(self, model, schedule, adapted=True):
        torch = self.torch
        root = self.sources / model
        check_source(root, model)
        code = (root / "pipeline/causal_inference.py").read_text()
        if adapted:
            code = adapted_pipeline_source(code, model)
        wrapper = types.ModuleType("utils.wan_wrapper")
        wrapper.WanDiffusionWrapper = wrapper.WanTextEncoder = wrapper.WanVAEWrapper = None
        memory = types.ModuleType("demo_utils.memory")
        for key in ["gpu", "get_cuda_free_memory_gb", "DynamicSwapInstaller", "move_model_to_device_with_memory_preservation"]:
            setattr(memory, key, None)
        namespace = {}
        with patch.dict(sys.modules, {"utils.wan_wrapper": wrapper, "demo_utils.memory": memory}):
            exec(compile(code, str(root / "pipeline/causal_inference.py"), "exec"), namespace)

        class Scheduler:
            timesteps = torch.linspace(1000, 1, 1000)

            def add_noise(self, clean, noise, timestep):
                return clean + noise * 0.01

        class Generator(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = types.SimpleNamespace(local_attn_size=-1)
                self.calls = []

            def get_scheduler(self):
                return Scheduler()

            def forward(self, noisy_image_or_video, timestep, current_start, **kwargs):
                self.calls.append((current_start, noisy_image_or_video.shape[1], float(timestep[0, 0])))
                return None, noisy_image_or_video * 0.9 + timestep[:, :, None, None, None] / 10000

        class Text(torch.nn.Module):
            def forward(self, text_prompts):
                return {"prompts": text_prompts}

        class VAE(torch.nn.Module):
            def decode_to_pixel(self, output, use_cache):
                return output

        config = load_config(root, model, schedule)
        pipeline = namespace["CausalInferencePipeline"](config, "cpu", Generator(), Text(), VAE())
        pipeline.num_transformer_blocks = 1
        pipeline.frame_seq_length = 1

        def init_kv(batch_size, dtype, device):
            pipeline.kv_cache1 = [{"global_end_index": torch.tensor([0]), "local_end_index": torch.tensor([0])}]

        pipeline._initialize_kv_cache = init_kv
        pipeline._initialize_crossattn_cache = lambda **kwargs: setattr(pipeline, "crossattn_cache", [{"is_init": False}])
        return pipeline

    def run_pipeline(self, pipeline):
        self.torch.manual_seed(42)
        return pipeline.inference(self.torch.randn(1, 21, 1, 2, 2), ["fixed rewrite"])

    def test_ffe_is_four_calls_then_six_single_calls_and_seven_cache_refreshes(self):
        for model in LOCK:
            with self.subTest(model=model):
                pipeline = self.pipeline(model, "ffe")
                first = self.run_pipeline(pipeline)
                calls = pipeline.generator.calls[:]
                steps = [call for call in calls if call[2] != 0]
                refreshes = [call for call in calls if call[2] == 0]
                self.assertEqual([c[2] for c in steps[:4]], [1000, 750, 500, 250])
                self.assertEqual([c[0] for c in steps], [0] * 4 + [3, 6, 9, 12, 15, 18])
                self.assertEqual([c[0] for c in refreshes], [0, 3, 6, 9, 12, 15, 18])
                self.assertEqual({c[1] for c in calls}, {3})
                # A subsequent sample resets its schedule/caches, independent of worker history.
                self.assertTrue(self.torch.equal(first, self.run_pipeline(pipeline)))

    def test_all4_adapter_is_bitwise_identical_to_the_official_loop(self):
        for model in LOCK:
            with self.subTest(model=model):
                official = self.pipeline(model, "all4", adapted=False)
                adapted = self.pipeline(model, "all4")
                self.assertTrue(self.torch.equal(self.run_pipeline(official), self.run_pipeline(adapted)))
                self.assertEqual(official.generator.calls, adapted.generator.calls)
                self.assertEqual(sum(t != 0 for _, _, t in adapted.generator.calls), 28)

    def test_strict_loading_rejects_a_partial_state(self):
        module = self.torch.nn.Linear(2, 2)
        with self.assertRaises(RuntimeError):
            module.load_state_dict(select_weights({"generator": {"weight": module.weight}}, "generator"), strict=True)


if __name__ == "__main__":
    unittest.main()
