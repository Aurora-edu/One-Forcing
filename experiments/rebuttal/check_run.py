#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path

from omegaconf import OmegaConf


def main():
    parser = argparse.ArgumentParser(
        description="Strictly validate a completed training smoke or full run."
    )
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--min_step", type=int, default=10)
    parser.add_argument("--expected_seed", type=int, default=None)
    parser.add_argument(
        "--allow_weights_only_resume",
        action="store_true",
        help=(
            "Allow runs resumed without optimizer state. This is for debugging only "
            "and must not be used for formal rebuttal results."
        ),
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    required = [
        run_dir / "resolved_config.yaml",
        run_dir / "runtime_seed.txt",
        run_dir / "metrics.jsonl",
        run_dir / "training.done",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing run artifacts: {missing}")

    config = OmegaConf.load(run_dir / "resolved_config.yaml")
    runtime_seed = int((run_dir / "runtime_seed.txt").read_text().strip())
    if args.expected_seed is not None and runtime_seed != args.expected_seed:
        raise AssertionError(
            f"Expected runtime seed {args.expected_seed}, got {runtime_seed}"
        )
    if bool(getattr(config, "randomize_seed", False)):
        raise AssertionError("Rebuttal run unexpectedly used randomize_seed=true")
    if (
        str(getattr(config, "resume_ckpt", "") or "")
        and not args.allow_weights_only_resume
    ):
        raise AssertionError(
            "Formal rebuttal validation rejects --resume_ckpt because current "
            "checkpoints do not restore optimizer state. Restart the formal run "
            "from step 0."
        )
    if int(config.seed) != runtime_seed:
        raise AssertionError(
            f"resolved_config seed={config.seed}, runtime seed={runtime_seed}"
        )

    records = []
    with open(run_dir / "metrics.jsonl", encoding="utf-8") as fp:
        for line in fp:
            if line.strip():
                records.append(json.loads(line))
    if not records:
        raise AssertionError("metrics.jsonl is empty")
    steps = [int(record["step"]) for record in records]
    if steps != sorted(set(steps)):
        raise AssertionError(f"Steps are not strictly increasing and unique: {steps}")
    if steps[-1] < args.min_step:
        raise AssertionError(f"Last logged step {steps[-1]} is below required {args.min_step}")
    for record in records:
        for key, value in record.items():
            if isinstance(value, (int, float)) and not math.isfinite(value):
                raise AssertionError(
                    f"Non-finite metric at step={record['step']}: {key}={value}"
                )

    completion = json.loads((run_dir / "training.done").read_text())
    if int(completion["final_step"]) != steps[-1]:
        raise AssertionError(
            f"training.done final_step={completion['final_step']} but metrics end at {steps[-1]}"
        )
    if int(completion["max_steps"]) != int(config.max_steps):
        raise AssertionError(
            f"training.done max_steps={completion['max_steps']} but "
            f"resolved_config has {config.max_steps}"
        )
    if not bool(getattr(config, "no_save", False)):
        interval = int(config.log_iters)
        expected_checkpoint_steps = list(range(interval, steps[-1] + 1, interval))
        if steps[-1] not in expected_checkpoint_steps:
            expected_checkpoint_steps.append(steps[-1])
        missing_checkpoints = [
            step
            for step in expected_checkpoint_steps
            if not (
                run_dir / f"checkpoint_model_{step:06d}" / "model.pt"
            ).is_file()
        ]
        if missing_checkpoints:
            raise FileNotFoundError(
                f"Missing expected checkpoints at steps {missing_checkpoints}"
            )
    print(
        f"PASS: {run_dir} reached step {steps[-1]} with seed={runtime_seed}; "
        f"{len(records)} finite metric records."
    )


if __name__ == "__main__":
    main()
