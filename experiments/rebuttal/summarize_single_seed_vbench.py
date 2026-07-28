#!/usr/bin/env python3
"""Audit and summarize raw/no-EMA, one-sample-per-prompt VBench runs."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.rebuttal.consolidate_results import (  # noqa: E402
    load_vbench_results,
    official_totals,
)


def parse_assignment(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"Expected LABEL=PATH, got {value!r}")
    label, path = value.split("=", 1)
    if not label or not path:
        raise ValueError(f"Expected non-empty LABEL=PATH, got {value!r}")
    return label, Path(path).resolve()


def audit_result(label: str, result_path: Path) -> dict:
    if not result_path.is_file():
        raise FileNotFoundError(result_path)
    suffix = "_eval_results.json"
    if not result_path.name.endswith(suffix):
        raise ValueError(f"Unexpected VBench result filename: {result_path}")
    name = result_path.name[: -len(suffix)]
    protocol_path = result_path.with_name(f"{name}_protocol.json")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("mode") != "vbench_standard":
        raise ValueError(f"{label}: expected vbench_standard protocol")
    if protocol.get("samples_per_prompt") != 1:
        raise ValueError(f"{label}: expected exactly one generated sample per prompt")
    if protocol.get("official_five_sample_protocol") is not False:
        raise ValueError(f"{label}: one-sample protocol metadata is inconsistent")

    condition_root = result_path.parent.parent
    export_path = condition_root / "videos" / "export.done"
    export = json.loads(export_path.read_text(encoding="utf-8"))
    if export.get("use_ema") is not False:
        raise ValueError(f"{label}: export is not audited no-EMA: {export_path}")
    if export.get("weight_source") != "generator":
        raise ValueError(f"{label}: export did not use raw generator weights")

    scores = load_vbench_results(result_path)
    totals = official_totals(scores)
    if totals is None:
        raise ValueError(f"{label}: complete 16-dimension VBench results are required")
    if not all(math.isfinite(value) for value in totals.values()):
        raise ValueError(f"{label}: non-finite aggregate score")
    return {
        "result_path": str(result_path),
        "protocol_path": str(protocol_path),
        "export_path": str(export_path),
        "checkpoint_path": export["checkpoint_path"],
        "samples_per_prompt": 1,
        "use_ema": False,
        "weight_source": "generator",
        "scores": scores,
        "normalized_aggregates": totals,
    }


def score_delta(right: dict, left: dict) -> dict:
    right_scores = right["scores"]
    left_scores = left["scores"]
    if set(right_scores) != set(left_scores):
        raise ValueError("Compared runs have different VBench dimensions")
    return {
        "scores": {
            dimension: right_scores[dimension] - left_scores[dimension]
            for dimension in sorted(right_scores)
        },
        "normalized_aggregates": {
            key: right["normalized_aggregates"][key]
            - left["normalized_aggregates"][key]
            for key in right["normalized_aggregates"]
        },
    }


def parse_labels(value: str, expected: int) -> list[str]:
    labels = [part.strip() for part in value.split(",")]
    if len(labels) != expected or any(not label for label in labels):
        raise ValueError(f"Expected {expected} comma-separated labels, got {value!r}")
    return labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", action="append", required=True, metavar="LABEL=PATH")
    parser.add_argument(
        "--comparison",
        action="append",
        default=[],
        metavar="NAME=RIGHT,LEFT",
        help="Report RIGHT minus LEFT.",
    )
    parser.add_argument(
        "--difference_in_differences",
        action="append",
        default=[],
        metavar="NAME=A,B,C,D",
        help="Report (A-B)-(C-D), for both totals and every dimension.",
    )
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()

    runs = {}
    for assignment in args.result:
        label, path = parse_assignment(assignment)
        if label in runs:
            raise ValueError(f"Duplicate result label: {label}")
        runs[label] = audit_result(label, path)

    comparisons = {}
    for assignment in args.comparison:
        name, expression = assignment.split("=", 1)
        right_label, left_label = parse_labels(expression, 2)
        comparisons[name] = {
            "formula": f"{right_label} - {left_label}",
            **score_delta(runs[right_label], runs[left_label]),
        }

    differences = {}
    for assignment in args.difference_in_differences:
        name, expression = assignment.split("=", 1)
        a_label, b_label, c_label, d_label = parse_labels(expression, 4)
        first = score_delta(runs[a_label], runs[b_label])
        second = score_delta(runs[c_label], runs[d_label])
        differences[name] = {
            "formula": f"({a_label} - {b_label}) - ({c_label} - {d_label})",
            "scores": {
                dimension: first["scores"][dimension] - second["scores"][dimension]
                for dimension in first["scores"]
            },
            "normalized_aggregates": {
                key: first["normalized_aggregates"][key]
                - second["normalized_aggregates"][key]
                for key in first["normalized_aggregates"]
            },
        }

    output = {
        "schema_version": 1,
        "protocol": (
            "Complete 16-dimension VBench scoring with one generated sample per "
            "prompt. This is intentionally not the official five-sample leaderboard protocol."
        ),
        "use_ema": False,
        "weight_source": "generator",
        "runs": runs,
        "comparisons": comparisons,
        "difference_in_differences": differences,
    }
    output_path = Path(args.output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(output, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, output_path)
    print(f"PASS: audited raw/no-EMA one-sample VBench summary: {output_path}")


if __name__ == "__main__":
    main()
