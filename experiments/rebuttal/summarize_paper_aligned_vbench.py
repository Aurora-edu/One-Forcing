#!/usr/bin/env python3
"""Compare only protocol-matched Qwen-conditioned VBench run summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from experiments.rebuttal.consolidate_results import official_totals
except ModuleNotFoundError:
    from consolidate_results import official_totals


PAIRED_FIELDS = (
    "full_info_sha256",
    "prompt_sha256",
    "rewrite_sha256",
    "manifest_sha256",
    "samples_per_prompt",
    "generation_seed_rule",
    "conditioning",
    "scoring_prompt",
)
SCORE_FIELDS = ("total_score", "quality_score", "semantic_score")


def parse_assignment(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"Expected LABEL=PATH, got {value!r}")
    label, path = value.split("=", 1)
    if not label or not path:
        raise ValueError(f"Expected LABEL=PATH, got {value!r}")
    return label, Path(path).resolve()


def summarize(run_assignments: list[str], comparison_assignments: list[str]) -> dict:
    if not run_assignments:
        raise ValueError("At least one --run is required")
    runs = {}
    reference = None
    reference_dimensions = None
    for assignment in run_assignments:
        label, path = parse_assignment(assignment)
        if label in runs:
            raise ValueError(f"Duplicate run label: {label}")
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("status") != "pass":
            raise ValueError(f"Run has no passing protocol audit: {path}")
        protocol = record.get("protocol", {})
        if protocol.get("conditioning") != "pinned_qwen_rewrite":
            raise ValueError(f"Run is not Qwen-conditioned: {path}")
        if protocol.get("official_five_sample_protocol") is not (
            protocol.get("samples_per_prompt") == 5
        ):
            raise ValueError(f"Sample-count label is inconsistent: {path}")
        if reference is None:
            reference = protocol
        for field in PAIRED_FIELDS:
            if protocol.get(field) != reference.get(field):
                raise ValueError(
                    f"Cannot compare runs with different {field}: {label}"
                )
        aggregates = record.get("normalized_aggregates", {})
        if any(field not in aggregates for field in SCORE_FIELDS):
            raise ValueError(f"Missing full VBench aggregate: {path}")
        scores = record.get("scores", {})
        if len(scores) != 16:
            raise ValueError(f"Run does not contain all 16 dimensions: {path}")
        computed = official_totals(scores)
        if computed is None or any(
            abs(computed[field] - aggregates[field]) > 1e-8
            for field in SCORE_FIELDS
        ):
            raise ValueError(f"Stored aggregate does not match dimension scores: {path}")
        if reference_dimensions is None:
            reference_dimensions = set(scores)
        elif set(scores) != reference_dimensions:
            raise ValueError(f"VBench dimensions differ across runs: {path}")
        runs[label] = {
            "summary_path": str(path),
            "checkpoint_path": record.get("checkpoint_path"),
            "method": record.get("method"),
            "schedule": record.get("schedule"),
            "weight_source": record.get("weight_source"),
            "normalized_aggregates": aggregates,
            "scores": scores,
        }
    comparisons = {}
    for assignment in comparison_assignments:
        if "=" not in assignment:
            raise ValueError(f"--comparison must be NAME=RIGHT,LEFT: {assignment}")
        name, expression = assignment.split("=", 1)
        labels = expression.split(",")
        if not name or name in comparisons:
            raise ValueError(f"Invalid or duplicate comparison name: {name!r}")
        if len(labels) != 2 or any(label not in runs for label in labels):
            raise ValueError(f"--comparison must be NAME=RIGHT,LEFT: {assignment}")
        right, left = labels
        comparisons[name] = {
            "formula": f"{right} - {left}",
            "normalized_aggregates": {
                field: runs[right]["normalized_aggregates"][field]
                - runs[left]["normalized_aggregates"][field]
                for field in SCORE_FIELDS
            },
            "scores": {
                dimension: runs[right]["scores"][dimension]
                - runs[left]["scores"][dimension]
                for dimension in sorted(runs[right]["scores"])
            },
        }
    return {
        "schema_version": 1,
        "status": "pass",
        "protocol": reference,
        "runs": runs,
        "comparisons": comparisons,
    }


def render_markdown(summary: dict) -> str:
    protocol = summary["protocol"]
    lines = [
        "# Qwen-aligned VBench results",
        "",
        f"Protocol: {protocol['prompt_count']} prompts × "
        f"{protocol['samples_per_prompt']} samples; 16 dimensions; "
        "shared Qwen rewrite and generation manifest.",
        "",
        "| Run | Blocks | Schedule | Weights | Total | Quality | Semantic |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    for label, run in summary["runs"].items():
        scores = run["normalized_aggregates"]
        lines.append(
            f"| {label} | {run['method']} | {run['schedule']} | {run['weight_source']} | "
            f"{scores['total_score'] * 100:.2f} | "
            f"{scores['quality_score'] * 100:.2f} | "
            f"{scores['semantic_score'] * 100:.2f} |"
        )
    if summary["comparisons"]:
        lines.extend([
            "", "| Difference (right − left) | Total | Quality | Semantic |",
            "|---|---:|---:|---:|",
        ])
        for name, comparison in summary["comparisons"].items():
            scores = comparison["normalized_aggregates"]
            lines.append(
                f"| {name}: {comparison['formula']} | "
                f"{scores['total_score'] * 100:+.2f} | "
                f"{scores['quality_score'] * 100:+.2f} | "
                f"{scores['semantic_score'] * 100:+.2f} |"
            )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, metavar="LABEL=SUMMARY_JSON")
    parser.add_argument("--comparison", action="append", default=[], metavar="NAME=RIGHT,LEFT")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_markdown", required=True)
    args = parser.parse_args()
    summary = summarize(args.run, args.comparison)
    json_path = Path(args.output_json).resolve()
    markdown_path = Path(args.output_markdown).resolve()
    for path in (json_path, markdown_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(summary), encoding="utf-8")
    print(f"PASS: paired Qwen VBench report: {markdown_path}")


if __name__ == "__main__":
    main()
