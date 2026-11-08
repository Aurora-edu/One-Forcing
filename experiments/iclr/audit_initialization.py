"""Audit actual loaded tensor identities without opening model checkpoints."""
import argparse
import json
from pathlib import Path

from experiments.iclr.protocol import LEGACY_PROFILE, PROFILE_RUN_SPECS


def audit(root, expected_runs=None, *, profile=LEGACY_PROFILE):
    specs = PROFILE_RUN_SPECS[profile]
    expected_runs = len(specs) if expected_runs is None else expected_runs
    if expected_runs != len(specs):
        raise ValueError("expected run count differs from selected experiment profile")
    paths = sorted(Path(root).glob("*/initialization.json"))
    if len(paths) != expected_runs:
        raise ValueError(f"expected {expected_runs} initialized runs, found {len(paths)}")
    rows = [json.loads(path.read_text()) for path in paths]
    for field in ("generator_checkpoint_sha256", "generator_tensor_sha256", "generator_source_key"):
        if len({row[field] for row in rows}) != 1:
            raise ValueError(f"initialization differs across runs: {field}")
    by_name = {row["experiment_id"]: row for row in rows}
    if set(by_name) != {name for name, _, _ in specs}:
        raise ValueError("initialization records must cover the declared frame/chunk runs")
    for seed in {s for _, _, s in specs}:
        pair = [by_name[name] for name, _, s in specs if s == seed]
        if any(row["initialization_seed"] != seed for row in pair):
            raise ValueError("initialization seed differs from declared training seed")
        if pair[0]["critic_tensor_sha256"] != pair[1]["critic_tensor_sha256"]:
            raise ValueError(f"critic_tensor_sha256 differs within training seed {seed}")
    if any(row.get("strict_load") is not True or row.get("use_ema") is not False for row in rows):
        raise ValueError("raw strict initialization not established")
    return {"num_runs": len(rows), "actual_tensor_identity": "matched",
        "generator_tensor_sha256": rows[0]["generator_tensor_sha256"],
        "critic_identity": "matched within each seed; model initialization may vary across seeds",
        "ode_provenance": "UNVERIFIED: require original ODE training records; tensor identity alone is insufficient"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", required=True)
    parser.add_argument("--profile", choices=tuple(PROFILE_RUN_SPECS), default=LEGACY_PROFILE)
    parser.add_argument("--expected-runs", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        count = args.expected_runs if args.expected_runs is not None else len(PROFILE_RUN_SPECS[args.profile])
        print(f"DRY RUN: compare raw generator and critic tensor hashes across {count} runs")
        return
    print(json.dumps(audit(args.runs, args.expected_runs, profile=args.profile), indent=2))


if __name__ == "__main__":
    main()
