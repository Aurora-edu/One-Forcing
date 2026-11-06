"""Audit actual loaded tensor identities without opening model checkpoints."""
import argparse
import json
from pathlib import Path


def audit(root, expected_runs=12):
    paths = sorted(Path(root).glob("*/initialization.json"))
    if len(paths) != expected_runs:
        raise ValueError(f"expected {expected_runs} initialized runs, found {len(paths)}")
    rows = [json.loads(path.read_text()) for path in paths]
    for field in ("generator_checkpoint_sha256", "generator_tensor_sha256", "critic_tensor_sha256",
                  "initialization_seed", "generator_source_key"):
        if len({row[field] for row in rows}) != 1:
            raise ValueError(f"initialization differs across runs: {field}")
    if any(row.get("strict_load") is not True or row.get("use_ema") is not False for row in rows):
        raise ValueError("raw strict initialization not established")
    return {"num_runs": len(rows), "actual_tensor_identity": "matched",
        "generator_tensor_sha256": rows[0]["generator_tensor_sha256"],
        "critic_tensor_sha256": rows[0]["critic_tensor_sha256"],
        "ode_provenance": "UNVERIFIED: require original ODE training records; tensor identity alone is insufficient"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", required=True)
    parser.add_argument("--expected-runs", type=int, default=12)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        print(f"DRY RUN: compare raw generator and critic tensor hashes across {args.expected_runs} runs")
        return
    print(json.dumps(audit(args.runs, args.expected_runs), indent=2))


if __name__ == "__main__":
    main()
