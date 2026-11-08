"""Select diagnostic rows unseen by the actual fixed-budget local CD training."""
import argparse
import json
from pathlib import Path

from experiments.iclr.cd_local_protocol import validate_done_pair, audit_metrics
from experiments.iclr.protocol import atomic_json, sha256_file


def choose_rows(dataset, seen_rows, count):
    if count < 1 or any(i < 0 or i >= len(dataset) for i in seen_rows):
        raise ValueError("invalid validation count or training row index")
    selected, selected_prompts = [], set()
    with dataset.env.begin() as txn:
        def prompt(index):
            value = txn.get(f"prompts_{index}_data".encode())
            if value is None:
                raise ValueError(f"missing prompt for row {index}")
            return value.decode()
        training_prompts = {prompt(i) for i in seen_rows}
        for index in range(len(dataset)):
            if index in seen_rows:
                continue
            text = prompt(index)
            if text in training_prompts or text in selected_prompts:
                continue
            selected.append(index); selected_prompts.add(text)
            if len(selected) == count:
                return selected
    raise ValueError("insufficient unseen, prompt-disjoint diagnostic rows; supply separate validation data instead of reusing training rows")


def selection(training_root, data, *, count=64, smoke=False):
    from experiments.rebuttal.curvature_intervention import CurvatureTrajectoryDataset, MANIFEST_NAME
    root, data = Path(training_root), Path(data)
    done = [json.loads((root / arm / "training.done").read_text()) for arm in ("curved", "rectified")]
    controls = validate_done_pair(*done, mode="smoke" if smoke else "formal")
    if sha256_file(data / MANIFEST_NAME) != controls["dataset_manifest_sha256"]:
        raise ValueError("validation source differs from the audited training dataset")
    seen = set()
    for arm, run in zip(("curved", "rectified"), done):
        audit_metrics(root / arm / "metrics.jsonl", run)
        for line in (root / arm / "metrics.jsonl").read_text().splitlines():
            seen.update(json.loads(line)["global_row_indices"])
    dataset = CurvatureTrajectoryDataset(str(data), "curved")
    try:
        rows = choose_rows(dataset, seen, count)
    finally:
        dataset.env.close()
    return {"row_indices": rows, "dataset_manifest_sha256": controls["dataset_manifest_sha256"],
            "global_sample_order_sha256": controls["global_sample_order_sha256"],
            "training_rows_seen": len(seen), "selection": "first unseen rows with prompts disjoint from all actually trained rows",
            "limitation": "unseen under these fixed updates; not a permanently held-out dataset split"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("validation selection already exists; preserve its sample identities")
    atomic_json(output, selection(args.training_root, args.data, count=args.count, smoke=args.smoke))


if __name__ == "__main__":
    main()
