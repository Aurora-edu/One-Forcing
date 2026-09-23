#!/usr/bin/env python3
"""Run only the ICLR 2027 VBench cells that still need Qwen alignment.

The paper's already Qwen-conditioned headline, FFE, long-video, and historical
four-step results are deliberately not scheduled here. No training is started:
the six inputs are existing, separately trained checkpoints. Main-text cells
are completed before appendix cells; an existing cell is reused only after a
full protocol audit, never merely because a result JSON happens to exist.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    from experiments.rebuttal.paper_vbench_protocol import audit_run
    from experiments.rebuttal.summarize_paper_aligned_vbench import (
        render_markdown,
        summarize,
    )
except ModuleNotFoundError:
    from paper_vbench_protocol import audit_run
    from summarize_paper_aligned_vbench import render_markdown, summarize


SCRIPT_DIR = Path(__file__).resolve().parent
SAMPLES_PER_PROMPT = 5


@dataclass(frozen=True)
class Cell:
    name: str
    checkpoint_key: str
    schedule: str
    phase: str
    question: str


CELLS = (
    Cell("full200_ffe", "full200", "ffe", "main", "adversarial objective"),
    Cell("dmd200_ffe", "dmd200", "ffe", "main", "adversarial objective"),
    Cell("curved300_all4", "curved300", "all4", "main", "trajectory rectification"),
    Cell("rectified300_all4", "rectified300", "all4", "main", "trajectory rectification"),
    Cell("full400_ffe", "full400", "ffe", "appendix", "training beyond 200 steps"),
    Cell("full600_ffe", "full600", "ffe", "appendix", "training beyond 200 steps"),
)


def parse_checkpoint_assignments(assignments: list[str]) -> dict[str, Path]:
    keys = {cell.checkpoint_key for cell in CELLS}
    paths: dict[str, Path] = {}
    for assignment in assignments:
        if "=" not in assignment:
            raise ValueError(f"Expected --checkpoint KEY=PATH, got {assignment!r}")
        key, raw_path = assignment.split("=", 1)
        if key not in keys or not raw_path or key in paths:
            raise ValueError(f"Invalid or duplicate checkpoint assignment: {assignment!r}")
        paths[key] = Path(raw_path).expanduser().resolve()
    return paths


def selected_cells(phase: str) -> tuple[Cell, ...]:
    if phase == "all":
        return CELLS
    return tuple(cell for cell in CELLS if cell.phase == phase)


def summary_path(root: Path, cell: Cell) -> Path:
    return root / cell.name / f"{cell.name}_paper_protocol_summary.json"


def audited_cell(root: Path, cell: Cell, checkpoint: Path) -> Path | None:
    path = summary_path(root, cell)
    if not path.is_file():
        return None
    record = audit_run(root / cell.name, cell.name, SAMPLES_PER_PROMPT, cell.schedule)
    if record["checkpoint_path"] != str(checkpoint):
        raise ValueError(f"Existing {cell.name} uses another checkpoint: {record['checkpoint_path']}")
    if record["checkpoint_size_bytes"] != checkpoint.stat().st_size:
        raise ValueError(f"Existing {cell.name} checkpoint changed size: {checkpoint}")
    stored = json.loads(path.read_text(encoding="utf-8"))
    if stored != record:
        raise ValueError(f"Existing {cell.name} summary differs from current audited results")
    return path


def run_cell(
    root: Path, cell: Cell, checkpoint: Path, gpus: str,
    python: Path, vbench_python: Path,
) -> Path:
    if not checkpoint.is_file():
        raise FileNotFoundError(f"{cell.checkpoint_key} checkpoint not found: {checkpoint}")
    prior = audited_cell(root, cell, checkpoint)
    if prior is not None:
        print(f"SKIP audited existing cell: {cell.name}", flush=True)
        return prior
    command = [
        "bash", str(SCRIPT_DIR / "run_paper_aligned_vbench.sh"),
        "--name", cell.name,
        "--checkpoint_path", str(checkpoint),
        "--schedule", cell.schedule,
        "--output_root", str(root / cell.name),
        "--samples_per_prompt", str(SAMPLES_PER_PROMPT),
        "--gpus", gpus,
        "--python", str(python),
        "--vbench_python", str(vbench_python),
    ]
    print(f"RUN {cell.phase}: {cell.name} — {cell.question}", flush=True)
    subprocess.run(command, check=True)
    completed = audited_cell(root, cell, checkpoint)
    if completed is None:
        raise RuntimeError(f"Expected audited result missing after {cell.name}")
    return completed


def write_comparison(root: Path, phase: str) -> tuple[Path, Path]:
    required = CELLS[:4] if phase == "main" else CELLS
    assignments = [f"{cell.name}={summary_path(root, cell)}" for cell in required]
    comparisons = [
        "gan_gain=full200_ffe,dmd200_ffe",
        "rectification_gain=rectified300_all4,curved300_all4",
    ]
    if phase == "appendix":
        comparisons.extend([
            "step400_minus_200=full400_ffe,full200_ffe",
            "step600_minus_200=full600_ffe,full200_ffe",
        ])
    payload = summarize(assignments, comparisons)
    json_path = root / f"{phase}_iclr2027_qwen_results.json"
    markdown_path = root / f"{phase}_iclr2027_qwen_results.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(payload), encoding="utf-8")
    print(f"PASS: {json_path}\nPASS: {markdown_path}", flush=True)
    return json_path, markdown_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("plan", "main", "appendix", "all"), required=True)
    parser.add_argument("--checkpoint", action="append", default=[], metavar="KEY=PATH")
    parser.add_argument("--output_root", default="eval/iclr2027_qwen")
    parser.add_argument("--gpus", default="all")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--vbench_python", default="")
    args = parser.parse_args()
    paths = parse_checkpoint_assignments(args.checkpoint)
    cells = selected_cells(args.phase) if args.phase != "plan" else CELLS
    if args.phase == "plan":
        for index, cell in enumerate(cells, 1):
            print(f"{index}. {cell.phase}: {cell.name} ({cell.checkpoint_key}, {cell.schedule})")
        print("Already aligned, omitted: headline; FFE; Qwen 4-step; 20-second generated videos.")
        print("Paper-only check: 20-second Self-Forcing table provenance conflicts with saved audits.")
        return
    missing = sorted({cell.checkpoint_key for cell in cells} - paths.keys())
    if missing:
        parser.error("Missing --checkpoint for: " + ", ".join(missing))
    if not args.vbench_python:
        parser.error("--vbench_python is required for an execution phase")
    python = Path(args.python).resolve()
    vbench_python = Path(args.vbench_python).resolve()
    if not python.is_file() or not vbench_python.is_file():
        parser.error("--python and --vbench_python must point to existing executables")
    root = Path(args.output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.phase == "appendix":
        for cell in CELLS[:4]:
            if not summary_path(root, cell).is_file():
                parser.error(f"Main-text cell must be complete before appendix: {cell.name}")
    for cell in cells:
        run_cell(root, cell, paths[cell.checkpoint_key], args.gpus, python, vbench_python)
        if cell.name == "rectified300_all4":
            write_comparison(root, "main")
    if args.phase in ("appendix", "all"):
        write_comparison(root, "appendix")


if __name__ == "__main__":
    main()
