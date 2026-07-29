#!/usr/bin/env python3
"""Fail fast on the exact Qwen-rewrite VBench prompt/seed manifest."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from summarize_qwen_4step_comparison import audit_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt_path", required=True)
    parser.add_argument("--qwen_rewrite_path", required=True)
    parser.add_argument("--manifest_path", required=True)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()

    payload = {
        "schema_version": 1,
        "status": "pass",
        "pairing": audit_manifest(
            Path(args.prompt_path).resolve(),
            Path(args.qwen_rewrite_path).resolve(),
            Path(args.manifest_path).resolve(),
        ),
    }
    output_path = Path(args.output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, output_path)
    print(f"PASS: Qwen VBench inputs audited: {output_path}")


if __name__ == "__main__":
    main()
