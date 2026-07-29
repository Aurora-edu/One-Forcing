# Pinned Qwen-rewrite VBench inputs

These files make the reviewer comparison independent of machine-local NFS
mounts:

- `shard00_pairs.jsonl` and `shard01_pairs.jsonl` are the exact 472+472
  original/rewrite pairs used by the historical Self-Forcing run. Their SHA256
  values are `53e85750f9fec2ff0a1af9b1d8ac9adf3c9e6b69dbf69cf529d3b56be4017d7e`
  and `a9126faa105e2aeb976b352877576f75a97b57e6784c78cb20d3b8c1d5dbdbb6`.
- `VBench_full_info.json` is the VBench prompt metadata used to reconstruct the
  canonical 944-prompt order. Its JSON content matches the historical input;
  the repository copy has SHA256
  `12d720a3f5ec60d7640edadd2272876056da098632171fc30356be25674c4deb`.

`run_one_forcing_qwen_4step_vbench.sh` uses these files by default. Command-line
paths remain available only for an explicitly audited override.
