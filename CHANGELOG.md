# Changelog

## Unreleased

- Benchmark local models alongside Copilot, Hermes, OpenCode, Claude Code, and
  Codex in one run with `-m HARNESS:MODEL`. CLI benchmarks report quality only.
- Use one optional `--judge` to grade open-ended answers from all benchmark
  targets. The judge can be a model on the default provider, Copilot, or Hermes.
- Replace `homebench copilot` with `homebench -m copilot:MODEL`.
- Grade only OpenCode's final completed answer, not text from intermediate
  steps or incomplete runs.
