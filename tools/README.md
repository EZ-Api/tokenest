# Tools

Helper utilities for evaluating and tuning `tokenest`. Each tool lives in its own module and may pull extra dependencies.

## accuracy
Compares TokenX/NewAPI/UltraFast/Fast/Weighted against `tiktoken` (`o200k_base`) and writes timestamped markdown + Excel reports.

Run:
```bash
cd tokenest/tools/accuracy
GOWORK=off go run .
```

Options:
- `-report-dir`: custom output directory (default: `tokenest/report`) for markdown + xlsx.

Notes:
- This tool uses `gpt-tokenizer` (Node). Run `npm install` in `tokenest/tools/accuracy` if needed.
- Excel reports require uv + Python deps. Run `cd tokenest/tools/report && uv sync` once to preinstall.

## fit
Fits Weighted coefficients using linear regression against `tiktoken`.

Run:
```bash
cd tokenest/tools/fit
GOWORK=off go run .
```

Output:
- Prints coefficients and per-sample errors.

## adversary
Stress-tests worst-case under/overestimation by generating adversarial text and comparing against `tiktoken`.

Run:
```bash
cd tokenest/tools/adversary
GOWORK=off go run . -samples 2000 -length 50000 -top 5 -workers 4
```

Options:
- `-save-top`: save top N worst cases for TokenX and Weighted (default: 5, set `0` to disable).
- `-save-dir`: output directory for saved samples (default: `tokenest/datasets/test`).
- `-workers`: max concurrency (auto-capped at 8 to avoid host saturation).
- `-report-dir`: output report directory for markdown + xlsx (default: `tokenest/report`, use `-` to disable).

## Reports
Accuracy reports are written to `tokenest/report/` with timestamps in the filename (.md + .xlsx).
