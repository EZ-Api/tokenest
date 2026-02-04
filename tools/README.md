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
- Excel output includes deviation bar charts.

## fit
Fits ZR/Weighted-style coefficients against `tiktoken` and can export a reusable ZR config JSON.

Run:
```bash
cd tokenest/tools/fit
GOWORK=off go run .
```

Output:
- Default (no flags): parallel grid search over thresholds on the curated dataset, prints best config + coefficients + train/test evaluation.
- With flags: supports robust losses (Huber/IRLS), ridge regularization, JSONL log training, and `-out-zr-config` export.

### Common workflows

**1) Legacy grid search (curated dataset)**
```bash
cd tokenest/tools/fit
GOWORK=off go run .
```

Export best config to JSON:
```bash
GOWORK=off go run . -out-zr-config tokenest/report/zr-config.json
```

**2) Fixed-config robust refit (curated dataset)**
Use this to try robust losses without paying the grid-search cost:
```bash
GOWORK=off go run . -no-grid -loss huber_rel -huber-delta 0.20 -irls-iters 5 -ridge-lambda 0.001 -out-zr-config tokenest/report/zr-config.json
```

**3) Train from JSONL logs (future dataset)**
JSONL mode expects one JSON object per line. You provide field selectors:
- `-jsonl-text`: dot-path to extracted text field
- `-jsonl-tokens`: dot-path to actual token field (optional; empty => compute with `tiktoken` o200k_base)
```bash
GOWORK=off go run . \
  -jsonl /path/to/logs.jsonl \
  -jsonl-text request.text \
  -jsonl-tokens usage.total_tokens \
  -val-pct 0.2 -split-salt tokenest \
  -loss huber_rel -huber-delta 0.20 -irls-iters 5 \
  -ridge-lambda 0.001 \
  -bucket-cap 5000 \
  -out-zr-config tokenest/report/zr-config.json
```

**Conservative (underestimate-averse) mode**
```bash
GOWORK=off go run . -no-grid -loss asym_huber_rel -asym-alpha 2.0 -huber-delta 0.20 -irls-iters 5
```

### Using the output
- Today: copy the exported coefficients/thresholds into `tokenest/strategy/strategyTest1_params.go` (ZR hardcoded params).
- Future: a separate change (`update-zr-explain-config`) is planned to allow loading ZRConfig JSON at runtime without recompiling.

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

Notes:
- Excel output includes ratio bar charts.

## Reports
Accuracy reports are written to `tokenest/report/` with timestamps in the filename (.md + .xlsx).
