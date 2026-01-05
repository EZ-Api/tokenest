# Accuracy vs tiktoken

`tokenest` is a heuristic estimator. It does **not** attempt to exactly match tokenizer outputs like `tiktoken`.
The goal is: **fast, stable, and reasonably close** for preflight and missing-usage fallback.

## Measured Results (o200k_base)
Measured with `tokenest/tools/accuracy` using `tiktoken-go` (`o200k_base`),
`gpt-tokenizer` (JS, o200k_base), and `tokenest` UltraFast/Fast/Weighted (`ProfileOpenAI`) plus tokenx/new-api scans.

Environment: `go1.24.5` on `linux/amd64`. Times are average per call.

| Description | Actual (tiktoken-go o200k_base) | GPT-Tokenizer | GPT-Tokenizer Dev | UltraFast | UltraFast Dev | Fast | Fast Dev | TokenX | TokenX Dev | NewAPI | NewAPI Dev | Weighted | Weighted Dev | tiktoken-go Avg Time | GPT-Tokenizer Avg Time | UltraFast Avg Time | Fast Avg Time | TokenX Avg Time | NewAPI Avg Time | Weighted Avg Time |
|------------|----------------------------------|---------------|-------------------|-----------|---------------|------|----------|--------|------------|--------|------------|----------|--------------|----------------------|------------------------|--------------------|--------------|-----------------|-----------------|-------------------|
| Short English text | 10 | 10 | +0.00% | 10 | +0.00% | 10 | +0.00% | 11 | +10.00% | 11 | +10.00% | 11 | +10.00% | 19.35us | 4.09us | 41ns | 127ns | 2.10us | 538ns | 2.09us |
| German text with umlauts | 48 | 48 | +0.00% | 41 | -14.58% | 42 | -12.50% | 49 | +2.08% | 29 | -39.58% | 47 | -2.08% | 52.22us | 10.11us | 54ns | 467ns | 5.80us | 2.04us | 4.28us |
| Metamorphosis by Franz Kafka (English) | 31795 | 31796 | +0.00% | 35521 | +11.72% | 35649 | +12.12% | 35708 | +12.31% | 39629 | +24.64% | 33772 | +6.22% | 54.09ms | 20.94ms | 71ns | 2.44us | 8.86ms | 2.09ms | 7.21ms |
| Die Verwandlung by Franz Kafka (German) | 35308 | 35309 | +0.00% | 36513 | +3.41% | 36634 | +3.76% | 35072 | -0.67% | 35411 | +0.29% | 33122 | -6.19% | 56.57ms | 24.92ms | 47ns | 1.73us | 8.96ms | 2.14ms | 7.64ms |
| \u9053\u5fb7\u7d93 by Laozi (Chinese) | 11711 | 11712 | +0.01% | 10392 | -11.26% | 10919 | -6.76% | 12062 | +3.00% | 10888 | -7.03% | 11626 | -0.73% | 21.14ms | 7.14ms | 50ns | 2.61us | 1.06ms | 511.62us | 1.44ms |
| TypeScript ES5 Type Declarations (~ 4000 loc) | 49293 | 49293 | +0.00% | 54610 | +10.79% | 57198 | +16.04% | 52340 | +6.18% | 63546 | +28.91% | 49315 | +0.04% | 89.41ms | 17.80ms | 82ns | 2.03us | 13.30ms | 2.82ms | 9.59ms |
| Mixed (EN+CJK+Code) | 34600 | 34602 | +0.01% | 35512 | +2.64% | 36673 | +5.99% | 36678 | +6.01% | 40741 | +17.75% | 34725 | +0.36% | 55.07ms | 18.90ms | 47ns | 1.80us | 7.24ms | 2.35ms | 6.33ms |

Notes:
- Deviation is signed percentage `(estimated - actual) / actual`, using tiktoken-go (o200k_base) as baseline.
- gpt-tokenizer uses the o200k_base encoding by default and closely matches tiktoken-go on this dataset.
- Results depend on the exact text source and preprocessing.

## Dataset & Normalization
The accuracy tool uses the same samples as the tokenx benchmark:
- Short English: `Hello, world! This is a short sentence.`
- German: `Die p\u00fcnktlich gew\u00fcnschte Tr\u00fcffelf\u00fcllung im \u00fcbergest\u00fclpten W\u00fcrzk\u00fcmmel-W\u00fcrfel ist k\u00fcmmerlich und d\u00fcrfte f\u00fcrderhin zu R\u00fcffeln in H\u00fclle und F\u00fclle f\u00fchren`
- Metamorphosis (English): `tokenx/test/fixtures/ebooks/pg5200.txt`
- Die Verwandlung (German): `tokenx/test/fixtures/ebooks/pg22367.txt`
- \u9053\u5fb7\u7d93 (Chinese): `tokenx/test/fixtures/ebooks/pg7337.txt`
- TypeScript ES5: `tokenx/node_modules/typescript/lib/lib.es5.d.ts` (fallback: `https://unpkg.com/typescript@5.9.3/lib/lib.es5.d.ts`)
- Mixed (EN+CJK+Code): 1/3 English + 1/3 Chinese + 1/3 TypeScript from the samples above.
Normalization: none (raw file content, BOM/CRLF preserved).

## Relative Accuracy (Qualitative)
In general (vs tiktoken-go o200k_base baseline):

1) **TokenX scan** is close on word-like text and long prose.
2) **NewAPI scan** is strict per-rune classification and can swing under/over depending on word length and symbol density.
3) **Weighted** starts from tokenx and adds light ratio adjustments; it is typically best for mixed CJK/code inputs.
4) **Fast** balances cost/accuracy for preflight.
5) **UltraFast** is the roughest but fastest.

Typical behavior:
- **English prose**: TokenX is typically closest; Weighted stays within a smaller range.
- **CJK-heavy**: Fast improves over UltraFast; Weighted reduces CJK skew vs TokenX.
- **Code-heavy / symbol-dense**: Weighted is closer when punctuation/digit ratios are high.

## Why it differs from tiktoken
- Tokenizers split on subword/BPE rules that vary by model and language.
- gpt-tokenizer (JS) and tiktoken-go can still diverge on edge cases even with o200k_base.
- Weighted uses tokenx segmentation + fitted ratio adjustments, which are not tokenizer-compatible.
- TokenX and NewAPI scans are heuristics from their respective sources, not tokenizer-compatible.
- Whitespace/newlines are counted as separate units in Weighted/NewAPI; tiktoken often merges them with adjacent tokens.
- This library intentionally avoids tokenizer dependencies for performance and simplicity.

## How to Measure (Recommended)
If you want numeric accuracy, compare against tiktoken on your own dataset:

### Go Tool (Bundled)
```
cd tokenest/tools/accuracy
GOWORK=off go run .
```

1) Prepare a dataset of representative prompts (English/CJK/code/mixed).
2) Compute tiktoken counts (Python example below).
3) Compute tokenest estimates and compare metrics (MAE / MAPE).

### Example (Python, external)
```python
import tiktoken

enc = tiktoken.get_encoding("o200k_base")

def count_tokens(text: str) -> int:
    return len(enc.encode(text))
```

### Suggested Metrics
- MAE (mean absolute error)
- MAPE (mean absolute percentage error)
- Error distribution by content type

## Decision Guidance
- **Need exact billing**: use tiktoken (or provider count_tokens).
- **Need fast fallback**: use tokenest Weighted.
- **Need high throughput preflight**: use UltraFast/Fast.
