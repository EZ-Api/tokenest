# tokenest

A zero-dependency Go library for coarse and fine-grained LLM token estimation.

## Highlights
- **Four strategies**: UltraFast, Fast, Weighted, ZR
- **Auto by default**: works without caller preprocessing
- **Provider-aware profiles**: OpenAI / Claude / Gemini (fallback to OpenAI for everything else)
- **Optional LRU cache** for long, stable text
- **Explainable output** with per-category breakdowns
- **No tokenizer dependency** (no tiktoken, no external deps)

## Install
```
go get github.com/EZ-Api/tokenest
```

## Quick Start
```go
package main

import (
    "fmt"
    "github.com/EZ-Api/tokenest"
)

func main() {
    res := tokenest.EstimateText("Hello 你好", tokenest.Options{}) // Auto -> Fast
    fmt.Println(res.Tokens)

    res = tokenest.EstimateBytes([]byte("{\"prompt\":\"hi\"}"), tokenest.Options{})
    fmt.Println(res.Tokens) // Auto -> UltraFast

    res = tokenest.EstimateText("123", tokenest.Options{
        Strategy: tokenest.StrategyWeighted,
        Profile:  tokenest.ProfileOpenAI,
        Explain:  true,
    })
    fmt.Println(res.Tokens, res.Breakdown)
}
```

## Strategies
| Strategy | Input | Complexity | Use Case |
|---------|-------|------------|----------|
| UltraFast | raw bytes | O(1) | coarse filtering, high QPS |
| Fast | extracted text | O(min(n,1000)) | preflight estimation |
| Weighted | extracted text | O(n) | missing-usage fallback |
| ZR | extracted text | O(n) | opt-in, fitted categorical tuning |

Auto strategy selection:
- **raw bytes** → UltraFast
- **extracted text** → Fast (unless you explicitly request Weighted)

## Weighted Strategy (TokenX)
Weighted starts from tokenx segmentation and applies light ratio tuning:
- **Base**: tokenx segmentation count
- **Adjustments**: CJK/punctuation/digit ratios with per-profile tuning
- **Clamp**: bounded to avoid extreme drift
- **Fallback**: unknown providers/models → OpenAI profile

## ZR Strategy
ZR is an opt-in strategy generated from the fit tool's latest parameters. It classifies text into categories and applies fitted coefficients. Use it when you want the current fit behavior without changing Weighted defaults.

## Profiles
Profile resolution order:
1) `Options.Profile` (if set)
2) `Options.ProviderType` (balancer-friendly)
3) `Options.Model` (contains "claude" / "gemini")
4) Default: OpenAI weights

## Optional Cache
Use the wrapper when caching long, stable text (e.g., system prompts):
```go
est := tokenest.WithCache(tokenest.DefaultEstimator(), 1024)
res := est.EstimateText(systemPrompt, tokenest.Options{})
```
Caching is **off by default** and only applies to text >= 512 bytes.

## Comparison
- **vs tokenx**: keeps tokenx segmentation but adds ratio tuning to reduce mixed-text skew.
- **vs new-api**: avoids per-word heuristics that swing on long words/compound words.
- **vs tokenizer (tiktoken)**: lighter and faster, but approximate by design.

## Accuracy vs tiktoken
`tokenest` is heuristic and will not exactly match tokenizer outputs. In general:

- **Weighted** is closest (best for mixed text/code/CJK)
- **ZR** is an opt-in alternative based on the latest fit
- **Fast** is good for English and improves CJK/code vs UltraFast
- **UltraFast** is coarse and may undercount CJK/code

For a structured comparison and evaluation steps, see `ACCURACY.md`.
To refit Weighted on your own corpus, see `tokenest/tools/fit`.

## Notes
- This library is intentionally **zero-dependency**.
- If you can preprocess text, accuracy improves, but it is not required.
