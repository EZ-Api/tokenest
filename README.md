# tokenest

A zero-dependency Go library for estimating token counts for LLM requests.

## Highlights
- **Three strategies**: UltraFast, Fast, Weighted
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

Auto strategy selection:
- **raw bytes** → UltraFast
- **extracted text** → Fast (unless you explicitly request Weighted)

## Weighted Strategy (Tokenx + New-API)
Weighted combines tokenx segmentation with new-api multipliers:
- **Segmentation**: whitespace/punctuation splitting + category classification
- **Weights**: OpenAI/Claude/Gemini profiles
- **Fallback**: unknown providers/models → OpenAI weights

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
- **vs tokenx**: adds provider-aware weights + counts whitespace/newlines, avoiding systematic underestimation.
- **vs new-api**: keeps tokenx-style segmentation (better for long words/code) while reusing the proven weight tables.
- **vs tokenizer (tiktoken)**: lighter and faster, but approximate by design.

## Notes
- This library is intentionally **zero-dependency**.
- If you can preprocess text, accuracy improves, but it is not required.
