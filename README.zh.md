# tokenest

一个**零依赖**的 Go 语言 Token 估算库，用于 LLM 请求的 tokens 预估。

## 特点
- **三种策略**：UltraFast / Fast / Weighted
- **默认自动选择**：无需调用方预处理也能用
- **供应商 Profile**：OpenAI / Claude / Gemini（其他模型默认回落到 OpenAI 权重）
- **可选 LRU 缓存**：适合系统提示词等稳定文本
- **可解释输出**：支持按类别的估算明细
- **不依赖 tokenizer**（不引入 tiktoken）

## 安装
```
go get github.com/EZ-Api/tokenest
```

## 快速使用
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

## 策略说明
| 策略 | 输入 | 复杂度 | 适用场景 |
|------|------|--------|----------|
| UltraFast | 原始 bytes | O(1) | 粗筛、高 QPS |
| Fast | 提取后的文本 | O(min(n,1000)) | 预校验 |
| Weighted | 提取后的文本 | O(n) | usage 缺失回填 |

默认自动策略：
- **raw bytes** → UltraFast
- **已提取文本** → Fast（除非显式指定 Weighted）

## Weighted（基于 TokenX）
- **基础**：沿用 tokenx 的分段/分类计数
- **调整**：按 CJK/标点/数字比例做轻量系数修正
- **限制**：结果做上下限夹紧，避免极端漂移
- **回落**：非“御三家”的模型统一回落到 OpenAI Profile

## Profile 解析顺序
1) `Options.Profile`（手动指定）
2) `Options.ProviderType`（balancer 可用）
3) `Options.Model`（包含 claude/gemini）
4) 默认：OpenAI 权重

## 可选缓存
```go
est := tokenest.WithCache(tokenest.DefaultEstimator(), 1024)
res := est.EstimateText(systemPrompt, tokenest.Options{})
```
默认不缓存，仅对 >=512 字节的文本启用缓存。

## 对比
- **相比 tokenx**：保留分段逻辑，并增加比例修正，减少混合文本偏差。
- **相比 new-api**：避免按单词计数导致的长词/复合词波动。
- **相比 tokenizer**：更轻更快，但本质是近似估算。

## 准确度 vs tiktoken
`tokenest` 是启发式估算，无法与 tokenizer 完全一致。通常情况下：

- **Weighted** 最接近（对混合文本/代码/CJK 最好）
- **Fast** 英文表现好，较 UltraFast 对 CJK/代码更稳
- **UltraFast** 最粗糙，可能低估 CJK/代码

更系统的对比方法和评估步骤见 `ACCURACY.md`。
如需基于自有语料重新拟合 Weighted，参考 `tokenest/tools/fit`。

## 说明
- 本库保持 **0 依赖**、轻量可移植。
- 调用方预处理能提升准确度，但不是必需条件。
