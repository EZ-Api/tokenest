# 工具集

这些工具用于评估与调优 `tokenest`。每个工具都有独立的 Go module，可能会下载额外依赖。

## accuracy
对比 TokenX/NewAPI/UltraFast/Fast/Weighted 与 `tiktoken`（o200k_base），并输出带时间戳的 Markdown + Excel 报告。

运行：
```bash
cd tokenest/tools/accuracy
GOWORK=off go run .
```

选项：
- `-report-dir`：自定义报告输出目录（默认 `tokenest/report`，同时输出 md + xlsx）。

说明：
- 依赖 `gpt-tokenizer`（Node），需要时在 `tokenest/tools/accuracy` 下执行 `npm install`。
- Excel 报告依赖 uv + Python 包，可先执行 `cd tokenest/tools/report && uv sync`。
- Excel 会包含偏差柱状图。

## fit
拟合 ZR/Weighted 风格的参数，使其更接近 `tiktoken`，并可导出可复用的 ZR config JSON。

运行：
```bash
cd tokenest/tools/fit
GOWORK=off go run .
```

输出：
- 默认（不带参数）：在内置语料上做并行 grid search，输出最佳阈值 + 系数，并打印 train/test 误差。
- 带参数：支持 robust loss（Huber/IRLS）、ridge 正则、从 JSONL 日志训练、`-out-zr-config` 导出。

### 常见用法

**1) 旧版默认 grid search（内置语料）**
```bash
cd tokenest/tools/fit
GOWORK=off go run .
```

导出最佳参数到 JSON：
```bash
GOWORK=off go run . -out-zr-config tokenest/report/zr-config.json
```

**2) 固定阈值 + robust refit（内置语料）**
适合快速试验 robust loss，不跑 grid search：
```bash
GOWORK=off go run . -no-grid -loss huber_rel -huber-delta 0.20 -irls-iters 5 -ridge-lambda 0.001 -out-zr-config tokenest/report/zr-config.json
```

**3) 从 JSONL 日志训练（未来数据集）**
JSONL：每行一个 JSON。需要你指定字段路径：
- `-jsonl-text`：提取后的文本字段（dot-path）
- `-jsonl-tokens`：真实 token 字段（可选；为空则用 `tiktoken` o200k_base 计算）
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

**保守模式（更厌恶低估）**
```bash
GOWORK=off go run . -no-grid -loss asym_huber_rel -asym-alpha 2.0 -huber-delta 0.20 -irls-iters 5
```

### 如何使用导出结果
- 现在：把导出的阈值/系数拷贝到 `tokenest/strategy/strategyTest1_params.go`（ZR 硬编码参数）。
- 未来：`update-zr-explain-config` 提案会支持运行时直接加载 ZR config JSON，无需重新编译。

## adversary
生成对抗样本，寻找最坏低估/高估情况，评估鲁棒性。

运行：
```bash
cd tokenest/tools/adversary
GOWORK=off go run . -samples 2000 -length 50000 -top 5 -workers 4
```

选项：
- `-save-top`：保存 TokenX 和 Weighted 最坏的前 N 个样本（默认 5，设为 0 关闭）。
- `-save-dir`：保存目录（默认 `tokenest/datasets/test`）。
- `-workers`：并发数（自动限制到 8，避免打满主机）。
- `-report-dir`：报告输出目录（默认 `tokenest/report`，同时输出 md + xlsx，用 `-` 关闭）。

说明：
- Excel 会包含比例柱状图。

## 报告
accuracy 报告会写入 `tokenest/report/`，文件名包含时间戳（.md + .xlsx）。
