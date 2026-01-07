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
使用线性回归拟合 Weighted 系数，使其更接近 `tiktoken`。

运行：
```bash
cd tokenest/tools/fit
GOWORK=off go run .
```

输出：
- 打印拟合系数与各样本误差。

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
