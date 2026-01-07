package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/EZ-Api/tokenest"
	"github.com/pkoukk/tiktoken-go"
)

type sample struct {
	name      string
	path      string
	url       string
	inline    string
	cacheFile string
	normalize func(string) string
}

type sampleData struct {
	sample sample
	text   string
}

func main() {
	reportDirFlag := flag.String("report-dir", "", "write markdown + xlsx reports to this directory (default: <repo>/tokenest/report)")
	flag.Parse()

	enc := mustEncoding()
	repoRoot := findRepoRoot()
	tokenxFixtureDir := filepath.Join(repoRoot, "tokenx", "test", "fixtures", "ebooks")
	tokenxTypescript := filepath.Join(repoRoot, "tokenx", "node_modules", "typescript", "lib", "lib.es5.d.ts")
	datasetDir := filepath.Join(repoRoot, "tokenest", "datasets", "test")
	reportDir := *reportDirFlag
	if reportDir == "" {
		reportDir = filepath.Join(repoRoot, "tokenest", "report")
	}

	samples := []sample{
		{
			name:   "Short English text",
			inline: "Hello, world! This is a short sentence.",
		},
		{
			name:   "German text with umlauts",
			inline: "Die p\u00fcnktlich gew\u00fcnschte Tr\u00fcffelf\u00fcllung im \u00fcbergest\u00fclpten W\u00fcrzk\u00fcmmel-W\u00fcrfel ist k\u00fcmmerlich und d\u00fcrfte f\u00fcrderhin zu R\u00fcffeln in H\u00fclle und F\u00fclle f\u00fchren",
		},
		{
			name: "Metamorphosis by Franz Kafka (English)",
			path: filepath.Join(tokenxFixtureDir, "pg5200.txt"),
		},
		{
			name: "Die Verwandlung by Franz Kafka (German)",
			path: filepath.Join(tokenxFixtureDir, "pg22367.txt"),
		},
		{
			name: "\u9053\u5fb7\u7d93 by Laozi (Chinese)",
			path: filepath.Join(tokenxFixtureDir, "pg7337.txt"),
		},
		{
			name:      "TypeScript ES5 Type Declarations (~ 4000 loc)",
			path:      tokenxTypescript,
			url:       "https://unpkg.com/typescript@5.9.3/lib/lib.es5.d.ts",
			cacheFile: "lib.es5.d.ts",
		},
	}
	samples = append(samples, loadDatasetSamples(datasetDir)...)

	loaded := make([]sampleData, 0, len(samples))
	for _, s := range samples {
		text := loadSample(s)
		if text == "" {
			fmt.Printf("%s\t%s\n", s.name, "failed to load")
			continue
		}
		loaded = append(loaded, sampleData{sample: s, text: text})
	}

	if mixed, ok := buildMixedSample(loaded); ok {
		loaded = append(loaded, mixed)
	}

	gptTokenizerResults, gptTokenizerErr := countGPTTokenizer(loaded)
	if gptTokenizerErr != nil {
		fmt.Fprintf(os.Stderr, "gpt-tokenizer error: %v\n", gptTokenizerErr)
	}

	header := []string{
		"Description",
		"Actual (tiktoken-go o200k_base)",
		"GPT-Tokenizer",
		"GPT-Tokenizer Deviation",
		"UltraFast",
		"UltraFast Deviation",
		"Fast",
		"Fast Deviation",
		"TokenX",
		"TokenX Deviation",
		"NewAPI",
		"NewAPI Deviation",
		"Weighted",
		"Weighted Deviation",
		"tiktoken-go Avg Time",
		"GPT-Tokenizer Avg Time",
		"UltraFast Avg Time",
		"Fast Avg Time",
		"TokenX Avg Time",
		"NewAPI Avg Time",
		"Weighted Avg Time",
	}
	fmt.Println(strings.Join(header, "\t"))

	rows := make([][]string, 0, len(loaded))
	for _, item := range loaded {
		s := item.sample
		text := item.text

		actual, gptAvg := timedCount(func() int {
			return len(enc.Encode(text, nil, nil))
		}, len(text))

		ultra, ultraAvg := timedCount(func() int {
			res := tokenest.EstimateText(text, tokenest.Options{
				Strategy: tokenest.StrategyUltraFast,
			})
			return res.Tokens
		}, len(text))

		fast, fastAvg := timedCount(func() int {
			res := tokenest.EstimateText(text, tokenest.Options{
				Strategy: tokenest.StrategyFast,
			})
			return res.Tokens
		}, len(text))

		tokenxCount, tokenxAvg := timedCount(func() int {
			return estimateTokenX(text)
		}, len(text))

		newapiCount, newapiAvg := timedCount(func() int {
			return estimateNewAPI(newAPIProviderOpenAI, text)
		}, len(text))

		weighted, weightedAvg := timedCount(func() int {
			res := tokenest.EstimateText(text, tokenest.Options{
				Strategy: tokenest.StrategyWeighted,
				Profile:  tokenest.ProfileOpenAI,
			})
			return res.Tokens
		}, len(text))

		gptTokenizerCount := 0
		var gptTokenizerAvg time.Duration
		if gptTokenizerErr == nil {
			if result, ok := gptTokenizerResults[s.name]; ok {
				gptTokenizerCount = result.Count
				gptTokenizerAvg = time.Duration(result.AvgNs)
			}
		}

		row := []string{
			s.name,
			strconv.Itoa(actual),
			strconv.Itoa(gptTokenizerCount),
			fmt.Sprintf("%.2f%%", deviationSigned(actual, gptTokenizerCount)),
			strconv.Itoa(ultra),
			fmt.Sprintf("%.2f%%", deviationSigned(actual, ultra)),
			strconv.Itoa(fast),
			fmt.Sprintf("%.2f%%", deviationSigned(actual, fast)),
			strconv.Itoa(tokenxCount),
			fmt.Sprintf("%.2f%%", deviationSigned(actual, tokenxCount)),
			strconv.Itoa(newapiCount),
			fmt.Sprintf("%.2f%%", deviationSigned(actual, newapiCount)),
			strconv.Itoa(weighted),
			fmt.Sprintf("%.2f%%", deviationSigned(actual, weighted)),
			formatDuration(gptAvg),
			formatDuration(gptTokenizerAvg),
			formatDuration(ultraAvg),
			formatDuration(fastAvg),
			formatDuration(tokenxAvg),
			formatDuration(newapiAvg),
			formatDuration(weightedAvg),
		}
		fmt.Println(strings.Join(row, "\t"))
		rows = append(rows, row)
	}

	if reportDir != "" {
		if err := writeReports(reportDir, header, rows); err != nil {
			fmt.Fprintf(os.Stderr, "report write error: %v\n", err)
		}
	}
}

func loadDatasetSamples(dir string) []sample {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil
	}

	paths := make([]string, 0, len(entries))
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		name := entry.Name()
		ext := strings.ToLower(filepath.Ext(name))
		if ext != ".txt" && ext != ".go" {
			continue
		}
		paths = append(paths, filepath.Join(dir, name))
	}
	if len(paths) == 0 {
		return nil
	}

	sort.Strings(paths)
	samples := make([]sample, 0, len(paths))
	for _, path := range paths {
		base := strings.TrimSuffix(filepath.Base(path), filepath.Ext(path))
		display := "Dataset: " + strings.ReplaceAll(base, "_", " ")
		samples = append(samples, sample{
			name: display,
			path: path,
		})
	}
	return samples
}

func mustEncoding() *tiktoken.Tiktoken {
	enc, err := tiktoken.GetEncoding("o200k_base")
	if err != nil {
		panic(err)
	}
	return enc
}

type gptTokenizerSample struct {
	Name string `json:"name"`
	Text string `json:"text"`
}

type gptTokenizerPayload struct {
	Samples []gptTokenizerSample `json:"samples"`
}

type gptTokenizerResult struct {
	Count int   `json:"count"`
	AvgNs int64 `json:"avg_ns"`
}

type gptTokenizerResponse struct {
	Results map[string]gptTokenizerResult `json:"results"`
}

func countGPTTokenizer(samples []sampleData) (map[string]gptTokenizerResult, error) {
	if len(samples) == 0 {
		return map[string]gptTokenizerResult{}, nil
	}

	payload := gptTokenizerPayload{Samples: make([]gptTokenizerSample, 0, len(samples))}
	for _, s := range samples {
		payload.Samples = append(payload.Samples, gptTokenizerSample{
			Name: s.sample.name,
			Text: s.text,
		})
	}

	body, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}

	cmd := exec.Command("node", "./gpt-tokenizer.mjs")
	cmd.Stdin = bytes.NewReader(body)

	var stdout bytes.Buffer
	var stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("gpt-tokenizer failed: %w: %s", err, strings.TrimSpace(stderr.String()))
	}

	var resp gptTokenizerResponse
	if err := json.Unmarshal(stdout.Bytes(), &resp); err != nil {
		return nil, err
	}

	return resp.Results, nil
}

func writeReports(dir string, header []string, rows [][]string) error {
	if dir == "" {
		return nil
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}

	now := time.Now().UTC()
	normalized := normalizeRows(header, rows)

	if err := writeMarkdownReport(dir, now, header, normalized); err != nil {
		return err
	}

	if err := writeXLSXReport(dir, now, header, normalized); err != nil {
		return err
	}

	return nil
}

func normalizeRows(header []string, rows [][]string) [][]string {
	out := make([][]string, 0, len(rows))
	for _, row := range rows {
		if len(row) < len(header) {
			padded := make([]string, len(header))
			copy(padded, row)
			row = padded
		}
		out = append(out, row)
	}
	return out
}

func writeMarkdownReport(dir string, now time.Time, header []string, rows [][]string) error {
	fileName := fmt.Sprintf("testAccuracy-%s.md", now.Format("20060102-150405Z"))
	path := filepath.Join(dir, fileName)

	sep := make([]string, len(header))
	for i := range sep {
		sep[i] = "---"
	}

	var b strings.Builder
	b.WriteString("# testAccuracy\n\n")
	b.WriteString("Generated by `tokenest/tools/accuracy` using `o200k_base`.\n")
	b.WriteString("Generated at: ")
	b.WriteString(now.Format(time.RFC3339))
	b.WriteString("\n\n")
	b.WriteString("| ")
	b.WriteString(strings.Join(header, " | "))
	b.WriteString(" |\n")
	b.WriteString("| ")
	b.WriteString(strings.Join(sep, " | "))
	b.WriteString(" |\n")

	for _, row := range rows {
		b.WriteString("| ")
		b.WriteString(strings.Join(row, " | "))
		b.WriteString(" |\n")
	}

	return os.WriteFile(path, []byte(b.String()), 0o644)
}

type xlsxColumn struct {
	Index int    `json:"index"`
	Title string `json:"title"`
}

type accuracyXLSXPayload struct {
	ReportType       string       `json:"report_type"`
	Title            string       `json:"title"`
	GeneratedAt      string       `json:"generated_at"`
	Note             string       `json:"note"`
	Header           []string     `json:"header"`
	Rows             [][]string   `json:"rows"`
	DeviationColumns []xlsxColumn `json:"deviation_columns"`
}

func writeXLSXReport(dir string, now time.Time, header []string, rows [][]string) error {
	payload := accuracyXLSXPayload{
		ReportType:  "accuracy",
		Title:       "testAccuracy",
		GeneratedAt: now.Format(time.RFC3339),
		Note:        "Generated by `tokenest/tools/accuracy` using `o200k_base`.",
		Header:      header,
		Rows:        rows,
	}

	for i, name := range header {
		if strings.Contains(name, "Deviation") {
			payload.DeviationColumns = append(payload.DeviationColumns, xlsxColumn{
				Index: i,
				Title: name,
			})
		}
	}

	outputName := fmt.Sprintf("testAccuracy-%s.xlsx", now.Format("20060102-150405Z"))
	outputPath := filepath.Join(dir, outputName)
	if absPath, err := filepath.Abs(outputPath); err == nil {
		outputPath = absPath
	}
	return runXLSXReport(outputPath, payload)
}

func runXLSXReport(outputPath string, payload any) error {
	repoRoot := findRepoRoot()
	reportDir := filepath.Join(repoRoot, "tokenest", "tools", "report")
	if _, err := os.Stat(reportDir); err != nil {
		return fmt.Errorf("report generator not found: %w", err)
	}

	tmpFile, err := os.CreateTemp(reportDir, "accuracy-report-*.json")
	if err != nil {
		return err
	}

	encoder := json.NewEncoder(tmpFile)
	if err := encoder.Encode(payload); err != nil {
		_ = tmpFile.Close()
		return err
	}

	if err := tmpFile.Close(); err != nil {
		return err
	}

	tmpPath := tmpFile.Name()

	if err := runReportScript(reportDir, tmpPath, outputPath); err != nil {
		return err
	}

	_ = os.Remove(tmpPath)
	return nil
}

func runReportScript(reportDir, inputPath, outputPath string) error {
	if err := runWithUV(reportDir, inputPath, outputPath); err == nil {
		return nil
	} else if fallbackErr := runWithPython(reportDir, inputPath, outputPath); fallbackErr != nil {
		return fmt.Errorf("uv run failed: %v; python fallback failed: %v", err, fallbackErr)
	}
	return nil
}

func runWithUV(reportDir, inputPath, outputPath string) error {
	cmd := exec.Command("uv", "run", "python", "report_xlsx.py", "--input", inputPath, "--output", outputPath)
	cmd.Dir = reportDir
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		msg := strings.TrimSpace(stderr.String())
		if msg == "" {
			msg = err.Error()
		}
		return fmt.Errorf("uv run error: %s", msg)
	}
	return nil
}

func runWithPython(reportDir, inputPath, outputPath string) error {
	cmd := exec.Command("python3", "report_xlsx.py", "--input", inputPath, "--output", outputPath)
	cmd.Dir = reportDir
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		msg := strings.TrimSpace(stderr.String())
		if msg == "" {
			msg = err.Error()
		}
		return fmt.Errorf("python error: %s", msg)
	}
	return nil
}

func findRepoRoot() string {
	wd, err := os.Getwd()
	if err != nil {
		return "."
	}
	return filepath.Clean(filepath.Join(wd, "..", "..", ".."))
}

func buildMixedSample(samples []sampleData) (sampleData, bool) {
	english, ok := findSampleText(samples, "Metamorphosis by Franz Kafka (English)")
	if !ok {
		return sampleData{}, false
	}
	cjk, ok := findSampleText(samples, "\u9053\u5fb7\u7d93 by Laozi (Chinese)")
	if !ok {
		return sampleData{}, false
	}
	code, ok := findSampleText(samples, "TypeScript ES5 Type Declarations (~ 4000 loc)")
	if !ok {
		return sampleData{}, false
	}

	mixed := strings.Join([]string{
		takeRunes(english, 0.33),
		takeRunes(cjk, 0.33),
		takeRunes(code, 0.33),
	}, "\n\n")

	return sampleData{
		sample: sample{name: "Mixed (EN+CJK+Code)"},
		text:   mixed,
	}, true
}

func findSampleText(samples []sampleData, name string) (string, bool) {
	for _, item := range samples {
		if item.sample.name == name {
			return item.text, true
		}
	}
	return "", false
}

func takeRunes(text string, fraction float64) string {
	if text == "" || fraction <= 0 {
		return ""
	}
	runeCount := utf8.RuneCountInString(text)
	if runeCount == 0 {
		return ""
	}
	target := int(math.Ceil(float64(runeCount) * fraction))
	if target <= 0 {
		return ""
	}
	if target >= runeCount {
		return text
	}
	count := 0
	for idx := range text {
		if count == target {
			return text[:idx]
		}
		count++
	}
	return text
}

func loadSample(s sample) string {
	text := s.inline
	if text == "" && s.path != "" {
		if data, err := os.ReadFile(s.path); err == nil {
			text = string(data)
		}
	}
	if text == "" && s.url != "" {
		var err error
		text, err = fetchText(s.url, s.cacheFile)
		if err != nil {
			return ""
		}
	}
	if s.normalize != nil {
		text = s.normalize(text)
	}
	return text
}

func fetchText(url string, cacheFile string) (string, error) {
	if cacheFile != "" {
		cachePath := filepath.Join("data", cacheFile)
		if cached, err := os.ReadFile(cachePath); err == nil {
			return string(cached), nil
		}
		if err := os.MkdirAll(filepath.Dir(cachePath), 0o755); err != nil {
			return "", err
		}
		body, err := downloadText(url)
		if err != nil {
			return "", err
		}
		if err := os.WriteFile(cachePath, []byte(body), 0o644); err != nil {
			return "", err
		}
		return body, nil
	}
	return downloadText(url)
}

func downloadText(url string) (string, error) {
	resp, err := http.Get(url)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return "", fmt.Errorf("http status %d", resp.StatusCode)
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", err
	}
	return string(body), nil
}

func timedCount(fn func() int, size int) (int, time.Duration) {
	iterations := pickIterations(size)
	_ = fn()
	start := time.Now()
	var count int
	for i := 0; i < iterations; i++ {
		count = fn()
	}
	elapsed := time.Since(start)
	if iterations == 0 {
		return count, 0
	}
	return count, elapsed / time.Duration(iterations)
}

func pickIterations(size int) int {
	switch {
	case size < 200:
		return 20000
	case size < 2000:
		return 2000
	case size < 20000:
		return 200
	case size < 200000:
		return 20
	default:
		return 5
	}
}

func deviationSigned(actual, estimated int) float64 {
	if actual == 0 {
		return 0
	}
	return (float64(estimated-actual) / float64(actual)) * 100
}

func formatDuration(d time.Duration) string {
	if d == 0 {
		return "0"
	}
	if d < time.Microsecond {
		return fmt.Sprintf("%dns", d.Nanoseconds())
	}
	if d < time.Millisecond {
		return fmt.Sprintf("%.2fus", float64(d.Nanoseconds())/1000.0)
	}
	return fmt.Sprintf("%.2fms", float64(d.Nanoseconds())/1e6)
}
