package main

import (
	"fmt"
	"io"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"unicode"
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
}

type sampleData struct {
	sample sample
	text   string
}

type tokenXStats struct {
	TotalRunes int
	CJKRunes   int
	PunctRunes int
	DigitRunes int
}

type featureRow struct {
	name   string
	actual float64
	base   float64
	feat   []float64
}

type searchConfig struct {
	charsPerToken  float64
	shortThreshold int
}

func main() {
	enc := mustEncoding()
	repoRoot := findRepoRoot()
	datasetsDir := filepath.Join(repoRoot, "tokenest", "datasets", "test")

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
			name: "Bible KJV (English)",
			path: filepath.Join(datasetsDir, "bible_kjv_en.txt"),
		},
		{
			name: "Capital (English)",
			path: filepath.Join(datasetsDir, "capital_en.txt"),
		},
		{
			name: "Candide (French)",
			path: filepath.Join(datasetsDir, "candide_fr.txt"),
		},
		{
			name: "Faust (German)",
			path: filepath.Join(datasetsDir, "faust_de.txt"),
		},
		{
			name: "Analects (Chinese)",
			path: filepath.Join(datasetsDir, "analects_zh.txt"),
		},
		{
			name: "Go HTTP Server (Code)",
			path: filepath.Join(datasetsDir, "golang_net_http_server.go"),
		},
		{name: "Mixed3 01", path: filepath.Join(datasetsDir, "mixed3_01_zh_en_code.txt")},
		{name: "Mixed3 02", path: filepath.Join(datasetsDir, "mixed3_02_zh_en_code.txt")},
		{name: "Mixed3 03", path: filepath.Join(datasetsDir, "mixed3_03_zh_en_code.txt")},
		{name: "Mixed5 01", path: filepath.Join(datasetsDir, "mixed5_01_zh_en_de_fr_code.txt")},
		{name: "Mixed5 02", path: filepath.Join(datasetsDir, "mixed5_02_zh_en_de_fr_code.txt")},
		{name: "Mixed5 03", path: filepath.Join(datasetsDir, "mixed5_03_zh_en_de_fr_code.txt")},
		{name: "Mixed5 04", path: filepath.Join(datasetsDir, "mixed5_04_zh_en_de_fr_code.txt")},
		{name: "Mixed5 05", path: filepath.Join(datasetsDir, "mixed5_05_zh_en_de_fr_code.txt")},
	}

	loaded := make([]sampleData, 0, len(samples))
	for _, s := range samples {
		text := loadSample(s)
		if text == "" {
			fmt.Printf("%s\t%s\n", s.name, "failed to load")
			continue
		}
		loaded = append(loaded, sampleData{sample: s, text: text})
	}

	// Prepare data splits
	var trainItems []sampleData
	var testItems []sampleData

	// We split each sample text into train/test parts to ensure coverage
	for _, item := range loaded {
		runes := []rune(item.text)
		if len(runes) == 0 {
			continue
		}

		splitIdx := int(float64(len(runes)) * 0.8)
		if splitIdx == 0 && len(runes) > 0 {
			splitIdx = len(runes)
		}

		trainText := string(runes[:splitIdx])
		testText := string(runes[splitIdx:])

		if len(trainText) > 0 {
			trainItems = append(trainItems, sampleData{sample: item.sample, text: trainText})
		}
		if len(testText) > 0 {
			testItems = append(testItems, sampleData{sample: item.sample, text: testText})
		}
	}

	// Grid Search
	var bestConfig searchConfig
	var bestCoeffs []float64
	bestTrainMAPE := math.MaxFloat64

	fmt.Println("Starting grid search for hyperparameters...")

	for chars := 3.0; chars <= 8.0; chars += 0.5 {
		for threshold := 1; threshold <= 6; threshold++ {
			cfg := searchConfig{
				charsPerToken:  chars,
				shortThreshold: threshold,
			}

			// Build train features
			trainRows := make([]featureRow, 0, len(trainItems))
			for _, item := range trainItems {
				trainRows = append(trainRows, makeFeatureRow(item.sample.name, item.text, enc, cfg))
			}

			// Fit
			x := make([][]float64, 0, len(trainRows))
			y := make([]float64, 0, len(trainRows))
			for _, row := range trainRows {
				x = append(x, row.feat)
				y = append(y, row.actual)
			}

			coeffs, err := solveLeastSquares(x, y)
			if err != nil {
				continue
			}

			// Evaluate on Train
			mape := calculateMAPE(trainRows, coeffs)
			if mape < bestTrainMAPE {
				bestTrainMAPE = mape
				bestConfig = cfg
				bestCoeffs = coeffs
			}
		}
	}

	fmt.Printf("\n=== BEST CONFIGURATION FOUND ===\n")
	fmt.Printf("Train MAPE: %.4f%%\n", bestTrainMAPE)
	fmt.Printf("CharsPerToken: %.1f\n", bestConfig.charsPerToken)
	fmt.Printf("ShortThreshold: %d\n", bestConfig.shortThreshold)

	fmt.Println("\nWeighted fit coefficients (o200k_base):")
	fmt.Printf("baseFactor=%.4f\n", bestCoeffs[0])
	fmt.Printf("cjkRatioFactor=%.4f\n", bestCoeffs[1])
	fmt.Printf("punctRatioFactor=%.4f\n", bestCoeffs[2])
	fmt.Printf("digitRatioFactor=%.4f\n", bestCoeffs[3])

	// Re-evaluate on Train with best config
	fmt.Println("\n=== TRAIN SET EVALUATION (Best Config) ===")
	finalTrainRows := make([]featureRow, 0, len(trainItems))
	for _, item := range trainItems {
		finalTrainRows = append(finalTrainRows, makeFeatureRow(item.sample.name, item.text, enc, bestConfig))
	}
	evaluate(finalTrainRows, bestCoeffs)

	// Re-evaluate on Test with best config
	fmt.Println("\n=== TEST SET EVALUATION (Best Config) ===")
	finalTestRows := make([]featureRow, 0, len(testItems))
	for _, item := range testItems {
		finalTestRows = append(finalTestRows, makeFeatureRow(item.sample.name, item.text, enc, bestConfig))
	}
	evaluate(finalTestRows, bestCoeffs)

	fmt.Println("\nSuggested tuning snippet:")
	fmt.Printf("// Update tokenest/weighted.go constants:\n")
	fmt.Printf("tokenXDefaultCharsPerToken = %.1f\n", bestConfig.charsPerToken)
	fmt.Printf("tokenXShortTokenThreshold  = %d\n", bestConfig.shortThreshold)
	fmt.Printf("\n// Update tuning coefficients:\n")
	fmt.Printf("baseFactor: %.4f,\n", bestCoeffs[0])
	fmt.Printf("cjkRatioFactor: %.4f,\n", bestCoeffs[1])
	fmt.Printf("punctRatioFactor: %.4f,\n", bestCoeffs[2])
	fmt.Printf("digitRatioFactor: %.4f,\n", bestCoeffs[3])

	fmt.Println("\nCurrent Weighted estimate (library, untuned) per sample (Full Text):")
	for _, item := range loaded {
		res := tokenest.EstimateText(item.text, tokenest.Options{
			Strategy: tokenest.StrategyWeighted,
			Profile:  tokenest.ProfileOpenAI,
		})
		fmt.Printf("%s\tactual=%d\tweighted=%d\n", item.sample.name, len(enc.Encode(item.text, nil, nil)), res.Tokens)
	}
}

func calculateMAPE(rows []featureRow, coeffs []float64) float64 {
	var totalAbsPct float64
	count := 0
	for _, row := range rows {
		pred := predict(coeffs, row.feat)
		if row.actual > 0 {
			totalAbsPct += math.Abs(pred-row.actual) / row.actual * 100
			count++
		}
	}
	if count == 0 {
		return 0
	}
	return totalAbsPct / float64(count)
}

func makeFeatureRow(name string, text string, enc *tiktoken.Tiktoken, cfg searchConfig) featureRow {
	actual := float64(len(enc.Encode(text, nil, nil)))
	baseTokens, stats := estimateTokenXWithStats(text, cfg)
	features := buildFeatures(baseTokens, stats)
	return featureRow{
		name:   name,
		actual: actual,
		base:   float64(baseTokens),
		feat:   features,
	}
}

func evaluate(rows []featureRow, coeffs []float64) {
	var totalAbsPct float64
	for _, row := range rows {
		pred := predict(coeffs, row.feat)
		pct := 0.0
		if row.actual > 0 {
			pct = math.Abs(pred-row.actual) / row.actual * 100
		}
		totalAbsPct += pct
		fmt.Printf("%s\tactual=%.0f\tpred=%.0f\tape=%.2f%%\n", row.name, row.actual, pred, pct)
	}
	if len(rows) > 0 {
		fmt.Printf("MAPE: %.2f%%\n", totalAbsPct/float64(len(rows)))
	}
}

func mustEncoding() *tiktoken.Tiktoken {
	enc, err := tiktoken.GetEncoding("o200k_base")
	if err != nil {
		panic(err)
	}
	return enc
}

func findRepoRoot() string {
	wd, err := os.Getwd()
	if err != nil {
		return "."
	}
	return filepath.Clean(filepath.Join(wd, "..", "..", ".."))
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

func buildFeatures(baseTokens int, stats tokenXStats) []float64 {
	if baseTokens <= 0 {
		return []float64{0, 0, 0, 0}
	}
	total := stats.TotalRunes
	if total == 0 {
		total = 1
	}
	base := float64(baseTokens)
	cjkRatio := float64(stats.CJKRunes) / float64(total)
	punctRatio := float64(stats.PunctRunes) / float64(total)
	digitRatio := float64(stats.DigitRunes) / float64(total)

	return []float64{
		base,
		base * cjkRatio,
		base * punctRatio,
		base * digitRatio,
	}
}

func predict(coeffs []float64, features []float64) float64 {
	sum := 0.0
	for i, coef := range coeffs {
		sum += coef * features[i]
	}
	return sum
}

func solveLeastSquares(x [][]float64, y []float64) ([]float64, error) {
	if len(x) == 0 {
		return nil, fmt.Errorf("empty dataset")
	}
	featureCount := len(x[0])
	xtx := make([][]float64, featureCount)
	for i := range xtx {
		xtx[i] = make([]float64, featureCount)
	}
	xty := make([]float64, featureCount)

	for row := 0; row < len(x); row++ {
		for i := 0; i < featureCount; i++ {
			xty[i] += x[row][i] * y[row]
			for j := 0; j < featureCount; j++ {
				xtx[i][j] += x[row][i] * x[row][j]
			}
		}
	}

	return solveLinearSystem(xtx, xty)
}

func solveLinearSystem(a [][]float64, b []float64) ([]float64, error) {
	n := len(b)
	for i := 0; i < n; i++ {
		maxRow := i
		maxVal := math.Abs(a[i][i])
		for r := i + 1; r < n; r++ {
			if v := math.Abs(a[r][i]); v > maxVal {
				maxVal = v
				maxRow = r
			}
		}
		if maxVal == 0 {
			return nil, fmt.Errorf("singular matrix")
		}
		if maxRow != i {
			a[i], a[maxRow] = a[maxRow], a[i]
			b[i], b[maxRow] = b[maxRow], b[i]
		}

		pivot := a[i][i]
		for j := i; j < n; j++ {
			a[i][j] /= pivot
		}
		b[i] /= pivot

		for r := 0; r < n; r++ {
			if r == i {
				continue
			}
			factor := a[r][i]
			for j := i; j < n; j++ {
				a[r][j] -= factor * a[i][j]
			}
			b[r] -= factor * b[i]
		}
	}
	return b, nil
}

func estimateTokenXWithStats(text string, cfg searchConfig) (int, tokenXStats) {
	stats := tokenXStats{}
	if text == "" {
		return 0, stats
	}

	baseTokens := 0
	segmentStart := 0
	segmentType := tokenXSegmentTypeNone
	first := true

	for idx, r := range text {
		currentType := tokenXSegmentTypeForRune(r)
		if first {
			first = false
			segmentType = currentType
			continue
		}

		if currentType != segmentType {
			baseTokens += estimateTokenXSegment(text[segmentStart:idx], &stats, cfg)
			segmentStart = idx
			segmentType = currentType
		}
	}

	if segmentStart < len(text) {
		baseTokens += estimateTokenXSegment(text[segmentStart:], &stats, cfg)
	}

	return baseTokens, stats
}

type tokenXSegmentType int

const (
	tokenXSegmentTypeNone tokenXSegmentType = iota
	tokenXSegmentTypeWhitespace
	tokenXSegmentTypePunctuation
	tokenXSegmentTypeOther
)

func tokenXSegmentTypeForRune(r rune) tokenXSegmentType {
	if unicode.IsSpace(r) {
		return tokenXSegmentTypeWhitespace
	}
	if isTokenXPunct(r) {
		return tokenXSegmentTypePunctuation
	}
	return tokenXSegmentTypeOther
}

func estimateTokenXSegment(segment string, stats *tokenXStats, cfg searchConfig) int {
	if segment == "" {
		return 0
	}

	if isTokenXWhitespace(segment) {
		return 0
	}

	runeCount := utf8.RuneCountInString(segment)
	stats.TotalRunes += runeCount

	for _, r := range segment {
		if isCJKRune(r) {
			stats.CJKRunes++
		}
		if isTokenXPunct(r) {
			stats.PunctRunes++
		}
		if r >= '0' && r <= '9' {
			stats.DigitRunes++
		}
	}

	if isCJKSegment(segment) {
		return runeCount
	}

	if isNumericSegment(segment) {
		return 1
	}

	if runeCount <= cfg.shortThreshold {
		return 1
	}

	if containsTokenXPunct(segment) {
		if runeCount > 1 {
			return int(math.Ceil(float64(runeCount) / 2.0))
		}
		return 1
	}

	if isAlphanumericSegment(segment) {
		avg := getLanguageSpecificCharsPerToken(segment)
		if avg <= 0 {
			avg = cfg.charsPerToken
		}
		return int(math.Ceil(float64(runeCount) / avg))
	}

	return runeCount
}

func isTokenXWhitespace(segment string) bool {
	for _, r := range segment {
		if !unicode.IsSpace(r) {
			return false
		}
	}
	return segment != ""
}

func containsTokenXPunct(segment string) bool {
	for _, r := range segment {
		if isTokenXPunct(r) {
			return true
		}
	}
	return false
}

func isTokenXPunct(r rune) bool {
	switch r {
	case '.', ',', '!', '?', ';', '(', ')', '{', '}', '[', ']', '<', '>', ':', '/', '\\', '|', '@', '#', '$', '%', '^', '&', '*', '+', '=', '`', '~', '_', '-':
		return true
	default:
		return false
	}
}

func isCJKSegment(segment string) bool {
	if segment == "" {
		return false
	}
	for _, r := range segment {
		if !isCJKRune(r) {
			return false
		}
	}
	return true
}

func isCJKRune(r rune) bool {
	switch {
	case r >= 0x4e00 && r <= 0x9fff:
		return true
	case r >= 0x3400 && r <= 0x4dbf:
		return true
	case r >= 0x3000 && r <= 0x303f:
		return true
	case r >= 0xff00 && r <= 0xffef:
		return true
	case r >= 0x30a0 && r <= 0x30ff:
		return true
	case r >= 0x2e80 && r <= 0x2eff:
		return true
	case r >= 0x31c0 && r <= 0x31ef:
		return true
	case r >= 0x3200 && r <= 0x32ff:
		return true
	case r >= 0x3300 && r <= 0x33ff:
		return true
	case r >= 0xac00 && r <= 0xd7af:
		return true
	case r >= 0x1100 && r <= 0x11ff:
		return true
	case r >= 0x3130 && r <= 0x318f:
		return true
	case r >= 0xa960 && r <= 0xa97f:
		return true
	case r >= 0xd7b0 && r <= 0xd7ff:
		return true
	default:
		return false
	}
}

func isNumericSegment(segment string) bool {
	hasDigit := false
	prevSeparator := false
	for _, r := range segment {
		if r >= '0' && r <= '9' {
			hasDigit = true
			prevSeparator = false
			continue
		}
		if r == '.' || r == ',' {
			if prevSeparator {
				return false
			}
			prevSeparator = true
			continue
		}
		return false
	}
	return hasDigit && !prevSeparator
}

func isAlphanumericSegment(segment string) bool {
	for _, r := range segment {
		if isLatinAlphaNum(r) {
			continue
		}
		return false
	}
	return true
}

func isLatinAlphaNum(r rune) bool {
	if r >= 'a' && r <= 'z' {
		return true
	}
	if r >= 'A' && r <= 'Z' {
		return true
	}
	if r >= '0' && r <= '9' {
		return true
	}
	if r >= 0x00c0 && r <= 0x00ff {
		return true
	}
	return false
}

type languageConfig struct {
	avgCharsPerToken float64
	set              map[rune]struct{}
}

func (c languageConfig) matches(segment string) bool {
	for _, r := range segment {
		if _, ok := c.set[unicode.ToLower(r)]; ok {
			return true
		}
	}
	return false
}

var defaultLanguageConfigs = []languageConfig{
	{
		avgCharsPerToken: 3,
		set: map[rune]struct{}{
			'\u00e4': {},
			'\u00f6': {},
			'\u00fc': {},
			'\u00df': {},
			'\u1e9e': {},
		},
	},
	{
		avgCharsPerToken: 3,
		set: map[rune]struct{}{
			'\u00e9': {},
			'\u00e8': {},
			'\u00ea': {},
			'\u00eb': {},
			'\u00e0': {},
			'\u00e2': {},
			'\u00ee': {},
			'\u00ef': {},
			'\u00f4': {},
			'\u00fb': {},
			'\u00f9': {},
			'\u00fc': {},
			'\u00ff': {},
			'\u00e7': {},
			'\u0153': {},
			'\u00e6': {},
			'\u00e1': {},
			'\u00ed': {},
			'\u00f3': {},
			'\u00fa': {},
			'\u00f1': {},
		},
	},
	{
		avgCharsPerToken: 3.5,
		set: map[rune]struct{}{
			'\u0105': {},
			'\u0107': {},
			'\u0119': {},
			'\u0142': {},
			'\u0144': {},
			'\u00f3': {},
			'\u015b': {},
			'\u017a': {},
			'\u017c': {},
			'\u011b': {},
			'\u0161': {},
			'\u010d': {},
			'\u0159': {},
			'\u017e': {},
			'\u00fd': {},
			'\u016f': {},
			'\u00fa': {},
			'\u010f': {},
			'\u0165': {},
			'\u0148': {},
		},
	},
}

func getLanguageSpecificCharsPerToken(segment string) float64 {
	for _, cfg := range defaultLanguageConfigs {
		if cfg.matches(segment) {
			return cfg.avgCharsPerToken
		}
	}
	return 0
}
