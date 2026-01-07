package main

import (
	"fmt"
	"io"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"sync"
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
	actual float64 // Pre-calculated ground truth
}

type tokenXStats struct {
	TotalRunes int
	CJKRunes   int
	PunctRunes int
	DigitRunes int
	SpaceRunes int
	UpperRunes int
	HexRunes   int
	CodePunct  int
}

type featureRow struct {
	name     string
	actual   float64
	base     float64
	feat     []float64
	category int
}

const (
	CatGeneral = iota
	CatCapital
	CatDense
	CatHex
	CatAlnum
	CatCode
	CatText
)

type searchConfig struct {
	charsPerToken       float64
	shortThreshold      int
	capitalThreshold    float64
	denseThreshold      float64
	hexThreshold        float64
	alnumPunctThreshold float64
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

		// Adversary TokenX
		{name: "Adversary TokenX 01", path: filepath.Join(datasetsDir, "adversary_tokenx_01_alnum_run.txt")},
		{name: "Adversary TokenX 02", path: filepath.Join(datasetsDir, "adversary_tokenx_02_alnum_run.txt")},
		{name: "Adversary TokenX 03", path: filepath.Join(datasetsDir, "adversary_tokenx_03_alnum_run.txt")},
		{name: "Adversary TokenX 04", path: filepath.Join(datasetsDir, "adversary_tokenx_04_alnum_run.txt")},
		{name: "Adversary TokenX 05", path: filepath.Join(datasetsDir, "adversary_tokenx_05_alnum_run.txt")},
		{name: "Adversary TokenX 05 Hex", path: filepath.Join(datasetsDir, "adversary_tokenx_05_hex_stream.txt")},

		// Adversary Weighted
		{name: "Adversary Weighted 01", path: filepath.Join(datasetsDir, "adversary_weighted_01_alnum_run.txt")},
		{name: "Adversary Weighted 02", path: filepath.Join(datasetsDir, "adversary_weighted_02_alnum_run.txt")},
		{name: "Adversary Weighted 03", path: filepath.Join(datasetsDir, "adversary_weighted_03_alnum_run.txt")},
		{name: "Adversary Weighted 04", path: filepath.Join(datasetsDir, "adversary_weighted_04_alnum_run.txt")},
		{name: "Adversary Weighted 05", path: filepath.Join(datasetsDir, "adversary_weighted_05_alnum_run.txt")},
		{name: "Adversary Weighted 05 Base64", path: filepath.Join(datasetsDir, "adversary_weighted_05_base64.txt")},

		// Toxic
		{name: "Toxic Base64", path: filepath.Join(datasetsDir, "toxic_base64.txt")},
		{name: "Toxic Log", path: filepath.Join(datasetsDir, "toxic_log.txt")},
		{name: "Toxic Markdown Table", path: filepath.Join(datasetsDir, "toxic_markdown_table.txt")},
		{name: "Toxic Minified JS", path: filepath.Join(datasetsDir, "toxic_minified_js.txt")},
		{name: "Toxic Minified JSON", path: filepath.Join(datasetsDir, "toxic_minified_json.txt")},
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

	// Pre-calculate actual tokens
	fmt.Println("Pre-calculating ground truth tokens...")
	for i := range trainItems {
		trainItems[i].actual = float64(len(enc.Encode(trainItems[i].text, nil, nil)))
	}
	for i := range testItems {
		testItems[i].actual = float64(len(enc.Encode(testItems[i].text, nil, nil)))
	}

	// Grid Search
	var bestConfig searchConfig
	var bestCoeffs map[int][]float64
	bestTrainMAPE := math.MaxFloat64

	fmt.Println("Starting parallel grid search for hyperparameters...")

	type jobResult struct {
		cfg    searchConfig
		mape   float64
		coeffs map[int][]float64
	}

	jobs := make(chan searchConfig, 1000)
	results := make(chan jobResult, 1000)
	var wg sync.WaitGroup

	// Start workers
	numWorkers := runtime.NumCPU()
	fmt.Printf("Using %d workers\n", numWorkers)

	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for cfg := range jobs {
				// Build train features and split by category
				trainRows := make([]featureRow, 0, len(trainItems))
				rowsByCat := make(map[int][]featureRow)

				for _, item := range trainItems {
					// Use pre-calculated actual
					row := makeFeatureRowWithActual(item.sample.name, item.text, item.actual, cfg)
					trainRows = append(trainRows, row)
					rowsByCat[row.category] = append(rowsByCat[row.category], row)
				}

				// Fit for each category
				coeffsByCat := make(map[int][]float64)

				// Helper to fit a subset
				fitSubset := func(rows []featureRow) ([]float64, error) {
					if len(rows) == 0 {
						return nil, fmt.Errorf("empty subset")
					}
					x := make([][]float64, 0, len(rows))
					y := make([]float64, 0, len(rows))
					for _, row := range rows {
						x = append(x, row.feat)
						y = append(y, row.actual)
					}
					return solveLeastSquares(x, y)
				}

				// 1. Fit General Category
				genCoeffs, err := fitSubset(rowsByCat[CatGeneral])
				if err != nil {
					// Fallback to fitting all rows if General subset fails
					genCoeffs, err = fitSubset(trainRows)
					if err != nil {
						continue // Skip this config
					}
				}
				coeffsByCat[CatGeneral] = genCoeffs

				// 2. Fit other categories
				for _, cat := range []int{CatCapital, CatDense, CatHex, CatAlnum} {
					rows := rowsByCat[cat]
					if len(rows) < 2 {
						// Fallback logic
						if cat == CatAlnum {
							if capCoeffs, ok := coeffsByCat[CatCapital]; ok && len(capCoeffs) > 0 {
								coeffsByCat[cat] = capCoeffs
							} else {
								coeffsByCat[cat] = genCoeffs
							}
						} else {
							coeffsByCat[cat] = genCoeffs
						}
						continue
					}
					catCoeffs, err := fitSubset(rows)
					if err != nil {
						simpleCoeffs, err2 := fitSimple(rows)
						if err2 == nil {
							coeffsByCat[cat] = simpleCoeffs
						} else {
							// Fallback on error
							if cat == CatAlnum {
								if capCoeffs, ok := coeffsByCat[CatCapital]; ok && len(capCoeffs) > 0 {
									coeffsByCat[cat] = capCoeffs
								} else {
									coeffsByCat[cat] = genCoeffs
								}
							} else {
								coeffsByCat[cat] = genCoeffs
							}
						}
					} else {
						coeffsByCat[cat] = catCoeffs
					}
				}

				// Evaluate
				mape := calculateMAPE(trainRows, coeffsByCat)
				results <- jobResult{cfg: cfg, mape: mape, coeffs: coeffsByCat}
			}
		}()
	}

	// Result collector
	done := make(chan bool)
	go func() {
		count := 0
		for res := range results {
			count++
			if count%1000 == 0 {
				fmt.Printf("Processed %d configs...\r", count)
			}
			if res.mape < bestTrainMAPE {
				bestTrainMAPE = res.mape
				bestConfig = res.cfg
				bestCoeffs = res.coeffs
			}
		}
		done <- true
	}()

	// Feed jobs
	for chars := 3.0; chars <= 5.0; chars += 0.5 {
		for threshold := 4; threshold <= 6; threshold++ {
			for capThresh := 0.3; capThresh <= 0.8; capThresh += 0.05 {
				for denseThresh := 0.01; denseThresh <= 0.05; denseThresh += 0.01 {
					for hexThresh := 0.90; hexThresh <= 0.99; hexThresh += 0.02 {
						for alnumThresh := 0.01; alnumThresh <= 0.10; alnumThresh += 0.02 {
							jobs <- searchConfig{
								charsPerToken:       chars,
								shortThreshold:      threshold,
								capitalThreshold:    capThresh,
								denseThreshold:      denseThresh,
								hexThreshold:        hexThresh,
								alnumPunctThreshold: alnumThresh,
							}
						}
					}
				}
			}
		}
	}
	close(jobs)
	wg.Wait()
	close(results)
	<-done

	fmt.Printf("\n=== BEST CONFIGURATION FOUND ===\n")
	fmt.Printf("Train MAPE: %.4f%%\n", bestTrainMAPE)
	fmt.Printf("CharsPerToken: %.1f\n", bestConfig.charsPerToken)
	fmt.Printf("ShortThreshold: %d\n", bestConfig.shortThreshold)
	fmt.Printf("CapitalThreshold: %.2f\n", bestConfig.capitalThreshold)
	fmt.Printf("DenseThreshold: %.2f\n", bestConfig.denseThreshold)
	fmt.Printf("HexThreshold: %.2f\n", bestConfig.hexThreshold)
	fmt.Printf("AlnumPunctThreshold: %.2f\n", bestConfig.alnumPunctThreshold)

	fmt.Println("\nWeighted fit coefficients (o200k_base):")
	printCoeffs("General", bestCoeffs[CatGeneral])
	printCoeffs("Capital", bestCoeffs[CatCapital])
	printCoeffs("Dense", bestCoeffs[CatDense])
	printCoeffs("Hex", bestCoeffs[CatHex])
	printCoeffs("Alnum", bestCoeffs[CatAlnum])

	// Re-evaluate on Train with best config
	fmt.Println("\n=== TRAIN SET EVALUATION (Best Config) ===")
	finalTrainRows := make([]featureRow, 0, len(trainItems))
	for _, item := range trainItems {
		finalTrainRows = append(finalTrainRows, makeFeatureRowWithActual(item.sample.name, item.text, item.actual, bestConfig))
	}
	evaluate(finalTrainRows, bestCoeffs)

	// Re-evaluate on Test with best config
	fmt.Println("\n=== TEST SET EVALUATION (Best Config) ===")
	finalTestRows := make([]featureRow, 0, len(testItems))
	for _, item := range testItems {
		finalTestRows = append(finalTestRows, makeFeatureRowWithActual(item.sample.name, item.text, item.actual, bestConfig))
	}
	evaluate(finalTestRows, bestCoeffs)

	fmt.Println("\nCurrent Weighted estimate (library, untuned) per sample (Full Text):")
	for _, item := range loaded {
		res := tokenest.EstimateText(item.text, tokenest.Options{
			Strategy: tokenest.StrategyWeighted,
			Profile:  tokenest.ProfileOpenAI,
		})
		fmt.Printf("%s\tactual=%d\tweighted=%d\n", item.sample.name, len(enc.Encode(item.text, nil, nil)), res.Tokens)
	}
}

func printCoeffs(label string, coeffs []float64) {
	if len(coeffs) < 4 {
		fmt.Printf("[%s] No coefficients (using default/fallback)\n", label)
		return
	}
	fmt.Printf("[%s]\n", label)
	fmt.Printf("  baseFactor: %.4f,\n", coeffs[0])
	fmt.Printf("  cjkRatioFactor: %.4f,\n", coeffs[1])
	fmt.Printf("  punctRatioFactor: %.4f,\n", coeffs[2])
	fmt.Printf("  digitRatioFactor: %.4f,\n", coeffs[3])
	if len(coeffs) > 4 {
		fmt.Printf("  cjkSqFactor: %.4f,\n", coeffs[4])
		fmt.Printf("  punctSqFactor: %.4f,\n", coeffs[5])
		fmt.Printf("  digitSqFactor: %.4f,\n", coeffs[6])
		fmt.Printf("  cjkPunctFactor: %.4f,\n", coeffs[7])
	}
}

func calculateMAPE(rows []featureRow, coeffsMap map[int][]float64) float64 {
	var totalAbsPct float64
	count := 0
	for _, row := range rows {
		coeffs := coeffsMap[row.category]
		if len(coeffs) == 0 {
			coeffs = coeffsMap[CatGeneral]
		}
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

func classify(stats tokenXStats, cfg searchConfig) int {
	total := float64(stats.TotalRunes)
	if total == 0 {
		return CatGeneral
	}

	// Safety: Short text is unstable for statistical classification.
	// Force General for very short texts to avoid misclassification (e.g. "Dense").
	if total < 50 {
		return CatGeneral
	}

	// Rule 1: Capital
	// If significant portion of content is uppercase
	// Note: TotalRunes includes everything (CJK, Punct, Digit, Letters).
	if float64(stats.UpperRunes)/total > cfg.capitalThreshold {
		return CatCapital
	}

	// Rule 2: Dense (Low whitespace)
	// In estimateTokenXWithStats, we increment stats.SpaceRunes when we see space,
	// BUT spaces are NOT included in segment processing (estimateTokenXSegment returns 0 for space segments).
	// So TotalRunes usually does NOT include spaces.
	// We need to be careful with the ratio denominator.
	// Let's look at space density relative to visible characters.
	if total > 0 {
		spaceRatio := float64(stats.SpaceRunes) / total
		// Normal text usually has ~0.15-0.2 spaces per char.
		// Minified code or hex dumps have very few.
		if spaceRatio < cfg.denseThreshold {
			// Check for Hex
			if float64(stats.HexRunes)/total > cfg.hexThreshold {
				return CatHex
			}
			// Check for Alnum (Low punctuation)
			// Minified JSON/JS has high punctuation.
			// Random alnum strings have low punctuation.
			if float64(stats.PunctRunes)/total < cfg.alnumPunctThreshold {
				return CatAlnum
			}

			return CatDense
		}
	}

	return CatGeneral
}

func makeFeatureRow(name string, text string, enc *tiktoken.Tiktoken, cfg searchConfig) featureRow {
	actual := float64(len(enc.Encode(text, nil, nil)))
	return makeFeatureRowWithActual(name, text, actual, cfg)
}

func makeFeatureRowWithActual(name string, text string, actual float64, cfg searchConfig) featureRow {
	baseTokens, stats := estimateTokenXWithStats(text, cfg)
	features := buildFeatures(baseTokens, stats)
	cat := classify(stats, cfg)
	return featureRow{
		name:     name,
		actual:   actual,
		base:     float64(baseTokens),
		feat:     features,
		category: cat,
	}
}

func evaluate(rows []featureRow, coeffsMap map[int][]float64) {
	var totalAbsPct float64
	for _, row := range rows {
		coeffs := coeffsMap[row.category]
		if len(coeffs) == 0 {
			coeffs = coeffsMap[CatGeneral]
		}
		pred := predict(coeffs, row.feat)
		pct := 0.0
		if row.actual > 0 {
			pct = math.Abs(pred-row.actual) / row.actual * 100
		}
		totalAbsPct += pct
		catName := "General"
		if row.category == CatCapital {
			catName = "Capital"
		} else if row.category == CatDense {
			catName = "Dense"
		} else if row.category == CatHex {
			catName = "Hex"
		} else if row.category == CatAlnum {
			catName = "Alnum"
		}
		fmt.Printf("%s [%s]\tactual=%.0f\tpred=%.0f\tape=%.2f%%\n", row.name, catName, row.actual, pred, pct)
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
		// Quadratic terms
		base * cjkRatio * cjkRatio,
		base * punctRatio * punctRatio,
		base * digitRatio * digitRatio,
		// Interaction terms
		base * cjkRatio * punctRatio,
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

func fitSimple(rows []featureRow) ([]float64, error) {
	// Fit only y = a * base
	var sumXY, sumXX float64
	for _, row := range rows {
		sumXY += row.base * row.actual
		sumXX += row.base * row.base
	}
	if sumXX == 0 {
		return nil, fmt.Errorf("singular")
	}
	a := sumXY / sumXX
	// Return 4 coeffs, others 0
	return []float64{a, 0, 0, 0}, nil
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
		if unicode.IsSpace(r) {
			stats.SpaceRunes++
		}
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
		if unicode.IsUpper(r) {
			stats.UpperRunes++
		}
		if isHexRune(r) {
			stats.HexRunes++
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

func isHexRune(r rune) bool {
	if r >= '0' && r <= '9' {
		return true
	}
	if r >= 'a' && r <= 'f' {
		return true
	}
	if r >= 'A' && r <= 'F' {
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
