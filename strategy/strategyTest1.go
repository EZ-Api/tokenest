package tokenest

import (
	"math"
	"unicode"
	"unicode/utf8"
)

type zrStats struct {
	TotalRunes int
	CJKRunes   int
	PunctRunes int
	DigitRunes int
	SpaceRunes int
	UpperRunes int
	HexRunes   int
}

func estimateZR(text string) int {
	if text == "" {
		return 0
	}

	baseTokens, stats := estimateZRTokenXWithStats(text, zrConfigDefault)
	if baseTokens == 0 {
		return 0
	}

	features := buildZRFeatures(baseTokens, stats)
	category := classifyZR(stats, zrConfigDefault)
	coeffs := zrCoefficientsByCategory[category]
	if len(coeffs) == 0 {
		coeffs = zrCoefficientsByCategory[zrCategoryGeneral]
	}

	pred := zrPredict(coeffs, features)
	if pred < 0 {
		return 0
	}
	return int(math.Ceil(pred))
}

func buildZRFeatures(baseTokens int, stats zrStats) []float64 {
	if baseTokens <= 0 {
		return []float64{0, 0, 0, 0, 0, 0, 0, 0}
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
		base * cjkRatio * cjkRatio,
		base * punctRatio * punctRatio,
		base * digitRatio * digitRatio,
		base * cjkRatio * punctRatio,
	}
}

func zrPredict(coeffs []float64, features []float64) float64 {
	limit := len(features)
	if len(coeffs) < limit {
		limit = len(coeffs)
	}

	sum := 0.0
	for i := 0; i < limit; i++ {
		sum += coeffs[i] * features[i]
	}
	return sum
}

func classifyZR(stats zrStats, cfg zrConfig) zrCategory {
	total := float64(stats.TotalRunes)
	if total == 0 {
		return zrCategoryGeneral
	}

	if total < 50 {
		return zrCategoryGeneral
	}

	if float64(stats.UpperRunes)/total > cfg.capitalThreshold {
		return zrCategoryCapital
	}

	spaceRatio := float64(stats.SpaceRunes) / total
	if spaceRatio < cfg.denseThreshold {
		if float64(stats.HexRunes)/total > cfg.hexThreshold {
			return zrCategoryHex
		}
		if float64(stats.PunctRunes)/total < cfg.alnumPunctThreshold {
			return zrCategoryAlnum
		}
		return zrCategoryDense
	}

	return zrCategoryGeneral
}

func estimateZRTokenXWithStats(text string, cfg zrConfig) (int, zrStats) {
	stats := zrStats{}
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
			baseTokens += estimateZRTokenXSegment(text[segmentStart:idx], &stats, cfg)
			segmentStart = idx
			segmentType = currentType
		}
	}

	if segmentStart < len(text) {
		baseTokens += estimateZRTokenXSegment(text[segmentStart:], &stats, cfg)
	}

	return baseTokens, stats
}

func estimateZRTokenXSegment(segment string, stats *zrStats, cfg zrConfig) int {
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
