package strategy

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

func EstimateZR(text string) int {
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
	segmentType := zrSegmentTypeNone
	first := true

	for idx, r := range text {
		if unicode.IsSpace(r) {
			stats.SpaceRunes++
		}
		currentType := zrSegmentTypeForRune(r)
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

type zrSegmentType int

const (
	zrSegmentTypeNone zrSegmentType = iota
	zrSegmentTypeWhitespace
	zrSegmentTypePunctuation
	zrSegmentTypeOther
)

func zrSegmentTypeForRune(r rune) zrSegmentType {
	if unicode.IsSpace(r) {
		return zrSegmentTypeWhitespace
	}
	if isTokenXPunct(r) {
		return zrSegmentTypePunctuation
	}
	return zrSegmentTypeOther
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
	case r >= 0x4E00 && r <= 0x9FFF:
		return true
	case r >= 0x3400 && r <= 0x4DBF:
		return true
	case r >= 0x3000 && r <= 0x303F:
		return true
	case r >= 0xFF00 && r <= 0xFFEF:
		return true
	case r >= 0x30A0 && r <= 0x30FF:
		return true
	case r >= 0x2E80 && r <= 0x2EFF:
		return true
	case r >= 0x31C0 && r <= 0x31EF:
		return true
	case r >= 0x3200 && r <= 0x32FF:
		return true
	case r >= 0x3300 && r <= 0x33FF:
		return true
	case r >= 0xAC00 && r <= 0xD7AF:
		return true
	case r >= 0x1100 && r <= 0x11FF:
		return true
	case r >= 0x3130 && r <= 0x318F:
		return true
	case r >= 0xA960 && r <= 0xA97F:
		return true
	case r >= 0xD7B0 && r <= 0xD7FF:
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
	if r >= 0x00C0 && r <= 0x00FF {
		return true
	}
	return false
}

func getLanguageSpecificCharsPerToken(segment string) float64 {
	for _, cfg := range defaultLanguageConfigs {
		if cfg.matches(segment) {
			return cfg.avgCharsPerToken
		}
	}
	return 0
}

type languageConfig struct {
	avgCharsPerToken float64
	set              map[rune]struct{}
}

func (c languageConfig) matches(segment string) bool {
	for _, r := range segment {
		if _, ok := c.set[r]; ok {
			return true
		}
	}
	return false
}

var defaultLanguageConfigs = []languageConfig{
	{
		avgCharsPerToken: 3,
		set: map[rune]struct{}{
			'\u00E4': {},
			'\u00F6': {},
			'\u00FC': {},
			'\u00DF': {},
			'\u1E9E': {},
		},
	},
	{
		avgCharsPerToken: 3,
		set: map[rune]struct{}{
			'\u00E9': {},
			'\u00E8': {},
			'\u00EA': {},
			'\u00EB': {},
			'\u00E0': {},
			'\u00E2': {},
			'\u00EE': {},
			'\u00EF': {},
			'\u00F4': {},
			'\u00FB': {},
			'\u00F9': {},
			'\u00FC': {},
			'\u00FF': {},
			'\u00E7': {},
			'\u0153': {},
			'\u00E6': {},
			'\u00E1': {},
			'\u00ED': {},
			'\u00F3': {},
			'\u00FA': {},
			'\u00F1': {},
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
			'\u00F3': {},
			'\u015B': {},
			'\u017A': {},
			'\u017C': {},
			'\u011B': {},
			'\u0161': {},
			'\u010D': {},
			'\u0159': {},
			'\u017E': {},
			'\u00FD': {},
			'\u016F': {},
			'\u00FA': {},
			'\u010F': {},
			'\u0165': {},
			'\u0148': {},
		},
	},
}
