package tokenest

import (
	"math"
	"unicode"
	"unicode/utf8"
)

const (
	weightedV2ClampMin = 0.85
	weightedV2ClampMax = 1.20
	tokenXShortTokenThreshold = 3
)

type weightedTuning struct {
	baseFactor       float64
	cjkRatioFactor   float64
	punctRatioFactor float64
	digitRatioFactor float64
	clampMin         float64
	clampMax         float64
}

func tuningForProfile(profile Profile) weightedTuning {
	switch profile {
	case ProfileClaude:
		return weightedTuning{
			baseFactor:       1.00,
			cjkRatioFactor:   0.00,
			punctRatioFactor: 0.00,
			digitRatioFactor: 0.00,
			clampMin:         weightedV2ClampMin,
			clampMax:         weightedV2ClampMax,
		}
	case ProfileGemini:
		return weightedTuning{
			baseFactor:       1.00,
			cjkRatioFactor:   0.00,
			punctRatioFactor: 0.00,
			digitRatioFactor: 0.00,
			clampMin:         weightedV2ClampMin,
			clampMax:         weightedV2ClampMax,
		}
	default:
		return weightedTuning{
			baseFactor:       1.00,
			cjkRatioFactor:   0.00,
			punctRatioFactor: 0.00,
			digitRatioFactor: 0.00,
			clampMin:         weightedV2ClampMin,
			clampMax:         weightedV2ClampMax,
		}
	}
}

const (
	weightedV2CategoryBase       = "base"
	weightedV2CategoryCJKRatio   = "ratio_cjk"
	weightedV2CategoryPunctRatio = "ratio_punct"
	weightedV2CategoryDigitRatio = "ratio_digit"
	weightedV2CategoryClamp      = "clamp"
)

var weightedV2BreakdownOrder = []string{
	weightedV2CategoryBase,
	weightedV2CategoryCJKRatio,
	weightedV2CategoryPunctRatio,
	weightedV2CategoryDigitRatio,
	weightedV2CategoryClamp,
}

type tokenXStats struct {
	TotalRunes    int
	CJKRunes      int
	PunctRunes    int
	DigitRunes    int
	Whitespace    int
	EmojiCount    int
	MathCount     int
	URLDelimCount int
	AtCount       int
}

func estimateWeighted(text string, profile Profile, explain bool, breakdown *[]CategoryBreakdown) int {
	if text == "" {
		return 0
	}

	baseTokens, stats := estimateTokenXWithStats(text)
	if baseTokens == 0 {
		return 0
	}

	tuning := tuningForProfile(profile)
	totalRunes := stats.TotalRunes
	if totalRunes == 0 {
		totalRunes = 1
	}

	cjkRatio := float64(stats.CJKRunes) / float64(totalRunes)
	punctRatio := float64(stats.PunctRunes) / float64(totalRunes)
	digitRatio := float64(stats.DigitRunes) / float64(totalRunes)

	base := float64(baseTokens)
	tokens := base*tuning.baseFactor +
		base*cjkRatio*tuning.cjkRatioFactor +
		base*punctRatio*tuning.punctRatioFactor +
		base*digitRatio*tuning.digitRatioFactor

	minTokens := base * tuning.clampMin
	maxTokens := base * tuning.clampMax
	if tokens < minTokens {
		tokens = minTokens
	}
	if tokens > maxTokens {
		tokens = maxTokens
	}

	if explain && breakdown != nil {
		items := make([]CategoryBreakdown, 0, len(weightedV2BreakdownOrder))
		appendBreakdownItem := func(category string, units float64, weight float64) {
			if units == 0 || weight == 0 {
				return
			}
			items = append(items, CategoryBreakdown{
				Category:  category,
				BaseUnits: units,
				Weight:    weight,
				Tokens:    units * weight,
			})
		}

		appendBreakdownItem(weightedV2CategoryBase, base, tuning.baseFactor)
		appendBreakdownItem(weightedV2CategoryCJKRatio, base*cjkRatio, tuning.cjkRatioFactor)
		appendBreakdownItem(weightedV2CategoryPunctRatio, base*punctRatio, tuning.punctRatioFactor)
		appendBreakdownItem(weightedV2CategoryDigitRatio, base*digitRatio, tuning.digitRatioFactor)

		sum := 0.0
		for _, item := range items {
			sum += item.Tokens
		}
		clampDelta := tokens - sum
		if clampDelta != 0 {
			items = append(items, CategoryBreakdown{
				Category:  weightedV2CategoryClamp,
				BaseUnits: clampDelta,
				Weight:    1,
				Tokens:    clampDelta,
			})
		}

		*breakdown = items
	}

	return int(math.Ceil(tokens))
}

func estimateTokenXWithStats(text string) (int, tokenXStats) {
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
			baseTokens += estimateTokenXSegment(text[segmentStart:idx], &stats)
			segmentStart = idx
			segmentType = currentType
		}
	}

	if segmentStart < len(text) {
		baseTokens += estimateTokenXSegment(text[segmentStart:], &stats)
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

func estimateTokenXSegment(segment string, stats *tokenXStats) int {
	if segment == "" {
		return 0
	}

	if isTokenXWhitespace(segment) {
		stats.Whitespace += utf8.RuneCountInString(segment)
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
		if isEmoji(r) {
			stats.EmojiCount++
		}
		if isMathSymbol(r) {
			stats.MathCount++
		}
		if isURLDelim(r) {
			stats.URLDelimCount++
		}
		if isAtSign(r) {
			stats.AtCount++
		}
	}

	if isCJKSegment(segment) {
		return runeCount
	}

	if isNumericSegment(segment) {
		return 1
	}

	if runeCount <= tokenXShortTokenThreshold {
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
			avg = defaultCharsPerToken
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
