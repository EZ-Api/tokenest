package main

import (
	"math"
	"unicode"
	"unicode/utf8"
)

const (
	tokenXDefaultCharsPerToken = 6.0
	tokenXShortTokenThreshold  = 3
)

type tokenXLanguageConfig struct {
	avgCharsPerToken float64
	set              map[rune]struct{}
}

var tokenXLanguageConfigs = []tokenXLanguageConfig{
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

func estimateTokenX(text string) int {
	if text == "" {
		return 0
	}

	segments := splitTokenXSegments(text)
	total := 0
	for _, segment := range segments {
		total += estimateTokenXSegment(segment)
	}
	return total
}

func splitTokenXSegments(text string) []string {
	if text == "" {
		return nil
	}

	var segments []string
	start := 0
	segmentType := tokenXSegmentTypeNone
	first := true

	for i, r := range text {
		currentType := tokenXSegmentTypeForRune(r)
		if first {
			first = false
			segmentType = currentType
			continue
		}

		if currentType != segmentType {
			if start < i {
				segments = append(segments, text[start:i])
			}
			start = i
			segmentType = currentType
		}
	}

	if start < len(text) {
		segments = append(segments, text[start:])
	}

	return segments
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

func estimateTokenXSegment(segment string) int {
	if segment == "" {
		return 0
	}

	if isTokenXWhitespace(segment) {
		return 0
	}

	if containsTokenXCJK(segment) {
		return utf8.RuneCountInString(segment)
	}

	if isTokenXNumeric(segment) {
		return 1
	}

	runeCount := utf8.RuneCountInString(segment)
	if runeCount <= tokenXShortTokenThreshold {
		return 1
	}

	if containsTokenXPunct(segment) {
		if runeCount > 1 {
			return int(math.Ceil(float64(runeCount) / 2.0))
		}
		return 1
	}

	if isTokenXAlphanumeric(segment) {
		avg := tokenXCharsPerToken(segment)
		if avg <= 0 {
			avg = tokenXDefaultCharsPerToken
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

func containsTokenXCJK(segment string) bool {
	for _, r := range segment {
		if isTokenXCJKRune(r) {
			return true
		}
	}
	return false
}

func isTokenXCJKRune(r rune) bool {
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

func isTokenXNumeric(segment string) bool {
	if segment == "" {
		return false
	}

	sawDigit := false
	prevSeparator := false
	for _, r := range segment {
		if r >= '0' && r <= '9' {
			sawDigit = true
			prevSeparator = false
			continue
		}
		if r == '.' || r == ',' {
			if !sawDigit || prevSeparator {
				return false
			}
			prevSeparator = true
			continue
		}
		return false
	}
	return sawDigit && !prevSeparator
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

func isTokenXAlphanumeric(segment string) bool {
	for _, r := range segment {
		if isTokenXAlphaNumRune(r) {
			continue
		}
		return false
	}
	return segment != ""
}

func isTokenXAlphaNumRune(r rune) bool {
	switch {
	case r >= 'a' && r <= 'z':
		return true
	case r >= 'A' && r <= 'Z':
		return true
	case r >= '0' && r <= '9':
		return true
	case r >= 0x00c0 && r <= 0x00d6:
		return true
	case r >= 0x00d8 && r <= 0x00f6:
		return true
	case r >= 0x00f8 && r <= 0x00ff:
		return true
	default:
		return false
	}
}

func tokenXCharsPerToken(segment string) float64 {
	for _, cfg := range tokenXLanguageConfigs {
		if cfg.matches(segment) {
			return cfg.avgCharsPerToken
		}
	}
	return 0
}

func (cfg tokenXLanguageConfig) matches(segment string) bool {
	for _, r := range segment {
		if _, ok := cfg.set[unicode.ToLower(r)]; ok {
			return true
		}
	}
	return false
}

type newAPIProvider string

const (
	newAPIProviderOpenAI newAPIProvider = "openai"
	newAPIProviderGemini newAPIProvider = "gemini"
	newAPIProviderClaude newAPIProvider = "claude"
)

type newAPIMultipliers struct {
	Word       float64
	Number     float64
	CJK        float64
	Symbol     float64
	MathSymbol float64
	URLDelim   float64
	AtSign     float64
	Emoji      float64
	Newline    float64
	Space      float64
	BasePad    int
}

var newAPIMultipliersMap = map[newAPIProvider]newAPIMultipliers{
	newAPIProviderGemini: {
		Word: 1.15, Number: 2.8, CJK: 0.68, Symbol: 0.38, MathSymbol: 1.05, URLDelim: 1.2, AtSign: 2.5, Emoji: 1.08, Newline: 1.15, Space: 0.2, BasePad: 0,
	},
	newAPIProviderClaude: {
		Word: 1.13, Number: 1.63, CJK: 1.21, Symbol: 0.4, MathSymbol: 4.52, URLDelim: 1.26, AtSign: 2.82, Emoji: 2.6, Newline: 0.89, Space: 0.39, BasePad: 0,
	},
	newAPIProviderOpenAI: {
		Word: 1.02, Number: 1.55, CJK: 0.85, Symbol: 0.4, MathSymbol: 2.68, URLDelim: 1.0, AtSign: 2.0, Emoji: 2.12, Newline: 0.5, Space: 0.42, BasePad: 0,
	},
}

func estimateNewAPI(provider newAPIProvider, text string) int {
	if text == "" {
		return 0
	}

	m, ok := newAPIMultipliersMap[provider]
	if !ok {
		m = newAPIMultipliersMap[newAPIProviderOpenAI]
	}

	var count float64
	type wordType int
	const (
		wordTypeNone wordType = iota
		wordTypeLatin
		wordTypeNumber
	)
	currentWordType := wordTypeNone

	for _, r := range text {
		if unicode.IsSpace(r) {
			currentWordType = wordTypeNone
			if r == '\n' || r == '\t' {
				count += m.Newline
			} else {
				count += m.Space
			}
			continue
		}

		if isNewAPICJK(r) {
			currentWordType = wordTypeNone
			count += m.CJK
			continue
		}

		if isNewAPIEmoji(r) {
			currentWordType = wordTypeNone
			count += m.Emoji
			continue
		}

		if isNewAPILatinOrNumber(r) {
			isNum := unicode.IsNumber(r)
			newType := wordTypeLatin
			if isNum {
				newType = wordTypeNumber
			}
			if currentWordType == wordTypeNone || currentWordType != newType {
				if newType == wordTypeNumber {
					count += m.Number
				} else {
					count += m.Word
				}
				currentWordType = newType
			}
			continue
		}

		currentWordType = wordTypeNone
		switch {
		case isNewAPIMathSymbol(r):
			count += m.MathSymbol
		case r == '@':
			count += m.AtSign
		case isNewAPIURLDelim(r):
			count += m.URLDelim
		default:
			count += m.Symbol
		}
	}

	return int(math.Ceil(count)) + m.BasePad
}

func isNewAPICJK(r rune) bool {
	return unicode.Is(unicode.Han, r) ||
		(r >= 0x3040 && r <= 0x30ff) ||
		(r >= 0xac00 && r <= 0xd7a3)
}

func isNewAPILatinOrNumber(r rune) bool {
	return unicode.IsLetter(r) || unicode.IsNumber(r)
}

func isNewAPIEmoji(r rune) bool {
	switch {
	case r >= 0x1f300 && r <= 0x1f9ff:
		return true
	case r >= 0x2600 && r <= 0x26ff:
		return true
	case r >= 0x2700 && r <= 0x27bf:
		return true
	case r >= 0x1f600 && r <= 0x1f64f:
		return true
	case r >= 0x1f900 && r <= 0x1f9ff:
		return true
	case r >= 0x1fa00 && r <= 0x1faff:
		return true
	default:
		return false
	}
}

const newAPIMathSymbols = "\u2211\u222b\u2202\u221a\u221e\u2264\u2265\u2260\u2248\u00b1\u00d7\u00f7\u2208\u2209\u220b\u220c\u2282\u2283\u2286\u2287\u222a\u2229\u2227\u2228\u00ac\u2200\u2203\u2204\u2205\u2206\u2207\u221d\u221f\u2220\u2221\u2222\u00b0\u2032\u2033\u2034\u207a\u207b\u207c\u207d\u207e\u207f\u2080\u2081\u2082\u2083\u2084\u2085\u2086\u2087\u2088\u2089\u208a\u208b\u208c\u208d\u208e\u00b2\u00b3\u00b9\u2074\u2075\u2076\u2077\u2078\u2079\u2070"

var newAPIMathSymbolSet = func() map[rune]struct{} {
	set := make(map[rune]struct{}, len(newAPIMathSymbols))
	for _, r := range newAPIMathSymbols {
		set[r] = struct{}{}
	}
	return set
}()

func isNewAPIMathSymbol(r rune) bool {
	if _, ok := newAPIMathSymbolSet[r]; ok {
		return true
	}
	if r >= 0x2200 && r <= 0x22ff {
		return true
	}
	if r >= 0x2a00 && r <= 0x2aff {
		return true
	}
	if r >= 0x1d400 && r <= 0x1d7ff {
		return true
	}
	return false
}

func isNewAPIURLDelim(r rune) bool {
	switch r {
	case '/', ':', '?', '&', '=', ';', '#', '%':
		return true
	default:
		return false
	}
}
