package tokenest

import (
	"math"
	"unicode"
	"unicode/utf8"
)

const defaultCharsPerToken = 6.0

func estimateWeighted(text string, profile Profile, explain bool, breakdown *[]CategoryBreakdown) int {
	if text == "" {
		return 0
	}

	weights := weightsForProfile(profile)
	var total float64
	var breakdownMap map[string]*CategoryBreakdown
	if explain && breakdown != nil {
		breakdownMap = make(map[string]*CategoryBreakdown)
	}

	addCategory := func(category string, units float64) {
		if units <= 0 {
			return
		}
		weight := weightForCategory(weights, category)
		tokens := units * weight
		total += tokens
		if breakdownMap == nil {
			return
		}
		entry, ok := breakdownMap[category]
		if !ok {
			entry = &CategoryBreakdown{Category: category, Weight: weight}
			breakdownMap[category] = entry
		}
		entry.BaseUnits += units
		entry.Tokens += tokens
	}

	for i := 0; i < len(text); {
		r, size := utf8.DecodeRuneInString(text[i:])
		if r == utf8.RuneError && size == 1 {
			addCategory(categorySymbol, 1)
			i++
			continue
		}

		switch {
		case unicode.IsSpace(r):
			start := i
			i += size
			for i < len(text) {
				rn, sz := utf8.DecodeRuneInString(text[i:])
				if !unicode.IsSpace(rn) {
					break
				}
				i += sz
			}
			addWhitespaceSegment(text[start:i], addCategory)
			continue
		case isEmoji(r):
			start := i
			i += size
			for i < len(text) {
				rn, sz := utf8.DecodeRuneInString(text[i:])
				if !isEmoji(rn) {
					break
				}
				i += sz
			}
			addCategory(categoryEmoji, float64(utf8.RuneCountInString(text[start:i])))
			continue
		case isPunctuation(r):
			start := i
			i += size
			for i < len(text) {
				rn, sz := utf8.DecodeRuneInString(text[i:])
				if !isPunctuation(rn) {
					break
				}
				i += sz
			}
			addPunctuationSegment(text[start:i], addCategory)
			continue
		default:
			start := i
			i += size
			for i < len(text) {
				rn, sz := utf8.DecodeRuneInString(text[i:])
				if unicode.IsSpace(rn) || isPunctuation(rn) || isEmoji(rn) {
					break
				}
				i += sz
			}
			addWordSegment(text[start:i], addCategory)
			continue
		}
	}

	tokens := int(math.Ceil(total))
	if breakdown != nil && breakdownMap != nil {
		items := make([]CategoryBreakdown, 0, len(breakdownOrder))
		for _, category := range breakdownOrder {
			entry, ok := breakdownMap[category]
			if !ok {
				continue
			}
			items = append(items, *entry)
		}
		*breakdown = items
	}

	return tokens
}

func addWhitespaceSegment(segment string, addCategory func(string, float64)) {
	spaceCount := 0
	newlineCount := 0
	for _, r := range segment {
		if r == '\n' {
			newlineCount++
			continue
		}
		if unicode.IsSpace(r) {
			spaceCount++
		}
	}
	addCategory(categoryNewline, float64(newlineCount))
	addCategory(categorySpace, float64(spaceCount))
}

func addPunctuationSegment(segment string, addCategory func(string, float64)) {
	symbolRun := 0
	flushSymbols := func() {
		if symbolRun == 0 {
			return
		}
		units := math.Ceil(float64(symbolRun) / 2.0)
		addCategory(categorySymbol, units)
		symbolRun = 0
	}

	for _, r := range segment {
		switch {
		case isAtSign(r):
			flushSymbols()
			addCategory(categoryAtSign, 1)
		case isURLDelim(r):
			flushSymbols()
			addCategory(categoryURLDelim, 1)
		case isMathSymbol(r):
			flushSymbols()
			addCategory(categoryMathSymbol, 1)
		default:
			symbolRun++
		}
	}
	flushSymbols()
}

func addWordSegment(segment string, addCategory func(string, float64)) {
	if segment == "" {
		return
	}

	if isCJKSegment(segment) {
		addCategory(categoryCJK, float64(utf8.RuneCountInString(segment)))
		return
	}

	if isNumericSegment(segment) {
		addCategory(categoryNumber, 1)
		return
	}

	runeCount := utf8.RuneCountInString(segment)
	if runeCount <= 3 {
		addCategory(categoryWord, 1)
		return
	}

	if isAlphanumericSegment(segment) {
		avg := getLanguageSpecificCharsPerToken(segment)
		if avg <= 0 {
			avg = defaultCharsPerToken
		}
		units := math.Ceil(float64(runeCount) / avg)
		addCategory(categoryWord, units)
		return
	}

	addCategory(categoryWord, float64(runeCount))
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

func isPunctuation(r rune) bool {
	switch r {
	case '.', ',', '!', '?', ';', '(', ')', '{', '}', '[', ']', '<', '>', ':', '/', '\\', '|', '@', '#', '$', '%', '^', '&', '*', '+', '=', '`', '~', '_', '-', '"', '\'':
		return true
	default:
		return false
	}
}

func isAtSign(r rune) bool {
	return r == '@'
}

func isURLDelim(r rune) bool {
	switch r {
	case ':', '/', '.', '?', '&', '=', '#', '%':
		return true
	default:
		return false
	}
}

func isMathSymbol(r rune) bool {
	switch r {
	case '+', '-', '*', '/', '=', '^', '<', '>':
		return true
	default:
		return false
	}
}

func isEmoji(r rune) bool {
	switch {
	case r >= 0x1F300 && r <= 0x1F5FF:
		return true
	case r >= 0x1F600 && r <= 0x1F64F:
		return true
	case r >= 0x1F680 && r <= 0x1F6FF:
		return true
	case r >= 0x1F700 && r <= 0x1F77F:
		return true
	case r >= 0x1F900 && r <= 0x1F9FF:
		return true
	case r >= 0x1FA00 && r <= 0x1FAFF:
		return true
	case r >= 0x2600 && r <= 0x26FF:
		return true
	case r >= 0x2700 && r <= 0x27BF:
		return true
	default:
		return false
	}
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
