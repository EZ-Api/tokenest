package tokenest

import "strings"

type weights struct {
	word       float64
	number     float64
	cjk        float64
	symbol     float64
	mathSymbol float64
	urlDelim   float64
	atSign     float64
	emoji      float64
	newline    float64
	space      float64
}

const (
	categoryWord       = "word"
	categoryNumber     = "number"
	categoryCJK        = "cjk"
	categorySymbol     = "symbol"
	categoryMathSymbol = "math_symbol"
	categoryURLDelim   = "url_delim"
	categoryAtSign     = "at_sign"
	categoryEmoji      = "emoji"
	categoryNewline    = "newline"
	categorySpace      = "space"
)

var breakdownOrder = []string{
	categoryWord,
	categoryNumber,
	categoryCJK,
	categorySymbol,
	categoryMathSymbol,
	categoryURLDelim,
	categoryAtSign,
	categoryEmoji,
	categoryNewline,
	categorySpace,
}

func weightsForProfile(profile Profile) weights {
	switch profile {
	case ProfileGemini:
		return weights{
			word:       1.15,
			number:     2.8,
			cjk:        0.68,
			symbol:     0.38,
			mathSymbol: 1.05,
			urlDelim:   1.2,
			atSign:     2.5,
			emoji:      1.08,
			newline:    1.15,
			space:      0.2,
		}
	case ProfileClaude:
		return weights{
			word:       1.13,
			number:     1.63,
			cjk:        1.21,
			symbol:     0.4,
			mathSymbol: 4.52,
			urlDelim:   1.26,
			atSign:     2.82,
			emoji:      2.6,
			newline:    0.89,
			space:      0.39,
		}
	default:
		return weights{
			word:       1.02,
			number:     1.55,
			cjk:        0.85,
			symbol:     0.4,
			mathSymbol: 2.68,
			urlDelim:   1.0,
			atSign:     2.0,
			emoji:      2.12,
			newline:    0.5,
			space:      0.42,
		}
	}
}

func weightForCategory(w weights, category string) float64 {
	switch category {
	case categoryWord:
		return w.word
	case categoryNumber:
		return w.number
	case categoryCJK:
		return w.cjk
	case categoryMathSymbol:
		return w.mathSymbol
	case categoryURLDelim:
		return w.urlDelim
	case categoryAtSign:
		return w.atSign
	case categoryEmoji:
		return w.emoji
	case categoryNewline:
		return w.newline
	case categorySpace:
		return w.space
	case categorySymbol:
		fallthrough
	default:
		return w.symbol
	}
}

func resolveProfile(opts Options) Profile {
	if opts.Profile != ProfileAuto {
		return opts.Profile
	}

	providerType := strings.ToLower(strings.TrimSpace(opts.ProviderType))
	switch {
	case providerType == "anthropic" || strings.Contains(providerType, "claude"):
		return ProfileClaude
	case providerType == "gemini" || providerType == "google" || strings.Contains(providerType, "gemini"):
		return ProfileGemini
	case providerType == "openai" || strings.Contains(providerType, "openai"):
		return ProfileOpenAI
	}

	model := strings.ToLower(strings.TrimSpace(opts.Model))
	switch {
	case strings.Contains(model, "claude"):
		return ProfileClaude
	case strings.Contains(model, "gemini"):
		return ProfileGemini
	default:
		return ProfileOpenAI
	}
}
