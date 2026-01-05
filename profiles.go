package tokenest

import "strings"

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
