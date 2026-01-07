package tokenest

import "math"

// Strategy defines the estimation algorithm to use.
type Strategy int

const (
	// StrategyAuto automatically selects the best strategy based on input type.
	// Raw bytes -> UltraFast; extracted text -> Fast.
	StrategyAuto Strategy = iota

	// StrategyUltraFast uses len(bytes)/4 for O(1) estimation.
	// Best for raw JSON bytes, coarse filtering, high QPS paths.
	StrategyUltraFast

	// StrategyFast uses head/mid/tail sampling to detect CJK and punctuation density.
	// O(min(n, 1000)) complexity, suitable for preflight estimation.
	StrategyFast

	// StrategyWeighted uses tokenx-style segmentation with lightweight profile tuning.
	// O(n) complexity, best balance of accuracy and throughput for usage fallback.
	StrategyWeighted

	// StrategyZR uses ZR tuning parameters for higher-fidelity estimation on mixed inputs.
	// O(n) complexity, opt-in alternative to Weighted.
	StrategyZR
)

func (s Strategy) String() string {
	switch s {
	case StrategyAuto:
		return "auto"
	case StrategyUltraFast:
		return "ultrafast"
	case StrategyFast:
		return "fast"
	case StrategyWeighted:
		return "weighted"
	case StrategyZR:
		return "ZR"
	default:
		return "unknown"
	}
}

// Profile defines the weight profile for weighted estimation.
type Profile int

const (
	// ProfileAuto automatically resolves profile from model name or provider type.
	ProfileAuto Profile = iota

	// ProfileOpenAI uses OpenAI-tuned weights (default fallback).
	ProfileOpenAI

	// ProfileClaude uses Claude-tuned weights.
	ProfileClaude

	// ProfileGemini uses Gemini-tuned weights.
	ProfileGemini
)

func (p Profile) String() string {
	switch p {
	case ProfileAuto:
		return "auto"
	case ProfileOpenAI:
		return "openai"
	case ProfileClaude:
		return "claude"
	case ProfileGemini:
		return "gemini"
	default:
		return "unknown"
	}
}

// Options configures the estimation behavior.
type Options struct {
	// Strategy selects the estimation algorithm. Default: StrategyAuto.
	Strategy Strategy

	// Profile selects the weight profile for weighted estimation. Default: ProfileAuto.
	Profile Profile

	// Model is used for automatic profile resolution (e.g., "claude-3-opus").
	Model string

	// ProviderType is used for automatic profile resolution (e.g., "anthropic", "google").
	ProviderType string

	// GlobalMultiplier applies a final multiplier to the result. Default: 1.0.
	GlobalMultiplier float64

	// Explain includes per-category breakdown in the result.
	Explain bool
}

// ImageCounts tracks images by detail level for accurate estimation.
type ImageCounts struct {
	LowDetail  int
	HighDetail int
	Unknown    int
}

// Total returns the total image count.
func (c ImageCounts) Total() int {
	return c.LowDetail + c.HighDetail + c.Unknown
}

// CategoryBreakdown provides per-category token details when Explain is enabled.
type CategoryBreakdown struct {
	Category  string
	BaseUnits float64
	Weight    float64
	Tokens    float64
}

// Result contains the estimation result and metadata.
type Result struct {
	// Tokens is the estimated token count.
	Tokens int

	// Strategy is the strategy that was used.
	Strategy Strategy

	// Profile is the profile that was used (for weighted estimation).
	Profile Profile

	// Breakdown provides per-category details when Explain is enabled.
	Breakdown []CategoryBreakdown
}

// Overhead constants for message formatting.
const (
	// BaseOverhead covers role tokens, separators, and JSON structure.
	BaseOverhead = 50

	// PerMessageOverhead covers per-message formatting overhead.
	PerMessageOverhead = 4

	// ImageTokensLow is the token cost for low-detail images (OpenAI: 85 tokens).
	ImageTokensLow = 85

	// ImageTokensHigh is the approximate token cost for high-detail images.
	ImageTokensHigh = 765

	// ImageTokensDefault is the default token cost when detail level is unknown.
	ImageTokensDefault = 500
)

// EstimateBytes estimates tokens from raw bytes (e.g., JSON request body).
// With StrategyAuto, this uses UltraFast estimation.
func EstimateBytes(data []byte, opts Options) Result {
	strategy := opts.Strategy
	if strategy == StrategyAuto {
		strategy = StrategyUltraFast
	}

	var tokens int
	var breakdown []CategoryBreakdown
	switch strategy {
	case StrategyUltraFast:
		tokens = estimateUltraFast(data)
	case StrategyFast:
		tokens = estimateFast(string(data))
	case StrategyWeighted:
		profile := resolveProfile(opts)
		if opts.Explain {
			breakdown = make([]CategoryBreakdown, 0)
		}
		tokens = estimateWeighted(string(data), profile, opts.Explain, &breakdown)
	case StrategyZR:
		tokens = estimateZR(string(data))
	default:
		tokens = estimateUltraFast(data)
	}

	tokens = applyMultiplier(tokens, opts.GlobalMultiplier)

	return Result{
		Tokens:    tokens,
		Strategy:  strategy,
		Profile:   resolveProfile(opts),
		Breakdown: breakdown,
	}
}

// EstimateText estimates tokens from extracted text content.
// With StrategyAuto, this uses Fast estimation.
func EstimateText(text string, opts Options) Result {
	strategy := opts.Strategy
	if strategy == StrategyAuto {
		strategy = StrategyFast
	}

	var tokens int
	var breakdown []CategoryBreakdown

	switch strategy {
	case StrategyUltraFast:
		tokens = estimateUltraFast([]byte(text))
	case StrategyFast:
		tokens = estimateFast(text)
	case StrategyWeighted:
		profile := resolveProfile(opts)
		if opts.Explain {
			breakdown = make([]CategoryBreakdown, 0)
		}
		tokens = estimateWeighted(text, profile, opts.Explain, &breakdown)
	case StrategyZR:
		tokens = estimateZR(text)
	default:
		tokens = estimateFast(text)
	}

	tokens = applyMultiplier(tokens, opts.GlobalMultiplier)

	return Result{
		Tokens:    tokens,
		Strategy:  strategy,
		Profile:   resolveProfile(opts),
		Breakdown: breakdown,
	}
}

// EstimateInput estimates input tokens including text, images, and message overhead.
func EstimateInput(text string, images ImageCounts, messageCount int, opts Options) Result {
	multiplier := opts.GlobalMultiplier
	opts.GlobalMultiplier = 1.0
	result := EstimateText(text, opts)

	imageTokens := images.LowDetail*ImageTokensLow +
		images.HighDetail*ImageTokensHigh +
		images.Unknown*ImageTokensDefault

	overhead := BaseOverhead + messageCount*PerMessageOverhead

	result.Tokens += imageTokens + overhead
	result.Tokens = applyMultiplier(result.Tokens, multiplier)

	return result
}

// EstimateOutput estimates output tokens from response text.
func EstimateOutput(text string, opts Options) Result {
	return EstimateText(text, opts)
}

func applyMultiplier(tokens int, multiplier float64) int {
	if multiplier <= 0 || multiplier == 1.0 {
		return tokens
	}
	return int(math.Ceil(float64(tokens) * multiplier))
}
