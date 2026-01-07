// Package tokenest provides lightweight token estimation for LLM API requests.
//
// It offers four estimation strategies with different accuracy/performance trade-offs:
//   - UltraFast: O(1) byte-based estimation for raw JSON, suitable for coarse filtering
//   - Fast: O(min(n,1000)) sampling-based estimation with CJK/punctuation density detection
//   - Weighted: O(n) tokenx-based estimation with ratio tuning (CJK/punct/digit) per profile
//   - ZR: O(n) categorical tuning with ZR coefficients for mixed inputs (opt-in)
//
// The library supports automatic strategy selection and model-aware profiles for
// different providers (OpenAI, Claude, Gemini).
//
// Basic usage:
//
//	result := tokenest.EstimateText("Hello 你好", tokenest.Options{})
//	fmt.Println(result.Tokens)
//
// With explicit strategy:
//
//	result := tokenest.EstimateText(text, tokenest.Options{
//	    Strategy: tokenest.StrategyWeighted,
//	    Profile:  tokenest.ProfileClaude,
//	})
package tokenest
