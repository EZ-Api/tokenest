package tokenest

import (
	"strings"
	"testing"
)

func TestEstimateUltraFast(t *testing.T) {
	data := []byte("abcd")
	res := EstimateBytes(data, Options{Strategy: StrategyUltraFast})
	if res.Tokens != 1 {
		t.Fatalf("expected 1 token, got %d", res.Tokens)
	}
}

func TestEstimateFastEnglish(t *testing.T) {
	text := "hello world"
	res := EstimateText(text, Options{Strategy: StrategyFast})
	if res.Tokens != 3 {
		t.Fatalf("expected 3 tokens, got %d", res.Tokens)
	}
}

func TestEstimateFastCJK(t *testing.T) {
	text := "\u4F60\u597D\u4E16\u754C" // "你好世界"
	res := EstimateText(text, Options{Strategy: StrategyFast})
	if res.Tokens != 5 {
		t.Fatalf("expected 5 tokens, got %d", res.Tokens)
	}
}

func TestEstimateInputAddsOverheadAndImages(t *testing.T) {
	text := "hello"
	images := ImageCounts{LowDetail: 1}
	res := EstimateInput(text, images, 2, Options{Strategy: StrategyUltraFast})
	want := 2 + ImageTokensLow + BaseOverhead + 2*PerMessageOverhead
	if res.Tokens != want {
		t.Fatalf("expected %d tokens, got %d", want, res.Tokens)
	}
}

func TestResolveProfileProviderType(t *testing.T) {
	res := EstimateText("hi", Options{Strategy: StrategyWeighted, ProviderType: "anthropic"})
	if res.Profile != ProfileClaude {
		t.Fatalf("expected ProfileClaude, got %v", res.Profile)
	}
}

func TestResolveProfileFallbackOpenAI(t *testing.T) {
	res := EstimateText("hi", Options{Strategy: StrategyWeighted, Model: "qwen-2.5"})
	if res.Profile != ProfileOpenAI {
		t.Fatalf("expected ProfileOpenAI fallback, got %v", res.Profile)
	}
}

func TestWeightedExplainBreakdown(t *testing.T) {
	res := EstimateText("123", Options{Strategy: StrategyWeighted, Profile: ProfileOpenAI, Explain: true})
	if res.Tokens == 0 {
		t.Fatalf("expected non-zero tokens")
	}
	found := false
	for _, item := range res.Breakdown {
		if item.Category == categoryNumber {
			found = true
			if item.BaseUnits != 1 {
				t.Fatalf("expected number base units 1, got %v", item.BaseUnits)
			}
			if item.Weight != weightsForProfile(ProfileOpenAI).number {
				t.Fatalf("unexpected number weight %v", item.Weight)
			}
		}
	}
	if !found {
		t.Fatalf("expected breakdown to include number category")
	}
}

type countEstimator struct {
	calls int
}

func (c *countEstimator) EstimateBytes(data []byte, opts Options) Result {
	c.calls++
	return EstimateBytes(data, opts)
}

func (c *countEstimator) EstimateText(text string, opts Options) Result {
	c.calls++
	return EstimateText(text, opts)
}

func (c *countEstimator) EstimateInput(text string, images ImageCounts, messageCount int, opts Options) Result {
	c.calls++
	return EstimateInput(text, images, messageCount, opts)
}

func (c *countEstimator) EstimateOutput(text string, opts Options) Result {
	c.calls++
	return EstimateOutput(text, opts)
}

func TestWithCacheHit(t *testing.T) {
	inner := &countEstimator{}
	cached := WithCache(inner, 4)
	text := strings.Repeat("a", defaultCacheMinTextBytes+64)
	opts := Options{Strategy: StrategyFast}

	cached.EstimateText(text, opts)
	cached.EstimateText(text, opts)

	if inner.calls != 1 {
		t.Fatalf("expected 1 inner call, got %d", inner.calls)
	}
}

func TestWithCacheBypassShortText(t *testing.T) {
	inner := &countEstimator{}
	cached := WithCache(inner, 4)
	text := "short text"
	opts := Options{Strategy: StrategyFast}

	cached.EstimateText(text, opts)
	cached.EstimateText(text, opts)

	if inner.calls != 2 {
		t.Fatalf("expected 2 inner calls, got %d", inner.calls)
	}
}

func TestWithCacheProfileKeying(t *testing.T) {
	inner := &countEstimator{}
	cached := WithCache(inner, 4)
	text := strings.Repeat("a", defaultCacheMinTextBytes+64)

	cached.EstimateText(text, Options{Strategy: StrategyWeighted, Profile: ProfileOpenAI})
	cached.EstimateText(text, Options{Strategy: StrategyWeighted, Profile: ProfileClaude})

	if inner.calls != 2 {
		t.Fatalf("expected 2 inner calls for different profiles, got %d", inner.calls)
	}
}

func TestAutoStrategyDefaults(t *testing.T) {
	bytesRes := EstimateBytes([]byte("hello"), Options{Strategy: StrategyAuto})
	if bytesRes.Strategy != StrategyUltraFast {
		t.Fatalf("expected StrategyUltraFast, got %v", bytesRes.Strategy)
	}

	textRes := EstimateText("hello", Options{Strategy: StrategyAuto})
	if textRes.Strategy != StrategyFast {
		t.Fatalf("expected StrategyFast, got %v", textRes.Strategy)
	}
}
