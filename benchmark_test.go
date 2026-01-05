package tokenest

import (
	"strings"
	"testing"
)

func BenchmarkUltraFast(b *testing.B) {
	data := []byte(strings.Repeat("a", 10*1024))
	opts := Options{Strategy: StrategyUltraFast}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = EstimateBytes(data, opts)
	}
}

func BenchmarkFast(b *testing.B) {
	text := strings.Repeat("a", 10*1024)
	opts := Options{Strategy: StrategyFast}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = EstimateText(text, opts)
	}
}

func BenchmarkWeighted(b *testing.B) {
	text := strings.Repeat("a", 4*1024) + strings.Repeat("/", 512) + "\u4F60\u597D\u4E16\u754C"
	opts := Options{Strategy: StrategyWeighted, Profile: ProfileOpenAI}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = EstimateText(text, opts)
	}
}
