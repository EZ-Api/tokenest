package strategy

import (
	"math"
	"strings"
	"testing"
)

func TestEstimateZRReturnsZeroOnEmpty(t *testing.T) {
	if got := EstimateZR(""); got != 0 {
		t.Fatalf("expected 0 for empty input, got %d", got)
	}
}

func TestEstimateZRSimpleLatin(t *testing.T) {
	if got := EstimateZR("hi"); got != 1 {
		t.Fatalf("expected 1 token for 'hi', got %d", got)
	}
}

func TestEstimateZRCapitalCategory(t *testing.T) {
	text := strings.Repeat("A", 60)
	baseTokens := int(math.Ceil(float64(len(text)) / zrConfigDefault.charsPerToken))
	expected := int(math.Ceil(float64(baseTokens) * zrCoefficientsByCategory[zrCategoryCapital][0]))
	if got := EstimateZR(text); got != expected {
		t.Fatalf("expected %d tokens for capital input, got %d", expected, got)
	}
}
