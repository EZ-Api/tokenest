package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestFitByCategory_HuberDownweightsOutlier(t *testing.T) {
	rows := make([]fitRow, 0, 101)
	for i := 0; i < 100; i++ {
		rows = append(rows, fitRow{
			actual:   1,
			feat:     [8]float64{1},
			category: CatGeneral,
		})
	}
	rows = append(rows, fitRow{
		actual:   1000,
		feat:     [8]float64{1},
		category: CatGeneral,
	})

	mseRes, err := fitByCategory(sliceSource{rows: rows}, LossConfig{Kind: lossMSE, MinActual: 1}, 0, nil)
	if err != nil {
		t.Fatalf("mse fit failed: %v", err)
	}
	mseA := mseRes.Coeffs[CatGeneral][0]
	if mseA < 10 || mseA > 12 {
		t.Fatalf("unexpected mse coeff=%.4f, want around 10.9", mseA)
	}

	huberRes, err := fitByCategory(sliceSource{rows: rows}, LossConfig{
		Kind:       lossHuber,
		HuberDelta: 1.0,
		IRLSIters:  5,
		MinActual:  1,
		AsymAlpha:  1,
	}, 0, nil)
	if err != nil {
		t.Fatalf("huber fit failed: %v", err)
	}
	huberA := huberRes.Coeffs[CatGeneral][0]
	if huberA < 0.95 || huberA > 1.10 {
		t.Fatalf("unexpected huber coeff=%.4f, want near 1.0", huberA)
	}
}

func TestFitByCategory_RidgeEnablesFullSolve(t *testing.T) {
	rows := make([]fitRow, 0, 10)
	for i := 0; i < 10; i++ {
		rows = append(rows, fitRow{
			actual:   2,
			feat:     [8]float64{1, 1},
			category: CatGeneral,
		})
	}

	noRidge, err := fitByCategory(sliceSource{rows: rows}, LossConfig{Kind: lossMSE, MinActual: 1}, 0, nil)
	if err != nil {
		t.Fatalf("no-ridge fit failed: %v", err)
	}
	if got := noRidge.Coeffs[CatGeneral][1]; got != 0 {
		t.Fatalf("expected fallback simple fit (beta1=0), got %.4f", got)
	}

	withRidge, err := fitByCategory(sliceSource{rows: rows}, LossConfig{Kind: lossMSE, MinActual: 1}, 1e-3, nil)
	if err != nil {
		t.Fatalf("ridge fit failed: %v", err)
	}
	if got := withRidge.Coeffs[CatGeneral][1]; got < 0.5 {
		t.Fatalf("expected ridge to produce non-trivial beta1, got %.4f", got)
	}
}

func TestJSONLSource_ParseAndBucketCap(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "data.jsonl")
	if err := os.WriteFile(path, []byte(
		"{\"text\":\"hello\",\"tokens\":10}\n"+
			"{\"text\":\"world\",\"tokens\":10}\n",
	), 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}

	src := jsonlSource{
		path:       path,
		textPath:   "text",
		tokensPath: "tokens",
		cfg: searchConfig{
			charsPerToken:       3,
			shortThreshold:      6,
			capitalThreshold:    0.3,
			denseThreshold:      0.01,
			hexThreshold:        0.9,
			alnumPunctThreshold: 0.03,
		},
		wantSplit:  splitAny,
		lenBounds:  []int{32, 64},
		bucketCap:  1,
		maxSamples: 0,
	}

	count := 0
	if err := src.Iterate(func(row fitRow) error {
		count++
		if row.actual != 10 {
			t.Fatalf("unexpected actual=%.0f", row.actual)
		}
		return nil
	}); err != nil {
		t.Fatalf("iterate: %v", err)
	}
	if count != 1 {
		t.Fatalf("expected bucket cap to keep 1 row, got %d", count)
	}
}

func TestWriteZRConfigFile_RoundTrip(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "zr.json")

	cfg := searchConfig{
		charsPerToken:       3.0,
		shortThreshold:      6,
		capitalThreshold:    0.30,
		denseThreshold:      0.01,
		hexThreshold:        0.90,
		alnumPunctThreshold: 0.03,
	}
	coeffs := map[int][]float64{
		CatGeneral: {1, 2, 3, 4, 5, 6, 7, 8},
		CatCapital: {9, 0, 0, 0, 0, 0, 0, 0},
		CatDense:   {1, 0, 0, 0, 0, 0, 0, 0},
		CatHex:     {1, 0, 0, 0, 0, 0, 0, 0},
		CatAlnum:   {9, 0, 0, 0, 0, 0, 0, 0},
	}
	meta := &zrFitMetadataJSON{
		Loss:  "mse",
		Ridge: 0.01,
	}
	if err := writeZRConfigFile(path, cfg, coeffs, meta); err != nil {
		t.Fatalf("writeZRConfigFile: %v", err)
	}

	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	var got zrConfigFileJSON
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if got.Thresholds.ShortThreshold != 6 {
		t.Fatalf("unexpected thresholds: %+v", got.Thresholds)
	}
	if len(got.Coefficients.General) != 8 {
		t.Fatalf("expected 8 general coeffs, got %d", len(got.Coefficients.General))
	}
	if got.Metadata == nil || got.Metadata.Loss != "mse" {
		t.Fatalf("unexpected metadata: %+v", got.Metadata)
	}
	if got.Metadata.CreatedAt == "" {
		t.Fatalf("expected created_at to be set")
	}
}
