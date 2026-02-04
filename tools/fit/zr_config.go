package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"time"
)

type zrThresholdsJSON struct {
	CharsPerToken       float64 `json:"chars_per_token"`
	ShortThreshold      int     `json:"short_threshold"`
	CapitalThreshold    float64 `json:"capital_threshold"`
	DenseThreshold      float64 `json:"dense_threshold"`
	HexThreshold        float64 `json:"hex_threshold"`
	AlnumPunctThreshold float64 `json:"alnum_punct_threshold"`
}

type zrCoefficientsJSON struct {
	General []float64 `json:"general"`
	Capital []float64 `json:"capital"`
	Dense   []float64 `json:"dense"`
	Hex     []float64 `json:"hex"`
	Alnum   []float64 `json:"alnum"`
}

type zrFitMetadataJSON struct {
	CreatedAt string `json:"created_at"`

	Loss       string  `json:"loss,omitempty"`
	HuberDelta float64 `json:"huber_delta,omitempty"`
	IRLSIters  int     `json:"irls_iters,omitempty"`
	Ridge      float64 `json:"ridge_lambda,omitempty"`
	AsymAlpha  float64 `json:"asym_alpha,omitempty"`

	Dataset   string  `json:"dataset,omitempty"`
	ValPct    float64 `json:"val_pct,omitempty"`
	BucketCap int     `json:"bucket_cap,omitempty"`
	LenBounds []int   `json:"len_bounds,omitempty"`

	Train  *Metrics `json:"train_metrics,omitempty"`
	Val    *Metrics `json:"val_metrics,omitempty"`
	Anchor *Metrics `json:"anchor_metrics,omitempty"`
}

type zrConfigFileJSON struct {
	Thresholds   zrThresholdsJSON   `json:"thresholds"`
	Coefficients zrCoefficientsJSON `json:"coefficients"`
	Metadata     *zrFitMetadataJSON `json:"metadata,omitempty"`
}

func writeZRConfigFile(path string, cfg searchConfig, coeffsMap map[int][]float64, meta *zrFitMetadataJSON) error {
	coeffs := zrCoefficientsJSON{
		General: coeffs8(coeffsMap[CatGeneral]),
		Capital: coeffs8(coeffsMap[CatCapital]),
		Dense:   coeffs8(coeffsMap[CatDense]),
		Hex:     coeffs8(coeffsMap[CatHex]),
		Alnum:   coeffs8(coeffsMap[CatAlnum]),
	}
	doc := zrConfigFileJSON{
		Thresholds: zrThresholdsJSON{
			CharsPerToken:       cfg.charsPerToken,
			ShortThreshold:      cfg.shortThreshold,
			CapitalThreshold:    cfg.capitalThreshold,
			DenseThreshold:      cfg.denseThreshold,
			HexThreshold:        cfg.hexThreshold,
			AlnumPunctThreshold: cfg.alnumPunctThreshold,
		},
		Coefficients: coeffs,
		Metadata:     meta,
	}
	if doc.Metadata != nil && doc.Metadata.CreatedAt == "" {
		doc.Metadata.CreatedAt = time.Now().UTC().Format(time.RFC3339)
	}

	data, err := json.MarshalIndent(doc, "", "  ")
	if err != nil {
		return err
	}
	data = append(data, '\n')

	dir := filepath.Dir(path)
	if dir != "." && dir != "" {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return err
		}
	}
	return os.WriteFile(path, data, 0o644)
}

func coeffs8(in []float64) []float64 {
	out := make([]float64, featureCount)
	copy(out, in)
	return out
}
