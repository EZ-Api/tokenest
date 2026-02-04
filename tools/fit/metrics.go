package main

import "math"

type Metrics struct {
	Count     int     `json:"count"`
	MAE       float64 `json:"mae"`
	MAPE      float64 `json:"mape"`
	P50APE    float64 `json:"p50_ape"`
	P90APE    float64 `json:"p90_ape"`
	UnderRate float64 `json:"under_rate"`
}

func computeMetrics(source RowSource, coeffsMap map[int][]float64) (Metrics, error) {
	var sumAbs float64
	var sumAPE float64
	under := 0
	count := 0

	q50 := newP2Quantile(0.50)
	q90 := newP2Quantile(0.90)

	if err := source.Iterate(func(row fitRow) error {
		coeffs := coeffsMap[row.category]
		if len(coeffs) == 0 {
			coeffs = coeffsMap[CatGeneral]
		}
		pred := predict(coeffs, row.feat)
		if pred < 0 {
			pred = 0
		}

		absErr := math.Abs(pred - row.actual)
		sumAbs += absErr

		ape := 0.0
		if row.actual > 0 {
			ape = absErr / row.actual * 100
		}
		sumAPE += ape
		q50.Add(ape)
		q90.Add(ape)

		if pred < row.actual {
			under++
		}
		count++
		return nil
	}); err != nil {
		return Metrics{}, err
	}

	if count == 0 {
		return Metrics{}, nil
	}

	m := Metrics{
		Count:     count,
		MAE:       sumAbs / float64(count),
		MAPE:      sumAPE / float64(count),
		UnderRate: float64(under) / float64(count),
	}
	if v, ok := q50.Value(); ok {
		m.P50APE = v
	}
	if v, ok := q90.Value(); ok {
		m.P90APE = v
	}
	return m, nil
}
