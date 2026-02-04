package main

import (
	"errors"
	"fmt"
	"math"
)

const featureCount = 8

type mat8 [featureCount][featureCount]float64
type vec8 [featureCount]float64

type categoryCoeffs struct {
	General []float64
	Capital []float64
	Dense   []float64
	Hex     []float64
	Alnum   []float64
}

type fitCoefficients struct {
	byCat map[int][]float64
}

func (c fitCoefficients) coeffsForCategory(cat int) []float64 {
	if c.byCat == nil {
		return nil
	}
	if coeffs := c.byCat[cat]; len(coeffs) > 0 {
		return coeffs
	}
	return c.byCat[CatGeneral]
}

type groupAcc struct {
	xtx   mat8
	xty   vec8
	sumXY float64
	sumXX float64
	count int
}

func (g *groupAcc) add(row fitRow, w float64) {
	g.count++
	base := row.feat[0]
	g.sumXY += w * base * row.actual
	g.sumXX += w * base * base

	for i := 0; i < featureCount; i++ {
		g.xty[i] += w * row.feat[i] * row.actual
		for j := 0; j < featureCount; j++ {
			g.xtx[i][j] += w * row.feat[i] * row.feat[j]
		}
	}
}

func solveGroup(g groupAcc, ridgeLambda float64) (vec8, error) {
	if g.count == 0 {
		return vec8{}, errors.New("empty group")
	}

	if ridgeLambda > 0 {
		for i := 0; i < featureCount; i++ {
			g.xtx[i][i] += ridgeLambda
		}
	}

	beta, err := solveLinearSystem8(g.xtx, g.xty)
	if err == nil && allFinite(beta) {
		return beta, nil
	}

	if g.sumXX == 0 {
		if err == nil {
			err = errors.New("singular")
		}
		return vec8{}, err
	}
	a := g.sumXY / g.sumXX
	return vec8{a, 0, 0, 0, 0, 0, 0, 0}, nil
}

func solveLinearSystem8(a mat8, b vec8) (vec8, error) {
	const n = featureCount
	for i := 0; i < n; i++ {
		maxRow := i
		maxVal := math.Abs(a[i][i])
		for r := i + 1; r < n; r++ {
			if v := math.Abs(a[r][i]); v > maxVal {
				maxVal = v
				maxRow = r
			}
		}
		if maxVal == 0 {
			return vec8{}, fmt.Errorf("singular matrix (col %d)", i)
		}

		if maxRow != i {
			a[i], a[maxRow] = a[maxRow], a[i]
			b[i], b[maxRow] = b[maxRow], b[i]
		}

		pivot := a[i][i]
		for j := i; j < n; j++ {
			a[i][j] /= pivot
		}
		b[i] /= pivot

		for r := 0; r < n; r++ {
			if r == i {
				continue
			}
			factor := a[r][i]
			if factor == 0 {
				continue
			}
			for j := i; j < n; j++ {
				a[r][j] -= factor * a[i][j]
			}
			b[r] -= factor * b[i]
		}
	}

	return b, nil
}

func allFinite(v vec8) bool {
	for _, x := range v {
		if math.IsNaN(x) || math.IsInf(x, 0) {
			return false
		}
	}
	return true
}

func dot(beta vec8, feat [8]float64) float64 {
	sum := 0.0
	for i := 0; i < featureCount; i++ {
		sum += beta[i] * feat[i]
	}
	return sum
}

func vec8ToSlice(v vec8) []float64 {
	out := make([]float64, featureCount)
	for i := 0; i < featureCount; i++ {
		out[i] = v[i]
	}
	return out
}

type fitResult struct {
	Coeffs map[int][]float64
	Counts map[int]int
}

func fitByCategory(source RowSource, loss LossConfig, ridgeLambda float64, bucketWeights []float64) (fitResult, error) {
	counts := map[int]int{}
	total := 0
	if err := source.Iterate(func(row fitRow) error {
		counts[row.category]++
		total++
		return nil
	}); err != nil {
		return fitResult{}, err
	}
	if total == 0 {
		return fitResult{}, errors.New("empty dataset")
	}

	generalUsesAll := counts[CatGeneral] == 0

	enabled := map[int]bool{
		CatCapital: counts[CatCapital] >= 2,
		CatDense:   counts[CatDense] >= 2,
		CatHex:     counts[CatHex] >= 2,
		CatAlnum:   counts[CatAlnum] >= 2,
	}

	weightsForBucket := func(bucket int) float64 {
		if bucket < 0 || bucket >= len(bucketWeights) || len(bucketWeights) == 0 {
			return 1
		}
		w := bucketWeights[bucket]
		if w <= 0 || math.IsNaN(w) || math.IsInf(w, 0) {
			return 1
		}
		return w
	}

	initLoss := baseLossForInit(loss)
	betaGeneral, betaCap, betaDense, betaHex, betaAlnum, err := solveOnceByCategory(source, initLoss, ridgeLambda, weightsForBucket, generalUsesAll, enabled)
	if err != nil {
		return fitResult{}, err
	}

	if loss.UsesIRLS() {
		iters := loss.IRLSIters
		if iters <= 0 {
			iters = 5
		}
		for i := 0; i < iters; i++ {
			betaGeneral, betaCap, betaDense, betaHex, betaAlnum, err = solveOnceByCategoryIRLS(
				source,
				loss,
				ridgeLambda,
				weightsForBucket,
				generalUsesAll,
				enabled,
				betaGeneral,
				betaCap,
				betaDense,
				betaHex,
				betaAlnum,
			)
			if err != nil {
				return fitResult{}, err
			}
		}
	}

	coeffs := map[int][]float64{
		CatGeneral: vec8ToSlice(betaGeneral),
	}

	// Apply fallback rules consistent with the legacy fit tool.
	if enabled[CatCapital] {
		coeffs[CatCapital] = vec8ToSlice(betaCap)
	}
	if enabled[CatDense] {
		coeffs[CatDense] = vec8ToSlice(betaDense)
	}
	if enabled[CatHex] {
		coeffs[CatHex] = vec8ToSlice(betaHex)
	}
	if enabled[CatAlnum] {
		coeffs[CatAlnum] = vec8ToSlice(betaAlnum)
	}

	for _, cat := range []int{CatCapital, CatDense, CatHex, CatAlnum} {
		if len(coeffs[cat]) > 0 {
			continue
		}
		switch cat {
		case CatAlnum:
			if len(coeffs[CatCapital]) > 0 {
				coeffs[cat] = coeffs[CatCapital]
			} else {
				coeffs[cat] = coeffs[CatGeneral]
			}
		default:
			coeffs[cat] = coeffs[CatGeneral]
		}
	}

	return fitResult{Coeffs: coeffs, Counts: counts}, nil
}

func solveOnceByCategory(
	source RowSource,
	loss LossConfig,
	ridgeLambda float64,
	bucketWeight func(int) float64,
	generalUsesAll bool,
	enabled map[int]bool,
) (general vec8, cap vec8, dense vec8, hex vec8, alnum vec8, _ error) {
	var genAcc, capAcc, denseAcc, hexAcc, alnumAcc groupAcc

	if err := source.Iterate(func(row fitRow) error {
		w := bucketWeight(row.bucket) * sampleWeight(loss, row.actual, 0)
		if generalUsesAll || row.category == CatGeneral {
			genAcc.add(row, w)
		}
		switch row.category {
		case CatCapital:
			if enabled[CatCapital] {
				capAcc.add(row, w)
			}
		case CatDense:
			if enabled[CatDense] {
				denseAcc.add(row, w)
			}
		case CatHex:
			if enabled[CatHex] {
				hexAcc.add(row, w)
			}
		case CatAlnum:
			if enabled[CatAlnum] {
				alnumAcc.add(row, w)
			}
		}
		return nil
	}); err != nil {
		return vec8{}, vec8{}, vec8{}, vec8{}, vec8{}, err
	}

	var err error
	general, err = solveGroup(genAcc, ridgeLambda)
	if err != nil {
		return vec8{}, vec8{}, vec8{}, vec8{}, vec8{}, err
	}

	if enabled[CatCapital] {
		cap, _ = solveGroup(capAcc, ridgeLambda)
	}
	if enabled[CatDense] {
		dense, _ = solveGroup(denseAcc, ridgeLambda)
	}
	if enabled[CatHex] {
		hex, _ = solveGroup(hexAcc, ridgeLambda)
	}
	if enabled[CatAlnum] {
		alnum, _ = solveGroup(alnumAcc, ridgeLambda)
	}

	return general, cap, dense, hex, alnum, nil
}

func solveOnceByCategoryIRLS(
	source RowSource,
	loss LossConfig,
	ridgeLambda float64,
	bucketWeight func(int) float64,
	generalUsesAll bool,
	enabled map[int]bool,
	betaGeneral vec8,
	betaCap vec8,
	betaDense vec8,
	betaHex vec8,
	betaAlnum vec8,
) (general vec8, cap vec8, dense vec8, hex vec8, alnum vec8, _ error) {
	var genAcc, capAcc, denseAcc, hexAcc, alnumAcc groupAcc

	if err := source.Iterate(func(row fitRow) error {
		if generalUsesAll || row.category == CatGeneral {
			pred := dot(betaGeneral, row.feat)
			w := bucketWeight(row.bucket) * sampleWeight(loss, row.actual, pred-row.actual)
			genAcc.add(row, w)
		}

		switch row.category {
		case CatCapital:
			if enabled[CatCapital] {
				pred := dot(betaCap, row.feat)
				w := bucketWeight(row.bucket) * sampleWeight(loss, row.actual, pred-row.actual)
				capAcc.add(row, w)
			}
		case CatDense:
			if enabled[CatDense] {
				pred := dot(betaDense, row.feat)
				w := bucketWeight(row.bucket) * sampleWeight(loss, row.actual, pred-row.actual)
				denseAcc.add(row, w)
			}
		case CatHex:
			if enabled[CatHex] {
				pred := dot(betaHex, row.feat)
				w := bucketWeight(row.bucket) * sampleWeight(loss, row.actual, pred-row.actual)
				hexAcc.add(row, w)
			}
		case CatAlnum:
			if enabled[CatAlnum] {
				pred := dot(betaAlnum, row.feat)
				w := bucketWeight(row.bucket) * sampleWeight(loss, row.actual, pred-row.actual)
				alnumAcc.add(row, w)
			}
		}
		return nil
	}); err != nil {
		return vec8{}, vec8{}, vec8{}, vec8{}, vec8{}, err
	}

	var err error
	general, err = solveGroup(genAcc, ridgeLambda)
	if err != nil {
		return vec8{}, vec8{}, vec8{}, vec8{}, vec8{}, err
	}

	cap = betaCap
	dense = betaDense
	hex = betaHex
	alnum = betaAlnum

	if enabled[CatCapital] {
		if v, err := solveGroup(capAcc, ridgeLambda); err == nil {
			cap = v
		}
	}
	if enabled[CatDense] {
		if v, err := solveGroup(denseAcc, ridgeLambda); err == nil {
			dense = v
		}
	}
	if enabled[CatHex] {
		if v, err := solveGroup(hexAcc, ridgeLambda); err == nil {
			hex = v
		}
	}
	if enabled[CatAlnum] {
		if v, err := solveGroup(alnumAcc, ridgeLambda); err == nil {
			alnum = v
		}
	}

	return general, cap, dense, hex, alnum, nil
}
