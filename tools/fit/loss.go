package main

import "math"

type lossKind string

const (
	lossMSE          lossKind = "mse"
	lossRelMSE       lossKind = "rel_mse"
	lossHuber        lossKind = "huber"
	lossHuberRel     lossKind = "huber_rel"
	lossAsymHuberRel lossKind = "asym_huber_rel"
)

type LossConfig struct {
	Kind       lossKind
	HuberDelta float64
	IRLSIters  int
	MinActual  float64
	AsymAlpha  float64
}

func (c LossConfig) UsesIRLS() bool {
	switch c.Kind {
	case lossHuber, lossHuberRel, lossAsymHuberRel:
		return true
	default:
		return false
	}
}

func (c LossConfig) IsRelative() bool {
	switch c.Kind {
	case lossRelMSE, lossHuberRel, lossAsymHuberRel:
		return true
	default:
		return false
	}
}

func baseLossForInit(loss LossConfig) LossConfig {
	switch loss.Kind {
	case lossHuber:
		loss.Kind = lossMSE
	case lossHuberRel, lossAsymHuberRel:
		loss.Kind = lossRelMSE
	}
	return loss
}

func sampleWeight(loss LossConfig, actual, residual float64) float64 {
	switch loss.Kind {
	case lossMSE:
		return 1
	case lossRelMSE:
		return relativeBaseWeight(actual, loss.MinActual)
	case lossHuber:
		return huberWeight(residual, loss.HuberDelta)
	case lossHuberRel, lossAsymHuberRel:
		denom := actual
		if denom < loss.MinActual {
			denom = loss.MinActual
		}
		if denom <= 0 {
			denom = 1
		}
		rel := residual / denom
		w := relativeBaseWeight(actual, loss.MinActual) * huberWeight(rel, loss.HuberDelta)
		if loss.Kind == lossAsymHuberRel && residual < 0 && loss.AsymAlpha > 1 {
			w *= loss.AsymAlpha
		}
		return w
	default:
		return 1
	}
}

func relativeBaseWeight(actual, minActual float64) float64 {
	denom := actual
	if denom < minActual {
		denom = minActual
	}
	if denom <= 0 {
		denom = 1
	}
	return 1 / (denom * denom)
}

func huberWeight(residual, delta float64) float64 {
	if delta <= 0 {
		return 1
	}
	abs := math.Abs(residual)
	if abs <= delta || abs == 0 {
		return 1
	}
	return delta / abs
}
