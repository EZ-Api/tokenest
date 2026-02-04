package main

import (
	"math"
	"sort"
)

// p2Quantile implements the P² streaming quantile estimator.
// It estimates the p-quantile without storing all observations.
//
// Reference: Jain & Chlamtac (1985), also described on Wikipedia.
type p2Quantile struct {
	p     float64
	count int

	// Marker heights.
	q [5]float64
	// Marker positions.
	n [5]int
	// Desired marker positions.
	np [5]float64
	// Desired position increments.
	dn [5]float64

	boot []float64
}

func newP2Quantile(p float64) *p2Quantile {
	return &p2Quantile{
		p:    p,
		boot: make([]float64, 0, 5),
	}
}

func (e *p2Quantile) Add(x float64) {
	e.count++
	if len(e.boot) < 5 {
		e.boot = append(e.boot, x)
		if len(e.boot) == 5 {
			sort.Float64s(e.boot)
			for i := 0; i < 5; i++ {
				e.q[i] = e.boot[i]
				e.n[i] = i + 1
			}
			p := e.p
			e.np[0] = 1
			e.np[1] = 1 + 2*p
			e.np[2] = 1 + 4*p
			e.np[3] = 3 + 2*p
			e.np[4] = 5
			e.dn[0] = 0
			e.dn[1] = p / 2
			e.dn[2] = p
			e.dn[3] = (1 + p) / 2
			e.dn[4] = 1
		}
		return
	}

	k := 0
	switch {
	case x < e.q[0]:
		e.q[0] = x
		k = 0
	case x < e.q[1]:
		k = 0
	case x < e.q[2]:
		k = 1
	case x < e.q[3]:
		k = 2
	case x <= e.q[4]:
		k = 3
	default:
		e.q[4] = x
		k = 3
	}

	for i := k + 1; i < 5; i++ {
		e.n[i]++
	}
	for i := 0; i < 5; i++ {
		e.np[i] += e.dn[i]
	}

	for i := 1; i <= 3; i++ {
		d := e.np[i] - float64(e.n[i])
		if (d >= 1 && e.n[i+1]-e.n[i] > 1) || (d <= -1 && e.n[i-1]-e.n[i] < -1) {
			ds := 1
			if d < 0 {
				ds = -1
			}

			qHat := e.parabolic(i, ds)
			if e.q[i-1] < qHat && qHat < e.q[i+1] && !math.IsNaN(qHat) && !math.IsInf(qHat, 0) {
				e.q[i] = qHat
			} else {
				e.q[i] = e.linear(i, ds)
			}
			e.n[i] += ds
		}
	}
}

func (e *p2Quantile) Value() (float64, bool) {
	if len(e.boot) == 0 {
		return 0, false
	}
	if len(e.boot) < 5 {
		boot := append([]float64(nil), e.boot...)
		sort.Float64s(boot)
		idx := int(math.Round(float64(len(boot)-1) * e.p))
		if idx < 0 {
			idx = 0
		}
		if idx >= len(boot) {
			idx = len(boot) - 1
		}
		return boot[idx], true
	}
	return e.q[2], true
}

func (e *p2Quantile) parabolic(i int, d int) float64 {
	ni := float64(e.n[i])
	ni1 := float64(e.n[i-1])
	ni2 := float64(e.n[i+1])

	qi := e.q[i]
	q1 := e.q[i-1]
	q2 := e.q[i+1]

	dn := float64(d)
	return qi + dn/(ni2-ni1)*((ni-ni1+dn)*(q2-qi)/(ni2-ni)+(ni2-ni-dn)*(qi-q1)/(ni-ni1))
}

func (e *p2Quantile) linear(i int, d int) float64 {
	if d > 0 {
		return e.q[i] + (e.q[i+1]-e.q[i])/float64(e.n[i+1]-e.n[i])
	}
	return e.q[i] - (e.q[i]-e.q[i-1])/float64(e.n[i]-e.n[i-1])
}
