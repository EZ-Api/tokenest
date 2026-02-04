package main

type fitRow struct {
	name     string
	actual   float64
	feat     [8]float64
	category int
	bucket   int
}

type RowSource interface {
	Iterate(func(fitRow) error) error
}

type sliceSource struct {
	rows []fitRow
}

func (s sliceSource) Iterate(fn func(fitRow) error) error {
	for _, row := range s.rows {
		if err := fn(row); err != nil {
			return err
		}
	}
	return nil
}
