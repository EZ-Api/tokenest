package tokenest

type zrCategory int

const (
	zrCategoryGeneral zrCategory = iota
	zrCategoryCapital
	zrCategoryDense
	zrCategoryHex
	zrCategoryAlnum
)

type zrConfig struct {
	charsPerToken       float64
	shortThreshold      int
	capitalThreshold    float64
	denseThreshold      float64
	hexThreshold        float64
	alnumPunctThreshold float64
}

var zrConfigDefault = zrConfig{
	charsPerToken:       3.0,
	shortThreshold:      6,
	capitalThreshold:    0.30,
	denseThreshold:      0.01,
	hexThreshold:        0.90,
	alnumPunctThreshold: 0.03,
}

var zrCoefficientsByCategory = map[zrCategory][]float64{
	zrCategoryGeneral: {0.9315, 0.6002, -1.1969, -0.6224, -0.4560, 1.7567, 3.1898, -4.6306},
	zrCategoryCapital: {2.0163, 0, 0, 0, 0, 0, 0, 0},
	zrCategoryDense:   {0.9315, 0.6002, -1.1969, -0.6224, -0.4560, 1.7567, 3.1898, -4.6306},
	zrCategoryHex:     {0.9315, 0.6002, -1.1969, -0.6224, -0.4560, 1.7567, 3.1898, -4.6306},
	zrCategoryAlnum:   {2.0163, 0, 0, 0, 0, 0, 0, 0},
}
