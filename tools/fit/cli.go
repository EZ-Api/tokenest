package main

import (
	"flag"
	"fmt"
	"strconv"
	"strings"
)

type cliOptions struct {
	Loss LossConfig

	RidgeLambda float64
	OutZRConfig string

	JSONLPath       string
	JSONLTextPath   string
	JSONLTokensPath string
	ValPct          float64
	SplitSalt       string

	LenBounds     []int
	BucketCap     int
	BucketWeights []float64
	MaxSamples    int

	NoGrid bool
	Select string

	FixedConfig searchConfig
}

func parseCLI() (cliOptions, error) {
	var (
		lossName      = flag.String("loss", string(lossMSE), "loss: mse|rel_mse|huber|huber_rel|asym_huber_rel")
		huberDelta    = flag.Float64("huber-delta", 0.20, "Huber delta; for *_rel this is relative residual threshold")
		irlsIters     = flag.Int("irls-iters", 5, "IRLS iterations for Huber-family losses")
		minActual     = flag.Float64("min-actual", 1.0, "Min actual tokens used in relative losses")
		asymAlpha     = flag.Float64("asym-alpha", 2.0, "Underestimation penalty multiplier for asym_huber_rel")
		ridgeLambda   = flag.Float64("ridge-lambda", 0.0, "Ridge regularization lambda (0 disables)")
		outZRConfig   = flag.String("out-zr-config", "", "Write ZR config JSON to path")
		selectMetric  = flag.String("select", "train_mape", "selection metric in grid mode: train_mape|val_mape")
		noGrid        = flag.Bool("no-grid", false, "Skip hyperparameter grid search and use fixed thresholds")
		jsonlPath     = flag.String("jsonl", "", "JSONL dataset path (one JSON object per line)")
		jsonlText     = flag.String("jsonl-text", "", "Dot path to extracted text field (required for -jsonl)")
		jsonlTokens   = flag.String("jsonl-tokens", "", "Dot path to actual token field (optional; empty -> compute with tiktoken)")
		valPct        = flag.Float64("val-pct", 0.20, "Validation split percent for -jsonl (0..1)")
		splitSalt     = flag.String("split-salt", "tokenest", "Salt used for deterministic hash split in -jsonl mode")
		lenBuckets    = flag.String("len-buckets", "32,64,128,256,512,1024,2048,4096,8192", "Comma-separated length bucket upper-bounds")
		bucketCap     = flag.Int("bucket-cap", 0, "Max samples per length bucket (0 disables; applied per Iterate pass)")
		bucketWeights = flag.String("bucket-weights", "", "Optional comma-separated bucket weights (len = buckets+1)")
		maxSamples    = flag.Int("max-samples", 0, "Max samples to read from -jsonl (0 unlimited)")
	)

	// Threshold overrides (used in -no-grid or -jsonl mode)
	charsPerToken := flag.Float64("chars-per-token", 3.0, "TokenX alnum chars-per-token")
	shortThreshold := flag.Int("short-threshold", 6, "TokenX short segment threshold")
	capitalThreshold := flag.Float64("capital-threshold", 0.30, "Capital category threshold")
	denseThreshold := flag.Float64("dense-threshold", 0.01, "Dense category whitespace threshold")
	hexThreshold := flag.Float64("hex-threshold", 0.90, "Hex category threshold")
	alnumPunctThreshold := flag.Float64("alnum-punct-threshold", 0.03, "Alnum category punctuation threshold")

	flag.Parse()

	loss := LossConfig{
		Kind:       lossKind(*lossName),
		HuberDelta: *huberDelta,
		IRLSIters:  *irlsIters,
		MinActual:  *minActual,
		AsymAlpha:  *asymAlpha,
	}
	if err := validateLoss(loss); err != nil {
		return cliOptions{}, err
	}
	if *ridgeLambda < 0 {
		return cliOptions{}, fmt.Errorf("-ridge-lambda must be >= 0")
	}
	if *valPct < 0 || *valPct > 1 {
		return cliOptions{}, fmt.Errorf("-val-pct must be in [0,1]")
	}
	if *bucketCap < 0 {
		return cliOptions{}, fmt.Errorf("-bucket-cap must be >= 0")
	}
	if *maxSamples < 0 {
		return cliOptions{}, fmt.Errorf("-max-samples must be >= 0")
	}

	bounds, err := parseIntCSV(*lenBuckets)
	if err != nil {
		return cliOptions{}, fmt.Errorf("invalid -len-buckets: %w", err)
	}

	var bw []float64
	if strings.TrimSpace(*bucketWeights) != "" {
		bw, err = parseFloatCSV(*bucketWeights)
		if err != nil {
			return cliOptions{}, fmt.Errorf("invalid -bucket-weights: %w", err)
		}
		expected := len(bounds) + 1
		if len(bw) != expected {
			return cliOptions{}, fmt.Errorf("-bucket-weights length must be %d (buckets+1)", expected)
		}
	}

	sel := strings.TrimSpace(*selectMetric)
	if sel == "" {
		sel = "train_mape"
	}
	switch sel {
	case "train_mape", "val_mape":
	default:
		return cliOptions{}, fmt.Errorf("invalid -select %q (use train_mape|val_mape)", sel)
	}

	return cliOptions{
		Loss:        loss,
		RidgeLambda: *ridgeLambda,
		OutZRConfig: strings.TrimSpace(*outZRConfig),
		JSONLPath:   strings.TrimSpace(*jsonlPath),

		JSONLTextPath:   strings.TrimSpace(*jsonlText),
		JSONLTokensPath: strings.TrimSpace(*jsonlTokens),
		ValPct:          *valPct,
		SplitSalt:       *splitSalt,

		LenBounds:     bounds,
		BucketCap:     *bucketCap,
		BucketWeights: bw,
		MaxSamples:    *maxSamples,

		NoGrid: *noGrid,
		Select: sel,

		FixedConfig: searchConfig{
			charsPerToken:       *charsPerToken,
			shortThreshold:      *shortThreshold,
			capitalThreshold:    *capitalThreshold,
			denseThreshold:      *denseThreshold,
			hexThreshold:        *hexThreshold,
			alnumPunctThreshold: *alnumPunctThreshold,
		},
	}, nil
}

func validateLoss(loss LossConfig) error {
	switch loss.Kind {
	case lossMSE, lossRelMSE, lossHuber, lossHuberRel, lossAsymHuberRel:
	default:
		return fmt.Errorf("unknown loss %q", loss.Kind)
	}
	if loss.MinActual <= 0 {
		return fmt.Errorf("-min-actual must be > 0")
	}
	if loss.UsesIRLS() {
		if loss.IRLSIters <= 0 {
			return fmt.Errorf("-irls-iters must be > 0")
		}
		if loss.HuberDelta <= 0 {
			return fmt.Errorf("-huber-delta must be > 0")
		}
	}
	if loss.Kind == lossHuber || loss.Kind == lossHuberRel || loss.Kind == lossAsymHuberRel {
		if loss.HuberDelta <= 0 {
			return fmt.Errorf("-huber-delta must be > 0")
		}
	}
	if loss.Kind == lossAsymHuberRel && loss.AsymAlpha < 1 {
		return fmt.Errorf("-asym-alpha must be >= 1")
	}
	return nil
}

func parseIntCSV(s string) ([]int, error) {
	s = strings.TrimSpace(s)
	if s == "" {
		return nil, nil
	}
	parts := strings.Split(s, ",")
	out := make([]int, 0, len(parts))
	prev := -1
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		v, err := strconv.Atoi(p)
		if err != nil {
			return nil, err
		}
		if v <= 0 {
			return nil, fmt.Errorf("bucket bound must be > 0, got %d", v)
		}
		if v <= prev {
			return nil, fmt.Errorf("bucket bounds must be strictly increasing")
		}
		prev = v
		out = append(out, v)
	}
	return out, nil
}

func parseFloatCSV(s string) ([]float64, error) {
	s = strings.TrimSpace(s)
	if s == "" {
		return nil, nil
	}
	parts := strings.Split(s, ",")
	out := make([]float64, 0, len(parts))
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		v, err := strconv.ParseFloat(p, 64)
		if err != nil {
			return nil, err
		}
		out = append(out, v)
	}
	return out, nil
}
