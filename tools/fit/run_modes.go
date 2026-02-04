package main

import (
	"fmt"

	"github.com/pkoukk/tiktoken-go"
)

func runFixedConfigFit(enc *tiktoken.Tiktoken, opts cliOptions, trainItems, testItems []sampleData, loaded []sampleData) error {
	cfg := opts.FixedConfig

	trainRows := make([]fitRow, 0, len(trainItems))
	for _, item := range trainItems {
		trainRows = append(trainRows, makeFeatureRowWithActual(item.sample.name, item.text, item.actual, cfg))
	}

	testRows := make([]fitRow, 0, len(testItems))
	for _, item := range testItems {
		testRows = append(testRows, makeFeatureRowWithActual(item.sample.name, item.text, item.actual, cfg))
	}

	fitRes, err := fitByCategory(sliceSource{rows: trainRows}, opts.Loss, opts.RidgeLambda, nil)
	if err != nil {
		return err
	}

	fmt.Printf("\n=== FIXED CONFIG FIT ===\n")
	fmt.Printf("Loss: %s\n", opts.Loss.Kind)
	if opts.Loss.UsesIRLS() {
		fmt.Printf("HuberDelta: %.4f, IRLSIters: %d\n", opts.Loss.HuberDelta, opts.Loss.IRLSIters)
	}
	if opts.RidgeLambda > 0 {
		fmt.Printf("RidgeLambda: %.6f\n", opts.RidgeLambda)
	}

	fmt.Printf("CharsPerToken: %.2f\n", cfg.charsPerToken)
	fmt.Printf("ShortThreshold: %d\n", cfg.shortThreshold)
	fmt.Printf("CapitalThreshold: %.2f\n", cfg.capitalThreshold)
	fmt.Printf("DenseThreshold: %.4f\n", cfg.denseThreshold)
	fmt.Printf("HexThreshold: %.2f\n", cfg.hexThreshold)
	fmt.Printf("AlnumPunctThreshold: %.4f\n", cfg.alnumPunctThreshold)

	fmt.Println("\nFitted coefficients (o200k_base):")
	printCoeffs("General", fitRes.Coeffs[CatGeneral])
	printCoeffs("Capital", fitRes.Coeffs[CatCapital])
	printCoeffs("Dense", fitRes.Coeffs[CatDense])
	printCoeffs("Hex", fitRes.Coeffs[CatHex])
	printCoeffs("Alnum", fitRes.Coeffs[CatAlnum])

	fmt.Println("\n=== TRAIN SET EVALUATION (Fixed Config) ===")
	evaluate(trainRows, fitRes.Coeffs)

	fmt.Println("\n=== TEST SET EVALUATION (Fixed Config) ===")
	evaluate(testRows, fitRes.Coeffs)

	anchorRows := make([]fitRow, 0, len(loaded))
	for _, item := range loaded {
		actual := float64(len(enc.Encode(item.text, nil, nil)))
		anchorRows = append(anchorRows, makeFeatureRowWithActual(item.sample.name, item.text, actual, cfg))
	}
	anchorMetrics, _ := computeMetrics(sliceSource{rows: anchorRows}, fitRes.Coeffs)
	fmt.Printf("\n=== ANCHOR EVAL (Full Text) ===\n")
	fmt.Printf("Anchor: count=%d mae=%.2f mape=%.2f%% p50=%.2f%% p90=%.2f%% under=%.2f%%\n",
		anchorMetrics.Count,
		anchorMetrics.MAE,
		anchorMetrics.MAPE,
		anchorMetrics.P50APE,
		anchorMetrics.P90APE,
		anchorMetrics.UnderRate*100,
	)

	if opts.OutZRConfig != "" {
		trainMetrics, _ := computeMetrics(sliceSource{rows: trainRows}, fitRes.Coeffs)
		valMetrics, _ := computeMetrics(sliceSource{rows: testRows}, fitRes.Coeffs)
		meta := &zrFitMetadataJSON{
			Loss:       string(opts.Loss.Kind),
			HuberDelta: opts.Loss.HuberDelta,
			IRLSIters:  opts.Loss.IRLSIters,
			Ridge:      opts.RidgeLambda,
			AsymAlpha:  opts.Loss.AsymAlpha,
			Dataset:    "curated-fixed",
			Train:      &trainMetrics,
			Val:        &valMetrics,
			Anchor:     &anchorMetrics,
		}
		if err := writeZRConfigFile(opts.OutZRConfig, cfg, fitRes.Coeffs, meta); err != nil {
			return err
		}
		fmt.Printf("\nWrote ZR config: %s\n", opts.OutZRConfig)
	}

	return nil
}

func runJSONLFit(enc *tiktoken.Tiktoken, opts cliOptions, loaded []sampleData) error {
	if err := validateJSONLConfig(opts.JSONLPath, opts.JSONLTextPath); err != nil {
		return err
	}

	cfg := opts.FixedConfig
	numBuckets := len(opts.LenBounds) + 1

	trainSource := jsonlSource{
		path:       opts.JSONLPath,
		textPath:   opts.JSONLTextPath,
		tokensPath: opts.JSONLTokensPath,
		enc:        enc,
		cfg:        cfg,
		wantSplit:  splitTrain,
		valPct:     opts.ValPct,
		splitSalt:  opts.SplitSalt,
		lenBounds:  opts.LenBounds,
		bucketCap:  opts.BucketCap,
		maxSamples: opts.MaxSamples,
	}
	valSource := jsonlSource{
		path:       opts.JSONLPath,
		textPath:   opts.JSONLTextPath,
		tokensPath: opts.JSONLTokensPath,
		enc:        enc,
		cfg:        cfg,
		wantSplit:  splitVal,
		valPct:     opts.ValPct,
		splitSalt:  opts.SplitSalt,
		lenBounds:  opts.LenBounds,
		bucketCap:  0,
		maxSamples: opts.MaxSamples,
	}

	bucketWeights := opts.BucketWeights
	if len(bucketWeights) == 0 {
		var err error
		bucketWeights, err = autoBucketWeights(trainSource, numBuckets)
		if err != nil {
			return err
		}
	}

	fitRes, err := fitByCategory(trainSource, opts.Loss, opts.RidgeLambda, bucketWeights)
	if err != nil {
		return err
	}

	trainMetrics, _ := computeMetrics(trainSource, fitRes.Coeffs)
	valMetrics, _ := computeMetrics(valSource, fitRes.Coeffs)

	fmt.Printf("\n=== JSONL FIT ===\n")
	fmt.Printf("Path: %s\n", opts.JSONLPath)
	fmt.Printf("Loss: %s\n", opts.Loss.Kind)
	if opts.Loss.UsesIRLS() {
		fmt.Printf("HuberDelta: %.4f, IRLSIters: %d\n", opts.Loss.HuberDelta, opts.Loss.IRLSIters)
	}
	if opts.RidgeLambda > 0 {
		fmt.Printf("RidgeLambda: %.6f\n", opts.RidgeLambda)
	}
	if opts.BucketCap > 0 {
		fmt.Printf("BucketCap: %d\n", opts.BucketCap)
	}
	fmt.Printf("ValPct: %.2f\n", opts.ValPct)

	fmt.Println("\nFitted coefficients (o200k_base):")
	printCoeffs("General", fitRes.Coeffs[CatGeneral])
	printCoeffs("Capital", fitRes.Coeffs[CatCapital])
	printCoeffs("Dense", fitRes.Coeffs[CatDense])
	printCoeffs("Hex", fitRes.Coeffs[CatHex])
	printCoeffs("Alnum", fitRes.Coeffs[CatAlnum])

	fmt.Printf("\nTrain: count=%d mae=%.2f mape=%.2f%% p50=%.2f%% p90=%.2f%% under=%.2f%%\n",
		trainMetrics.Count,
		trainMetrics.MAE,
		trainMetrics.MAPE,
		trainMetrics.P50APE,
		trainMetrics.P90APE,
		trainMetrics.UnderRate*100,
	)
	fmt.Printf("Val:   count=%d mae=%.2f mape=%.2f%% p50=%.2f%% p90=%.2f%% under=%.2f%%\n",
		valMetrics.Count,
		valMetrics.MAE,
		valMetrics.MAPE,
		valMetrics.P50APE,
		valMetrics.P90APE,
		valMetrics.UnderRate*100,
	)

	anchorRows := make([]fitRow, 0, len(loaded))
	for _, item := range loaded {
		actual := float64(len(enc.Encode(item.text, nil, nil)))
		anchorRows = append(anchorRows, makeFeatureRowWithActual(item.sample.name, item.text, actual, cfg))
	}
	anchorMetrics, _ := computeMetrics(sliceSource{rows: anchorRows}, fitRes.Coeffs)
	fmt.Printf("\nAnchor: count=%d mae=%.2f mape=%.2f%% p50=%.2f%% p90=%.2f%% under=%.2f%%\n",
		anchorMetrics.Count,
		anchorMetrics.MAE,
		anchorMetrics.MAPE,
		anchorMetrics.P50APE,
		anchorMetrics.P90APE,
		anchorMetrics.UnderRate*100,
	)

	if opts.OutZRConfig != "" {
		meta := &zrFitMetadataJSON{
			Loss:       string(opts.Loss.Kind),
			HuberDelta: opts.Loss.HuberDelta,
			IRLSIters:  opts.Loss.IRLSIters,
			Ridge:      opts.RidgeLambda,
			AsymAlpha:  opts.Loss.AsymAlpha,
			Dataset:    "jsonl",
			ValPct:     opts.ValPct,
			BucketCap:  opts.BucketCap,
			LenBounds:  opts.LenBounds,
			Train:      &trainMetrics,
			Val:        &valMetrics,
			Anchor:     &anchorMetrics,
		}
		if err := writeZRConfigFile(opts.OutZRConfig, cfg, fitRes.Coeffs, meta); err != nil {
			return err
		}
		fmt.Printf("\nWrote ZR config: %s\n", opts.OutZRConfig)
	}

	return nil
}

func autoBucketWeights(source RowSource, numBuckets int) ([]float64, error) {
	counts := make([]int, numBuckets)
	total := 0
	if err := source.Iterate(func(row fitRow) error {
		if row.bucket >= 0 && row.bucket < numBuckets {
			counts[row.bucket]++
			total++
		}
		return nil
	}); err != nil {
		return nil, err
	}
	nonEmpty := 0
	for _, c := range counts {
		if c > 0 {
			nonEmpty++
		}
	}
	weights := make([]float64, numBuckets)
	if total == 0 || nonEmpty == 0 {
		for i := range weights {
			weights[i] = 1
		}
		return weights, nil
	}

	scale := float64(total) / float64(nonEmpty)
	for i, c := range counts {
		if c <= 0 {
			weights[i] = 1
			continue
		}
		weights[i] = scale / float64(c)
	}
	return weights, nil
}
