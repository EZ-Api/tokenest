package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"hash/fnv"
	"os"
	"strconv"
	"strings"

	"github.com/pkoukk/tiktoken-go"
)

type splitKind int

const (
	splitAny splitKind = iota
	splitTrain
	splitVal
)

type jsonlSource struct {
	path       string
	textPath   string
	tokensPath string
	enc        *tiktoken.Tiktoken

	cfg       searchConfig
	wantSplit splitKind
	valPct    float64
	splitSalt string

	lenBounds  []int
	bucketCap  int
	maxSamples int
}

func (s jsonlSource) Iterate(fn func(fitRow) error) error {
	f, err := os.Open(s.path)
	if err != nil {
		return err
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	// API logs can have very large lines; raise scanner limits.
	scanner.Buffer(make([]byte, 64*1024), 16*1024*1024)

	numBuckets := len(s.lenBounds) + 1
	var bucketCounts []int
	if s.bucketCap > 0 {
		bucketCounts = make([]int, numBuckets)
	}

	seen := 0
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}

		var obj any
		if err := json.Unmarshal([]byte(line), &obj); err != nil {
			continue
		}

		rawText, ok := extractJSONPath(obj, s.textPath)
		if !ok {
			continue
		}
		text, ok := rawText.(string)
		if !ok || text == "" {
			continue
		}

		isVal := s.isVal(text)
		if s.wantSplit == splitTrain && isVal {
			continue
		}
		if s.wantSplit == splitVal && !isVal {
			continue
		}

		actual, ok := s.extractActualTokens(obj, text)
		if !ok || actual <= 0 {
			continue
		}

		baseTokens, stats := estimateTokenXWithStats(text, s.cfg)
		if baseTokens <= 0 {
			continue
		}

		category := classify(stats, s.cfg)
		features := buildFeatures(baseTokens, stats)
		bucket := lengthBucket(int(actual), s.lenBounds)
		if bucket < 0 {
			bucket = 0
		}
		if bucket >= numBuckets {
			bucket = numBuckets - 1
		}

		if s.bucketCap > 0 {
			if bucketCounts[bucket] >= s.bucketCap {
				continue
			}
			bucketCounts[bucket]++
		}

		row := fitRow{
			actual:   actual,
			feat:     features,
			category: category,
			bucket:   bucket,
		}
		if err := fn(row); err != nil {
			return err
		}

		seen++
		if s.maxSamples > 0 && seen >= s.maxSamples {
			break
		}
	}
	if err := scanner.Err(); err != nil {
		return err
	}
	return nil
}

func (s jsonlSource) isVal(text string) bool {
	if s.wantSplit == splitAny || s.valPct <= 0 {
		return false
	}
	if s.valPct >= 1 {
		return true
	}
	h := hashSplit(s.splitSalt, text)
	// deterministic: compare on 10k buckets.
	p := float64(h%10_000) / 10_000.0
	return p < s.valPct
}

func (s jsonlSource) extractActualTokens(obj any, text string) (float64, bool) {
	if s.tokensPath == "" {
		if s.enc == nil {
			return 0, false
		}
		return float64(len(s.enc.Encode(text, nil, nil))), true
	}
	v, ok := extractJSONPath(obj, s.tokensPath)
	if !ok {
		return 0, false
	}

	switch t := v.(type) {
	case float64:
		return t, true
	case string:
		f, err := strconv.ParseFloat(strings.TrimSpace(t), 64)
		if err != nil {
			return 0, false
		}
		return f, true
	case json.Number:
		f, err := t.Float64()
		if err != nil {
			return 0, false
		}
		return f, true
	default:
		// Try integer-like types if user pre-parsed.
		if i, ok := asInt64(t); ok {
			return float64(i), true
		}
		return 0, false
	}
}

func extractJSONPath(obj any, path string) (any, bool) {
	if path == "" {
		return nil, false
	}
	cur := obj
	parts := strings.Split(path, ".")
	for _, part := range parts {
		if part == "" {
			continue
		}
		switch node := cur.(type) {
		case map[string]any:
			v, ok := node[part]
			if !ok {
				return nil, false
			}
			cur = v
		case []any:
			i, err := strconv.Atoi(part)
			if err != nil || i < 0 || i >= len(node) {
				return nil, false
			}
			cur = node[i]
		default:
			return nil, false
		}
	}
	return cur, true
}

func asInt64(v any) (int64, bool) {
	switch t := v.(type) {
	case int:
		return int64(t), true
	case int64:
		return t, true
	case int32:
		return int64(t), true
	case uint64:
		if t > uint64(^uint64(0)>>1) {
			return 0, false
		}
		return int64(t), true
	case uint32:
		return int64(t), true
	default:
		return 0, false
	}
}

func hashSplit(salt, text string) uint64 {
	h := fnv.New64a()
	_, _ = h.Write([]byte(salt))
	_, _ = h.Write([]byte{0})
	_, _ = h.Write([]byte(text))
	return h.Sum64()
}

func lengthBucket(length int, bounds []int) int {
	if length < 0 {
		length = 0
	}
	for i, b := range bounds {
		if length <= b {
			return i
		}
	}
	return len(bounds)
}

func validateJSONLConfig(path, textPath string) error {
	if path == "" {
		return fmt.Errorf("-jsonl is required")
	}
	if textPath == "" {
		return fmt.Errorf("-jsonl-text is required")
	}
	return nil
}
