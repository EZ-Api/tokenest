package main

import (
	"encoding/base64"
	"flag"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"sync"
	"time"
	"unicode"
	"unicode/utf8"

	"github.com/EZ-Api/tokenest"
	"github.com/pkoukk/tiktoken-go"
)

const (
	tokenXDefaultCharsPerToken = 6.0
	tokenXShortTokenThreshold  = 3
)

type tokenXLanguageConfig struct {
	avgCharsPerToken float64
	set              map[rune]struct{}
}

var tokenXLanguageConfigs = []tokenXLanguageConfig{
	{
		avgCharsPerToken: 3,
		set: map[rune]struct{}{
			'\u00e4': {},
			'\u00f6': {},
			'\u00fc': {},
			'\u00df': {},
			'\u1e9e': {},
		},
	},
	{
		avgCharsPerToken: 3,
		set: map[rune]struct{}{
			'\u00e9': {},
			'\u00e8': {},
			'\u00ea': {},
			'\u00eb': {},
			'\u00e0': {},
			'\u00e2': {},
			'\u00ee': {},
			'\u00ef': {},
			'\u00f4': {},
			'\u00fb': {},
			'\u00f9': {},
			'\u00fc': {},
			'\u00ff': {},
			'\u00e7': {},
			'\u0153': {},
			'\u00e6': {},
			'\u00e1': {},
			'\u00ed': {},
			'\u00f3': {},
			'\u00fa': {},
			'\u00f1': {},
		},
	},
	{
		avgCharsPerToken: 3.5,
		set: map[rune]struct{}{
			'\u0105': {},
			'\u0107': {},
			'\u0119': {},
			'\u0142': {},
			'\u0144': {},
			'\u00f3': {},
			'\u015b': {},
			'\u017a': {},
			'\u017c': {},
			'\u011b': {},
			'\u0161': {},
			'\u010d': {},
			'\u0159': {},
			'\u017e': {},
			'\u00fd': {},
			'\u016f': {},
			'\u00fa': {},
			'\u010f': {},
			'\u0165': {},
			'\u0148': {},
		},
	},
}

type candidate struct {
	Kind string
	Name string
	Text string
}

type scored struct {
	Name   string
	Kind   string
	Length int
	Actual int
	Est    int
	Diff   int
	Ratio  float64
	Sample string
}

func main() {
	var (
		length  = flag.Int("length", 50000, "target length for generated strings")
		samples = flag.Int("samples", 200, "number of random candidates")
		top     = flag.Int("top", 8, "top N underestimation cases to report")
		seed    = flag.Int64("seed", time.Now().UnixNano(), "random seed")
		workers = flag.Int("workers", 0, "max concurrent workers (default: auto)")
		saveDir = flag.String("save-dir", "", "directory to save worst-case samples (default: <repo>/tokenest/datasets/test)")
		saveTop = flag.Int("save-top", 5, "save top N samples for TokenX and Weighted (0 disables)")
	)
	flag.Parse()

	rng := rand.New(rand.NewSource(*seed))
	workerCount := resolveWorkers(*workers)
	repoRoot := findRepoRoot()
	resolvedSaveDir := *saveDir
	if *saveTop > 0 && resolvedSaveDir == "" {
		resolvedSaveDir = filepath.Join(repoRoot, "tokenest", "datasets", "test")
	}

	kinds := []string{
		"minified_json",
		"minified_js",
		"base64",
		"markdown_table",
		"log_data",
		"hex_stream",
		"punct_run",
		"alnum_run",
		"uuid_stream",
	}

	candidates := make([]candidate, 0, *samples+len(kinds))
	for _, kind := range kinds {
		candidates = append(candidates, candidate{
			Kind: kind,
			Name: kind + "_seed",
			Text: generate(kind, *length, rng),
		})
	}

	for i := 0; i < *samples; i++ {
		kind := kinds[rng.Intn(len(kinds))]
		candidates = append(candidates, candidate{
			Kind: kind,
			Name: fmt.Sprintf("%s_%03d", kind, i+1),
			Text: generate(kind, *length, rng),
		})
	}

	textByName := make(map[string]string, len(candidates))
	for _, c := range candidates {
		textByName[c.Name] = c.Text
	}

	var tokenxScores []scored
	var weightedScores []scored

	jobs := make(chan candidate)
	results := make(chan scorePair, workerCount)

	var wg sync.WaitGroup
	wg.Add(workerCount)
	for i := 0; i < workerCount; i++ {
		go func() {
			defer wg.Done()
			enc := mustEncoding()
			for c := range jobs {
				actual := len(enc.Encode(c.Text, nil, nil))
				tokenxEst := estimateTokenX(c.Text)
				weightedEst := estimateWeighted(c.Text)

				var res scorePair
				if actual > tokenxEst {
					res.tokenx = buildScore(c, actual, tokenxEst)
					res.tokenxOk = true
				}
				if actual > weightedEst {
					res.weighted = buildScore(c, actual, weightedEst)
					res.weightedOk = true
				}
				if res.tokenxOk || res.weightedOk {
					results <- res
				}
			}
		}()
	}

	go func() {
		wg.Wait()
		close(results)
	}()

	go func() {
		for _, c := range candidates {
			jobs <- c
		}
		close(jobs)
	}()

	for res := range results {
		if res.tokenxOk {
			tokenxScores = append(tokenxScores, res.tokenx)
		}
		if res.weightedOk {
			weightedScores = append(weightedScores, res.weighted)
		}
	}

	sort.Slice(tokenxScores, func(i, j int) bool { return tokenxScores[i].Ratio > tokenxScores[j].Ratio })
	sort.Slice(weightedScores, func(i, j int) bool { return weightedScores[i].Ratio > weightedScores[j].Ratio })

	fmt.Printf("TokenX worst underestimation (top %d)\n", min(*top, len(tokenxScores)))
	printScores(tokenxScores, *top)

	fmt.Printf("\nWeighted worst underestimation (top %d)\n", min(*top, len(weightedScores)))
	printScores(weightedScores, *top)

	fmt.Println()
	fmt.Printf("Weighted max underestimation ratio: %.2f%%\n", maxRatio(weightedScores)*100)

	if resolvedSaveDir != "" && *saveTop > 0 {
		if err := saveWorstCases(resolvedSaveDir, *saveTop, tokenxScores, weightedScores, textByName); err != nil {
			fmt.Fprintf(os.Stderr, "save error: %v\n", err)
		} else {
			fmt.Printf("Saved top %d worst cases to %s\n", *saveTop, resolvedSaveDir)
		}
	}
}

type scorePair struct {
	tokenx     scored
	tokenxOk   bool
	weighted   scored
	weightedOk bool
}

func buildScore(c candidate, actual, est int) scored {
	diff := actual - est
	ratio := float64(diff) / float64(actual)
	return scored{
		Name:   c.Name,
		Kind:   c.Kind,
		Length: len(c.Text),
		Actual: actual,
		Est:    est,
		Diff:   diff,
		Ratio:  ratio,
		Sample: preview(c.Text, 96),
	}
}

func preview(text string, max int) string {
	text = strings.ReplaceAll(text, "\n", " ")
	if len(text) <= max {
		return text
	}
	return text[:max] + "..."
}

func printScores(scores []scored, top int) {
	limit := min(top, len(scores))
	for i := 0; i < limit; i++ {
		s := scores[i]
		fmt.Printf("%d) %s len=%d actual=%d est=%d diff=%d ratio=%.2f%%\n", i+1, s.Name, s.Length, s.Actual, s.Est, s.Diff, s.Ratio*100)
		fmt.Printf("   sample: %s\n", s.Sample)
	}
	if limit == 0 {
		fmt.Println("(no underestimation cases found)")
	}
}

func maxRatio(scores []scored) float64 {
	max := 0.0
	for _, s := range scores {
		if s.Ratio > max {
			max = s.Ratio
		}
	}
	return max
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func resolveWorkers(requested int) int {
	maxWorkers := runtime.NumCPU()
	if maxWorkers < 1 {
		maxWorkers = 1
	}
	if maxWorkers > 8 {
		maxWorkers = 8
	}
	if requested <= 0 || requested > maxWorkers {
		return maxWorkers
	}
	return requested
}

func findRepoRoot() string {
	wd, err := os.Getwd()
	if err != nil {
		return "."
	}
	return filepath.Clean(filepath.Join(wd, "..", "..", ".."))
}

func saveWorstCases(dir string, top int, tokenxScores, weightedScores []scored, textByName map[string]string) error {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	if err := saveScoreList(dir, "adversary_tokenx", top, tokenxScores, textByName); err != nil {
		return err
	}
	if err := saveScoreList(dir, "adversary_weighted", top, weightedScores, textByName); err != nil {
		return err
	}
	return nil
}

func saveScoreList(dir, prefix string, top int, scores []scored, textByName map[string]string) error {
	limit := min(top, len(scores))
	for i := 0; i < limit; i++ {
		score := scores[i]
		content := textByName[score.Name]
		if content == "" {
			continue
		}
		kind := sanitizeName(score.Kind)
		name := fmt.Sprintf("%s_%02d_%s.txt", prefix, i+1, kind)
		path := filepath.Join(dir, name)
		if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
			return err
		}
	}
	return nil
}

func sanitizeName(name string) string {
	if name == "" {
		return "unknown"
	}
	var b strings.Builder
	for _, r := range name {
		switch {
		case r >= 'a' && r <= 'z':
			b.WriteRune(r)
		case r >= 'A' && r <= 'Z':
			b.WriteRune(r)
		case r >= '0' && r <= '9':
			b.WriteRune(r)
		default:
			b.WriteByte('_')
		}
	}
	return b.String()
}

func estimateWeighted(text string) int {
	res := tokenest.EstimateText(text, tokenest.Options{
		Strategy: tokenest.StrategyWeighted,
		Profile:  tokenest.ProfileOpenAI,
	})
	return res.Tokens
}

func mustEncoding() *tiktoken.Tiktoken {
	enc, err := tiktoken.GetEncoding("o200k_base")
	if err != nil {
		panic(err)
	}
	return enc
}

func generate(kind string, length int, rng *rand.Rand) string {
	if length <= 0 {
		return ""
	}

	var base string
	switch kind {
	case "minified_json":
		base = genMinifiedJSON()
	case "minified_js":
		base = genMinifiedJS()
	case "base64":
		base = genBase64(rng)
	case "markdown_table":
		base = genMarkdownTable()
	case "log_data":
		base = genLogData()
	case "hex_stream":
		base = genHexStream(rng)
	case "punct_run":
		base = genPunctRun(rng)
	case "alnum_run":
		base = genAlnumRun(rng)
	case "uuid_stream":
		base = genUUIDStream(rng)
	default:
		base = genAlnumRun(rng)
	}

	if len(base) >= length {
		return base[:length]
	}

	repeat := length/len(base) + 1
	return strings.Repeat(base, repeat)[:length]
}

func genMinifiedJSON() string {
	parts := make([]string, 0, 200)
	for i := 0; i < 200; i++ {
		parts = append(parts,
			fmt.Sprintf("{\"id\":%d,\"u\":\"user_%d\",\"ok\":%t,\"tags\":[%d,%d,%d,%d,%d],\"meta\":{\"v\":%d,\"s\":\"%s\"}}",
				i,
				i,
				i%2 == 0,
				i%10,
				(i+1)%10,
				(i+2)%10,
				(i+3)%10,
				(i+4)%10,
				i%9,
				strings.Repeat("x", 12),
			))
	}
	return "{\"items\":[" + strings.Join(parts, ",") + "],\"count\":200,\"ok\":true,\"ts\":1700000000}"
}

func genMinifiedJS() string {
	chunks := make([]string, 0, 200)
	for i := 0; i < 200; i++ {
		chunks = append(chunks,
			fmt.Sprintf("function f%d(a){return a.map(function(x){return x*%d}).join(',')}", i, i%7+1),
			fmt.Sprintf("var a%d=[%s];var b%d=f%d(a%d);", i, joinInts(i, 20), i, i, i),
		)
	}
	return strings.Join(chunks, "")
}

func genBase64(rng *rand.Rand) string {
	buf := make([]byte, 24000)
	for i := range buf {
		buf[i] = byte(rng.Intn(256))
	}
	return base64.StdEncoding.EncodeToString(buf)
}

func genMarkdownTable() string {
	header := "| id | ts | level | message | code |\n|---:|:---:|:-----:|:--------|----:|\n"
	rows := make([]string, 0, 2000)
	for i := 0; i < 2000; i++ {
		rows = append(rows, fmt.Sprintf("| %d | 2023-10-01 12:%02d:%02d | INFO | value=%d step=%d | %d |", i, i%60, (i*7)%60, i, i%10, 1000+i))
	}
	return header + strings.Join(rows, "\n")
}

func genLogData() string {
	lines := make([]string, 0, 3000)
	for i := 0; i < 3000; i++ {
		lines = append(lines, fmt.Sprintf("2023-10-01 12:%02d:%02d [WARN] req_id=%d user=%d cost_ms=%d bytes=%d", i%60, (i*13)%60, 100000+i, i%5000, i%120, 1000+i%9000))
	}
	return strings.Join(lines, "\n")
}

func genHexStream(rng *rand.Rand) string {
	buf := make([]byte, 2000)
	for i := range buf {
		buf[i] = byte(rng.Intn(256))
	}
	out := make([]byte, 0, len(buf)*2)
	for _, b := range buf {
		out = append(out, hexDigit(b>>4), hexDigit(b&0x0f))
	}
	return string(out)
}

func genPunctRun(rng *rand.Rand) string {
	punct := []rune("{}[]()<>,.;:!?@#$%^&*+-=~/\\|_`)")
	var sb strings.Builder
	for i := 0; i < 2000; i++ {
		sb.WriteRune(punct[rng.Intn(len(punct))])
	}
	return sb.String()
}

func genAlnumRun(rng *rand.Rand) string {
	chars := []rune("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
	var sb strings.Builder
	for i := 0; i < 2000; i++ {
		sb.WriteRune(chars[rng.Intn(len(chars))])
	}
	return sb.String()
}

func genUUIDStream(rng *rand.Rand) string {
	var parts []string
	for i := 0; i < 500; i++ {
		parts = append(parts, fmt.Sprintf("%08x-%04x-%04x-%04x-%012x", rng.Uint32(), rng.Uint32()&0xffff, rng.Uint32()&0xffff, rng.Uint32()&0xffff, rng.Uint64()&0xffffffffffff))
	}
	return strings.Join(parts, "")
}

func joinInts(seed, count int) string {
	values := make([]string, 0, count)
	for i := 0; i < count; i++ {
		values = append(values, fmt.Sprintf("%d", (seed+i)%100))
	}
	return strings.Join(values, ",")
}

func hexDigit(v byte) byte {
	if v < 10 {
		return '0' + v
	}
	return 'a' + (v - 10)
}

func estimateTokenX(text string) int {
	if text == "" {
		return 0
	}

	segments := splitTokenXSegments(text)
	total := 0
	for _, segment := range segments {
		total += estimateTokenXSegment(segment)
	}
	return total
}

func splitTokenXSegments(text string) []string {
	if text == "" {
		return nil
	}

	var segments []string
	start := 0
	segmentType := tokenXSegmentTypeNone
	first := true

	for i, r := range text {
		currentType := tokenXSegmentTypeForRune(r)
		if first {
			first = false
			segmentType = currentType
			continue
		}

		if currentType != segmentType {
			if start < i {
				segments = append(segments, text[start:i])
			}
			start = i
			segmentType = currentType
		}
	}

	if start < len(text) {
		segments = append(segments, text[start:])
	}

	return segments
}

type tokenXSegmentType int

const (
	tokenXSegmentTypeNone tokenXSegmentType = iota
	tokenXSegmentTypeWhitespace
	tokenXSegmentTypePunctuation
	tokenXSegmentTypeOther
)

func tokenXSegmentTypeForRune(r rune) tokenXSegmentType {
	if unicode.IsSpace(r) {
		return tokenXSegmentTypeWhitespace
	}
	if isTokenXPunct(r) {
		return tokenXSegmentTypePunctuation
	}
	return tokenXSegmentTypeOther
}

func estimateTokenXSegment(segment string) int {
	if segment == "" {
		return 0
	}

	if isTokenXWhitespace(segment) {
		return 0
	}

	if containsTokenXCJK(segment) {
		return utf8.RuneCountInString(segment)
	}

	if isTokenXNumeric(segment) {
		return 1
	}

	runeCount := utf8.RuneCountInString(segment)
	if runeCount <= tokenXShortTokenThreshold {
		return 1
	}

	if containsTokenXPunct(segment) {
		if runeCount > 1 {
			return int(math.Ceil(float64(runeCount) / 2.0))
		}
		return 1
	}

	if isTokenXAlphanumeric(segment) {
		avg := tokenXCharsPerToken(segment)
		if avg <= 0 {
			avg = tokenXDefaultCharsPerToken
		}
		return int(math.Ceil(float64(runeCount) / avg))
	}

	return runeCount
}

func isTokenXWhitespace(segment string) bool {
	for _, r := range segment {
		if !unicode.IsSpace(r) {
			return false
		}
	}
	return segment != ""
}

func containsTokenXCJK(segment string) bool {
	for _, r := range segment {
		if isTokenXCJKRune(r) {
			return true
		}
	}
	return false
}

func isTokenXCJKRune(r rune) bool {
	if r >= 0x4E00 && r <= 0x9FFF {
		return true
	}
	if r >= 0x3400 && r <= 0x4DBF {
		return true
	}
	return false
}

func isTokenXNumeric(segment string) bool {
	sawDigit := false
	prevSeparator := false
	for _, r := range segment {
		if r >= '0' && r <= '9' {
			sawDigit = true
			prevSeparator = false
			continue
		}
		if r == '.' || r == ',' {
			if prevSeparator {
				return false
			}
			prevSeparator = true
			continue
		}
		return false
	}
	return sawDigit && !prevSeparator
}

func containsTokenXPunct(segment string) bool {
	for _, r := range segment {
		if isTokenXPunct(r) {
			return true
		}
	}
	return false
}

func isTokenXPunct(r rune) bool {
	switch r {
	case '.', ',', '!', '?', ';', '(', ')', '{', '}', '[', ']', '<', '>', ':', '/', '\\', '|', '@', '#', '$', '%', '^', '&', '*', '+', '=', '`', '~', '_', '-':
		return true
	default:
		return false
	}
}

func isTokenXAlphanumeric(segment string) bool {
	for _, r := range segment {
		if isTokenXAlphaNumRune(r) {
			continue
		}
		return false
	}
	return segment != ""
}

func isTokenXAlphaNumRune(r rune) bool {
	switch {
	case r >= 'a' && r <= 'z':
		return true
	case r >= 'A' && r <= 'Z':
		return true
	case r >= '0' && r <= '9':
		return true
	case r >= 0x00C0 && r <= 0x00FF:
		return true
	default:
		return false
	}
}

func tokenXCharsPerToken(segment string) float64 {
	for _, cfg := range tokenXLanguageConfigs {
		if cfg.matches(segment) {
			return cfg.avgCharsPerToken
		}
	}
	return 0
}

func (cfg tokenXLanguageConfig) matches(segment string) bool {
	for _, r := range segment {
		if _, ok := cfg.set[r]; ok {
			return true
		}
	}
	return false
}
