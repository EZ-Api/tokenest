package tokenest

import "math"

const (
	fastSampleTotal = 1000
	fastHeadSize    = 256
	fastMidSize     = 256
	fastTailSize    = 256
)

func estimateUltraFast(data []byte) int {
	if len(data) == 0 {
		return 0
	}
	return (len(data) + 3) / 4
}

func estimateFast(text string) int {
	if text == "" {
		return 0
	}

	sample := sampleFastText(text)
	if sample == "" {
		return 0
	}

	totalRunes := 0
	cjkCount := 0
	punctCount := 0
	for _, r := range sample {
		totalRunes++
		if isCJKFast(r) {
			cjkCount++
		}
		if isFastPunct(r) {
			punctCount++
		}
	}
	if totalRunes == 0 {
		return 0
	}

	cjkRatio := float64(cjkCount) / float64(totalRunes)
	punctRatio := float64(punctCount) / float64(totalRunes)

	divisor := 4.0 - (cjkRatio * 1.5) - (punctRatio * 1.0)
	if divisor < 2.0 {
		divisor = 2.0
	}
	if divisor > 4.0 {
		divisor = 4.0
	}

	bytesLen := float64(len(text))
	return int(math.Ceil(bytesLen / divisor))
}

func sampleFastText(text string) string {
	if len(text) <= fastSampleTotal {
		return text
	}

	head := safeSlice(text, 0, fastHeadSize)
	midStart := len(text)/2 - fastMidSize/2
	midEnd := midStart + fastMidSize
	mid := safeSlice(text, midStart, midEnd)
	tail := safeSlice(text, len(text)-fastTailSize, len(text))

	return head + mid + tail
}

func safeSlice(text string, start, end int) string {
	if start < 0 {
		start = 0
	}
	if end < 0 {
		end = 0
	}
	if end > len(text) {
		end = len(text)
	}
	if start > end {
		start = end
	}

	start = adjustLeftToRuneBoundary(text, start)
	end = adjustRightToRuneBoundary(text, end)
	if end < start {
		end = start
	}

	return text[start:end]
}

func adjustLeftToRuneBoundary(text string, idx int) int {
	for idx < len(text) && isContinuationByte(text[idx]) {
		idx++
	}
	return idx
}

func adjustRightToRuneBoundary(text string, idx int) int {
	for idx > 0 && idx <= len(text) && isContinuationByte(text[idx-1]) {
		idx--
	}
	return idx
}

func isContinuationByte(b byte) bool {
	return (b & 0xC0) == 0x80
}

func isCJKFast(r rune) bool {
	return r >= 0x4E00 && r <= 0x9FFF
}

func isFastPunct(r rune) bool {
	switch r {
	case '{', '}', '[', ']', '(', ')', '<', '>', ';', ':', ',', '.', '!', '?', '@', '#', '$', '%', '^', '&', '*', '=', '+', '-', '/', '\\', '|', '~', '`', '"', '\'', '_':
		return true
	default:
		return false
	}
}
