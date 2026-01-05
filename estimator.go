package tokenest

// Estimator defines the token estimation interface for optional wrapping (e.g., caching).
type Estimator interface {
	EstimateBytes(data []byte, opts Options) Result
	EstimateText(text string, opts Options) Result
	EstimateInput(text string, images ImageCounts, messageCount int, opts Options) Result
	EstimateOutput(text string, opts Options) Result
}

type defaultEstimator struct{}

// DefaultEstimator returns the default estimator backed by the package-level functions.
func DefaultEstimator() Estimator {
	return defaultEstimator{}
}

func (defaultEstimator) EstimateBytes(data []byte, opts Options) Result {
	return EstimateBytes(data, opts)
}

func (defaultEstimator) EstimateText(text string, opts Options) Result {
	return EstimateText(text, opts)
}

func (defaultEstimator) EstimateInput(text string, images ImageCounts, messageCount int, opts Options) Result {
	return EstimateInput(text, images, messageCount, opts)
}

func (defaultEstimator) EstimateOutput(text string, opts Options) Result {
	return EstimateOutput(text, opts)
}
