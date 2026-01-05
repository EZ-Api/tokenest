package tokenest

import (
	"container/list"
	"encoding/binary"
	"hash/maphash"
	"math"
	"sync"
)

const defaultCacheMinTextBytes = 512

var cacheSeed = maphash.MakeSeed()

type cacheEntry struct {
	key   uint64
	value Result
}

type lruCache struct {
	mu    sync.Mutex
	cap   int
	ll    *list.List
	items map[uint64]*list.Element
}

func newLRU(size int) *lruCache {
	if size <= 0 {
		return nil
	}
	return &lruCache{
		cap:   size,
		ll:    list.New(),
		items: make(map[uint64]*list.Element, size),
	}
}

func (c *lruCache) Get(key uint64) (Result, bool) {
	if c == nil {
		return Result{}, false
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if elem, ok := c.items[key]; ok {
		c.ll.MoveToFront(elem)
		return elem.Value.(cacheEntry).value, true
	}
	return Result{}, false
}

func (c *lruCache) Add(key uint64, value Result) {
	if c == nil {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()

	if elem, ok := c.items[key]; ok {
		elem.Value = cacheEntry{key: key, value: value}
		c.ll.MoveToFront(elem)
		return
	}

	elem := c.ll.PushFront(cacheEntry{key: key, value: value})
	c.items[key] = elem

	if c.ll.Len() > c.cap {
		back := c.ll.Back()
		if back != nil {
			c.ll.Remove(back)
			entry := back.Value.(cacheEntry)
			delete(c.items, entry.key)
		}
	}
}

// WithCache wraps an estimator with an LRU cache. Caching is opt-in and disabled by default.
func WithCache(inner Estimator, size int) Estimator {
	if inner == nil {
		inner = DefaultEstimator()
	}
	cache := newLRU(size)
	if cache == nil {
		return inner
	}
	return &cachedEstimator{
		inner:       inner,
		cache:       cache,
		minTextSize: defaultCacheMinTextBytes,
	}
}

type cachedEstimator struct {
	inner       Estimator
	cache       *lruCache
	minTextSize int
}

func (c *cachedEstimator) EstimateBytes(data []byte, opts Options) Result {
	if len(data) < c.minTextSize {
		return c.inner.EstimateBytes(data, opts)
	}
	key := cacheKeyBytes(data, opts)
	if val, ok := c.cache.Get(key); ok {
		return val
	}
	val := c.inner.EstimateBytes(data, opts)
	c.cache.Add(key, val)
	return val
}

func (c *cachedEstimator) EstimateText(text string, opts Options) Result {
	if len(text) < c.minTextSize {
		return c.inner.EstimateText(text, opts)
	}
	key := cacheKeyText(text, opts)
	if val, ok := c.cache.Get(key); ok {
		return val
	}
	val := c.inner.EstimateText(text, opts)
	c.cache.Add(key, val)
	return val
}

func (c *cachedEstimator) EstimateInput(text string, images ImageCounts, messageCount int, opts Options) Result {
	if len(text) < c.minTextSize {
		return c.inner.EstimateInput(text, images, messageCount, opts)
	}
	key := cacheKeyInput(text, images, messageCount, opts)
	if val, ok := c.cache.Get(key); ok {
		return val
	}
	val := c.inner.EstimateInput(text, images, messageCount, opts)
	c.cache.Add(key, val)
	return val
}

func (c *cachedEstimator) EstimateOutput(text string, opts Options) Result {
	return c.EstimateText(text, opts)
}

func cacheKeyBytes(data []byte, opts Options) uint64 {
	strategy := effectiveBytesStrategy(opts.Strategy)
	profile := resolveProfile(opts)
	return hashKey(strategy, profile, opts, data, ImageCounts{}, 0, 'b')
}

func cacheKeyText(text string, opts Options) uint64 {
	strategy := effectiveTextStrategy(opts.Strategy)
	profile := resolveProfile(opts)
	return hashKey(strategy, profile, opts, []byte(text), ImageCounts{}, 0, 't')
}

func cacheKeyInput(text string, images ImageCounts, messageCount int, opts Options) uint64 {
	strategy := effectiveTextStrategy(opts.Strategy)
	profile := resolveProfile(opts)
	return hashKey(strategy, profile, opts, []byte(text), images, messageCount, 'i')
}

func effectiveBytesStrategy(strategy Strategy) Strategy {
	if strategy == StrategyAuto {
		return StrategyUltraFast
	}
	return strategy
}

func effectiveTextStrategy(strategy Strategy) Strategy {
	if strategy == StrategyAuto {
		return StrategyFast
	}
	return strategy
}

func hashKey(strategy Strategy, profile Profile, opts Options, data []byte, images ImageCounts, messageCount int, kind byte) uint64 {
	var h maphash.Hash
	h.SetSeed(cacheSeed)

	writeUint64(&h, uint64(kind))
	writeUint64(&h, uint64(strategy))
	writeUint64(&h, uint64(profile))
	writeUint64(&h, math.Float64bits(opts.GlobalMultiplier))
	writeUint64(&h, boolToUint64(opts.Explain))
	writeUint64(&h, uint64(messageCount))
	writeUint64(&h, uint64(images.LowDetail))
	writeUint64(&h, uint64(images.HighDetail))
	writeUint64(&h, uint64(images.Unknown))
	writeUint64(&h, uint64(BaseOverhead))
	writeUint64(&h, uint64(PerMessageOverhead))
	writeUint64(&h, uint64(ImageTokensLow))
	writeUint64(&h, uint64(ImageTokensHigh))
	writeUint64(&h, uint64(ImageTokensDefault))

	h.Write(data)

	return h.Sum64()
}

func writeUint64(h *maphash.Hash, v uint64) {
	var buf [8]byte
	binary.LittleEndian.PutUint64(buf[:], v)
	h.Write(buf[:])
}

func boolToUint64(v bool) uint64 {
	if v {
		return 1
	}
	return 0
}
