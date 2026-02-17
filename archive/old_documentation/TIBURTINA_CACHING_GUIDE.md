# Tiburtina Caching System

## Problem Solved

**Before:** Page 8 took 30-60 seconds to load because it fetched data from external APIs on every render.

**After:** Data is cached and reused, making Page 8 load instantly after the first fetch.

---

## How It Works

### Caching Strategy

Three separate caches with different TTLs (Time-To-Live):

| Data Type | TTL | Rationale |
|-----------|-----|-----------|
| **Macro** | 30 minutes | Economic indicators update slowly (daily/weekly) |
| **Crypto** | 5 minutes | Crypto prices change frequently |
| **News** | 10 minutes | News updates moderately fast |

### Cache Behavior

**First Access:**
1. Dashboard tries to load from cache
2. Cache is empty → fetches from API (slow)
3. Stores result in cache with timestamp
4. Returns data

**Subsequent Access (within TTL):**
1. Dashboard tries to load from cache
2. Cache has fresh data → returns immediately (fast!)
3. No API call needed

**After TTL Expires:**
1. Dashboard tries to load from cache
2. Cache is stale → fetches from API (slow)
3. Updates cache with new data
4. Returns refreshed data

### Error Handling

**If API fetch fails:**
- Returns cached data even if stale
- Better to show old data than no data!
- Displays error message alongside cached data

**If user cancels (Ctrl+C):**
- Returns cached data if available
- Allows interrupting slow API calls

---

## Using the Cache

### In Dashboard

**Page 8 automatically uses caching:**
```bash
python3 dashboard.py
# Press '8' to view Tiburtina page
# First load: ~30-60s (fetching data)
# Second load: <1s (cached data)
```

**Cache Status Display:**
```
Cache Status:
  Macro      FRESH (2m ago)
  Crypto     FRESH (30s ago)
  News       stale (12m ago)
```

### Manual Cache Control

**Force refresh (bypass cache):**
```python
from tiburtina_integration import get_tiburtina_client

client = get_tiburtina_client()

# Force fresh data (slow)
macro = client.get_macro_snapshot(use_cache=False)
crypto = client.get_crypto_overview(use_cache=False)
news = client.get_news_summary(use_cache=False)
```

**Use cache (default):**
```python
# Use cached data if available (fast)
macro = client.get_macro_snapshot()  # use_cache=True by default
crypto = client.get_crypto_overview()
news = client.get_news_summary()
```

### Check Cache Status

```python
from tiburtina_integration import get_tiburtina_client

client = get_tiburtina_client()
status = client.get_cache_status()

# Output:
# {
#   "macro": {
#     "has_data": True,
#     "is_stale": False,
#     "age_seconds": 120.5
#   },
#   "crypto": {
#     "has_data": True,
#     "is_stale": False,
#     "age_seconds": 45.2
#   },
#   "news": {
#     "has_data": True,
#     "is_stale": True,  # >10 minutes old
#     "age_seconds": 650.0
#   }
# }
```

---

## Performance Impact

### Before Caching
```
Page 8 load time: 30-60 seconds
- Macro API: 15-30s (FRED API slow)
- Crypto API: 5-10s (CoinGecko)
- News API: 10-20s (NewsAPI/RSS)
```

### After Caching
```
First load: 30-60 seconds (same as before)
Subsequent loads: <1 second (instant!)

Dashboard refresh every 5s: no lag
Page navigation: smooth
```

### Cache Hit Rates

**Expected performance:**
- Macro: >95% hit rate (30min TTL, rarely stale)
- Crypto: ~80% hit rate (5min TTL, refreshes occasionally)
- News: ~85% hit rate (10min TTL, refreshes moderately)

---

## Troubleshooting

### Cache Not Working

**Check if data is being cached:**
```bash
python3 -c "
from tiburtina_integration import get_tiburtina_client
client = get_tiburtina_client()
print(client.get_cache_status())
"
```

**Expected output:**
```
{'macro': {'has_data': False, 'is_stale': True, 'age_seconds': None}, ...}
```

**If `has_data` is always False:**
- API fetch is failing
- Check error messages in dashboard
- Verify API keys in Tiburtina's `.env`

### Stale Data Not Refreshing

**Caches refresh automatically:**
- When TTL expires
- On next page load

**Force refresh manually:**
```python
from tiburtina_integration import get_tiburtina_client
client = get_tiburtina_client()

# Bypass cache
macro = client.get_macro_snapshot(use_cache=False)
```

### Memory Usage

**Cache memory footprint:**
- Macro: ~5-10 KB
- Crypto: ~20-30 KB (10 cryptos × 3KB each)
- News: ~50-100 KB (10 articles × 5-10KB each)
- **Total: ~75-140 KB** (negligible)

**Cache lifetime:**
- Lives as long as dashboard process
- Cleared when dashboard restarts
- No disk persistence (in-memory only)

---

## Advanced Usage

### Pre-fetching Data

**Load cache before showing dashboard:**
```python
from tiburtina_integration import get_tiburtina_client

client = get_tiburtina_client()

# Pre-fetch all data (takes 30-60s once)
client.prefetch_data()

# Now dashboard loads instantly
```

**Example startup script:**
```bash
#!/bin/bash
# prefetch_dashboard.sh

echo "Pre-loading Tiburtina data..."
python3 -c "
from tiburtina_integration import get_tiburtina_client
client = get_tiburtina_client()
if client.is_available():
    client.prefetch_data()
    print('Cache pre-loaded!')
else:
    print('Tiburtina not available')
"

# Start dashboard
python3 dashboard.py
```

### Custom TTLs

**Modify cache lifetimes** (in `tiburtina_integration.py`):
```python
# In TiburtinaClient.__init__()
self._macro_cache = CachedData(ttl_seconds=1800)  # 30 min (default)
self._crypto_cache = CachedData(ttl_seconds=300)  # 5 min (default)
self._news_cache = CachedData(ttl_seconds=600)    # 10 min (default)

# Adjust as needed:
# - Increase TTL for slower refreshes (less API calls)
# - Decrease TTL for faster refreshes (more current data)
```

---

## Implementation Details

### CachedData Class

**Simple cache with TTL:**
```python
class CachedData:
    def __init__(self, ttl_seconds: int = 300):
        self.ttl_seconds = ttl_seconds
        self.data = None
        self.timestamp = None

    def is_stale(self) -> bool:
        """Check if cache is older than TTL."""
        if self.data is None or self.timestamp is None:
            return True
        age = time.time() - self.timestamp
        return age > self.ttl_seconds

    def set(self, data):
        """Store data with current timestamp."""
        self.data = data
        self.timestamp = time.time()

    def get(self):
        """Return data if fresh, None if stale."""
        if self.is_stale():
            return None
        return self.data
```

### Cache Logic Flow

```
get_macro_snapshot(use_cache=True)
    ↓
Is cache enabled?
    ├─ No → fetch from API
    └─ Yes → check cache
        ├─ Cache fresh? → return cached data (fast!)
        └─ Cache stale? → fetch from API
            ├─ Success → update cache, return data
            └─ Error → return stale cache (fallback)
```

---

## Best Practices

1. **Default to cached data** - Always use `use_cache=True` (default) unless you need guaranteed fresh data

2. **Let TTLs work** - Don't force refresh unnecessarily, caches auto-refresh when stale

3. **Monitor cache status** - Check cache age/staleness on Page 8

4. **Handle errors gracefully** - Cached data returned even on API errors

5. **Pre-fetch for UX** - Consider pre-loading cache before showing dashboard

---

## Summary

**Caching makes Page 8 usable!**

- ✅ First load: 30-60s (unavoidable, external APIs)
- ✅ Subsequent loads: <1s (instant!)
- ✅ Automatic refresh when data is stale
- ✅ Graceful fallback on API errors
- ✅ Minimal memory footprint (~100KB)
- ✅ Cache status visible in dashboard

**Result:** Page 8 is now practical for real-time monitoring!
