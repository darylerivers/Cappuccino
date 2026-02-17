#!/bin/bash
# Pre-fetch Tiburtina data in the background
# This populates the cache so Page 8 loads instantly

echo "Pre-fetching Tiburtina data in background..."

python3 << 'EOF' &
from tiburtina_integration import get_tiburtina_client
import sys

try:
    client = get_tiburtina_client()

    if not client.is_available():
        print("Tiburtina not available:", client.get_error(), file=sys.stderr)
        exit(1)

    # Prefetch all data (this will take 30-60s)
    print("Fetching crypto data...", file=sys.stderr)
    client.get_crypto_overview(use_cache=False)

    print("Fetching news...", file=sys.stderr)
    client.get_news_summary(use_cache=False)

    print("Fetching macro data...", file=sys.stderr)
    client.get_macro_snapshot(use_cache=False)

    print("✓ Tiburtina data cached!", file=sys.stderr)

except Exception as e:
    print(f"✗ Prefetch failed: {e}", file=sys.stderr)
    exit(1)
EOF

PREFETCH_PID=$!
echo "Prefetch running in background (PID: $PREFETCH_PID)"
echo "Dashboard will load instantly once prefetch completes (~30-60s)"
echo ""
echo "To check progress: tail -f /proc/$PREFETCH_PID/fd/2 2>/dev/null || echo 'Prefetch complete'"
echo ""
