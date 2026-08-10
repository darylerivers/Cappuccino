#!/usr/bin/env bash
# Split-tunnel: PIA VPN for exchange traffic, direct for Claude API
# Usage: ./vpn_splittunnel.sh [connect|disconnect|status]
#
# Strategy: PIA takes over default route via tun0.
# We punch bypass routes for Anthropic API IPs through the original gateway.

set -euo pipefail

PHYSICAL_GW="10.0.0.1"
PHYSICAL_DEV="wlan0"
PIA_REGION="singapore"

# Anthropic API IPs (api.anthropic.com resolves to 160.79.104.10 / Cloudflare)
# Add the /22 block so any Cloudflare failover IP is also covered
ANTHROPIC_ROUTES=(
    "160.79.104.0/22"   # api.anthropic.com (Cloudflare block)
    "104.16.0.0/12"     # Cloudflare main range (covers Claude web too)
)

wait_for_vpn() {
    echo "Waiting for VPN..."
    for i in $(seq 1 30); do
        state=$(piactl get connectionstate 2>/dev/null)
        if [[ "$state" == "Connected" ]]; then
            echo "VPN connected. VPN IP: $(piactl get vpnip)"
            return 0
        fi
        sleep 1
    done
    echo "ERROR: VPN did not connect within 30s (state: $state)"
    return 1
}

add_bypass_routes() {
    echo "Adding Anthropic bypass routes via $PHYSICAL_GW ($PHYSICAL_DEV)..."
    for route in "${ANTHROPIC_ROUTES[@]}"; do
        ip route replace "$route" via "$PHYSICAL_GW" dev "$PHYSICAL_DEV" 2>/dev/null && \
            echo "  ✓ $route → direct" || \
            echo "  ✗ $route → failed (may need sudo)"
    done
}

remove_bypass_routes() {
    echo "Removing bypass routes..."
    for route in "${ANTHROPIC_ROUTES[@]}"; do
        ip route del "$route" 2>/dev/null && echo "  removed $route" || true
    done
}

verify() {
    echo ""
    echo "=== Verification ==="
    pub_ip=$(curl -s --max-time 5 https://ifconfig.me 2>/dev/null || echo "failed")
    echo "Public IP (should be VPN):       $pub_ip"

    anthropic_route=$(ip route get 160.79.104.10 2>/dev/null | head -1)
    echo "Route to Anthropic (want wlan0): $anthropic_route"

    # Test Bybit
    bybit_code=$(curl -s --max-time 8 -o /dev/null -w "%{http_code}" \
        "https://api.bybit.com/v5/market/tickers?category=spot&symbol=BTCUSDT" 2>/dev/null)
    echo "Bybit API status:                HTTP $bybit_code (want 200)"

    # Test Binance futures
    binance_code=$(curl -s --max-time 8 -o /dev/null -w "%{http_code}" \
        "https://fapi.binance.com/fapi/v1/ticker/price?symbol=BTCUSDT" 2>/dev/null)
    echo "Binance Futures API status:      HTTP $binance_code (want 200)"

    # Test Claude API reachability (just connectivity, no auth)
    claude_code=$(curl -s --max-time 8 -o /dev/null -w "%{http_code}" \
        "https://api.anthropic.com" 2>/dev/null)
    echo "Anthropic API reachable:         HTTP $claude_code (want 401/200, not 403)"
}

case "${1:-connect}" in
    connect)
        echo "--- PIA Split-Tunnel Setup ---"
        piactl set region "$PIA_REGION"
        piactl connect
        wait_for_vpn
        add_bypass_routes
        verify
        echo ""
        echo "Split-tunnel active. Trading traffic → VPN, Claude API → direct."
        echo "Run './vpn_splittunnel.sh disconnect' to clean up."
        ;;
    disconnect)
        remove_bypass_routes
        piactl disconnect
        echo "VPN disconnected."
        ;;
    status)
        state=$(piactl get connectionstate)
        vpnip=$(piactl get vpnip 2>/dev/null)
        echo "VPN state: $state | VPN IP: $vpnip"
        verify
        ;;
    routes-only)
        # Re-add bypass routes without reconnecting (useful after VPN reconnect)
        add_bypass_routes
        verify
        ;;
    *)
        echo "Usage: $0 [connect|disconnect|status|routes-only]"
        exit 1
        ;;
esac
