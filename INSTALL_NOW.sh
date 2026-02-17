#!/bin/bash
# Quick installation script for enhanced safety system

set -e

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                                                                ║"
echo "║     Cappuccino Enhanced Safety System Installation            ║"
echo "║                                                                ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Check if running with sudo
if [[ $EUID -ne 0 ]]; then
   echo "⚠️  This script needs sudo privileges to install systemd service"
   echo ""
   echo "Please run:"
   echo "    sudo bash INSTALL_NOW.sh"
   echo ""
   exit 1
fi

echo "📋 Installation Steps:"
echo "  1. Install systemd watchdog service"
echo "  2. Enable auto-start on boot"
echo "  3. Start watchdog service"
echo "  4. Verify everything is running"
echo ""

read -p "Continue with installation? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Installation cancelled"
    exit 0
fi

echo ""
echo "🔧 Installing systemd service..."
bash systemd/install_services.sh

echo ""
echo "✅ Installation complete!"
echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                      What's Protected Now                      ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "✅ Layer 1: Emergency brake (kills trials at low memory)"
echo "✅ Layer 2: Smart worker restart (after 10 consecutive brakes)"
echo "✅ Layer 3: Systemd watchdog (survives reboots, auto-restarts)"
echo "✅ Layer 4: Memory leak fix (clears PyTorch tensors)"
echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                       Quick Commands                           ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "Check status:"
echo "  sudo systemctl status cappuccino-watchdog"
echo ""
echo "View logs:"
echo "  sudo journalctl -u cappuccino-watchdog -f"
echo "  tail -f logs/watchdog.log"
echo ""
echo "Start training:"
echo "  ./start_safe_workers.sh"
echo ""
echo "Monitor training:"
echo "  python scripts/automation/dashboard_detailed.py --loop"
echo ""
