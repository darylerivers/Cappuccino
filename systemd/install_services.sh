#!/bin/bash
# Install Cappuccino systemd services

set -e

echo "========================================="
echo "  Installing Cappuccino Systemd Services"
echo "========================================="
echo ""

# Check if running as root or with sudo access
if [[ $EUID -ne 0 ]]; then
   echo "This script must be run with sudo privileges"
   echo "Usage: sudo bash systemd/install_services.sh"
   exit 1
fi

SERVICE_DIR="/etc/systemd/system"
SOURCE_DIR="/opt/user-data/experiment/cappuccino/systemd"

# Install watchdog service
echo "Installing cappuccino-watchdog.service..."
cp "${SOURCE_DIR}/cappuccino-watchdog.service" "${SERVICE_DIR}/"
chmod 644 "${SERVICE_DIR}/cappuccino-watchdog.service"
echo "✅ Watchdog service installed"

# Reload systemd
echo ""
echo "Reloading systemd daemon..."
systemctl daemon-reload
echo "✅ Systemd reloaded"

# Enable service (auto-start on boot)
echo ""
echo "Enabling watchdog service for auto-start on boot..."
systemctl enable cappuccino-watchdog.service
echo "✅ Watchdog service enabled"

# Start service
echo ""
echo "Starting watchdog service..."
systemctl start cappuccino-watchdog.service
echo "✅ Watchdog service started"

# Check status
echo ""
echo "========================================="
echo "  Service Status"
echo "========================================="
systemctl status cappuccino-watchdog.service --no-pager -l

echo ""
echo "========================================="
echo "  Installation Complete!"
echo "========================================="
echo ""
echo "Useful commands:"
echo "  Check status:    sudo systemctl status cappuccino-watchdog"
echo "  View logs:       sudo journalctl -u cappuccino-watchdog -f"
echo "  Restart:         sudo systemctl restart cappuccino-watchdog"
echo "  Stop:            sudo systemctl stop cappuccino-watchdog"
echo "  Disable:         sudo systemctl disable cappuccino-watchdog"
echo ""
echo "Watchdog logs also available at:"
echo "  logs/watchdog.log"
echo "  logs/watchdog_service.log"
echo "  logs/watchdog_service_error.log"
echo ""
