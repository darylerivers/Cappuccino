#!/bin/bash
# Install systemd service to keep GPU in high performance mode

echo "Creating systemd service for GPU performance..."

sudo tee /etc/systemd/system/amd-gpu-performance.service > /dev/null << 'EOF'
[Unit]
Description=AMD GPU High Performance Mode
After=multi-user.target

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/bin/bash -c 'echo on > /sys/class/drm/card1/device/power/control'
ExecStart=/bin/bash -c 'echo high > /sys/class/drm/card1/device/power_dpm_force_performance_level'
ExecStart=/bin/bash -c 'echo COMPUTE > /sys/class/drm/card1/device/pp_power_profile_mode || true'
ExecStart=/bin/bash -c 'echo 0 > /sys/class/drm/card1/device/power/runtime_auto_suspend_delay_ms || true'

[Install]
WantedBy=multi-user.target
EOF

echo "Enabling service..."
sudo systemctl daemon-reload
sudo systemctl enable amd-gpu-performance.service
sudo systemctl start amd-gpu-performance.service

echo ""
echo "✅ Service installed and started"
echo "   GPU will now stay in high performance mode across reboots"
echo ""
sudo systemctl status amd-gpu-performance.service
