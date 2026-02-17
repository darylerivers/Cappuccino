#!/bin/bash

# Define the ROCm installation directory and tool names
ROCM_DIR="/opt/rocm"
TOOL_NAMES=("hip" "hcc" "hsa" "cl" "rocminfo" "rocprof")

# Check if ROCm is already installed
if [ -d "$ROCM_DIR" ]; then
    echo "ROCm is already installed in $ROCM_DIR."
    # Verify if the tools are present within ROCm installation directory
    for tool in "${TOOL_NAMES[@]}"; do
        if [ ! -x "$ROCM_DIR/$tool" ]; then
            echo "The following ROCm tool is missing: $tool"
        else
            echo "$tool is installed."
        fi
    done
else
    # Define the ROCm installer script URL
    ROCM_INSTALLER_URL="https://rocm.github.io/rocm-releases.html"

    # Download and install ROCm
    echo "ROCm is not installed. Checking for ROCm installation script..."
    if wget -q -O - "$ROCM_INSTALLER_URL" | sudo bash -s; then
        echo "ROCm installation script downloaded successfully."
        read -p "Press [Enter] to continue..."
    else
        echo "Failed to download ROCm installation script."
        exit 1
    fi
fi

# End of script