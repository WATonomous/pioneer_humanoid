#!/bin/bash
# Purpose: Sets up a CAN interface using slcand (daemon for serial CAN interfaces)
# This script configures a CAN interface by:
# 1. Starting the slcand daemon to create a network interface from a serial device
# 2. Bringing up the CAN interface
# 3. Setting the transmission queue length
# 
# The script takes three parameters all found in the src/interfacing/can/config/params.yaml:
# - device_path: Serial device path (e.g., /dev/canable)
# - interface_name: Name for the CAN interface (e.g., can0)
# - bitrate_code: Bitrate configuration code (e.g., -s6 for 500k, -s8 for 1M)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DEVICE_PATH="$1"
INTERFACE_NAME="$2"
BITRATE_CODE="$3" # e.g., -s6 for 500k, -s8 for 1M 

# Check if slcand is installed
if ! command -v slcand &> /dev/null; then
    echo "slcand could not be found. Please install can-utils."
    exit 1
fi

if ! command -v ifconfig &> /dev/null; then
    echo "ifconfig could not be found. Please install net-tools."
    exit 1
fi

if [ -z "$DEVICE_PATH" ] || [ -z "$INTERFACE_NAME" ] || [ -z "$BITRATE_CODE" ]; then
    echo "Usage: $0 <device_path> <interface_name> <bitrate_code>"
    echo "Example: $0 /dev/canable can0 -s8"
    echo ""
    exit 1
fi

echo "Attempting to set up CAN interface $INTERFACE_NAME on $DEVICE_PATH with bitrate code $BITRATE_CODE..."

# Check if slcand is already running for this device/interface to avoid errors
if pgrep -f "slcand.*$DEVICE_PATH.*$INTERFACE_NAME"; then
    echo "slcand appears to be already running for $DEVICE_PATH and $INTERFACE_NAME. Assuming interface is (or will be) up."
else
    echo "Starting slcand: slcand -o -c $BITRATE_CODE $DEVICE_PATH $INTERFACE_NAME"
    slcand -o -c "$BITRATE_CODE" "$DEVICE_PATH" "$INTERFACE_NAME"
    # Add a small delay to allow slcand to establish the interface
    sleep 1
fi

echo "Bringing up interface: ifconfig $INTERFACE_NAME up"
ifconfig "$INTERFACE_NAME" up

echo "Setting txqueuelen: ifconfig $INTERFACE_NAME txqueuelen 1000" # Transmisson queue length
ifconfig "$INTERFACE_NAME" txqueuelen 1000

echo "CAN interface $INTERFACE_NAME setup complete."
exit 0