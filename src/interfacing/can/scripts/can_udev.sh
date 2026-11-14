#!/usr/bin/env bash
set -e

UDEV_RULE_FILE="/etc/udev/rules.d/99-canable.rules"
UDEV_VENDOR_ID="16d0"
UDEV_PRODUCT_ID="117e"
UDEV_SYMLINK_DEFAULT="canable"

SUDO=""
if [ "${EUID:-$(id -u)}" -ne 0 ]; then
    if command -v sudo >/dev/null 2>&1; then
        SUDO="sudo"
    fi
fi

usage() {
    echo "Usage: $0 install [symlink] [device]"
    echo "       $0 remove [symlink]"
}

install_rule() {
    local symlink="$1"
    local device="${2:-}" path_match="" usb_path

    if [ -z "$symlink" ]; then
        symlink="$UDEV_SYMLINK_DEFAULT"
    fi

    if [ -z "$SUDO" ] && [ "${EUID:-$(id -u)}" -ne 0 ]; then
        echo "This command requires root privileges (or sudo)."
        exit 1
    fi

    if [ "$symlink" = "canable1" ] || [ "$symlink" = "canable2" ]; then
        if [ -z "$device" ]; then
            echo "Specify the device for $symlink, e.g. /dev/ttyACM0"
            exit 1
        fi
        usb_path=$(udevadm info --query=property --name="$device" | sed -n 's/^ID_PATH=//p')
        if [ -z "$usb_path" ]; then
            echo "No USB port path found for $device"
            exit 1
        fi
        path_match="ENV{ID_PATH}==\"$usb_path\", "
        UDEV_RULE_FILE="/etc/udev/rules.d/99-$symlink.rules"
    fi

    echo "Setting up CANable udev rule..."

    cat <<EOF | $SUDO tee "$UDEV_RULE_FILE" > /dev/null
SUBSYSTEM=="tty", ATTRS{idVendor}=="$UDEV_VENDOR_ID", ATTRS{idProduct}=="$UDEV_PRODUCT_ID", ${path_match}SYMLINK+="$symlink"
EOF

    echo "Reloading udev rules..."
    $SUDO udevadm control --reload-rules
    $SUDO udevadm trigger
    $SUDO udevadm settle

    if [ -e "/dev/$symlink" ]; then
        echo "✔ /dev/$symlink created successfully"
    elif usb_present "$UDEV_VENDOR_ID" "$UDEV_PRODUCT_ID"; then
        echo "⚠ CANable ($UDEV_VENDOR_ID:$UDEV_PRODUCT_ID) is plugged in but /dev/$symlink is missing — try unplugging and replugging it"
    elif usb_present 1d50 606f; then
        echo "⚠ This adapter runs candleLight/gs_usb firmware (1d50:606f): it is a native can0, not a serial device, so there is no /dev/$symlink."
        echo "  The rule is installed but not needed. Use setup_socketcan.sh instead (see src/interfacing/can/README.md)."
    else
        echo "⚠ No CANable ($UDEV_VENDOR_ID:$UDEV_PRODUCT_ID) found on USB. The rule is installed and applies once it is plugged in."
        echo "  If it is plugged in, check 'lsusb': a different ID means this rule does not match that adapter."
    fi
}

usb_present() {
    local d
    for d in /sys/bus/usb/devices/*; do
        [ "$(cat "$d/idVendor" 2>/dev/null)" = "$1" ] && [ "$(cat "$d/idProduct" 2>/dev/null)" = "$2" ] && return 0
    done
    return 1
}

remove_rule() {
    if [ "${1:-}" = "canable1" ] || [ "${1:-}" = "canable2" ]; then
        UDEV_RULE_FILE="/etc/udev/rules.d/99-$1.rules"
    fi

    if [ -z "$SUDO" ] && [ "${EUID:-$(id -u)}" -ne 0 ]; then
        echo "This command requires root privileges (or sudo)."
        exit 1
    fi

    if [ -e "$UDEV_RULE_FILE" ]; then
        echo "Removing udev rule: $UDEV_RULE_FILE"
        $SUDO rm -f "$UDEV_RULE_FILE"
        echo "Reloading udev rules..."
        $SUDO udevadm control --reload-rules
        $SUDO udevadm trigger
    else
        echo "udev rule not present: $UDEV_RULE_FILE"
    fi
}

CMD="$1"
case "$CMD" in
    install)
        install_rule "${2:-}" "${3:-}"
        ;;
    remove)
        remove_rule "${2:-}"
        ;;
    -h|--help|help|"" )
        usage
        exit 1
        ;;
    *)
        usage
        exit 1
        ;;
esac
