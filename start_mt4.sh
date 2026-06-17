#!/bin/bash
# Start MetaTrader 4 (HFM) under Wine — runs ALONGSIDE MT5 in an isolated win32 prefix.
#
# MT4 has no rpyc bridge: automation is via the PyBridge.mq4 EA (file bridge). This script
# only launches the terminal; attach PyBridge to a chart once (see ./setup_mt4.sh output).
#
# Prerequisites: run ./setup_mt4.sh first (creates the win32 prefix + installs HFM MT4 + EA).

set -e

export WINEPREFIX="$HOME/.mt4"
# Current MetaQuotes MT4 build is 64-bit → win64 prefix (same as MT5). Do NOT force WINEARCH
# (a mismatch errors on an existing prefix). Auto-detect terminal.exe wherever it installed.
TERMINAL="$(find "$WINEPREFIX/drive_c" -iname 'terminal.exe' -path '*MetaTrader 4*' 2>/dev/null | head -1)"

check_wine() {
    if ! command -v wine &>/dev/null; then
        echo "ERROR: wine not found. Install: sudo apt install winehq-stable"
        exit 1
    fi
}

start_terminal() {
    if pgrep -f "MetaTrader 4.*terminal.exe\|mt4_launcher" >/dev/null 2>&1; then
        echo "MT4 terminal already running."
        return
    fi
    if [ ! -f "$TERMINAL" ]; then
        echo "ERROR: MT4 terminal.exe not found at:"
        echo "  $TERMINAL"
        echo "MT4 is broker-distributed (no CDN auto-download). Run ./setup_mt4.sh first,"
        echo "or download HFM MT4 from https://www.hfm.com/int/en/platforms/mt4-terminal"
        exit 1
    fi
    echo "Starting MT4 terminal..."
    # Run from a temp copy so liveupdate can replace the Program Files binary (Wine file-lock)
    local TEMP_LAUNCHER="$WINEPREFIX/drive_c/users/$USER/AppData/Local/Temp/mt4_launcher"
    mkdir -p "$TEMP_LAUNCHER"
    cp "$TERMINAL" "$TEMP_LAUNCHER/terminal.exe"
    WINEDEBUG=-all wine "$TEMP_LAUNCHER/terminal.exe" &
    echo "MT4 terminal launched (PID: $!)"
    echo "  → Log in to your HFM MT4 account"
    echo "  → Attach PyBridge EA to ONE chart (smiley face = running)"
    echo "  → Enable: Tools > Options > Expert Advisors > Allow automated trading"
    sleep 15
}

check_wine
echo "=== Starting MT4 (HFM) on Ubuntu — prefix $WINEPREFIX (win32) ==="
echo "Note: MT5 uses ~/.mt5 (win64) and is unaffected by this."
echo ""
start_terminal
echo ""
echo "=== Ready ==="
echo "  conda activate envmt5"
echo "  BOT_PLATFORM=mt4 python -c 'from src.core.mt4_connector import MT4Connector; MT4Connector().connect()'"
