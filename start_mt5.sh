#!/bin/bash
# Start MetaTrader 5 and the mt5linux rpyc bridge on Ubuntu.
#
# What this script does:
#   1. Launches terminal64.exe under Wine (~/.mt5 prefix)
#   2. Launches the rpyc classic server inside Wine Python (port 18812)
#      so native Linux Python can call MT5 via mt5linux
#
# Prerequisites: run ./setup.sh first (installs Wine Python + bridge deps)

set -e

WINEPREFIX="$HOME/.mt5"
MT5_DIR="$WINEPREFIX/drive_c/Program Files/MetaTrader 5"
TERMINAL="$MT5_DIR/terminal64.exe"
BRIDGE_PORT=18812

# Wine Python installed by setup.sh
WINE_PYTHON_DIR="$WINEPREFIX/drive_c/Python310"
WINE_PYTHON="$WINE_PYTHON_DIR/python.exe"
WINE_PIP="$WINE_PYTHON_DIR/Scripts/pip.exe"

export WINEPREFIX

# ─── helpers ───────────────────────────────────────────────────────────────
check_wine() {
    if ! command -v wine &>/dev/null; then
        echo "ERROR: wine not found. Install: sudo apt install winehq-stable"
        exit 1
    fi
}

install_terminal() {
    # Download terminal64.exe directly from MetaQuotes CDN — no installer GUI needed
    local dest="$MT5_DIR/terminal64.exe"
    local url="https://download.mql5.com/cdn/web/metaquotes.software.corp/mt5/terminal64.exe"
    echo "Downloading terminal64.exe from MetaQuotes CDN..."
    mkdir -p "$MT5_DIR"
    curl -L "$url" -o "$dest" --progress-bar
    if [ -f "$dest" ]; then
        echo "OK  terminal64.exe downloaded ($(du -sh "$dest" | cut -f1))"
    else
        echo "ERROR: Download failed. Check your internet connection."
        exit 1
    fi
}

start_terminal() {
    if pgrep -f "terminal64.exe" >/dev/null 2>&1; then
        echo "MT5 terminal already running."
        return
    fi

    if [ ! -f "$TERMINAL" ]; then
        echo "terminal64.exe not found — downloading from MetaQuotes CDN..."
        install_terminal

        if [ ! -f "$TERMINAL" ]; then
            echo ""
            echo "ERROR: terminal64.exe still not found after install."
            exit 1
        fi
    fi

    echo "Starting MT5 terminal..."
    # Run from a temp copy so the liveupdate can replace the Program Files version
    # (Wine blocks writing to a file that is currently executing — running from a copy avoids this)
    local TEMP_LAUNCHER="$WINEPREFIX/drive_c/users/$USER/AppData/Local/Temp/mt5_launcher"
    mkdir -p "$TEMP_LAUNCHER"
    cp "$TERMINAL" "$TEMP_LAUNCHER/terminal64.exe"
    WINEPREFIX="$WINEPREFIX" WINEDEBUG=-all wine "$TEMP_LAUNCHER/terminal64.exe" &
    echo "MT5 terminal launched (PID: $!)"
    echo "  → Log in to your broker account"
    echo "  → Enable: Tools > Options > Expert Advisors > Allow algorithmic trading"
    sleep 15
}

start_bridge() {
    if pgrep -f "SlaveService\|rpyc.*18812" >/dev/null 2>&1; then
        echo "rpyc bridge already running on port $BRIDGE_PORT."
        return
    fi

    if [ ! -f "$WINE_PYTHON" ]; then
        echo "ERROR: Wine Python not found at $WINE_PYTHON"
        echo "Run setup.sh first to install Python inside Wine."
        exit 1
    fi

    echo "Starting mt5linux rpyc bridge on port $BRIDGE_PORT..."
    # Run rpyc SlaveService inside Wine Python — mt5linux (Linux) connects to this
    WINEPREFIX="$WINEPREFIX" WINEDEBUG=-all wine "$WINE_PYTHON" -c \
      "from rpyc.utils.server import ThreadedServer; from rpyc.core import SlaveService; ThreadedServer(SlaveService, hostname='127.0.0.1', port=$BRIDGE_PORT, reuse_addr=True).start()" &
    BRIDGE_PID=$!
    sleep 3

    if kill -0 $BRIDGE_PID 2>/dev/null; then
        echo "Bridge running (PID: $BRIDGE_PID) — localhost:$BRIDGE_PORT"
    else
        echo "WARNING: Bridge may have failed to start. Check Wine output above."
    fi
}

# ─── main ──────────────────────────────────────────────────────────────────
check_wine
echo "=== Starting MT5 on Ubuntu ==="
echo "WINEPREFIX: $WINEPREFIX"
echo ""
start_terminal
start_bridge
echo ""
echo "=== Ready ==="
echo "  conda activate envmt5"
echo "  python tests/test_connection.py   # verify connection"
echo "  python src/example_bot.py         # run example bot"
