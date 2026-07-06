#!/bin/bash
# One-time setup for MT4 (HFM) under Wine, alongside the existing MT5 (~/.mt5) install.
#
# What this does:
#   1. Enables 32-bit Wine and creates an isolated win32 prefix at ~/.mt4
#   2. Installs base Wine deps (corefonts)
#   3. Runs the HFM MT4 installer (you must supply it — broker-distributed, no CDN)
#   4. Detects the terminal's MQL4 data dir and installs PyBridge.mq4 + the bridge folders
#
# The file bridge needs NO DLLs and NO "Allow DLL imports" — only "Allow automated trading".
# Run once, then ./start_mt4.sh, then attach PyBridge to a chart.

set -e

export WINEPREFIX="$HOME/.mt4"
export WINEARCH=win32
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
EA_SRC="$REPO_DIR/mt4/PyBridge.mq4"

# HFM MT4 installer: drop the .exe here, or pass MT4_SETUP_URL / MT4_SETUP to override.
MT4_SETUP="${MT4_SETUP:-$REPO_DIR/downloads/hfm4setup.exe}"
MT4_SETUP_URL="${MT4_SETUP_URL:-}"

echo "=== MT4 setup — prefix $WINEPREFIX (win32), alongside MT5 (~/.mt5) ==="

# ── step 1: 32-bit Wine + prefix ────────────────────────────────────────────
if ! dpkg --print-foreign-architectures | grep -q i386; then
    echo "Enabling i386 architecture (needs sudo)…"
    sudo dpkg --add-architecture i386
    sudo apt-get update -qq || true
fi
if [ ! -d "$WINEPREFIX" ]; then
    echo "Creating win32 Wine prefix at $WINEPREFIX …"
    WINEARCH=win32 WINEPREFIX="$WINEPREFIX" wineboot --init
else
    echo "OK  prefix $WINEPREFIX already exists"
fi

# ── step 2: base deps ───────────────────────────────────────────────────────
if command -v winetricks &>/dev/null; then
    WINEPREFIX="$WINEPREFIX" winetricks -q corefonts || echo "WARN corefonts failed (non-fatal)"
fi

# ── step 3: HFM MT4 installer ───────────────────────────────────────────────
MT4_DIR="$WINEPREFIX/drive_c/Program Files/MetaTrader 4"
if [ ! -f "$MT4_DIR/terminal.exe" ]; then
    if [ -n "$MT4_SETUP_URL" ] && [ ! -f "$MT4_SETUP" ]; then
        echo "Downloading HFM MT4 installer from $MT4_SETUP_URL …"
        mkdir -p "$(dirname "$MT4_SETUP")"
        curl -fL "$MT4_SETUP_URL" -o "$MT4_SETUP" || true
    fi
    if [ ! -f "$MT4_SETUP" ]; then
        echo ""
        echo "  HFM MT4 installer not found at: $MT4_SETUP"
        echo "  Download it (PC) from your HFM client portal or:"
        echo "    https://www.hfm.com/int/en/platforms/mt4-terminal"
        echo "  then save it there (or run with MT4_SETUP=/path/to/installer.exe ./setup_mt4.sh)."
        exit 1
    fi
    echo "Running HFM MT4 installer under Wine (complete the GUI wizard)…"
    WINEDEBUG=-all wine "$MT4_SETUP" || true
else
    echo "OK  MT4 already installed: $MT4_DIR/terminal.exe"
fi

# ── step 4: install PyBridge EA + bridge folders ────────────────────────────
# Detect the MQL4 tree (Program Files vs Roaming/Terminal/<hash>); pick the most-recent.
echo "Detecting MQL4 data dir…"
MQL4_DIR="$(find "$WINEPREFIX/drive_c" -type d -ipath "*/MQL4/Experts" 2>/dev/null \
            | xargs -r -I{} dirname {} | xargs -r ls -dt 2>/dev/null | head -1)"
if [ -z "$MQL4_DIR" ]; then
    echo "WARN  MQL4 dir not found yet — launch MT4 once (./start_mt4.sh), then re-run this script."
else
    echo "OK  MQL4 dir: $MQL4_DIR"
    mkdir -p "$MQL4_DIR/Experts" "$MQL4_DIR/Files/pybridge/cmd" "$MQL4_DIR/Files/pybridge/res"
    cp "$EA_SRC" "$MQL4_DIR/Experts/PyBridge.mq4"
    echo "OK  PyBridge.mq4 → $MQL4_DIR/Experts/"
    echo "OK  bridge dir   → $MQL4_DIR/Files/pybridge   (set this in config.yaml mt4.file_bridge_dir, or leave 'auto')"
fi

echo ""
echo "=== Next steps ==="
echo "  1. ./start_mt4.sh                       # launch MT4 alongside MT5"
echo "  2. Log in to the HFM MT4 account"
echo "  3. MetaEditor: open PyBridge.mq4 → Compile (F7) to produce PyBridge.ex4"
echo "  4. Drag PyBridge onto any chart; enable Tools>Options>Expert Advisors>Allow automated trading"
echo "  5. conda activate envmt5 && BOT_PLATFORM=mt4 python tests/test_connection.py"
