#!/bin/bash
# One-time setup for MT5 Python bot development on Ubuntu.
#
# What this does:
#   1. Installs Python deps in the 'envmt5' conda env (native Linux)
#   2. Downloads Python 3.10 for Windows and installs it inside Wine (~/.mt5)
#   3. Installs MetaTrader5 + rpyc inside Wine Python (needed for the bridge)
#
# Run once, then use start_mt5.sh to launch MT5 + the bridge.

set -e

CONDA_ENV="envmt5"
CONDA_PIP="$HOME/anaconda3/envs/$CONDA_ENV/bin/pip"
CONDA_PYTHON="$HOME/anaconda3/envs/$CONDA_ENV/bin/python"

WINEPREFIX="$HOME/.mt5"
WINE_PYTHON_DIR="$WINEPREFIX/drive_c/Python310"
WINE_PYTHON="$WINE_PYTHON_DIR/python.exe"
WINE_PIP="$WINE_PYTHON_DIR/Scripts/pip.exe"

# Windows Python 3.10.11 (64-bit) — compatible with MetaTrader5 package
PY_WIN_URL="https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe"
PY_WIN_INSTALLER="/tmp/python-3.10.11-amd64.exe"

export WINEPREFIX

# ─── step 1: Linux Python deps ─────────────────────────────────────────────
echo "=== Step 1: Installing Linux Python packages in conda env '$CONDA_ENV' ==="
if [ ! -f "$CONDA_PYTHON" ]; then
    echo "ERROR: conda env '$CONDA_ENV' not found."
    echo "Create it: conda create -n $CONDA_ENV python=3.10"
    exit 1
fi
"$CONDA_PIP" install -r requirements.txt -q
echo "OK  Linux packages installed."
echo ""

# ─── step 2: Wine Python ───────────────────────────────────────────────────
echo "=== Step 2: Installing Python 3.10 inside Wine ==="
if [ -f "$WINE_PYTHON" ]; then
    echo "OK  Wine Python already installed: $WINE_PYTHON"
else
    if [ ! -f "$PY_WIN_INSTALLER" ]; then
        echo "Downloading Python 3.10.11 for Windows..."
        curl -L "$PY_WIN_URL" -o "$PY_WIN_INSTALLER"
    fi
    echo "Installing Python 3.10 inside Wine (silent, to C:\\Python310)..."
    wine "$PY_WIN_INSTALLER" /quiet InstallAllUsers=0 TargetDir="C:\\Python310" PrependPath=0
    echo "OK  Wine Python installed."
fi
echo ""

# ─── step 3: MetaTrader5 + rpyc inside Wine Python ────────────────────────
echo "=== Step 3: Installing MetaTrader5 + rpyc inside Wine Python ==="

# Check if already installed
MT5_CHECK=$(wine "$WINE_PYTHON" -c "import MetaTrader5; print('ok')" 2>/dev/null || true)
if [ "$MT5_CHECK" = "ok" ]; then
    echo "OK  MetaTrader5 already installed in Wine Python."
else
    echo "Installing packages inside Wine Python..."
    wine "$WINE_PYTHON" -m pip install --upgrade pip -q
    wine "$WINE_PYTHON" -m pip install MetaTrader5 rpyc -q
    echo "OK  MetaTrader5 + rpyc installed in Wine Python."
fi
echo ""

# ─── summary ───────────────────────────────────────────────────────────────
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  1. Start MT5 + bridge:    ./start_mt5.sh"
echo "  2. Log in to broker in the MT5 terminal window"
echo "  3. Enable algo trading:   Tools > Options > Expert Advisors"
echo "  4. Activate conda env:    conda activate $CONDA_ENV"
echo "  5. Test connection:       python tests/test_connection.py"
echo "  6. Run the example bot:   python src/example_bot.py"
