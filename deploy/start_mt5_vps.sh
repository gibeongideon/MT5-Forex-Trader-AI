#!/bin/bash
# Headless MT5 launcher (VPS): terminal (Program Files) + rpyc bridge. Idempotent.
#
# MULTI-INSTANCE SAFE (fixed 2026-07-20): the old `pgrep -f terminal64.exe` matched
# ANY instance — with the cent terminal (~/.mt5b) running it wrongly concluded this
# instance was already up and skipped launching it, leaving a dead bridge (18812).
# The check is now scoped to THIS WINEPREFIX via /proc/<pid>/environ. Required for
# running FundingPips (.mt5) + cent (.mt5b) + FTMO (.mt5c) side by side.
set -u
export DISPLAY=:99 WINEPREFIX=$HOME/.mt5 WINEARCH=win64 WINEDEBUG=-all
export WINEDLLOVERRIDES="mscoree,mshtml="
TERM_EXE="$HOME/.mt5/drive_c/Program Files/MetaTrader 5/terminal64.exe"
WINPY="$HOME/.mt5/drive_c/Python310/python.exe"
PORT=18812

# true only if a terminal64.exe is running under THIS wine prefix
term_running() {
  local p
  for p in $(pgrep -f terminal64.exe 2>/dev/null); do
    if tr '\0' '\n' < "/proc/$p/environ" 2>/dev/null | grep -qx "WINEPREFIX=$WINEPREFIX"; then
      return 0
    fi
  done
  return 1
}

if ! term_running; then
  wine "$TERM_EXE" "C:\\mt5_login.ini" >/dev/null 2>&1 &
  sleep 25
fi
if ! ss -tlnp 2>/dev/null | grep -q "127.0.0.1:$PORT"; then
  wine "$WINPY" -c "from rpyc.utils.server import ThreadedServer; from rpyc.core import SlaveService; ThreadedServer(SlaveService, hostname='127.0.0.1', port=$PORT, reuse_addr=True).start()" >/dev/null 2>&1 &
  sleep 4
fi
