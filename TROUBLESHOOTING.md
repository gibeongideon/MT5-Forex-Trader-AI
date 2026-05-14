# MT5 on Ubuntu — Troubleshooting Notes

## Problem: "Something went wrong" on every launch

**Symptom**  
Every time `terminal64.exe` starts on Ubuntu/Wine it shows the "MetaTrader 5 Setup (64 bit)"
installation screen, downloads companion packages (~78 MB), then fails with a
"Something went wrong" dialog and never opens the main trading window.

**Root cause**  
MT5's liveupdate mechanism tries to replace `terminal64.exe` with a newer version as part of
the installation. Under Wine, writing to a file that is the currently-running executable is
blocked with `STATUS_SHARING_VIOLATION` (`c0000043`). The update fails, MT5 reports the error.

Confirmed via Wine debug logging (`WINEDEBUG=warn+all`):

```
081c:warn:file:CreateFileW Unable to create file
    L"C:\\Program Files\\MetaTrader 5\\terminal64.exe" (status c0000043)
```

Windows normally works around this using `PendingFileRenameOperations` (schedule the file
replacement for next boot). Wine's implementation of that registry mechanism is incomplete,
so the fallback also fails.

**Fix**  
Run `terminal64.exe` from a **temporary copy** instead of directly from `Program Files`.
Because the running process is the copy in `Temp\`, the file at
`C:\Program Files\MetaTrader 5\terminal64.exe` is not locked, and the liveupdate can write
the updated terminal there successfully.

```bash
TEMP_LAUNCHER="$WINEPREFIX/drive_c/users/$USER/AppData/Local/Temp/mt5_launcher"
mkdir -p "$TEMP_LAUNCHER"
cp "$TERMINAL" "$TEMP_LAUNCHER/terminal64.exe"
WINEPREFIX="$WINEPREFIX" WINEDEBUG=-all wine "$TEMP_LAUNCHER/terminal64.exe" &
```

`start_mt5.sh` already uses this approach — just run `./start_mt5.sh`.

**What the liveupdate downloads**

| Package file | Extracted to |
|---|---|
| `mt5clwideavx264.png` | `MetaEditor64.exe` (109 MB) |
| `mt5clwtstavx264.png` | `metatester64.exe` (21 MB) |
| `mt5onnxavx264.png` | `onnxruntime.dll`, `onnxruntime_providers_shared.dll`, `openblas.dll` |
| `mt5clwdata.png` | `Bases/` history data, mail templates (573 files) |
| *(5th package via direct write)* | `terminal64.exe` itself |

These are ZIP archives disguised as PNG files, stored temporarily in:
`~/.mt5/drive_c/users/$USER/AppData/Roaming/MetaQuotes/WebInstall/`

**Other things tried (did NOT fix the issue alone)**

- `[LiveUpdate] Disable=1` in `terminal.ini` — the terminal ignores this key
- Setting `LastBuild=5847` in `terminal.ini` — terminal still runs the update check
- Blocking `download.mql5.com` in the Wine Windows hosts file — Wine uses the host OS DNS resolver, not the Wine prefix hosts file; downloads still proceeded
- Manually extracting the companion packages before launch — still failed because the terminal itself needed to be replaced

---

## Problem: terminal64.exe missing from Wine prefix

**Symptom**  
`start_mt5.sh` errors: `terminal64.exe not found`.

**Cause**  
A previous liveupdate attempt removed the old `terminal64.exe` and failed to write the new one
(sharing violation), leaving the install directory without the executable.

**Fix**  
`start_mt5.sh` automatically downloads a fresh copy from the MetaQuotes CDN:

```
https://download.mql5.com/cdn/web/metaquotes.software.corp/mt5/terminal64.exe
```

---

## Problem: Wine "debugger is running" popup during MT5 installer

**Symptom**  
Running the MetaQuotes setup `.exe` under Wine pops up a dialog:
"A debugger is running in your system. Please unload it from memory and restart."

**Cause**  
Wine registers itself as the system debugger via the `AeDebug` registry key. MT5's installer
detects this and refuses to proceed.

**Fix**  
Bypass the installer entirely — download `terminal64.exe` directly from the CDN URL above
(it is the full terminal, not a stub installer). No installer GUI needed.

---

## rpyc bridge (mt5linux) setup

The `MetaTrader5` Python package has no Linux wheels. The workaround is
[mt5linux](https://github.com/lucas-campagna/mt5linux): a thin rpyc bridge that lets Linux
Python call the Windows-only `MetaTrader5` package running in Wine Python.

**Bridge architecture:**
```
Linux Python (envmt5)
  → rpyc socket  localhost:18812
    → Wine Python (~/.mt5/drive_c/Python310/python.exe)
      → MetaTrader5 (Windows pip package)
        → terminal64.exe (IPC named pipe)
```

**Start the bridge (run after `./start_mt5.sh`):**

```bash
WINEPREFIX="$HOME/.mt5" WINEDEBUG=-all wine \
  "$HOME/.mt5/drive_c/Python310/python.exe" -c \
  "from rpyc.utils.server import ThreadedServer; from rpyc.core import SlaveService; \
   ThreadedServer(SlaveService, hostname='127.0.0.1', port=18812, reuse_addr=True).start()"
```

`start_mt5.sh` does this automatically.

**Version requirement:** Wine Python rpyc must match Linux rpyc (`==5.2.3`).
If mismatched, reinstall: `wine python.exe -m pip install "rpyc==5.2.3"`