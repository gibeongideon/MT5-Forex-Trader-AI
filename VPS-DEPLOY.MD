# VPS-DEPLOY — FundingPips basket challenge bot on a Linux VPS

Runbook to deploy the diversified basket challenge bot (`v5_basket_challenge_exec.py`,
2-Step Standard @ 7% vol) on a headless VPS. The bot = MT5 terminal under Wine +
an rpyc bridge (localhost:18812) + Python (conda `envmt5`) + systemd user timers.

> The bot is a **daily/H4 trend follower** — it reconciles hourly, holds for days.
> It is NOT latency-sensitive, so a cheap VPS near any region is fine; broker
> co-location is irrelevant. What matters is **uptime + RAM + a virtual display**.

---

## 0. What's different on a VPS vs the dev desktop

| | Dev desktop | VPS (headless) |
|---|---|---|
| Display for the MT5 GUI | real X (`DISPLAY=:0`) | **virtual X — Xvfb (`DISPLAY=:99`)** |
| Initial account login | click in the GUI | one-time via **VNC over SSH tunnel** |
| Absolute paths in scripts | `/home/rock/...` | edit to the VPS user/path |
| conda | `~/anaconda3` | `~/miniconda3` (or anaconda) — adjust |

The three wrapper scripts (`scripts/xau_*_cron.sh`) and the systemd units contain
**hardcoded paths** (`/home/rock/Desktop/2026_Projects/Trader36/MT5`,
`/home/rock/anaconda3/bin/conda`, `DISPLAY=:0`). Every one must be updated for the
VPS. Step 7 lists them.

---

## 1. Provision the VPS

- **OS:** Ubuntu 22.04 or 24.04 LTS, **x86-64** (Wine does not run the MT5 x64
  terminal on ARM — do NOT pick an ARM/Graviton instance).
- **RAM:** ≥ 4 GB (MT5 under Wine peaked ~2.4 GB here; 8 GB comfortable with headroom).
- **vCPU:** 2. **Disk:** ≥ 30 GB SSD. **Swap:** add 2 GB if RAM = 4 GB.
- Providers: Hetzner CX22/CX32, Contabo, DigitalOcean, Vultr, OVH — any x86 KVM VPS.

## 2. Base hardening (first login as root)

```bash
adduser trader && usermod -aG sudo trader
# SSH key auth only:
mkdir -p /home/trader/.ssh && cp ~/.ssh/authorized_keys /home/trader/.ssh/ 2>/dev/null
chown -R trader:trader /home/trader/.ssh && chmod 700 /home/trader/.ssh
sed -i 's/^#\?PasswordAuthentication.*/PasswordAuthentication no/' /etc/ssh/sshd_config
sed -i 's/^#\?PermitRootLogin.*/PermitRootLogin no/' /etc/ssh/sshd_config
systemctl restart ssh
apt update && apt -y upgrade
apt -y install ufw fail2ban
ufw allow OpenSSH && ufw --force enable        # only SSH open; bridge is localhost-only
timedatectl set-ntp true                        # clock sync (guards use UTC+3 platform time)
```

Re-login as `trader` for everything below. **Enable linger** so user services run
without an active login and survive reboots:

```bash
sudo loginctl enable-linger trader
```

## 3. Install dependencies

```bash
sudo dpkg --add-architecture i386 && sudo apt update
sudo apt -y install --install-recommends winehq-stable || sudo apt -y install wine wine64 wine32
sudo apt -y install winbind xvfb x11vnc curl git build-essential
wine --version        # sanity: expect wine-8.x or 9.x
```

Install conda (Miniconda) and create the env:

```bash
curl -L https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/mc.sh
bash /tmp/mc.sh -b -p $HOME/miniconda3
$HOME/miniconda3/bin/conda create -y -n envmt5 python=3.10
```

## 4. Get the code + run setup

```bash
# copy the repo to the VPS (git clone your remote, OR rsync from the dev box):
#   rsync -av --exclude .git --exclude '*.log' MT5/ trader@VPS:~/MT5/
cd ~/MT5
$HOME/miniconda3/bin/conda run -n envmt5 pip install -r requirements.txt
# setup.sh installs Wine Python 3.10 + MetaTrader5 + rpyc inside the ~/.mt5 prefix.
# It needs a display -> run it under Xvfb:
export WINEPREFIX=$HOME/.mt5
xvfb-run -a -s "-screen 0 1280x1024x24" bash setup.sh
```

## 5. Virtual display (Xvfb) as a service

The MT5 GUI needs an X server. Create a persistent virtual display on `:99`:

```bash
mkdir -p ~/.config/systemd/user
cat > ~/.config/systemd/user/xvfb.service <<'EOF'
[Unit]
Description=Xvfb virtual display :99 for headless MT5
[Service]
ExecStart=/usr/bin/Xvfb :99 -screen 0 1280x1024x24 -nolisten tcp
Restart=always
[Install]
WantedBy=default.target
EOF
systemctl --user daemon-reload
systemctl --user enable --now xvfb.service
```

## 6. One-time MT5 account login (via VNC over SSH tunnel)

You must log into the **FundingPips** account once in the GUI and enable algo
trading. Do it over an encrypted tunnel — never expose VNC to the internet.

```bash
# on the VPS: attach a VNC server to the Xvfb display, bound to localhost only
DISPLAY=:99 x11vnc -localhost -rfbport 5900 -nopw -forever -bg
DISPLAY=:99 WINEPREFIX=$HOME/.mt5 WINEDEBUG=-all \
  wine "$HOME/.mt5/drive_c/Program Files/MetaTrader 5/terminal64.exe" &
```
```bash
# on YOUR laptop: tunnel + connect a VNC viewer to localhost:5900
ssh -L 5900:localhost:5900 trader@VPS
```
In the VNC session:
1. **File → Login to Trade Account** → enter the FundingPips credentials + server.
2. **Tools → Options → Expert Advisors → ✅ Allow algorithmic trading**.
3. Confirm the symbols exist (US30/NAS100/GER40/BTCUSD/XAGUSD/XAUUSD…) in Market Watch.
4. Close the terminal (the persistent service in step 7 will own it from now on).
5. `pkill x11vnc` when done.

## 7. Install the bot's systemd units (with VPS paths)

Copy `mt5-terminal.service`, `xau-basket-dry.{service,timer}` from
`~/.config/systemd/user/` on the dev box (or recreate them). Then **fix the paths
and display** for the VPS:

```bash
cd ~/.config/systemd/user
# repo path, conda path, and DISPLAY :0 -> :99, in the units AND the wrapper:
sed -i "s#/home/rock/Desktop/2026_Projects/Trader36/MT5#$HOME/MT5#g; \
        s#/home/rock/anaconda3#$HOME/miniconda3#g; \
        s#DISPLAY=:0#DISPLAY=:99#g" \
    mt5-terminal.service xau-basket-dry.service
sed -i "s#/home/rock/Desktop/2026_Projects/Trader36/MT5#$HOME/MT5#g; \
        s#/home/rock/anaconda3#$HOME/miniconda3#g" \
    ~/MT5/scripts/xau_basket_dry_cron.sh
# make mt5-terminal wait for Xvfb:
sed -i 's/^After=.*/After=xvfb.service\nRequires=xvfb.service/' mt5-terminal.service

systemctl --user daemon-reload
systemctl --user enable --now mt5-terminal.service      # keeps MT5 + bridge up
sleep 20
systemctl --user enable --now xau-basket-dry.timer      # hourly dry-run pass
```

Verify the terminal + bridge are alive:
```bash
ss -tlnp | grep 18812                                   # bridge listening (localhost)
$HOME/miniconda3/bin/conda run -n envmt5 python -c \
  "from src.core.mt5_connector import MT5Connector as C; c=C(); c.connect(); \
   a=c.account_info(); print('OK', a.login, a.balance, a.currency); c.disconnect()"
```

## 8. Map the FundingPips symbols + point at the account

Edit `configs/v5_basket_challenge.json`: fill each `fp_symbol` with the exact
Market-Watch name from step 6 (e.g. `"SPX": {"fp_symbol": "US500", ...}`,
`"BTC": {"fp_symbol": "BTCUSD", ...}`). Leave `contract_size` to the live
`symbol_info` (read automatically). Any symbol left `null` simply won't trade.

Reset the dry-run state so the anchor snapshots the FundingPips balance, not a
stale one:
```bash
rm -f ~/MT5/data/v5_runs/basket_challenge_dry_state.json
```

## 9. Dry-run FIRST, then go live

**Always dry-run on the real account for 1–2 sessions before `--execute`:**
```bash
cd ~/MT5
$HOME/miniconda3/bin/conda run -n envmt5 python scripts/v5_basket_challenge_exec.py \
  --state data/v5_runs/basket_challenge_dry_state.json \
  --paper-csv data/v5_runs/basket_challenge_dry_log.csv
```
Confirm: guards show `action=trade`, targets list real `tgt lot` (not UNMAPPED),
and the plan matches expectation.

**Go live** — edit `scripts/xau_basket_dry_cron.sh` (or make a new
`xau_basket_cron.sh`) to add `--live --execute`, and point it at a *live* state
file (`basket_challenge_state.json`). Keep the vol dial at 7% — **never raise it
mid-challenge.** Phase-2 promotion is manual after the firm issues new creds:
```bash
$HOME/miniconda3/bin/conda run -n envmt5 python scripts/v5_basket_challenge_exec.py \
  --state data/v5_runs/basket_challenge_state.json --advance-phase
```

## 10. Monitor & maintain

```bash
journalctl --user -u xau-basket-dry.service -f          # live pass output
tail -f ~/MT5/data/v5_runs/v5-basket-dry-cron.log
systemctl --user list-timers                            # cadence + last/next
tail ~/MT5/data/v5_runs/basket_challenge_dry_log.csv    # per-pass ledger
```

- **Inactivity rule (30 days):** the basket trades often enough, but add a watch
  alert if `action` stays flat > 21 days.
- **Auto-recovery:** `mt5-terminal.service` + `xvfb.service` restart on boot;
  if MT5 wedges (rare "no driver" after a Wine update) run
  `wineserver -k && systemctl --user restart mt5-terminal.service`.
- **Backups:** the only irreplaceable state is `data/v5_runs/*state*.json` and the
  MT5 login. `rsync` the `data/v5_runs/` dir off-box daily.
- **Updates:** avoid unattended Wine/MT5 upgrades mid-challenge; pin versions.
- **Clock:** keep `timedatectl` NTP on — the daily guard resets on platform
  midnight (UTC+3); a drifting clock mis-anchors the daily loss line.

## 11. Security checklist

- [ ] SSH key-only, root login disabled, `ufw` = only OpenSSH open.
- [ ] Bridge (18812) binds `127.0.0.1` — **never** expose it; no `ufw allow 18812`.
- [ ] VNC only via SSH tunnel (`-localhost`), killed after login.
- [ ] `fail2ban` on. Distinct FundingPips terminal — do NOT reuse the cent-account
      credentials/terminal on this box.
- [ ] Journal + CSV don't contain credentials (they don't) before sharing logs.

---

### Files this deployment touches
Engine `scripts/v5_basket_challenge.py` · executor `scripts/v5_basket_challenge_exec.py`
· config `configs/v5_basket_challenge.json` · guards `src/v5/challenge_guards.py`
· wrapper `scripts/xau_basket_dry_cron.sh` · units `mt5-terminal.service`,
`xvfb.service`, `xau-basket-dry.{service,timer}` · launcher `start_mt5.sh` · `setup.sh`.

_Plan rationale + pass-rate sims: CHALLENGEBOT.MD. Last updated 2026-07-15._
