# VPS deploy action log — basket challenge bot

Target VPS: 64.227.159.213 (first deploy). Every command + outcome recorded here
so a second VPS is a replay of `deploy/vps_provision.sh`. Key: ~/.ssh/vps_basket_ed25519.

## Actions

### 2026-07-15 INCIDENT — SSH went down during hardening
- Connected OK as root (Ubuntu 22.04.5, x86_64, 4GB RAM/3GB swap, 2 cores).
- Created `trader` user (sudo NOPASSWD, key auth, linger) — VERIFIED working.
- Ran hardening heredoc (sed sshd_config, `systemctl restart ssh`, apt update,
  install ufw/fail2ban, ufw enable). Local ssh client hit the 2-min Bash timeout
  DURING apt → remote chain SIGHUP'd partway. Afterward: **port 22 connection
  REFUSED** (sshd not listening). Box is UP (RST), sshd DOWN.
- ROOT-CAUSE candidates: sshd restart left it down; OR fail2ban banned the
  desktop IP after the earlier username-probe failures; OR ufw enabled w/o 22.
- RECOVERY: via DigitalOcean web console (out-of-band).

### LESSONS for the replica provisioning script (avoid this)
1. NEVER restart sshd inside a long apt command; run `sshd -t` FIRST, restart in
   its own quick step, re-verify connectivity before anything else.
2. Keep a root session policy: allow 22 in ufw BEFORE `ufw enable`; use
   `ufw allow 22/tcp` explicitly (not only the OpenSSH profile).
3. Whitelist the admin IP in fail2ban (`ignoreip`) BEFORE enabling it, and enable
   it LAST, after SSH access is confirmed stable.
4. Run apt installs in the background or with a long (10-min) timeout, separate
   from anything that touches sshd/ufw.

### 2026-07-15 FRESH DEPLOY on 68.183.91.240 (clean box)
- Ubuntu 24.04.4 LTS, x86_64, 4GB RAM, 0 swap, 2 cores, 75G free. Only sshd running.
- Admin desktop IP (fail2ban ignore): 102.0.5.216
- Lessons applied: swap added; sshd hardening deferred to LAST + tested; heavy apt in background.

### Wine setup gotchas (Ubuntu 24.04, wine-11.0) — for the replica
- conda (new Miniconda) needs `conda tos accept` for pkgs/main + pkgs/r before `conda create`.
- setup.sh hardcodes `~/anaconda3` — symlink `~/anaconda3 -> ~/miniconda3`.
- setup.sh ran the Windows-Python installer on an un-booted prefix → `wine: could not
  load kernel32.dll (c0000135)`. FIX: `WINEARCH=win64 WINEDLLOVERRIDES="mscoree,mshtml="
  wineboot --init` (+ `wineserver -w`) FIRST, then install. Skips mono/gecko hang too.

### GOTCHA: wine process comm = "main" (not terminal64.exe)
- Under wine-11, the MT5 terminal's process `comm` shows as "main", so
  `pgrep -c terminal64.exe` (comm match) returns 0 even while it runs. Use
  `pgrep -f terminal64.exe` (full cmdline). The launcher already uses -f. Hours
  lost chasing a phantom "terminal exits" — it never did; the LiveUpdate churn +
  bad pgrep misled diagnosis.

### SECURITY: exclude secrets from rsync (IMPORTANT for replica)
- rsync carried desktop `.env` (LIVE cent creds + ANTHROPIC_API_KEY) + config.yaml
  to the VPS. FIX: always `--exclude '.env' --exclude 'config.yaml'` in the sync,
  then write fresh demo-only .env (chmod 600) + patch config.yaml on the target.
- ACTION: rotate the exposed Anthropic API key.

### MT5 headless login: ini auto-login INSUFFICIENT on a fresh terminal
- A brand-new MT5 terminal has no broker server address-book, so it can't resolve
  the server name (HFMarketsKE-Demo2) — log shows terminal starts but ZERO
  network/login activity. `initialize()` -> (-10005 IPC timeout).
- FIX (one-time): VNC in (x11vnc on :99, localhost:5900 via SSH tunnel), manually
  File->Login (save account) + enable Algo Trading. Terminal then stores servers +
  account; headless auto-reconnect works on every restart after.
- For the replica: either do this one-time VNC login, OR copy a prewarmed
  ~/.mt5 prefix that already has the broker servers + saved account.

### THE KEY FIX: connector must ATTACH, not relaunch (mt5.initialize path arg)
- With a terminal already running + logged in, `mt5.initialize(path=..., login=...)`
  tries to relaunch/switch -> (-10005 IPC timeout). Raw `initialize()` (no args)
  attaches instantly and works.
- FIX: patched src/core/mt5_connector.py to omit `path` when terminal_path is empty;
  set config.yaml mt5.terminal_path: "" -> pure attach. All bots then connect.
- Login that finally worked: 57482374 @ HFMarketsKE-Demo2 (the HFM demo DID work once
  servers.dat was copied + user logged in via VNC with save-account). $10M KES demo,
  trade_allowed=True. Attach mode means .env creds can stay empty on the VPS.

### ✅ DEPLOY COMPLETE (68.183.91.240) — 2026-07-15
- Full stack live: Ubuntu 24.04 / trader / 2G swap / Wine 11 / Xvfb :99 / envmt5 / repo.
- MT5 terminal + rpyc bridge (127.0.0.1:18812), persistent mt5-terminal.service,
  AUTO-RECONNECTS after restart (verified) — survives reboots.
- Demo 57482374 @ HFMarketsKE-Demo2 ($10M KES, trade_allowed) connected via ATTACH mode.
- Basket executor runs on VPS: connect -> state(init 10M, +8% target) -> guards(action=trade)
  -> 12 targets computed -> dry-run (unmapped, no orders). Runs INDEPENDENTLY via
  xau-basket-dry.timer (hourly :46, systemd-triggered pass verified).
- Hardened: ufw (22 only; bridge+VNC localhost), fail2ban (ignoreip admin), sshd key-only.
- Replayable: deploy/vps_provision.sh (all fixes baked in; excludes secrets).
- REMAINING for LIVE: fill fp_symbol names in configs/v5_basket_challenge.json + set
  MODEL, add --live --execute to the wrapper, reset state. Symbols currently unmapped=dry.
- Old app box 64.227.159.213 still has the `trader` user I created (cleanup rejected).

### 🟢 LIVE TRADING on the demo (2026-07-15 ~17:57 UTC)
- Symbols mapped (10/12; crypto BTC/ETH not on HFM demo): XAUUSD, XAGUSD, US500.F,
  US100.F, US30.F, GER40, UK100, EU50.F, JPN225, AUS200.
- FILLING MODE: HFM=FOK (connector _fill_type already auto-detects). IOC -> 10030.
- SIZING FIX: account is KES; patched target_lots to use order_calc_margin (account
  ccy) + account leverage -> correct lots (was ~130x oversized). Below-min-lot sleeves
  skip. (FundingPips is USD so no conversion needed there, but the fix is general.)
- LIVE PASS: opened 10 long positions (retcode 10009), margin level 22595% (safe),
  floating -3110 KES (entry spread). Reconcile HOLDS when in sync (no churn).
- AUTONOMOUS: xau-basket-dry.timer now runs scripts/vps_basket_live_cron.sh
  (--live --execute, state basket_challenge_live_state.json) hourly :46.
- Sizing note: on a ~$77k-equiv acct @7% vol some sleeves round near min-lot;
  bigger USD account (FundingPips $100k) sizes cleaner.

### PROTECTION + REPORTING added (2026-07-15)
- REAL-TIME guard: executor `--guard-only` mode (check equity vs daily/overall limits,
  flatten on breach, skip reconcile). Timer `xau-basket-guard.timer` every 60s.
  -> max unmonitored exposure 60s (was 60min). Uses EQUITY incl floating (firm basis).
- Cadence stack: guard 1-min · reconcile hourly :46 · daily report 20:55 UTC.
- DAILY EMAIL: `scripts/vps_daily_report.py` -> kipngenol@gmail.com. Reports balance/
  equity, today gain, total P&L, phase progress %, and RULE ADHERENCE % (daily-loss +
  max-loss budget used/headroom, worst intraday DD from guard log, violations flag).
  SMTP creds in ~/MT5/.env.mail (600) — Gmail app password TO FILL. Timer xau-basket-report.

### 2nd INSTANCE — live CENT account migrated to VPS (2026-07-15)
- Two independent MT5 instances on one VPS:
  * inst1: prefix ~/.mt5,  Xvfb :99,  bridge 18812 -> DEMO 57482374 (basket challenge)
  * inst2: prefix ~/.mt5b, Xvfb :100, bridge 18813 -> LIVE CENT 54939391 (dual ls+champ)
- Enabler: MT5Connector honors MT5_BRIDGE_PORT env (attach mode, no path). ~/.mt5b =
  rsync copy of ~/.mt5 with Config/accounts.dat + mt5_login.ini removed (so it doesn't
  grab the demo session). servers.dat kept -> knows HFMarketsKE-Live2.
- Separate display :100 for cent so the one-time VNC login can't hit the demo terminal.
- Services: mt5-terminal-cent.service (persistent, :100/18813), xau-dual-cent.timer
  (Mon-Fri hourly :16) -> scripts/vps_xau_dual_cent_cron.sh (MT5_BRIDGE_PORT=18813,
  --live --execute ls+champ). Verified: recognizes existing 360541 short, PLAN in sync.
- CUTOVER: desktop `xau-dual.timer` DISABLED (was trading cent); VPS now sole live trader.
- NOTE: desktop mt5-terminal still logged into cent (harmless, dry-only timers). If broker
  session tug-of-war appears, log the desktop terminal out of cent. RAM on 4GB: two MT5
  instances — monitor (idle ~800MB-1.5GB; launch peaks strain, 2GB swap covers).

### Vol-targeting added to basket (2026-07-16)
- Engine `v5_basket_challenge.py`: VOL_TARGET on. `risk_scalar()` = causal (trailing EWMA hl20,
  shifted) vol-target x drawdown-scaler on the book's own returns; `target_leverage()` scales
  all per-symbol leverages by the latest scalar. Backtest: eval SR 1.27->1.43, FP pass 92%->94%
  @7%, faster. Synced to VPS; live reconcile adjusted to vol-targeted leverages (retcode 10009).
