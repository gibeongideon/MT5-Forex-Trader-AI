#!/bin/bash
# vps_provision.sh — replicate the basket-challenge bot on a fresh VPS.
# Run FROM the desktop (drives the VPS over ssh). Idempotent-ish; safe to re-run.
# Captures the WORKING sequence from the first deploy (68.183.91.240) with every
# gotcha fixed — see deploy/vps-deploy-log.md for the war stories.
#
#   ./deploy/vps_provision.sh <VPS_IP> [SSH_KEY]
#
# Requires: a fresh Ubuntu 22.04/24.04 x86_64 VPS with your SSH pubkey on root.
# One MANUAL step remains (MT5 broker login via VNC) — the script pauses for it.
set -euo pipefail

VPS_IP="${1:?usage: vps_provision.sh <VPS_IP> [SSH_KEY]}"
KEY="${2:-$HOME/.ssh/vps_basket_ed25519}"
REPO="/home/rock/Desktop/2026_Projects/Trader36/MT5"
PUBKEY="$(cat "$KEY.pub")"
MYIP="$(ssh -i "$KEY" -o StrictHostKeyChecking=accept-new root@"$VPS_IP" 'echo $SSH_CLIENT' | awk '{print $1}')"
R="ssh -i $KEY -o BatchMode=yes root@$VPS_IP"
T="ssh -i $KEY -o BatchMode=yes trader@$VPS_IP"
say(){ echo -e "\n=== $* ==="; }

say "1. swap (MT5+Wine peaks ~2.4G) + trader user + i386"
$R bash -s <<EOF
set -e
swapon --show | grep -q /swapfile || { fallocate -l 2G /swapfile; chmod 600 /swapfile; mkswap /swapfile >/dev/null; swapon /swapfile; grep -q /swapfile /etc/fstab || echo '/swapfile none swap sw 0 0' >>/etc/fstab; }
id trader &>/dev/null || { adduser --disabled-password --gecos "" trader; usermod -aG sudo trader; echo 'trader ALL=(ALL) NOPASSWD:ALL' >/etc/sudoers.d/trader-nopasswd; chmod 440 /etc/sudoers.d/trader-nopasswd; }
install -d -m700 -o trader -g trader /home/trader/.ssh
echo "$PUBKEY" >/home/trader/.ssh/authorized_keys; chown trader:trader /home/trader/.ssh/authorized_keys; chmod 600 /home/trader/.ssh/authorized_keys
loginctl enable-linger trader
export DEBIAN_FRONTEND=noninteractive; dpkg --add-architecture i386; apt-get update -q >/dev/null
EOF

say "2. Wine 11 (winehq) + Xvfb + x11vnc + tools  [~5 min]"
$R bash -c '
set -e; export DEBIAN_FRONTEND=noninteractive
mkdir -p /etc/apt/keyrings
wget -qO /etc/apt/keyrings/winehq-archive.key https://dl.winehq.org/wine-builds/winehq.key
. /etc/os-release; wget -qNP /etc/apt/sources.list.d/ https://dl.winehq.org/wine-builds/ubuntu/dists/$VERSION_CODENAME/winehq-$VERSION_CODENAME.sources
apt-get update -q >/dev/null
apt-get install -y --install-recommends winehq-stable >/dev/null 2>&1 || apt-get install -y wine wine64 wine32 >/dev/null 2>&1
apt-get install -y winbind xvfb x11vnc curl git build-essential rsync cabextract iconv >/dev/null 2>&1 || apt-get install -y winbind xvfb x11vnc curl git build-essential rsync cabextract >/dev/null 2>&1
wine --version'

say "3. Miniconda + envmt5 (python 3.10) + requirements  [conda ToS + anaconda3 symlink!]"
# rsync repo FIRST (EXCLUDE secrets .env/config.yaml + heavy unused files)
rsync -az --delete -e "ssh -i $KEY -o BatchMode=yes" \
  --exclude '.git' --exclude '*.log' --exclude 'logs/' --exclude '.env' --exclude 'config.yaml' \
  --exclude 'XAUUSD_M15_long.csv' --exclude 'XAUUSD_M30_long.csv' --exclude 'live_trades.db' --exclude '__pycache__' \
  "$REPO/" "trader@$VPS_IP:/home/trader/MT5/"
$T bash -s <<'EOF'
set -e; cd ~
curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/mc.sh
[ -d ~/miniconda3 ] || bash /tmp/mc.sh -b -p $HOME/miniconda3
export PATH=$HOME/miniconda3/bin:$PATH
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true
conda env list | grep -q envmt5 || conda create -y -n envmt5 python=3.10 >/dev/null
$HOME/miniconda3/envs/envmt5/bin/pip install -q -r $HOME/MT5/requirements.txt
ln -sfn ~/miniconda3 ~/anaconda3   # setup.sh + wrappers hardcode ~/anaconda3
EOF

say "4. Wine prefix (BOOT FIRST!) + Wine Python + MetaTrader5 + rpyc"
$T bash -s <<'EOF'
set -e; export DISPLAY=:99 WINEPREFIX=$HOME/.mt5 WINEARCH=win64 WINEDEBUG=-all WINEDLLOVERRIDES="mscoree,mshtml="
mkdir -p ~/.config/systemd/user
cat > ~/.config/systemd/user/xvfb.service <<U
[Unit]
Description=Xvfb :99
[Service]
ExecStart=/usr/bin/Xvfb :99 -screen 0 1280x1024x24 -nolisten tcp
Restart=always
[Install]
WantedBy=default.target
U
export XDG_RUNTIME_DIR=/run/user/$(id -u); systemctl --user daemon-reload; systemctl --user enable --now xvfb.service; sleep 2
# CRITICAL: boot the prefix before any installer (else kernel32 c0000135)
[ -f ~/.mt5/drive_c/windows/system32/kernel32.dll ] || { wineboot --init >/dev/null 2>&1; wineserver -w; }
PY=/tmp/python-3.10.11-amd64.exe
[ -f ~/.mt5/drive_c/Python310/python.exe ] || { curl -fsSL https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe -o $PY; wine $PY /quiet InstallAllUsers=0 TargetDir='C:\Python310' PrependPath=0 Include_test=0 >/dev/null 2>&1; wineserver -w; }
wine ~/.mt5/drive_c/Python310/python.exe -m pip install -q MetaTrader5 rpyc mt5linux >/dev/null 2>&1
wine ~/.mt5/drive_c/Python310/python.exe -c "import MetaTrader5,rpyc; print('wine MT5+rpyc OK')"
EOF

say "5. Copy MT5 terminal + servers.dat from desktop (MetaQuotes CDN is 404)"
DTP="$HOME/.mt5/drive_c/Program Files/MetaTrader 5"
VDP="/home/trader/.mt5/drive_c/Program Files/MetaTrader 5"
$T "mkdir -p '$VDP/Config'"
rsync -az -e "ssh -i $KEY -o BatchMode=yes" --exclude 'Bases/' --exclude 'logs/' --exclude 'Config/' \
  --exclude 'Profiles/' --exclude 'MetaEditor64.exe' --exclude 'metatester64.exe' --exclude 'uninstall.exe' \
  "$DTP/" "trader@$VPS_IP:$VDP/"
scp -i "$KEY" "$DTP/Config/servers.dat" "trader@$VPS_IP:$VDP/Config/servers.dat"   # broker address book (NOT accounts.dat)

say "6. Launcher + persistent terminal service + path symlink + connector attach-patch"
$T bash -s <<'EOF'
set -e; export XDG_RUNTIME_DIR=/run/user/$(id -u)
# hardcoded /home/rock path -> symlink to this repo
sudo mkdir -p /home/rock/Desktop/2026_Projects/Trader36; sudo ln -sfn /home/trader/MT5 /home/rock/Desktop/2026_Projects/Trader36/MT5
sudo chmod 755 /home/rock /home/rock/Desktop /home/rock/Desktop/2026_Projects /home/rock/Desktop/2026_Projects/Trader36
# connector: ATTACH (omit path) — never pass path/login to a running terminal (IPC timeout)
python3 - <<'PY'
p="/home/trader/MT5/src/core/mt5_connector.py"; s=open(p).read()
o='        kwargs: dict = {"path": self.terminal_path, "timeout": self.timeout}'
n='        kwargs: dict = {"timeout": self.timeout}\n        if self.terminal_path:\n            kwargs["path"] = self.terminal_path'
open(p,"w").write(s.replace(o,n)) if o in s else None
PY
# config.yaml (fresh, attach-mode) + demo .env (fill in) — NEVER from desktop
cat > /home/trader/MT5/config.yaml <<Y
mt5:
  terminal_path: ""
  wine_prefix: "/home/trader/.mt5"
  login: 0
  password: ""
  server: ""
  timeout: 60000
Y
cat > /home/trader/MT5/.env <<E
MT5_LOGIN=
MT5_PASSWORD=
MT5_SERVER=
E
chmod 600 /home/trader/MT5/.env
# headless launcher (terminal Program Files + rpyc bridge; pgrep -f, not -c!)
cat > /home/trader/MT5/scripts/start_mt5_vps.sh <<'S'
#!/bin/bash
set -u; export DISPLAY=:99 WINEPREFIX=$HOME/.mt5 WINEARCH=win64 WINEDEBUG=-all WINEDLLOVERRIDES="mscoree,mshtml="
TE="$HOME/.mt5/drive_c/Program Files/MetaTrader 5/terminal64.exe"; WP="$HOME/.mt5/drive_c/Python310/python.exe"; P=18812
pgrep -f terminal64.exe >/dev/null || { wine "$TE" >/dev/null 2>&1 & sleep 25; }
ss -tlnp 2>/dev/null | grep -q "127.0.0.1:$P" || { wine "$WP" -c "from rpyc.utils.server import ThreadedServer; from rpyc.core import SlaveService; ThreadedServer(SlaveService, hostname='127.0.0.1', port=$P, reuse_addr=True).start()" >/dev/null 2>&1 & sleep 4; }
S
chmod +x /home/trader/MT5/scripts/start_mt5_vps.sh
cat > ~/.config/systemd/user/mt5-terminal.service <<U
[Unit]
Description=Persistent MT5 terminal + rpyc bridge
After=xvfb.service
Requires=xvfb.service
[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/home/trader/MT5
ExecStart=/home/trader/MT5/scripts/start_mt5_vps.sh
ExecStop=/bin/bash -c 'pkill -f "rpyc.*18812"; pkill -f terminal64.exe; wineserver -k || true'
KillMode=none
TimeoutStartSec=180
[Install]
WantedBy=default.target
U
systemctl --user daemon-reload; systemctl --user enable --now mt5-terminal.service; sleep 25
echo "terminal: $(pgrep -f terminal64.exe | wc -l) | bridge: $(ss -tlnp 2>/dev/null | grep -q 127.0.0.1:18812 && echo UP || echo down)"
EOF

cat <<MANUAL

=== 7. MANUAL STEP — one-time MT5 broker login (VNC) ===
The fresh terminal can't auto-login until it's connected once. On the VPS run:
  DISPLAY=:99 x11vnc -display :99 -localhost -rfbport 5900 -passwd 'Vnc4Login!' -forever -bg
On your desktop:
  ssh -i $KEY -L 5901:localhost:5900 trader@$VPS_IP    # keep open
  vncviewer localhost:5901                             # pw Vnc4Login!
In MT5: File->Login (your demo Login/Server, tick SAVE ACCOUNT) + enable Algo Trading.
Then: pkill x11vnc. The account is saved -> headless auto-reconnects on every restart.

=== 8. after login, wire the dry-run timer + harden ===
  $T 'systemctl --user enable --now xau-basket-dry.timer'   # (create units per repo)
  fail2ban ignoreip=$MYIP, ufw allow 22/tcp, PasswordAuthentication no (sshd -t THEN restart)
MANUAL
echo "PROVISION AUTOMATED PORTION DONE for $VPS_IP"
