sudo cp /tmp/mt5-eurusd.service /etc/systemd/system/
sudo cp /tmp/mt5-usdjpy.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now mt5-eurusd.service mt5-usdjpy.service


sudo loginctl enable-linger rock
systemctl --user daemon-reload
systemctl --user enable --now mt5-eurusd.service mt5-usdjpy.service

# Status
systemctl status mt5-eurusd mt5-usdjpy

# Live logs
journalctl -u mt5-eurusd -f
journalctl -u mt5-usdjpy -f

# Stop / start manually
systemctl stop mt5-eurusd
systemctl start mt5-eurusd

# Disable autostart
systemctl disable mt5-eurusd mt5-usdjpy
