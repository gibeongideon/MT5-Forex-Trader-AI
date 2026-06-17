"""
MT4 connection manager for Ubuntu/Wine via a file-based bridge.

MT4 has no official Python API (no mt5linux equivalent), and the ZeroMQ EA route is
unsupported under Wine. So automation goes through a tiny MQL4 EA (MQL4/Experts/PyBridge.mq4)
that exchanges plain-text files under the terminal's MQL4/Files/pybridge/ directory:

  Linux Python (this code)  ── writes cmd/<id>.req, reads res/<id>.res ──┐
                                                                          │  (shared folder
  PyBridge.mq4 inside terminal.exe under Wine (~/.mt4)  ──────────────────┘   = normal Linux dir)

This class mirrors MT5Connector's public interface so bots can target MT4 unchanged.
Hot reads (account/positions) are served from snapshot files the EA writes every ~100ms;
ticks, rates and order ops are request/response. No DLLs, no DLL-import permission required.
"""
from __future__ import annotations

import os
import time
import uuid
from collections import namedtuple
from pathlib import Path
from types import SimpleNamespace

import yaml
from dotenv import load_dotenv

load_dotenv()

_CONFIG_PATH = Path(__file__).parent.parent.parent / "config.yaml"

MT4Position = namedtuple(
    "MT4Position",
    "ticket symbol type volume price_open price_current sl tp profit magic comment",
)


def _load_config() -> dict:
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _resolve_bridge_dir(configured: str, wine_prefix: str) -> Path:
    """Return the absolute pybridge dir. If `configured` is an explicit path, use it; if it is
    'auto' (or missing), locate the terminal's MQL4/Files under the Wine prefix — handles both
    'Program Files/MetaTrader 4/MQL4/Files' and the Roaming/Terminal/<hash>/MQL4/Files layout,
    picking the most-recently-modified."""
    if configured and configured != "auto":
        return Path(os.path.expanduser(configured))
    root = Path(os.path.expanduser(wine_prefix)) / "drive_c"
    candidates = [p for p in root.glob("**/MQL4/Files") if p.is_dir()]
    if not candidates:
        # not found yet — return the conventional path so connect() can report a clear error
        return root / "Program Files" / "MetaTrader 4" / "MQL4" / "Files" / "pybridge"
    newest = max(candidates, key=lambda p: p.stat().st_mtime)
    return newest / "pybridge"


class _FileTransport:
    """Atomic-write command files, poll for result files terminated by an `eof=1` marker."""

    def __init__(self, bridge_dir: Path, timeout: float = 10.0, poll: float = 0.05):
        self.dir = Path(bridge_dir)
        self.cmd = self.dir / "cmd"
        self.res = self.dir / "res"
        self.timeout = timeout
        self.poll = poll
        for d in (self.cmd, self.res):
            d.mkdir(parents=True, exist_ok=True)

    def alive(self, max_age: float = 10.0) -> bool:
        acct = self.dir / "account.txt"
        return acct.exists() and (time.time() - acct.stat().st_mtime) < max_age

    @staticmethod
    def _parse_kv(text: str) -> dict:
        out = {}
        for line in text.splitlines():
            if "=" in line:
                k, _, v = line.partition("=")
                out[k.strip()] = v.strip()
        return out

    def request(self, payload: dict) -> dict:
        rid = uuid.uuid4().hex[:16]
        payload = {"id": rid, **payload}
        body = "".join(f"{k}={v}\n" for k, v in payload.items())
        tmp = self.cmd / f"{rid}.tmp"
        tmp.write_text(body)
        os.replace(tmp, self.cmd / f"{rid}.req")          # atomic publish

        res_path = self.res / f"{rid}.res"
        deadline = time.time() + self.timeout
        while time.time() < deadline:
            if res_path.exists():
                text = res_path.read_text()
                if text.rstrip().endswith("eof=1"):        # fully written
                    kv = self._parse_kv(text)
                    try:
                        res_path.unlink()
                    except OSError:
                        pass
                    if kv.get("ok") != "1":
                        raise RuntimeError(f"MT4 bridge error [{payload.get('action')}]: {kv.get('error', 'unknown')}")
                    return kv
            time.sleep(self.poll)
        raise TimeoutError(f"MT4 bridge timeout ({self.timeout}s) for action={payload.get('action')} "
                           "— is PyBridge EA attached and automated trading enabled?")


class MT4Connector:
    """High-level MT4 wrapper. Public interface identical to MT5Connector."""

    def __init__(self):
        cfg = _load_config()
        mt4 = cfg["mt4"]
        self.bridge_dir = _resolve_bridge_dir(mt4.get("file_bridge_dir", "auto"),
                                              mt4.get("wine_prefix", "~/.mt4"))
        self.login = int(os.getenv("MT4_LOGIN") or mt4.get("login") or 0)
        self.server = os.getenv("MT4_SERVER") or mt4.get("server") or ""
        self.timeout = float(mt4.get("request_timeout", 10.0))
        self._t = _FileTransport(self.bridge_dir, timeout=self.timeout)
        self._connected = False

    # ------------------------------------------------------------------ lifecycle
    def connect(self) -> None:
        if not self._t.alive():
            raise ConnectionError(
                f"MT4 file bridge not responding at {self.bridge_dir}\n"
                "Make sure:\n"
                "  1. MT4 terminal is running        (./start_mt4.sh)\n"
                "  2. PyBridge EA is attached to a chart (smiley face, not X)\n"
                "  3. Tools > Options > Expert Advisors > Allow automated trading is ON\n"
                "  4. You are logged in to the HFM MT4 account"
            )
        self._connected = True
        info = self.account_info()
        if info and getattr(info, "login", 0):
            print(f"Connected MT4: account={info.login}  balance={info.balance:.2f} {info.currency}  server={info.server}")
        else:
            print("Connected to MT4 bridge (account not yet logged in)")

    def disconnect(self) -> None:
        self._connected = False

    def reset_singleton(self) -> None:
        """File transport has no cached connection; re-create it so a fresh attempt re-checks the dir."""
        self._t = _FileTransport(self.bridge_dir, timeout=self.timeout)

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *_):
        self.disconnect()

    # ------------------------------------------------------------------ account (snapshot read)
    def account_info(self):
        acct = self.bridge_dir / "account.txt"
        if not acct.exists():
            return None
        kv = _FileTransport._parse_kv(acct.read_text())
        return SimpleNamespace(
            login=int(float(kv.get("login", 0))),
            balance=float(kv.get("balance", 0.0)),
            equity=float(kv.get("equity", 0.0)),
            margin_free=float(kv.get("margin_free", 0.0)),
            currency=kv.get("currency", ""),
            server=kv.get("server", self.server),
        )

    def account_balance(self) -> float:
        info = self.account_info()
        return info.balance if info else 0.0

    # ------------------------------------------------------------------ market data
    def get_rates(self, symbol: str, timeframe: str, count: int = 500):
        import pandas as pd
        kv = self._t.request({"action": "rates", "symbol": symbol,
                              "tf": timeframe.upper(), "count": count})
        csv_path = self.bridge_dir / "res" / kv.get("file", "")
        if not kv.get("file") or not csv_path.exists():
            raise RuntimeError(f"No rate data for {symbol} {timeframe}")
        df = pd.read_csv(csv_path)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.set_index("time", inplace=True)
        try:
            csv_path.unlink()
        except OSError:
            pass
        return df

    def get_tick(self, symbol: str):
        kv = self._t.request({"action": "tick", "symbol": symbol})
        return SimpleNamespace(
            ask=float(kv.get("ask", 0.0)), bid=float(kv.get("bid", 0.0)),
            last=float(kv.get("last", 0.0)), time=int(float(kv.get("time", 0))),
        )

    def symbol_info(self, symbol: str):
        kv = self._t.request({"action": "symbol", "symbol": symbol})
        return SimpleNamespace(
            point=float(kv.get("point", 0.0)),
            trade_tick_value=float(kv.get("tick_value", 0.0)),
            trade_tick_size=float(kv.get("tick_size", 0.0)),
            volume_min=float(kv.get("volume_min", 0.01)),
            volume_max=float(kv.get("volume_max", 100.0)),
            volume_step=float(kv.get("volume_step", 0.01)),
            digits=int(float(kv.get("digits", 5))),
        )

    # ------------------------------------------------------------------ orders & positions
    def open_position(self, symbol: str, order_type: str, volume: float,
                      sl: float = 0.0, tp: float = 0.0, comment: str = "", magic: int = 0) -> dict:
        kv = self._t.request({
            "action": "open", "symbol": symbol, "type": order_type,
            "volume": volume, "sl": sl, "tp": tp, "magic": magic, "comment": comment,
        })
        result = {"retcode": int(float(kv.get("retcode", 0))),
                  "order": int(float(kv.get("ticket", 0))),
                  "price": float(kv.get("price", 0.0))}
        print(f"Opened {order_type.upper()} {volume} {symbol} @ {result['price']}  ticket={result['order']}")
        return result

    def close_position(self, position) -> dict:
        kv = self._t.request({"action": "close", "ticket": position.ticket,
                              "volume": position.volume})
        print(f"Closed ticket={position.ticket} {position.symbol}")
        return {"retcode": int(float(kv.get("retcode", 0))), "order": position.ticket,
                "price": float(kv.get("price", 0.0))}

    def close_position_partial(self, position, volume: float) -> dict:
        kv = self._t.request({"action": "close", "ticket": position.ticket, "volume": volume})
        print(f"Partial-closed {volume} of ticket={position.ticket} {position.symbol}")
        return {"retcode": int(float(kv.get("retcode", 0))), "order": position.ticket,
                "price": float(kv.get("price", 0.0))}

    def get_positions(self, symbol: str | None = None, magic: int | None = None):
        pos_file = self.bridge_dir / "positions.csv"
        if not pos_file.exists():
            return []
        import csv
        out = []
        with open(pos_file) as f:
            for row in csv.DictReader(f):
                if not row.get("ticket"):
                    continue
                p = MT4Position(
                    ticket=int(row["ticket"]), symbol=row["symbol"], type=int(row["type"]),
                    volume=float(row["volume"]), price_open=float(row["price_open"]),
                    price_current=float(row["price_current"]), sl=float(row["sl"]),
                    tp=float(row["tp"]), profit=float(row["profit"]),
                    magic=int(row["magic"]), comment=row.get("comment", ""),
                )
                if symbol and p.symbol != symbol:
                    continue
                if magic is not None and p.magic != magic:
                    continue
                out.append(p)
        return out

    def modify_position(self, ticket: int, sl: float, tp: float) -> dict:
        kv = self._t.request({"action": "modify", "ticket": ticket, "sl": sl, "tp": tp})
        return {"retcode": int(float(kv.get("retcode", 0))), "order": ticket}

    def calc_lot_size(self, symbol: str, sl_pips: float, risk_pct: float = 0.01) -> float:
        info = self.symbol_info(symbol)
        account = self.account_info()
        if info is None or account is None:
            return 0.01
        pip_value = info.trade_tick_value * (info.point / info.trade_tick_size) if info.trade_tick_size else 0.0
        sl_value = sl_pips * pip_value * 10
        risk_amount = account.balance * risk_pct
        lot = risk_amount / sl_value if sl_value > 0 else info.volume_min
        lot = max(info.volume_min, min(info.volume_max, lot))
        step = info.volume_step or 0.01
        return round(round(lot / step) * step, 2)
