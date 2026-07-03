"""
SQLite-backed trade journal.

Records every completed trade with full context: model used, confidence score,
entry/exit reason, P&L. Provides summary statistics and DataFrame export.

Usage:
    journal = TradeJournal()                    # default: data/trades.db
    journal = TradeJournal("data/my_trades.db")

    journal.record({
        "bot":        "random_bot",
        "symbol":     "EURUSD",
        "direction":  "buy",
        "entry_time": "2024-01-15 09:00:00",
        "entry_price": 1.08500,
        "exit_time":  "2024-01-15 10:15:00",
        "exit_price":  1.08800,
        "pnl_pips":   30.0,
        "pnl_dollars": 30.0,
        "model":      "random",
        "confidence": 0.5,
        "entry_reason": "random",
        "exit_reason":  "tp",
        "volume":     0.01,
        "sl_pips":    30.0,
        "tp_pips":    60.0,
    })

    df = journal.get_trades()
    journal.print_summary()
"""

import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd


_DEFAULT_DB = Path(__file__).parent.parent / "data" / "trades.db"

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS trades (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    bot           TEXT    NOT NULL DEFAULT '',
    symbol        TEXT    NOT NULL DEFAULT '',
    direction     TEXT    NOT NULL DEFAULT '',
    entry_time    TEXT,
    entry_price   REAL,
    exit_time     TEXT,
    exit_price    REAL,
    pnl_pips      REAL    DEFAULT 0,
    pnl_dollars   REAL    DEFAULT 0,
    model         TEXT    DEFAULT '',
    confidence    REAL    DEFAULT 0.5,
    entry_reason  TEXT    DEFAULT '',
    exit_reason   TEXT    DEFAULT '',
    volume        REAL    DEFAULT 0,
    sl_pips       REAL    DEFAULT 0,
    tp_pips       REAL    DEFAULT 0,
    magic         INTEGER,
    run_id        TEXT    DEFAULT '',
    dry_run       INTEGER DEFAULT 0,
    created_at    TEXT    DEFAULT (datetime('now'))
)
"""


class TradeJournal:

    def __init__(self, db_path: str | Path = _DEFAULT_DB):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ------------------------------------------------------------------ #

    def record(self, trade: dict) -> int:
        """Insert a completed trade. Returns the new row id."""
        cols = [
            "bot", "symbol", "direction",
            "entry_time", "entry_price",
            "exit_time", "exit_price",
            "pnl_pips", "pnl_dollars",
            "model", "confidence",
            "entry_reason", "exit_reason",
            "volume", "sl_pips", "tp_pips",
            "magic", "run_id", "dry_run",
        ]
        values = [trade.get(c) for c in cols]
        placeholders = ", ".join(["?"] * len(cols))
        col_list = ", ".join(cols)
        sql = f"INSERT INTO trades ({col_list}) VALUES ({placeholders})"
        with self._connect() as conn:
            cur = conn.execute(sql, values)
            return cur.lastrowid

    def get_trades(self, bot: str = None, symbol: str = None) -> pd.DataFrame:
        """Return all trades as a DataFrame, optionally filtered."""
        conditions, params = [], []
        if bot:
            conditions.append("bot = ?")
            params.append(bot)
        if symbol:
            conditions.append("symbol = ?")
            params.append(symbol)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        sql = f"SELECT * FROM trades {where} ORDER BY id"
        with self._connect() as conn:
            return pd.read_sql_query(sql, conn, params=params)

    def print_summary(self, bot: str = None, symbol: str = None) -> None:
        df = self.get_trades(bot=bot, symbol=symbol)
        if df.empty:
            print("No trades recorded yet.")
            return

        wins = df[df["pnl_pips"] > 0]
        losses = df[df["pnl_pips"] <= 0]
        win_rate = len(wins) / len(df) * 100
        gross_profit = wins["pnl_dollars"].sum()
        gross_loss = abs(losses["pnl_dollars"].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        print(f"\n{'─' * 50}")
        print(f"  Trade Journal Summary")
        if bot:
            print(f"  Bot: {bot}")
        print(f"{'─' * 50}")
        print(f"  Total trades  : {len(df)}")
        print(f"  Win rate      : {win_rate:.1f}%  ({len(wins)}W / {len(losses)}L)")
        print(f"  Avg pnl (pips): {df['pnl_pips'].mean():.1f}")
        print(f"  Profit factor : {profit_factor:.2f}")
        print(f"  Total P&L ($) : {df['pnl_dollars'].sum():+.2f}")
        print(f"{'─' * 50}\n")

    # ------------------------------------------------------------------ #

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(_CREATE_TABLE)
            self._ensure_columns(conn)

    @staticmethod
    def _ensure_columns(conn) -> None:
        existing = {row[1] for row in conn.execute("PRAGMA table_info(trades)").fetchall()}
        migrations = {
            "magic": "ALTER TABLE trades ADD COLUMN magic INTEGER",
            "run_id": "ALTER TABLE trades ADD COLUMN run_id TEXT DEFAULT ''",
            "dry_run": "ALTER TABLE trades ADD COLUMN dry_run INTEGER DEFAULT 0",
        }
        for column, sql in migrations.items():
            if column not in existing:
                conn.execute(sql)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)
