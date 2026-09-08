"""SQLite schema and lookup helpers for Store Releases."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


def connect(path: str | Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    return connection


def initialise_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA foreign_keys = ON;

        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS variant_aliases (
            alias TEXT NOT NULL,
            variant_index INTEGER NOT NULL,
            PRIMARY KEY(alias, variant_index)
        );
        """
    )


def set_metadata(connection: sqlite3.Connection, key: str, value: Any) -> None:
    payload = json.dumps(value, sort_keys=True)
    connection.execute(
        "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
        (key, payload),
    )


def get_metadata(connection: sqlite3.Connection, key: str, default: Any = None) -> Any:
    row = connection.execute("SELECT value FROM metadata WHERE key = ?", (key,)).fetchone()
    if row is None:
        return default
    return json.loads(row["value"])


def count_rows(connection: sqlite3.Connection, table: str) -> int:
    row = connection.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()
    return int(row["n"])
