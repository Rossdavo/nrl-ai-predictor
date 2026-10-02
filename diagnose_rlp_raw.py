#!/usr/bin/env python3
"""
diagnose_rlp_raw.py

One-purpose diagnostic for the Rugby League Project 2026 player page.

It DOES NOT:
- write player population CSVs
- alter predictions
- alter ratings
- alter any existing data file

It downloads the page once and prints small diagnostics showing how the
player table is represented in the raw HTTP response seen by GitHub Actions.
"""

from __future__ import annotations

import re
import sys
import time

import requests

URL = "https://www.rugbyleagueproject.org/seasons/nrl-2026/players.html"
CONNECT_TIMEOUT = 5
READ_TIMEOUT = 15

NEEDLES = [
    "ADDO-CARR",
    "Josh Addo-Carr",
    "PAR-19",
    "BURTON",
    "CBY-19",
    "All Players",
]

TAG_PATTERNS = [
    "<tr",
    "<td",
    "<table",
    "<div",
    "<script",
    "<li",
    "<span",
]


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def show_context(text: str, needle: str, radius: int = 700) -> None:
    lower = text.lower()
    idx = lower.find(needle.lower())

    print(f"\n===== CONTEXT: {needle} =====", flush=True)

    if idx < 0:
        print("NOT FOUND", flush=True)
        return

    start = max(0, idx - radius)
    end = min(len(text), idx + len(needle) + radius)
    snippet = text[start:end]

    # Keep the output readable without changing the underlying evidence.
    snippet = snippet.replace("\r", "\\r")
    print(snippet, flush=True)


def main() -> int:
    try:
        log("RLP RAW RESPONSE DIAGNOSTIC")
        log(f"Fetching: {URL}")

        response = requests.get(
            URL,
            headers={
                "User-Agent": "Mozilla/5.0 NRL-AI-Predictor diagnostic",
                "Accept": "text/html,application/xhtml+xml",
                "Connection": "close",
            },
            timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
        )
        response.raise_for_status()

        text = response.text

        log(f"HTTP status: {response.status_code}")
        log(f"Response URL: {response.url}")
        log(f"Content-Type: {response.headers.get('content-type', '')}")
        log(f"Characters: {len(text):,}")
        log(f"Bytes: {len(response.content):,}")

        print("\n===== NEEDLE CHECKS =====", flush=True)
        for needle in NEEDLES:
            idx = text.lower().find(needle.lower())
            print(
                f"{needle!r}: {'FOUND at ' + str(idx) if idx >= 0 else 'NOT FOUND'}",
                flush=True,
            )

        print("\n===== RAW TAG COUNTS =====", flush=True)
        lower = text.lower()
        for tag in TAG_PATTERNS:
            print(f"{tag}: {lower.count(tag)}", flush=True)

        print("\n===== OTHER STRUCTURE COUNTS =====", flush=True)
        checks = {
            "pipe characters |": text.count("|"),
            "PAR-number tokens": len(re.findall(r"\bPAR-\d+\b", text)),
            "CBY-number tokens": len(re.findall(r"\bCBY-\d+\b", text)),
            "player href fragments": lower.count("/players/"),
            "data- attributes": lower.count("data-"),
            "JSON script type": lower.count("application/json"),
            "JavaScript variables": lower.count("var "),
        }
        for label, count in checks.items():
            print(f"{label}: {count}", flush=True)

        # Most useful evidence first.
        for needle in ["ADDO-CARR", "PAR-19", "All Players"]:
            show_context(text, needle)

        print("\n===== FIRST 1500 RAW CHARACTERS =====", flush=True)
        print(text[:1500].replace("\r", "\\r"), flush=True)

        log("DIAGNOSTIC COMPLETE")
        return 0

    except Exception as exc:
        log(f"FAILED: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
