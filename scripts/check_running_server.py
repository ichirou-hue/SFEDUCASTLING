"""Detect an existing SFEDUCASTLING server on the local development port."""

from __future__ import annotations

import socket
from urllib.error import URLError
from urllib.request import urlopen


HOST = "127.0.0.1"
PORT = 8005


def main() -> int:
    try:
        with socket.create_connection((HOST, PORT), timeout=0.5):
            pass
    except OSError:
        return 1

    try:
        with urlopen(f"http://{HOST}:{PORT}/", timeout=1) as response:
            page = response.read(16384)
    except (OSError, URLError):
        return 2

    return 0 if b"<title>SFEDUCASTLING" in page else 2


if __name__ == "__main__":
    raise SystemExit(main())
