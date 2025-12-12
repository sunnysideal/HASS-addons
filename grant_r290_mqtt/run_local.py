#!/usr/bin/env python3
"""Helper to load .env-style settings and run the MQTT publisher locally."""

import os
from pathlib import Path


def load_env(paths):
    """Load key=value pairs into the current process environment."""
    for path in paths:
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            os.environ[key] = value


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    repo_root = here.parent

    # Prefer .env at repo root; fall back to one alongside this script
    load_env([repo_root / ".env", here / ".env"])

    import mqtt_publisher

    mqtt_publisher.main()
