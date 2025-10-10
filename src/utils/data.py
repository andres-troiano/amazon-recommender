"""Data utilities: download helpers for public datasets.

Provides a simple, dependency-light downloader with progress logging.
"""

from __future__ import annotations

import math
import shutil
from pathlib import Path
from typing import Optional

import requests
from loguru import logger


def download_file(url: str, dest_path: Path, chunk_size: int = 1 << 20) -> Path:
    """Download a file from `url` to `dest_path` with basic progress logging.

    Creates parent directories if needed. Overwrites existing file.
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading dataset from {url} → {dest_path}")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        downloaded = 0
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    percent = (downloaded / total) * 100
                    if downloaded == total or downloaded % (10 * chunk_size) < chunk_size:
                        logger.info(f"Downloaded {downloaded // (1<<20)}MB / {total // (1<<20)}MB ({percent:.1f}%)")
    logger.info("Download complete.")
    return dest_path


__all__ = ["download_file"]
