"""Small shared helpers: logging, timing, atomic file replacement, hashing."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

LOG = logging.getLogger("orderpulse")


def setup_logging(level: int = logging.INFO) -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@contextmanager
def timed(step: str) -> Iterator[None]:
    t0 = time.perf_counter()
    LOG.info("%s: start", step)
    try:
        yield
    except Exception:
        LOG.error("%s: failed after %.1fs", step, time.perf_counter() - t0)
        raise
    LOG.info("%s: done in %.1fs", step, time.perf_counter() - t0)


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def short_hash(obj: Any, n: int = 10) -> str:
    payload = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:n]


def tmp_path_for(target: Path) -> Path:
    return target.with_name(target.name + ".tmp")


def atomic_replace(tmp: Path, target: Path) -> None:
    """Move a fully written temporary file over the target in one step.

    On Windows this fails if another process (e.g. a running Streamlit app)
    holds the target open; the error is re-raised with a clear message and the
    previous target is left untouched.
    """
    try:
        os.replace(tmp, target)
    except PermissionError as exc:  # pragma: no cover - platform specific
        raise PermissionError(
            f"Cannot replace {target}: the file is in use (stop the Streamlit app and retry). "
            f"The new version is kept at {tmp}."
        ) from exc


def write_json_atomic(obj: Any, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = tmp_path_for(target)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
    atomic_replace(tmp, target)


def write_text_atomic(text: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = tmp_path_for(target)
    tmp.write_text(text, encoding="utf-8")
    atomic_replace(tmp, target)


def remove_if_exists(path: Path) -> None:
    if path.exists():
        path.unlink()
