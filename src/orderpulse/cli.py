"""Command line entry point: ``python -m orderpulse <command>``."""

from __future__ import annotations

import argparse
import subprocess
import sys

from .config import PACKAGE_ROOT, load_config
from .utils import setup_logging, timed

STEPS = ["ingest", "quality", "marts", "train", "evaluate", "predict"]


def _run_step(step: str, cfg) -> None:
    if step == "ingest":
        from .ingest import ingest
        ingest(cfg)
    elif step == "quality":
        from .quality import run_quality
        run_quality(cfg)
    elif step == "marts":
        from .build_marts import build_warehouse
        build_warehouse(cfg)
    elif step == "train":
        from .train import train
        train(cfg)
    elif step == "evaluate":
        from .evaluate import evaluate
        evaluate(cfg)
    elif step == "predict":
        from .predict import predict
        predict(cfg)
    elif step == "tail-audit":
        from .reliability import EXP_DIR
        from .tail_audit import tail_audit
        tail_audit(cfg, cfg.root / EXP_DIR)
    elif step == "reliability":
        from .reliability import run_reliability
        run_reliability(cfg)
    else:  # pragma: no cover
        raise ValueError(step)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="orderpulse", description="OrderPulse pipeline")
    parser.add_argument("command", choices=STEPS + ["pipeline", "app", "test", "tail-audit", "reliability"])
    parser.add_argument("--root", help="project root (default: repository root or $ORDERPULSE_ROOT)")
    parser.add_argument("--config-dir", help="directory with data.yaml and model.yaml")
    parser.add_argument("--raw-dir", help="directory with the Olist CSV files (overrides data.yaml)")
    args = parser.parse_args(argv)
    setup_logging()

    if args.command == "app":
        return subprocess.call([sys.executable, "-m", "streamlit", "run", str(PACKAGE_ROOT / "app" / "main.py")],
                               cwd=PACKAGE_ROOT)
    if args.command == "test":
        return subprocess.call([sys.executable, "-m", "pytest"], cwd=PACKAGE_ROOT)

    cfg = load_config(root=args.root, config_dir=args.config_dir, raw_dir=args.raw_dir)
    try:
        if args.command == "pipeline":
            with timed("pipeline"):
                for step in STEPS:
                    _run_step(step, cfg)
        else:
            _run_step(args.command, cfg)
    except Exception as exc:  # expected setup errors get a short hint instead of a traceback
        hint = _setup_hint(exc, cfg)
        if hint is None:
            raise
        print(f"orderpulse {args.command}: {hint}", file=sys.stderr)
        return 2
    return 0


def _setup_hint(exc: Exception, cfg) -> str | None:
    """Readable message for missing CSV, database or artifacts (None for any other error)."""
    import duckdb

    from .ingest import SchemaError

    if isinstance(exc, SchemaError):
        missing = str(exc).count("missing file")
        if missing:
            return (f"{missing} Olist CSV file(s) not found in {cfg.raw_dir}. Download the dataset "
                    "(see README, 'Где взять данные') and put the CSV there, or pass --raw-dir <folder>.")
        return f"input data contract violated: {exc}"
    if isinstance(exc, (FileNotFoundError, duckdb.IOException)):
        return (f"{exc}. Earlier steps have not been run: build everything with "
                "`python -m orderpulse pipeline` (needs the Olist CSV).")
    return None


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
