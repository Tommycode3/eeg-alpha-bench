"""Command-line interface: ``alphabench --eo EO_FILE --ec EC_FILE``."""

from __future__ import annotations

import argparse
import json
import sys

from .analysis import PASS, analyze
from .config import Config
from .io import READERS, load
from .report import to_json, write_report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="alphabench",
        description="Does this EEG setup see alpha? Compares an eyes-open and an "
                    "eyes-closed recording (Berger effect) and returns PASS / FAIL / INCONCLUSIVE.",
    )
    p.add_argument("--eo", required=True, help="eyes-open recording")
    p.add_argument("--ec", required=True, help="eyes-closed recording")
    p.add_argument("--format", default="auto", choices=["auto", "edf", "bdf", "fif", *READERS],
                   help="input format (default: guess from file)")
    p.add_argument("--sfreq", type=float, help="sampling rate (Hz) if the file does not say")
    p.add_argument("--units", default="uV", choices=["uV", "mV", "V", "nV"],
                   help="units of generic CSV data (default uV)")
    p.add_argument("--channels", help="comma-separated channels to analyse (default: all)")
    p.add_argument("--line-freq", type=float, default=50.0,
                   help="mains frequency for the notch, 50 or 60 (0 disables)")
    p.add_argument("--report", default="alphabench_report.html",
                   help="report path; .html (with figure) or .md")
    p.add_argument("--json", help="also write machine-readable results here")
    p.add_argument("--strict", action="store_true",
                   help="exit with status 1 unless the verdict is PASS (useful in hardware CI)")
    return p


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    kw: dict = {}
    if a.sfreq:
        kw["sfreq"] = a.sfreq
    if a.format in {"auto", "csv"}:
        kw["units"] = a.units
    eo = load(a.eo, a.format, **kw)
    ec = load(a.ec, a.format, **kw)
    cfg = Config(line_freq=a.line_freq or None)
    chans = [c.strip() for c in a.channels.split(",")] if a.channels else None
    res = analyze(eo, ec, cfg, channels=chans)
    write_report(res, a.report)
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(to_json(res), fh, indent=2)
    print(f"{res.verdict}")
    for line in res.reasons:
        print(f"  - {line}")
    print(f"report: {a.report}")
    return 1 if (a.strict and res.verdict != PASS) else 0


if __name__ == "__main__":
    sys.exit(main())
