"""Single-file run reports: Markdown (text only) or HTML (with an inline PSD figure)."""

from __future__ import annotations

import base64
import html
import io as _io
import json
from datetime import datetime, timezone
from pathlib import Path


def _name(src: str) -> str:
    return Path(src).name or src

import numpy as np

from . import __version__
from .analysis import Result

COLUMNS = [
    ("n_eo", "clean EO", "{:.0f}"), ("n_ec", "clean EC", "{:.0f}"),
    ("alpha_eo_uv2", "alpha EO (uV^2)", "{:.2f}"), ("alpha_ec_uv2", "alpha EC (uV^2)", "{:.2f}"),
    ("reactivity_abs", "EC/EO abs", "{:.2f}"), ("reactivity_rel", "EC/EO rel", "{:.2f}"),
    ("specificity", "alpha/flank", "{:.2f}"),
    ("low_ratio_eo_ec", "EO/EC 1-4 Hz", "{:.2f}"),
    ("p_mwu", "p (MWU)", "{:.2g}"), ("cliffs_delta", "Cliff's d", "{:.2f}"),
    ("iaf_hz", "IAF (Hz)", "{:.2f}"), ("peak_db", "peak (dB)", "{:.1f}"),
    ("flags", "flags", "{}"), ("pass", "pass", "{}"),
]


def _fmt(v, f: str) -> str:  # noqa: ANN001
    if isinstance(v, float) and not np.isfinite(v):
        return "-"
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else ""
    return f.format(v)


def _threshold_lines(r: Result) -> list[str]:
    c = r.config
    return [
        f"clean epochs per condition >= {c.min_epochs} ({c.epoch_s:g} s; rejected if ptp < {c.flat_ptp_uv:g} uV, "
        f"ptp > {c.reject_ptp_uv:g} uV, or robust z of log ptp > {c.reject_z:g})",
        f"one-sided Mann-Whitney p < {c.alpha_level:g} / (number of eligible channels)",
        f"absolute alpha reactivity EC/EO >= {c.min_reactivity:g}",
        f"alpha specificity (alpha reactivity / flank-band reactivity) >= {c.min_specificity:g}",
        f"Cliff's delta >= {c.min_cliffs_delta:g}",
        f"eyes-closed alpha peak >= {c.min_peak_db:g} dB above the 1/f fit",
        "channel is not frontal (Fp/AF/F): ocular artifact",
    ]


def to_markdown(r: Result) -> str:
    lines = [f"# alphabench report: **{r.verdict}**", ""]
    lines += [f"- {x}" for x in r.reasons] + [""]
    lines += ["## Inputs", "",
              f"- Eyes open: `{_name(r.meta['eo_source'])}` ({r.meta['eo_duration_s']:.1f} s @ {r.meta['eo_sfreq']:g} Hz)",
              f"- Eyes closed: `{_name(r.meta['ec_source'])}` ({r.meta['ec_duration_s']:.1f} s @ {r.meta['ec_sfreq']:g} Hz)",
              f"- Alpha band {r.config.alpha_band[0]:g}-{r.config.alpha_band[1]:g} Hz, "
              f"band-pass {r.config.l_freq:g}-{r.config.h_freq:g} Hz, notch {r.config.line_freq or 'off'}", ""]
    if r.meta.get("missing_channels"):
        lines += [f"- Requested but not found: {', '.join(r.meta['missing_channels'])}", ""]
    lines += ["## Per-channel results", ""]
    hdr = ["channel"] + [h for _, h, _ in COLUMNS]
    lines.append("| " + " | ".join(hdr) + " |")
    lines.append("|" + "---|" * len(hdr))
    for ch, row in r.channels.iterrows():
        lines.append("| " + " | ".join([ch] + [_fmt(row[k], f) for k, _, f in COLUMNS]) + " |")
    lines += ["", "## PASS criteria (all must hold on at least one channel)", ""]
    lines += [f"- {x}" for x in _threshold_lines(r)]
    lines += ["", "Flags: `frontal` excluded from verdict; `eog_suspect` EO low-frequency power >= "
              f"{r.config.eog_low_ratio:g}x EC; `relative_only` relative alpha rose but absolute did not "
              "(denominator artifact); `few_epochs` not enough clean data.", ""]
    lines += [f"_alphabench {__version__}, {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC_", ""]
    return "\n".join(lines)


def _psd_png(r: Result) -> str:
    from .plotting import psd_figure

    ch = r.passing_channels or [c for c in r.channels.index[r.channels["eligible"]]] or list(r.channels.index)
    idx = [list(r.channels.index).index(c) for c in ch]
    eo, ec = np.nanmean(r.mean_psd_eo[idx], 0), np.nanmean(r.mean_psd_ec[idx], 0)
    fig = psd_figure(r.f, eo, ec, f"Mean PSD over {', '.join(ch)}", r.config.alpha_band)
    buf = _io.BytesIO()
    fig.savefig(buf, format="png")
    import matplotlib.pyplot as plt

    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


CSS = """body{font:15px/1.5 -apple-system,Segoe UI,Helvetica,Arial,sans-serif;max-width:1100px;margin:2em auto;padding:0 1em;color:#222}
table{border-collapse:collapse;font-size:13px}td,th{border-bottom:1px solid #ddd;padding:4px 8px;text-align:right}
th:first-child,td:first-child{text-align:left}.PASS{color:#1b7a3a}.FAIL{color:#b3261e}.INCONCLUSIVE{color:#a66a00}
tr.pass td{background:#eef7f0}code{background:#f4f4f4;padding:1px 4px}"""


def to_html(r: Result) -> str:
    e = html.escape
    rows = []
    for ch, row in r.channels.iterrows():
        cls = ' class="pass"' if row["pass"] else ""
        rows.append(f"<tr{cls}><td>{e(ch)}</td>" + "".join(
            f"<td>{e(_fmt(row[k], f))}</td>" for k, _, f in COLUMNS) + "</tr>")
    head = "<tr><th>channel</th>" + "".join(f"<th>{e(h)}</th>" for _, h, _ in COLUMNS) + "</tr>"
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>alphabench: {r.verdict}</title>
<style>{CSS}</style></head><body>
<h1>alphabench verdict: <span class="{r.verdict}">{r.verdict}</span></h1>
<ul>{''.join(f'<li>{e(x)}</li>' for x in r.reasons)}</ul>
<p>Eyes open: <code>{e(_name(r.meta['eo_source']))}</code> ({r.meta['eo_duration_s']:.1f} s @ {r.meta['eo_sfreq']:g} Hz)<br>
Eyes closed: <code>{e(_name(r.meta['ec_source']))}</code> ({r.meta['ec_duration_s']:.1f} s @ {r.meta['ec_sfreq']:g} Hz)</p>
<img alt="PSD eyes open vs eyes closed" style="max-width:100%" src="data:image/png;base64,{_psd_png(r)}">
<h2>Per-channel results</h2><table>{head}{''.join(rows)}</table>
<h2>PASS criteria (all must hold on at least one channel)</h2>
<ul>{''.join(f'<li>{e(x)}</li>' for x in _threshold_lines(r))}</ul>
<details><summary>Full configuration</summary><pre>{e(json.dumps(r.config.to_dict(), indent=2))}</pre></details>
<p><small>alphabench {__version__}, {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC</small></p>
</body></html>"""


def write_report(r: Result, path: str | Path) -> Path:
    path = Path(path)
    text = to_html(r) if path.suffix.lower() in {".html", ".htm"} else to_markdown(r)
    path.write_text(text, encoding="utf-8")
    return path


def to_json(r: Result) -> dict:
    df = r.channels.reset_index()
    return {
        "verdict": r.verdict, "reasons": r.reasons, "meta": r.meta,
        "config": r.config.to_dict(),
        "channels": json.loads(df.to_json(orient="records")),
        "version": __version__,
    }
