import json

import numpy as np
import pandas as pd
import pytest

from alphabench import load
from alphabench.cli import main
from alphabench.io import detect_format
from alphabench.synthetic import make_pair


def _write_csv(rec, path):
    df = pd.DataFrame(rec.data.T, columns=rec.ch_names)
    df.insert(0, "time", np.arange(rec.data.shape[1]) / rec.sfreq)
    df.to_csv(path, index=False)


def test_generic_csv_roundtrip(tmp_path):
    eo, _ = make_pair(duration_s=10, sfreq=200.0)
    _write_csv(eo, tmp_path / "eo.csv")
    rec = load(tmp_path / "eo.csv")
    assert rec.sfreq == pytest.approx(200.0)
    assert rec.ch_names == eo.ch_names
    np.testing.assert_allclose(rec.data, eo.data, rtol=1e-9, atol=1e-9)


def test_csv_units(tmp_path):
    pd.DataFrame({"time": np.arange(10) / 100, "Cz": np.ones(10) * 1e-5}).to_csv(tmp_path / "v.csv", index=False)
    assert load(tmp_path / "v.csv", units="V").data[0, 0] == pytest.approx(10.0)


def test_muse_muselsl(tmp_path):
    n = 512
    df = pd.DataFrame({"timestamps": 1.7e9 + np.arange(n) / 256, "TP9": 1.0, "AF7": 2.0,
                       "AF8": 3.0, "TP10": 4.0, "Right AUX": 0.0})
    df.to_csv(tmp_path / "m.csv", index=False)
    assert detect_format(tmp_path / "m.csv") == "muse"
    rec = load(tmp_path / "m.csv")
    assert rec.ch_names == ["TP9", "AF7", "AF8", "TP10"] and rec.sfreq == pytest.approx(256, rel=1e-3)
    assert rec.data[3, 0] == 4.0


def test_muse_mind_monitor(tmp_path):
    n = 300
    df = pd.DataFrame({"TimeStamp": ["2026-01-01 10:00:00.000"] * n, "Delta_TP9": 0.1,
                       "RAW_TP9": 800.0, "RAW_AF7": 810.0, "RAW_AF8": 820.0, "RAW_TP10": 830.0,
                       "Elements": pd.Series([None] * n, dtype=object)})
    df.loc[5, ["RAW_TP9", "RAW_AF7", "RAW_AF8", "RAW_TP10"]] = np.nan
    df.loc[5, "Elements"] = "/muse/elements/blink"
    df.to_csv(tmp_path / "mm.csv", index=False)
    rec = load(tmp_path / "mm.csv")
    assert rec.meta["flavour"] == "mindmonitor" and rec.data.shape == (4, n - 1) and rec.sfreq == 256


def test_openbci_txt(tmp_path):
    hdr = ("%OpenBCI Raw EXG Data\n%Number of channels = 8\n%Sample Rate = 250 Hz\n%Board = OpenBCI_GUI$BoardCytonSerial\n")
    cols = ["Sample Index"] + [f"EXG Channel {i}" for i in range(8)] + ["Accel Channel 0", "Timestamp"]
    rows = "\n".join(", ".join([str(k)] + [f"{i + 0.5:.1f}" for i in range(8)] + ["0.0", "1.7e9"]) for k in range(100))
    (tmp_path / "o.txt").write_text(hdr + ", ".join(cols) + "\n" + rows + "\n")
    assert detect_format(tmp_path / "o.txt") == "openbci"
    rec = load(tmp_path / "o.txt")
    assert rec.sfreq == 250 and rec.ch_names[0] == "EXG0" and rec.data.shape == (8, 100)
    assert rec.data[7, 0] == 7.5


def test_fif_via_mne(tmp_path):
    mne = pytest.importorskip("mne")
    eo, _ = make_pair(duration_s=5)
    info = mne.create_info(eo.ch_names, eo.sfreq, "eeg")
    mne.io.RawArray(eo.data * 1e-6, info, verbose="ERROR").save(tmp_path / "x_raw.fif", verbose="ERROR")
    rec = load(tmp_path / "x_raw.fif")
    np.testing.assert_allclose(rec.data, eo.data, rtol=1e-5, atol=1e-3)


def test_cli_end_to_end(tmp_path, capsys):
    eo, ec = make_pair(alpha_amp_uv=6.0, seed=0)
    _write_csv(eo, tmp_path / "eo.csv"); _write_csv(ec, tmp_path / "ec.csv")
    rc = main(["--eo", str(tmp_path / "eo.csv"), "--ec", str(tmp_path / "ec.csv"),
               "--report", str(tmp_path / "r.html"), "--json", str(tmp_path / "r.json"), "--strict"])
    assert rc == 0
    assert "PASS" in capsys.readouterr().out
    html = (tmp_path / "r.html").read_text()
    assert "data:image/png;base64" in html and "PASS" in html
    assert json.loads((tmp_path / "r.json").read_text())["verdict"] == "PASS"
    main(["--eo", str(tmp_path / "eo.csv"), "--ec", str(tmp_path / "ec.csv"), "--report", str(tmp_path / "r.md")])
    assert (tmp_path / "r.md").read_text().startswith("# alphabench report")


def test_cli_strict_exit_code_on_noise(tmp_path):
    eo, ec = make_pair(seed=1)
    _write_csv(eo, tmp_path / "eo.csv"); _write_csv(ec, tmp_path / "ec.csv")
    rc = main(["--eo", str(tmp_path / "eo.csv"), "--ec", str(tmp_path / "ec.csv"),
               "--report", str(tmp_path / "r.md"), "--strict"])
    assert rc == 1
