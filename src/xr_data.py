"""XR / AR-SSVEP dataset loading, manifest and structural audit.

Dataset (READ-ONLY, never copied/moved/unzipped/modified):
    data/xr
Fast local cache of the same zips (SHA256-verified identical, see data_audit):
    /tmp/ar_cache/SUBJECT_xx.zip

Structural checks reuse the *criteria* of src/uac_min/uac_data.py (which is the
earlier audit of this dataset).  uac_data is used ONLY for zip access and the
structural facts - none of UAC / codebook selection / FBCCA is imported here.

Facts (re-verified from the real files by audit_xr_external.py):
    30 subjects x 5 cycles x 8 frequencies (8..15 Hz) = 1200 trials
    each .mat holds one variable of shape (9, N)
    row 0 is an all-zero placeholder, rows 1..8 are the 8 EEG channels
    channel order (README): PO7 PO3 POz PO4 PO8 O2 Oz O1   (8 channels, no Pz)
    label = the frequency in the FILENAME (EEG_08.mat -> 8 Hz)
    _frequency_order.mat is a permutation of 8..15, used ONLY for the
    completeness check - the presentation order is NEVER the label.

Epoching (frozen, MPGLA_AF_XR_EXTERNAL_SPEC.md section 1):
    the A1 filter bank is applied to the FULL 5 s record first
    (baseband 8-88 Hz, then sub-bands (8,88) (16,88) (24,88)),
    and only then is [0.14, 0.74) = 150 samples sliced out.
    Slicing the 150 samples before filtering is forbidden.
"""
import io
import os
import re
import sys
import zipfile

import numpy as np
import scipy.io as sio
from scipy.signal import sosfiltfilt

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)

from core import _BASE_SOS, _SUB_SOS, SUB_WEIGHTS          # noqa: E402

FS = 250
LATENCY = 0.14
D = 0.6
START = int(round(LATENCY * FS))                 # 35
NS = int(round(D * FS))                          # 150
FREQS = list(range(8, 16))                       # 8..15 Hz
N_FREQ = len(FREQS)                              # 8
N_SUB = len(_SUB_SOS)                            # 3
N_CH = 8
N_SUBJECTS = 30
N_CYCLES = 5
CH_NAMES = ["PO7", "PO3", "POZ", "PO4", "PO8", "O2", "OZ", "O1"]
N_CHANNELS_RAW = 9                               # 1 placeholder + 8 EEG

CACHE = os.environ.get("AF_MPGLA_XR_CACHE", "data/xr")
SOURCE = os.environ.get("AF_MPGLA_XR_DIR", "data/xr")


def zip_path(subject):
    for base in (CACHE, SOURCE):
        p = os.path.join(base, "SUBJECT_%02d.zip" % subject)
        if os.path.exists(p):
            return p
    raise FileNotFoundError("SUBJECT_%02d.zip not found in cache or source"
                            % subject)


def _read(zf, name):
    d = sio.loadmat(io.BytesIO(zf.read(name)))
    k = [x for x in d if not x.startswith("__")][0]
    return k, d[k]


# --------------------------------------------------------------- manifest
def build_manifest(verbose=False):
    """One row per (subject, cycle, frequency).  Label = the filename freq."""
    rows = []
    for s in range(1, N_SUBJECTS + 1):
        zf = zipfile.ZipFile(zip_path(s))
        names = zf.namelist()
        cycles = sorted({m.group(1) for n in names
                         for m in [re.match(r"SUBJECT_\d+/(CYCLE_\d+)/", n)]
                         if m})
        for c in cycles:
            pref = "SUBJECT_%02d/%s/" % (s, c)
            order_file = [n for n in names
                          if n.startswith(pref) and "frequency_order" in n]
            order = None
            if order_file:
                _, v = _read(zf, order_file[0])
                order = tuple(int(x) for x in np.ravel(v))
            for f in FREQS:
                cand = [n for n in names if n.startswith(pref)
                        and re.search(r"EEG_%02d\.mat$" % f, n)]
                if not cand:
                    continue
                key, arr = _read(zf, cand[0])
                rows.append(dict(
                    subject_id=s, cycle_id=int(c[-2:]), frequency_hz=f,
                    file_path=cand[0], mat_key=key,
                    n_channels_raw=int(arr.shape[0]),
                    n_samples=int(arr.shape[1]), sampling_rate=FS,
                    row0_all_zero=bool(np.allclose(arr[0], 0.0)),
                    rows1_8_nonzero=bool(np.any(np.abs(arr[1:9]) > 0)),
                    order_in_cycle=(order.index(f) + 1
                                    if order and f in order else None),
                    frequency_order=str(list(order)) if order else None))
        zf.close()
        if verbose and s % 10 == 0:
            print("  manifest: subject %d done" % s, flush=True)
    return rows


def audit_manifest(rows):
    """The frozen structural assertions (spec section 0).  No silent repair."""
    res = {}
    res["dataset"] = SOURCE
    res["cache"] = CACHE
    res["sampling_rate_hz"] = FS
    res["expected"] = dict(n_subjects=N_SUBJECTS, cycles_per_subject=N_CYCLES,
                           classes=N_FREQ, freqs=FREQS, n_trials=1200,
                           n_channels_raw=N_CHANNELS_RAW, eeg_channels=N_CH,
                           window_d_s=D, latency_s=LATENCY,
                           epoch=[LATENCY, LATENCY + D],
                           samples=NS, start_sample=START,
                           min_samples_needed=START + NS)
    subj = {r["subject_id"] for r in rows}
    res["n_subjects"] = len(subj)
    res["n_subjects_ok"] = len(subj) == N_SUBJECTS
    per_subj = {}
    for r in rows:
        per_subj.setdefault(r["subject_id"], set()).add(r["cycle_id"])
    res["cycles_per_subject_ok"] = all(len(v) == N_CYCLES
                                       for v in per_subj.values())
    res["n_cycles_per_subject"] = {str(k): len(v)
                                   for k, v in sorted(per_subj.items())}
    per_cyc = {}
    for r in rows:
        per_cyc.setdefault((r["subject_id"], r["cycle_id"]), set()).add(
            r["frequency_hz"])
    res["freqs_per_cycle_ok"] = all(v == set(FREQS)
                                    for v in per_cyc.values())
    res["n_trials"] = len(rows)
    res["n_trials_ok"] = len(rows) == 1200
    res["n_channels_ok"] = all(r["n_channels_raw"] == N_CHANNELS_RAW
                               for r in rows)
    res["row0_zero_ok"] = all(r["row0_all_zero"] for r in rows)
    res["rows1_8_nonzero_ok"] = all(r["rows1_8_nonzero"] for r in rows)
    res["len_covers_window_ok"] = all(r["n_samples"] >= START + NS
                                      for r in rows)
    keys = [(r["subject_id"], r["cycle_id"], r["frequency_hz"]) for r in rows]
    res["no_duplicates"] = len(keys) == len(set(keys))
    res["no_missing_labels"] = all(r["frequency_hz"] in FREQS for r in rows)
    res["label_from_filename_ok"] = all(
        re.search(r"EEG_%02d\.mat$" % r["frequency_hz"], r["file_path"])
        for r in rows)
    res["min_samples"] = int(min(r["n_samples"] for r in rows))
    res["max_samples"] = int(max(r["n_samples"] for r in rows))
    res["median_samples"] = int(np.median([r["n_samples"] for r in rows]))
    orders = {r["frequency_order"] for r in rows if r["frequency_order"]}
    res["n_distinct_frequency_orders"] = len(orders)
    res["frequency_order_is_permutation"] = all(
        sorted(int(x) for x in o.strip("[]").split(",")) == FREQS
        for o in orders)
    res["order_present_ok"] = all(r["order_in_cycle"] is not None
                                  for r in rows)
    # the presentation order must NOT equal the label index in general
    res["order_equals_label_somewhere"] = any(
        r["order_in_cycle"] == r["frequency_hz"] - 7 for r in rows)

    bool_keys = [k for k, v in res.items() if k.endswith("_ok")
                 or k in ("no_duplicates", "no_missing_labels")]
    res["failures"] = [k for k in bool_keys if not res[k]]
    res["passed"] = not res["failures"]
    return res


# ------------------------------------------------------------------ loader
def load_trial(zf, file_path, assert_placeholder=True):
    """One trial -> (N_sub, Nc, Ns)  float64.

    Reads the FULL record, drops row 0 (the all-zero placeholder), filters the
    full 5 s with the A1 filter bank, then slices [0.14, 0.74).
    """
    _, arr = _read(zf, file_path)
    arr = np.asarray(arr, dtype=np.float64)
    if assert_placeholder:
        if arr.shape[0] != N_CHANNELS_RAW:
            raise AssertionError(
                "expected %d rows (1 placeholder + %d EEG), got %d"
                % (N_CHANNELS_RAW, N_CH, arr.shape[0]))
        if not np.allclose(arr[0], 0.0):
            raise AssertionError(
                "row 0 is not the all-zero placeholder in %s" % file_path)
    if arr.shape[1] < START + NS:
        raise AssertionError("record too short in %s (%d < %d)"
                             % (file_path, arr.shape[1], START + NS))
    rec = arr[1:1 + N_CH]                       # (8, N) full record
    base = sosfiltfilt(_BASE_SOS, rec, axis=1)
    out = np.empty((N_SUB, N_CH, NS))
    for m, sos in enumerate(_SUB_SOS):
        x = sosfiltfilt(sos, base, axis=1)      # sub-band of the base signal
        out[m] = x[:, START:START + NS]         # slice AFTER filtering
    return out


def load_subject_epochs(subject, assert_placeholder=True, verbose=False):
    """(N_sub, N_freq, N_cycles, N_ch, Ns) for one subject, all 5 cycles."""
    zf = zipfile.ZipFile(zip_path(subject))
    names = zf.namelist()
    E = np.empty((N_SUB, N_FREQ, N_CYCLES, N_CH, NS))
    for c in range(1, N_CYCLES + 1):
        pref = "SUBJECT_%02d/CYCLE_%02d/" % (subject, c)
        for j, f in enumerate(FREQS):
            cand = [n for n in names if n.startswith(pref)
                    and re.search(r"EEG_%02d\.mat$" % f, n)]
            if not cand:
                raise AssertionError("missing %sEEG_%02d.mat" % (pref, f))
            E[:, j, c - 1] = load_trial(zf, cand[0], assert_placeholder)
    zf.close()
    if verbose:
        print("  epochs S%02d %s" % (subject, E.shape), flush=True)
    return E


def load_all(subjects=None, verbose=True):
    subs = list(range(1, N_SUBJECTS + 1)) if subjects is None else list(subjects)
    E = {}
    for s in subs:
        E[s] = load_subject_epochs(s, verbose=verbose)
    return E


def load_raw_for_test(subject, cycle, freq):
    """Raw (N_ROW, N) record of one trial - used only by the tests/audit."""
    zf = zipfile.ZipFile(zip_path(subject))
    pref = "SUBJECT_%02d/CYCLE_%02d/" % (subject, cycle)
    cand = [n for n in zf.namelist() if n.startswith(pref)
            and re.search(r"EEG_%02d\.mat$" % freq, n)]
    _, arr = _read(zf, cand[0])
    zf.close()
    return np.asarray(arr, dtype=np.float64)
