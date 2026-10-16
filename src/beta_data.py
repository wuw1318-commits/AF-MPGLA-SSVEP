"""BETA SSVEP loader, frozen to the semantics of ``core.preprocess``.

Dimension semantics (fixed, and asserted by the unit tests)
----------------------------------------------------------
One subject's epochs are returned as

    E[m, i, h, c, n]      shape (Nm, Nf, Nb, Nc, Ns)

        m  sub-band index        Nm = 3      (M3 filter bank, [8m, 88] Hz)
        i  stimulus/condition    Nf = 40
        h  block                 Nb = 4
        c  channel               Nc = len(ch_idx), typically 9
        n  sample                Ns = round(250 * d)

That is the *same* layout ``core.preprocess`` produces for the Tsinghua
Benchmark, so ``run_sweep.source_trcs`` / ``target_model`` / ``similarity`` /
``mask_for`` / ``evaluate_subject`` are reused byte for byte.  The only thing
that differs between the two datasets is the axis order of the raw file:

    Tsinghua Benchmark  ``data[64, 1500, 40, 6]`` = (chan, time, class, block)
    BETA                ``data.EEG[64, T, 4, 40]`` = (chan, time, block, class)

Everything else - filters, sub-bands, weights, onset, channel list - is
imported from :mod:`core` rather than re-derived, so the two datasets cannot
drift apart.

Read-only, no disk cache
------------------------
The raw directory is only ever opened for reading.  No cache file is written
anywhere: a full 70-subject d=0.6 cache would be ~363 MB for no benefit, since
preprocessing one subject takes well under a second of CPU.  ``_MEMO`` is an
in-process cache only, and is keyed so that a different ``(ch_idx, d)`` can
never collide.

Class order
-----------
``data.suppl_info.freqs`` is NOT ascending: it runs 8.6, 8.8, ..., 15.8, then
8.0, 8.2, 8.4.  The loader reproduces that order exactly and refuses to accept
a numerically sorted vector (:func:`class_mapping`).  Note that none of this
round's three methods uses a reference signal, so the class order is a
labelling convention only - it cannot change an accuracy.  It is enforced
because the specification requires it, not because it is load-bearing.
"""
import os

import numpy as np
import scipy.io as sio
from scipy.signal import sosfiltfilt

import sys
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
ROOT = os.path.dirname(SRC)
if SRC not in sys.path:
    sys.path.insert(0, SRC)

# Imported UNCHANGED from the frozen core - never re-derived here.
from core import (  # noqa: E402
    FS, N_FREQ, ONSET_IDX, ONSET_S, SUB_WEIGHTS, CH_NAMES9, CH_IDX9,
    _BASE_SOS, _SUB_SOS,
)

N_BLOCK = 4                                   # BETA has 4 blocks per subject
N_SUBJECTS = 70
SHORT_UPTO = 15                               # S1..S15 have 2 s of flicker
SHORT_SAMPLES, LONG_SAMPLES = 750, 1000       # 3.0 s and 4.0 s at 250 Hz

DATA_DIR = os.environ.get("AF_MPGLA_BETA_DIR", "data/beta")

# Row indices into the 64-channel BETA tensor, resolved by NAME from
# data.suppl_info.chan.  Values are identical to the frozen Tsinghua CH_IDX9
# because BETA ships the same 64-channel layout, but they are re-derived here
# by name and asserted, never assumed.
CH_IDX9_BETA = [47, 52, 54, 55, 56, 58, 60, 61, 62]

_MEMO = {}
_CLASS_MAP = None
_CHAN_MAP = None


def mat_path(subject):
    if not 1 <= int(subject) <= N_SUBJECTS:
        raise ValueError("subject %r out of range 1..%d" % (subject, N_SUBJECTS))
    return os.path.join(DATA_DIR, "S%d.mat" % int(subject))


def expected_samples(subject):
    return SHORT_SAMPLES if int(subject) <= SHORT_UPTO else LONG_SAMPLES


def expected_shape(subject):
    return (64, expected_samples(subject), N_BLOCK, N_FREQ)


def _suppl(subject):
    return sio.loadmat(mat_path(subject))["data"]["suppl_info"][0, 0]


def load_subject_raw(subject):
    """Raw EEG tensor ``(64, T, 4, 40)`` for one subject.  Read-only."""
    d = sio.loadmat(mat_path(subject))["data"]
    eeg = np.asarray(d["EEG"][0, 0], dtype=np.float64)
    exp = expected_shape(subject)
    if eeg.shape != exp:
        raise ValueError("S%d: EEG shape %s, expected %s"
                         % (subject, eeg.shape, exp))
    if not np.isfinite(eeg).all():
        raise ValueError("S%d: non-finite values in EEG" % subject)
    return eeg


def class_mapping(subject=1):
    """``(freqs, phases)`` in the file's condition order, verified.

    Raises if the vector is ascending, because an ascending vector is the
    classic sign of someone having sorted the conditions by frequency value
    instead of reading ``suppl_info``.
    """
    global _CLASS_MAP
    if _CLASS_MAP is None:
        si = _suppl(subject)
        freqs = np.ravel(si["freqs"][0, 0]).astype(float)
        phases = np.ravel(si["phases"][0, 0]).astype(float)
        if freqs.shape != (N_FREQ,) or phases.shape != (N_FREQ,):
            raise ValueError("freqs %s / phases %s, expected %d entries"
                             % (freqs.shape, phases.shape, N_FREQ))
        _CLASS_MAP = (freqs, phases)
    freqs, phases = _CLASS_MAP
    if np.all(np.diff(freqs) > 0):
        raise ValueError(
            "suppl_info.freqs is ascending - refuses to accept a sorted "
            "condition order; BETA ships 8.6, 8.8, ..., 15.8, 8.0, 8.2, 8.4")
    return freqs.copy(), phases.copy()


def verify_class_mapping_all_subjects():
    """Every subject must ship the identical freqs/phases vectors."""
    ref = None
    for s in range(1, N_SUBJECTS + 1):
        si = _suppl(s)
        f = np.ravel(si["freqs"][0, 0]).astype(float)
        p = np.ravel(si["phases"][0, 0]).astype(float)
        if ref is None:
            ref = (f, p)
        elif not (np.array_equal(f, ref[0]) and np.array_equal(p, ref[1])):
            raise ValueError("S%d: freqs/phases differ from S1" % s)
    return ref[0].copy(), ref[1].copy()


def channel_mapping():
    """``(names, indices)`` for the nine paper electrodes, resolved by name."""
    global _CHAN_MAP
    if _CHAN_MAP is None:
        si = _suppl(1)
        chan = si["chan"][0, 0]
        names = [str(np.ravel(chan[i][3])[0]) for i in range(chan.shape[0])]
        upper = {n.upper(): i for i, n in enumerate(names)}
        missing = [n for n in CH_NAMES9 if n.upper() not in upper]
        if missing:
            raise ValueError("BETA lacks paper electrode(s): %s" % missing)
        idx = [int(upper[n.upper()]) for n in CH_NAMES9]
        if len(set(idx)) != len(idx):
            raise ValueError("duplicate channel index resolved: %s" % idx)
        if idx != CH_IDX9_BETA or idx != CH_IDX9:
            raise ValueError("resolved indices %s != BETA %s / Tsinghua %s"
                             % (idx, CH_IDX9_BETA, CH_IDX9))
        _CHAN_MAP = (list(CH_NAMES9), idx)
    return _CHAN_MAP[0][:], list(_CHAN_MAP[1])


def preprocess_beta(subject, ch_idx, d):
    """Band-pass 8-88 Hz, split into M3 sub-bands, segment [0.64, 0.64+d] s.

    Returns ``(Nm, Nf, Nb, Nc, Ns)`` - identical semantics to
    ``core.preprocess``.  Filtering happens on the WHOLE continuous trial and
    only then is the analysis window cut, exactly as the frozen Tsinghua
    implementation does.

    S1-S15 (750 samples) and S16-S70 (1000 samples) therefore yield analysis
    windows of the SAME length, because the window starts at the same sample
    for both and 160 + 250*1.0 = 410 <= 750.
    """
    ch_idx = list(ch_idx)
    n_samp = int(round(d * FS))
    key = (int(subject), tuple(ch_idx), n_samp)
    if key in _MEMO:
        return _MEMO[key]

    raw = load_subject_raw(subject)[ch_idx]                 # (Nc, T, Nb, Nf)
    t = raw.shape[1]
    if ONSET_IDX + n_samp > t:
        raise ValueError(
            "S%d: analysis window [%d, %d) exceeds the %d samples available - "
            "refusing to truncate silently"
            % (subject, ONSET_IDX, ONSET_IDX + n_samp, t))

    raw = np.ascontiguousarray(sosfiltfilt(_BASE_SOS, raw, axis=1))
    out = np.empty((len(_SUB_SOS), N_FREQ, N_BLOCK, len(ch_idx), n_samp),
                   dtype=np.float64)
    for m, sos in enumerate(_SUB_SOS):
        x = sosfiltfilt(sos, raw, axis=1)
        x = x[:, ONSET_IDX:ONSET_IDX + n_samp, :, :]        # (Nc, Ns, Nb, Nf)
        out[m] = np.transpose(x, (3, 2, 0, 1))              # (Nf, Nb, Nc, Ns)
    if not np.isfinite(out).all():
        raise ValueError("S%d: non-finite values after preprocessing" % subject)
    _MEMO[key] = out
    return out


def from_core_imports():
    """Diagnostic: which frozen-core objects this loader actually reuses."""
    return dict(filters_from_core=True, sosfiltfilt=sosfiltfilt,
                weights=SUB_WEIGHTS, onset=ONSET_S, onset_idx=ONSET_IDX)


def clear_memo():
    """Drop the in-process epoch cache (70 subjects at d=0.6 is ~363 MB)."""
    _MEMO.clear()
