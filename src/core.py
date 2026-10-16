"""Core implementation of iTRCA / SS-iTRCA (Wang et al., IEEE JBHI 2025).

Faithful re-implementation from the paper:
  - TRCA spatial filter ............ eqs (1)-(6)
  - source instance / TRC .......... eqs (7)-(8)
  - CCA common latent space ........ eq (9)
  - subject-general / -specific .... eqs (10)-(12)
  - M3 filter bank ................. eq (13)
  - recognition .................... eq (14)
  - TRC-based subject selection .... eqs (15)-(21)
  - ITR ............................ eq (22)

No reference (sine/cosine) signals are used anywhere: every quantity in the
paper is data driven, so stimulus frequency/phase information is only needed
for the ITR bookkeeping (Nf = 40).
"""
import os
import numpy as np
from scipy.signal import cheby1, sosfiltfilt
from scipy.linalg import eigh

# ---------------------------------------------------------------- constants
FS = 250                 # Benchmark sampling rate (Hz)
N_FREQ = 40              # stimuli per dataset (Table I)
N_BLOCK = 6              # blocks per subject (Table I)
TRIAL_LEN = 1500         # samples per trial (6 s)

# 9 occipital/parietal electrodes, paper order: Pz PO7 PO3 POz PO4 PO8 O1 Oz O2
# Row indices into the 64-row Benchmark tensor, taken from the authoritative
# 64-channels.loc shipped with the dataset (M1=33, M2=43, CB1=60, CB2=64 1-based)
# and cross-checked against MetaBCI's Wang2016 ch_names construction.
CH_IDX9 = [47, 52, 54, 55, 56, 58, 60, 61, 62]
CH_NAMES9 = ["Pz", "PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2"]

# Segmentation window [0.64, 0.64 + d] s measured from trial start, i.e.
# 0.5 s gaze shift + 0.14 s latency delay -> 0.14 s after flicker onset.
ONSET_S = 0.64
ONSET_IDX = int(round(ONSET_S * FS))          # 160

# M3 filter bank (Chen et al. 2015): sub-band m = [m*8, 88] Hz.
# The paper's stated 8-88 Hz passband is exactly sub-band m = 1 of this family.
SUB_BANDS = [(8.0, 88.0), (16.0, 88.0), (24.0, 88.0)]
SUB_WEIGHTS = np.array([m ** (-1.25) + 0.25 for m in (1, 2, 3)])

FILTER_ORDER = 4
FILTER_RP = 0.5          # dB passband ripple

DATA_DIR = os.environ.get("AF_MPGLA_BENCHMARK_DIR", "data/benchmark")
N_SUBJECTS = 35


# ------------------------------------------------------------- data loading
_BASE_SOS = cheby1(FILTER_ORDER, FILTER_RP, [8.0, 88.0], btype="bandpass",
                   fs=FS, output="sos")
_SUB_SOS = [cheby1(FILTER_ORDER, FILTER_RP, list(b), btype="bandpass",
                   fs=FS, output="sos") for b in SUB_BANDS]


def _mat_path(subject):
    return os.path.join(DATA_DIR, "S%d.mat" % subject)


def load_subject_raw(subject):
    """Return the raw (64, 1500, 40, 6) tensor for one subject (in microvolts)."""
    import scipy.io as sio
    return sio.loadmat(_mat_path(subject))["data"].astype(np.float64)


def preprocess(subject, ch_idx, d):
    """Band-pass 8-88 Hz, split into M3 sub-bands, segment [0.64, 0.64+d] s.

    Returns ``E`` with shape (N_sub, N_FREQ, N_BLOCK, Nc, Ns).
    """
    n_samp = int(round(d * FS))
    raw = load_subject_raw(subject)[ch_idx]            # (Nc, 1500, 40, 6)
    raw = sosfiltfilt(_BASE_SOS, raw, axis=1)          # dataset level 8-88 Hz
    out = np.empty((len(SUB_BANDS), N_FREQ, N_BLOCK, len(ch_idx), n_samp),
                   dtype=np.float64)
    for m, sos in enumerate(_SUB_SOS):
        x = sosfiltfilt(sos, raw, axis=1)
        x = x[:, ONSET_IDX:ONSET_IDX + n_samp, :, :]   # (Nc, Ns, Nf, Nb)
        out[m] = np.transpose(x, (2, 3, 0, 1))         # (Nf, Nb, Nc, Ns)
    return out


# --------------------------------------------------------------------- TRCA
def trca_sq(trials):
    """Build the TRCA matrices S (eq 2) and Q (eq 4) for one stimulus.

    trials : (Nb, Nc, Ns) raw trials of one stimulus from one subject.
    Because Cov(Z_h1, Z_h2) factorises, S and Q are cheap sums of the
    per-block Gram matrices, which is what makes sub-block-set TRCA fast.
    """
    nb, nc, ns = trials.shape
    z = trials - trials.mean(axis=2, keepdims=True)        # centre over time
    pz = np.einsum("hcn,hkn->hck", z, z)                   # per-block Z_h Z_h^T
    zs = z.sum(axis=0)                                     # sum_h Z_h
    total = zs @ zs.T                                      # sum_{h1,h2}
    q = pz.sum(axis=0) / (ns - 1.0)                        # eq (4)
    s = (total - pz.sum(axis=0)) / (ns - 1.0)              # eq (2)
    return s, q, pz, zs


def trca_filter(trials):
    """TRCA spatial filter, eqs (1)-(6) (eigenvector of Q^-1 S, largest eig.)."""
    s, q = trca_sq(trials)[:2]
    try:
        _, v = eigh(s, q)
        return v[:, -1]
    except np.linalg.LinAlgError:                          # singular Q fallback
        _, v = np.linalg.eigh(np.linalg.pinv(q) @ s)
        return v[:, -1]


def trca_many(trials_stack):
    """TRCA filters for all stimuli at once. trials_stack: (Nf, Nb, Nc, Ns)."""
    return np.stack([trca_filter(t) for t in trials_stack])


# ---------------------------------------------------------------------- CCA
def cca(X, Y, ridge=1e-10):
    """Canonical correlation analysis, eq (9).

    X : (dx, N)   source template  Y^S_i   (dx = number of selected sources)
    Y : (dy, N)   target template  Xbar^Ta_i (dy = Nc)
    Returns (w_x, w_y) for the first (largest) canonical pair, with the
    convention corr(w_x^T X, w_y^T Y) >= 0.
    """
    Xc = X - X.mean(axis=1, keepdims=True)
    Yc = Y - Y.mean(axis=1, keepdims=True)
    Cxx = Xc @ Xc.T
    Cyy = Yc @ Yc.T
    Cxy = Xc @ Yc.T
    Cxx = Cxx + ridge * np.trace(Cxx) / Cxx.shape[0] * np.eye(Cxx.shape[0])
    Cyy = Cyy + ridge * np.trace(Cyy) / Cyy.shape[0] * np.eye(Cyy.shape[0])

    ex, vx = np.linalg.eigh(Cxx)
    ey, vy = np.linalg.eigh(Cyy)
    wx_half = vx @ np.diag(1.0 / np.sqrt(np.maximum(ex, 1e-12))) @ vx.T
    wy_half = vy @ np.diag(1.0 / np.sqrt(np.maximum(ey, 1e-12))) @ vy.T

    A = wx_half @ Cxy @ wy_half                       # (dx, dy)
    u, s, vt = np.linalg.svd(A, full_matrices=False)
    w_x = wx_half @ u[:, 0]
    w_y = wy_half @ vt[0, :]
    return w_x, w_y


def cca_from_grams(Cxx, Cyy, Cxy, ridge=1e-10):
    """Same as :func:`cca` but from pre-computed (centred) Gram matrices."""
    n = Cxx.shape[0]
    Cxx = Cxx + ridge * np.trace(Cxx) / n * np.eye(n)
    Cyy = Cyy + ridge * np.trace(Cyy) / Cyy.shape[0] * np.eye(Cyy.shape[0])
    ex, vx = np.linalg.eigh(Cxx)
    ey, vy = np.linalg.eigh(Cyy)
    wx_half = vx @ np.diag(1.0 / np.sqrt(np.maximum(ex, 1e-12))) @ vx.T
    wy_half = vy @ np.diag(1.0 / np.sqrt(np.maximum(ey, 1e-12))) @ vy.T
    A = wx_half @ Cxy @ wy_half
    u, s, vt = np.linalg.svd(A, full_matrices=False)
    return wx_half @ u[:, 0], wy_half @ vt[0, :]


# ------------------------------------------------------------------ helpers
def corr_rows(a, b):
    """Pearson correlation between two 1-D arrays."""
    a = a - a.mean()
    b = b - b.mean()
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-15 or nb < 1e-15:
        return 0.0
    return float(a @ b / (na * nb))


def itr_bits_per_min(acc, n_freq=N_FREQ, d=1.0):
    """ITR, eq (22). T = d + 0.5 s (gaze time + gaze shifting)."""
    p = np.clip(acc, 1e-12, 1 - 1e-12)
    t = d + 0.5
    bits = np.log2(n_freq) + p * np.log2(p) + (1 - p) * np.log2((1 - p) / (n_freq - 1))
    return bits * 60.0 / t
