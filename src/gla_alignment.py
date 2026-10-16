"""GLA: Global Latency-Aligned source selection for SS-iTRCA.

Frozen definition (GLA_MINIMAL_SPEC.md section 3):

    L = 10 samples, lag_grid = {-10..+10}, Fs = 250 Hz  (max +-40 ms)
    J = arange(L, Ns-L)                      # arange(10,140), len 130

    q_{s,t,m}(tau) = mean_i | corr( xTa_{m,i,J} , y^S_s_{m,i,J-tau} ) |
    tau*_{s,t,m}   = argmax_tau q

    tie-break: largest q -> smallest |tau| -> smallest tau (never random)

One lag per (source, target, sub-band); all 40 classes share it.  Estimation
uses the target CALIBRATION blocks only - the test block is guarded by an
explicit assertion, not by a comment.

The branch arithmetic mirrors run_sweep.evaluate_subject exactly, so the
uncropped path reproduces A1 bit-for-bit (verified in baseline_check).
"""
import numpy as np

from core import SUB_WEIGHTS, cca_from_grams
from run_sweep import target_model, similarity, mask_for, _pearson

L = 10
LAG_GRID = np.arange(-L, L + 1)
# frozen tie-break order: |tau| ascending, then tau ascending
TIE_ORDER = sorted(LAG_GRID.tolist(), key=lambda t: (abs(t), t))
GAMMA = 0.5
CLB = {"trca": 1.0, "itrca": 0.0, "ss_itrca": 0.9}


def make_J(ns, l=L):
    return np.arange(l, ns - l)


def assert_blocks_clean(cal, forbidden, n_block=6):
    """Guard: calibration blocks must be legal and disjoint from forbidden.

    Enforced, not documented - a violation raises AssertionError.
    """
    cal = [int(x) for x in cal]
    assert len(set(cal)) == len(cal), "duplicate calibration blocks"
    assert all(0 <= b < n_block for b in cal), "calibration block out of range"
    bad = sorted(set(cal) & set(int(x) for x in forbidden))
    assert not bad, \
        "calibration blocks %s overlap forbidden (test/unused) blocks" % bad


def estimate_lags_guarded(xTa, Y_source, J, cal, forbidden):
    """estimate_lags with the frozen leakage guard applied first."""
    assert_blocks_clean(cal, forbidden)
    return estimate_lags(xTa, Y_source, J)


# ------------------------------------------------------------ lag estimation
def lag_scores_m(xTa_m, Y_m, J):
    """q for one sub-band.  xTa_m (Nf,Ns); Y_m (Nsub,Nf,Ns) -> (n_lag, Nsub)."""
    xc = xTa_m[:, J]
    xc = xc - xc.mean(axis=1, keepdims=True)
    xn = np.linalg.norm(xc, axis=1)                      # (Nf,)
    out = np.empty((len(LAG_GRID), Y_m.shape[0]))
    for k, tau in enumerate(LAG_GRID):
        Yc = Y_m[:, :, J - tau]
        Yc = Yc - Yc.mean(axis=2, keepdims=True)
        yn = np.linalg.norm(Yc, axis=2)                  # (Nsub,Nf)
        num = np.einsum("in,min->mi", xc, Yc)
        r = np.abs(num / np.maximum(xn[None, :] * yn, 1e-15))
        out[k] = r.mean(axis=1)                          # mean over 40 classes
    return out


def choose_tau(scores):
    """Frozen tie-break.  scores (n_lag, Nsub) -> (Nsub,) int."""
    idx = {int(t): i for i, t in enumerate(LAG_GRID)}
    nsub = scores.shape[1]
    tau = np.zeros(nsub, dtype=int)
    for n in range(nsub):
        col = scores[:, n]
        mx = col.max()
        for t in TIE_ORDER:
            if col[idx[t]] == mx:
                tau[n] = t
                break
    return tau


def estimate_lags(xTa, Y_source, J):
    """tau* per (source, sub-band).  xTa (Nm,Nf,Ns); Y (Nm,Nsub,Nf,Ns).

    Returns (Nm, Nsub) int array.  Uses only the arrays handed in - the caller
    is responsible for passing calibration-only data (enforced downstream).
    """
    nm = xTa.shape[0]
    taus = np.zeros((nm, Y_source.shape[1]), dtype=int)
    for m in range(nm):
        taus[m] = choose_tau(lag_scores_m(xTa[m], Y_source[m], J))
    return taus


def align(Y_source, taus, J):
    """Y_aligned[m,s,i,:] = Y_source[m,s,i, J - tau[s,m]].  taus (Nm,Nsub)."""
    nm, nsub, nf, ns = Y_source.shape
    out = np.empty((nm, nsub, nf, len(J)))
    for m in range(nm):
        for s in range(nsub):
            out[m, s] = Y_source[m, s][:, J - taus[m, s]]
    return out


# ---------------------------------------------------------------- branches
def mask_for_clb(xTa, Yt, clb, gamma=GAMMA):
    if clb >= 1.0:
        return np.zeros((xTa.shape[0], Yt.shape[1], xTa.shape[1]), dtype=bool)
    _, ctilde, _, cmax_signed = similarity(xTa, Yt)
    return mask_for(ctilde, cmax_signed, gamma, clb)


def branches(test, wTa, xTa, xbar, Yt, Yc, grams, mask):
    """(g1, g2) = (sign(rho1)rho1^2, sign(rho2)rho2^2); mirrors run_sweep."""
    nm, nf, nc, ns = xbar.shape
    proj_ta = np.einsum("mtcn,mic->mtin", test, wTa)
    rho2 = _pearson(proj_ta, xTa[:, None, :, :])
    rho1 = np.zeros_like(rho2)
    if mask.any():
        for m in range(nm):
            ycm = Yc[m]
            wgt = np.empty((nf, nc))
            wsrc = np.empty((nf, ns))
            for i in range(nf):
                sel = mask[m, :, i]
                if not sel.any():
                    wgt[i] = 0.0
                    wsrc[i] = 0.0
                    continue
                Ys = ycm[sel, i]
                xs = xbar[m, i] - xbar[m, i].mean(axis=1, keepdims=True)
                Cxx = grams[m][np.ix_(sel, sel)][:, :, i]
                wx, wy = cca_from_grams(Cxx, xs @ xs.T, Ys @ xs.T)
                wgt[i] = wy
                wsrc[i] = wx @ Yt[m][sel, i]
            proj = np.einsum("tcn,ic->tin", test[m], wgt)
            pc = proj - proj.mean(axis=2, keepdims=True)
            pn = np.linalg.norm(pc, axis=2)
            wc = wsrc - wsrc.mean(axis=1, keepdims=True)
            wn = np.linalg.norm(wc, axis=1)
            rho1[m] = np.einsum("tin,in->ti", pc, wc) / \
                np.maximum(pn * wn[None, :], 1e-15)
    return np.sign(rho1) * rho1 ** 2, np.sign(rho2) * rho2 ** 2


def fused_pred(g1, g2):
    rho = g1 + g2
    feat = (rho * SUB_WEIGHTS[:, None, None]).sum(axis=0)
    return feat.argmax(axis=1)


def grams_of(Yc):
    return np.einsum("msin,mtin->msti", Yc, Yc)


# ------------------------------------------------------------------ driver
def eval_context(epochs_u, Yt, b, cal, clb, use_crop, tau=None, J=None,
                 forbidden=()):
    """One context for one method.

    use_crop : False -> original 150-sample A1 path (M0/M1/M2)
               True  -> central 130 samples; tau=None means tau=0 (crop
                        controls M3/M4/M5), else the estimated lags (M6/M7)
    Yt : (Nm,Nsub,Nf,Ns) full-length source TRCs
    """
    nf = epochs_u.shape[1]
    assert b not in set(int(x) for x in cal), "test block inside calibration"
    assert not (set(int(x) for x in cal) & set(int(x) for x in forbidden)), \
        "calibration overlaps forbidden blocks"
    wTa, xTa, xbar = target_model(epochs_u, list(cal))
    test = epochs_u[:, :, b]

    if not use_crop:
        Yc = Yt - Yt.mean(axis=3, keepdims=True)
        grams = grams_of(Yc)
        mask = mask_for_clb(xTa, Yt, clb)
        g1, g2 = branches(test, wTa, xTa, xbar, Yt, Yc, grams, mask)
        return fused_pred(g1, g2), None

    J = J if J is not None else make_J(epochs_u.shape[4])
    taus = np.zeros((xTa.shape[0], Yt.shape[1]), dtype=int) if tau is None else tau
    Y_al = align(Yt, taus, J)
    xTa_c = xTa[:, :, J]
    xbar_c = xbar[:, :, :, J]
    test_c = test[:, :, :, J]
    Yc = Y_al - Y_al.mean(axis=3, keepdims=True)
    grams = grams_of(Yc)
    mask = mask_for_clb(xTa_c, Y_al, clb)
    g1, g2 = branches(test_c, wTa, xTa_c, xbar_c, Y_al, Yc, grams, mask)
    return fused_pred(g1, g2), (taus, mask.sum(axis=1).mean())
