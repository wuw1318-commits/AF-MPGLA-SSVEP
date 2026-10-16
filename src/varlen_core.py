"""Length-generic wrapper for the MP-GLA x AF combination.

The two mechanisms are NOT re-implemented.  Imported UNCHANGED:

    estimate_lags / mask_for / branches / fused_pred / grams_of / make_J
        / assert_blocks_clean                        <- gla_alignment
    similarity / target_model / source_trcs          <- run_sweep
    cca_from_grams / SUB_WEIGHTS / CH_IDX9 / preprocess <- core

The ONLY thing this module adds is a length-generic copy of the mean-padded
shift, because `mp_gla.mean_padded_alignment.mean_padded_shift` carries

    assert ns == 150, "this frozen design assumes Ns = 150"

at runtime (line 39) - the single hard-coded sample count in the whole
pipeline.  `mean_padded_shift_varlen` below is the same mathematics with that
one assertion lifted; a unit test proves it is BIT-IDENTICAL to the original
whenever ns == 150.  The original module is not modified.

Frozen choices (MPGLA_AF_TIMEWINDOW_SPEC.md section 1):
    L = 10 SAMPLES (+-40 ms at 250 Hz) - the lag grid does NOT scale with d
    ONSET_IDX = 160 for every d - same start, same latency compensation
    gamma = 0.5, clb = 0.9 - unchanged at every d
    no cropping at any d: the aligned TRCs keep the full length of that window
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
ROOT = os.path.dirname(SRC)
sys.path.insert(0, SRC)
sys.path.insert(0, os.path.join(SRC, "gla"))
sys.path.insert(0, os.path.join(SRC, "mp_gla"))
sys.path.insert(0, HERE)

import gla_alignment as G                                   # noqa: E402
import mean_padded_alignment as MP                          # noqa: E402
from core import CH_IDX9, FS, ONSET_IDX, preprocess         # noqa: E402
from run_sweep import (N_BLOCK, N_FREQ, similarity,         # noqa: E402
                       source_trcs, target_model)

NC, NTB = 9, 3
GAMMA = 0.5
CLB = 0.9
L = G.L                                                     # 10 samples
D_GRID = (0.2, 0.4, 0.6, 0.8, 1.0)
CELLS = ("M00", "M01", "M10", "M11")
ALIGNED = {"M00": False, "M01": False, "M10": True, "M11": True}
METHODS = ("TRCA", "iTRCA") + CELLS


def n_samples(d):
    """The frozen d -> number of samples rule, straight from core.preprocess."""
    return int(round(d * FS))


def mean_padded_shift_varlen(Y_source, taus):
    """Frozen MP-GLA mean-padded shift, for ANY window length.

    Bit-identical to mp_gla.mean_padded_alignment.mean_padded_shift when
    Y_source.shape[3] == 150; identical mathematics otherwise.  Edges lost to
    a shift are filled with that TRC's own temporal mean.  Never circular,
    never NaN, never cropped.
    """
    nm, nsub, nf, ns = Y_source.shape
    assert taus.shape == (nm, nsub), "taus must be (Nm, Nsource)"
    out = np.empty_like(Y_source)
    for m in range(nm):
        for s in range(nsub):
            tau = int(taus[m, s])
            assert -L <= tau <= L, "lag out of the frozen grid"
            blk = Y_source[m, s]
            mu = blk.mean(axis=1, keepdims=True)
            a = np.broadcast_to(mu, blk.shape).copy()
            if tau > 0:
                a[:, tau:] = blk[:, :-tau]
            elif tau < 0:
                k = -tau
                a[:, :-k] = blk[:, k:]
            else:
                a = blk.copy()
            out[m, s] = a
    return out


def masks_b0_b1(ctilde, cmax_signed, cmax_abs, gamma=GAMMA, clb=CLB):
    """(mask_B0, mask_B1, triggered, anomalies) - unchanged from the factorial."""
    triggered = cmax_signed > gamma
    mask_b0 = np.where(triggered[:, None, :], ctilde > clb, True)
    mask_b1 = ctilde > clb
    anom = dict(n_nan_inf=0, n_all_zero=0, n_empty=0, n_fallback_cells=0)
    nm, nsub, nf = ctilde.shape
    for m in range(nm):
        for i in range(nf):
            bad_finite = not np.all(np.isfinite(ctilde[m, :, i]))
            all_zero = bool(cmax_abs[m, i] == 0.0)
            empty = not bool(mask_b1[m, :, i].any())
            if bad_finite:
                anom["n_nan_inf"] += 1
            if all_zero:
                anom["n_all_zero"] += 1
            if empty:
                anom["n_empty"] += 1
            if bad_finite or all_zero or empty:
                anom["n_fallback_cells"] += 1
                mask_b1[m, :, i] = mask_b0[m, :, i]
    return mask_b0, mask_b1, triggered, anom


def assert_split_clean(cal, forbidden=(), n_block=N_BLOCK):
    cal = [int(x) for x in cal]
    assert len(set(cal)) == len(cal), "duplicate calibration blocks"
    assert all(0 <= b < n_block for b in cal), "calibration block out of range"
    bad = sorted(set(cal) & set(int(x) for x in forbidden))
    assert not bad, \
        "calibration blocks %s overlap forbidden (test) blocks" % bad


def prepare(ep_u, Yt, cal, forbidden=(), taus_override=None):
    """Everything the four cells need for one (target, calibration) split.

    The window length comes from the data, never from a constant.  Depends on
    the calibration tuple only, so tau, the aligned TRCs and all four masks are
    automatically test-block independent.
    """
    ns = ep_u.shape[4]
    assert Yt.shape[3] == ns, "target and source traces have different lengths"
    assert_split_clean(cal, forbidden)
    J = G.make_J(ns)
    wTa, xTa, xbar = target_model(ep_u, list(cal))
    taus = (G.estimate_lags(xTa, Yt, J) if taus_override is None
            else np.asarray(taus_override))
    Y_al = mean_padded_shift_varlen(Yt, taus)

    _, ct_o, ca_o, cs_o = similarity(xTa, Yt)
    m00, m01, trg_o, an_o = masks_b0_b1(ct_o, cs_o, ca_o)
    _, ct_a, ca_a, cs_a = similarity(xTa, Y_al)
    m10, m11, trg_a, an_a = masks_b0_b1(ct_a, cs_a, ca_a)

    Yc_o = Yt - Yt.mean(axis=3, keepdims=True)
    Yc_a = Y_al - Y_al.mean(axis=3, keepdims=True)
    it_mask = G.mask_for(ct_o, cs_o, GAMMA, 0.0)          # iTRCA = clb 0.0
    return dict(ns=ns, J=J, wTa=wTa, xTa=xTa, xbar=xbar, taus=taus,
                Yt=Yt, Y_al=Y_al, Yc_o=Yc_o, Yc_a=Yc_a,
                grams_o=G.grams_of(Yc_o), grams_a=G.grams_of(Yc_a),
                masks={"M00": m00, "M01": m01, "M10": m10, "M11": m11,
                       "iTRCA": it_mask},
                triggered={"orig": trg_o, "aligned": trg_a},
                anom={"orig": an_o, "aligned": an_a},
                ctilde={"orig": ct_o, "aligned": ct_a},
                cmax={"orig": cs_o, "aligned": cs_a})


def predict_mask(prep, test, mask, aligned):
    """The shared branch arithmetic with an explicitly supplied mask."""
    if aligned:
        Yt, Yc, grams = prep["Y_al"], prep["Yc_a"], prep["grams_a"]
    else:
        Yt, Yc, grams = prep["Yt"], prep["Yc_o"], prep["grams_o"]
    g1, g2 = G.branches(test, prep["wTa"], prep["xTa"], prep["xbar"],
                        Yt, Yc, grams, mask)
    return G.fused_pred(g1, g2)


def predict(prep, cell, test):
    return predict_mask(prep, test, prep["masks"][cell], ALIGNED[cell])


def predict_trca(prep, test):
    """TRCA = the shared branch arithmetic with an all-False mask."""
    dummy = np.zeros_like(prep["masks"]["M00"])
    return predict_mask(prep, test, dummy, False)


def eval_context(ep_u, prep, b, cells=CELLS, want_trca=False,
                 want_itrc=True):
    """One outer context: all requested methods share the same `prep`."""
    test = ep_u[:, :, b]
    out = {c: predict(prep, c, test) for c in cells}
    if want_trca:
        out["TRCA"] = predict_trca(prep, test)
    if want_itrc:
        out["iTRCA"] = predict_mask(prep, test, prep["masks"]["iTRCA"], False)
    return out


def accuracy(pred):
    return float((pred == np.arange(N_FREQ)).mean())


def assert_no_crop(prep):
    """Guard: alignment preserves the window length of that d."""
    assert prep["Y_al"].shape == prep["Yt"].shape, \
        "alignment changed the window length"
    assert prep["Y_al"].shape[3] == prep["ns"]


def assert_pad_only_edges(Y_orig, Y_aligned, taus):
    """Guard: the elementwise frozen shift+pad, and nothing else."""
    nm, nsub, nf, ns = Y_orig.shape
    for m in range(nm):
        for s in range(nsub):
            y = Y_orig[m, s]
            tau = int(taus[m, s])
            mu = y.mean(axis=1, keepdims=True)
            exp = np.broadcast_to(mu, y.shape).copy()
            if tau > 0:
                exp[:, tau:] = y[:, :-tau]
            elif tau < 0:
                k = -tau
                exp[:, :-k] = y[:, k:]
            else:
                exp = y.copy()
            assert np.array_equal(Y_aligned[m, s], exp), \
                "aligned TRC is not the frozen shift+pad at (m=%d, s=%d)" % (m, s)
    assert not np.array_equal(Y_aligned, -Y_orig), "polarity flip detected"


def assert_b1_differs_only_untriggered(trig_aligned, m10, m11):
    diff = m10 != m11
    inside = diff & np.broadcast_to(trig_aligned[:, None, :], diff.shape)
    assert not inside.any(), \
        "M11 changed the mask inside an aligned-triggered cell (%d cells)" \
        % int(inside.any(axis=1).sum())


def assert_same_tau(t1, t2):
    assert np.array_equal(t1, t2), "tau differs between the two aligned cells"


def assert_L_unchanged():
    """Guard: the lag grid is not rescaled by d."""
    assert G.L == 10 and len(G.LAG_GRID) == 21, \
        "the lag grid has been changed; it must stay +-10 samples"
