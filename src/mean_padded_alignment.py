"""MP-GLA: mean-padded, full-length global latency alignment.

V2 of the GLA line.  V1 cropped both sides by L=10 samples to make the shift
legal, which cost 13.3% of the analysis window; V2 keeps the target template,
the target TRC and the test trial at the full 150 samples and fills only the
edge samples that a source TRC actually loses (mean |tau| = 3.5258 samples,
2.35% of the window).

The lag estimator, the tie-break, the branch arithmetic and the fusion are
imported UNCHANGED from gla_alignment (which itself mirrors run_sweep), so the
only new piece of mathematics here is mean_padded_shift.

Frozen definition (MP_GLA_MINIMAL_SPEC.md section 4):

    mu = y.mean()                     # this TRC's own temporal mean
    aligned = full(150, mu)
    if tau > 0:   aligned[tau:] = y[:-tau]
    elif tau < 0: aligned[:-k]  = y[k:]        # k = -tau
    else:         aligned[:]    = y

Equivalent to aligned[j] = y[j - tau] on the legal region, with the missing
edge filled by the TRC's own mean.  Never circular, never NaN.
"""
import numpy as np

import gla_alignment as G
from run_sweep import target_model

L = G.L
LAG_GRID = G.LAG_GRID
TIE_ORDER = G.TIE_ORDER
GAMMA = G.GAMMA


def mean_padded_shift(Y_source, taus):
    """(Nm, Nsource, Nfreq, Ns) + (Nm, Nsource) -> (Nm, Nsource, Nfreq, Ns)."""
    nm, nsub, nf, ns = Y_source.shape
    assert taus.shape == (nm, nsub), "taus must be (Nm, Nsource)"
    assert ns == 150, "this frozen design assumes Ns = 150"
    out = np.empty_like(Y_source)
    for m in range(nm):
        for s in range(nsub):
            tau = int(taus[m, s])
            assert -L <= tau <= L, "lag out of the frozen grid"
            blk = Y_source[m, s]                       # (Nf, Ns)
            mu = blk.mean(axis=1, keepdims=True)       # per-TRC temporal mean
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


METHODS = {
    "M0_TRCA":        dict(clb=1.0, sel="orig",    cca="orig"),
    "M1_iTRCA":       dict(clb=0.0, sel="orig",    cca="orig"),
    "M2_SSiTRCA":     dict(clb=0.9, sel="orig",    cca="orig"),
    "M3_LagSelSS":    dict(clb=0.9, sel="aligned", cca="orig"),
    "M4_AlignOnlySS": dict(clb=0.9, sel="orig",    cca="aligned"),
    "M5_MPGLASS":     dict(clb=0.9, sel="aligned", cca="aligned"),
    "M6_MPGLAiTRCA":  dict(clb=0.0, sel="orig",    cca="aligned"),
}
METHOD_NAMES = list(METHODS)


def eval_context(ep_u, Y_orig, Y_aligned, b, cal, clb, sel, cca, forbidden=()):
    """One context for one method.  Everything stays at the full 150 samples.

    Returns (prediction, n_selected, selection_changed_frac).
    """
    nf = ep_u.shape[1]
    G.assert_blocks_clean(cal, forbidden)
    wTa, xTa, xbar = target_model(ep_u, list(cal))
    Ysel = Y_aligned if sel == "aligned" else Y_orig
    Ycca = Y_aligned if cca == "aligned" else Y_orig

    mask_sel = G.mask_for_clb(xTa, Ysel, clb)
    if sel == "aligned":
        # how much did lag alignment change the selected source set?
        mask_ref = G.mask_for_clb(xTa, Y_orig, clb)
        frac = float((mask_sel != mask_ref).mean())
    else:
        frac = 0.0

    Yc = Ycca - Ycca.mean(axis=3, keepdims=True)
    grams = G.grams_of(Yc)
    test = ep_u[:, :, b]                              # full 150 samples
    g1, g2 = G.branches(test, wTa, xTa, xbar, Ycca, Yc, grams, mask_sel)
    return G.fused_pred(g1, g2), float(mask_sel.sum(axis=1).mean()), frac
