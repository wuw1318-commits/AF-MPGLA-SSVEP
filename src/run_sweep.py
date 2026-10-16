"""iTRCA / SS-iTRCA / TRCA - full LOSO+LOBO evaluation.

Protocol (Section II-E of the paper):
  outer : leave-one-subject-out.  Each of the 35 subjects is the target once;
          the other 34 are sources (Nsrc = 34), each contributing all of its
          blocks (Nsb = Nb = 6).
  inner : leave-one-block-out on the target.  Each block is the test block
          once; Ntb of the remaining Nb-1 blocks train the target subject.

The three compared methods are the same code path with different clb
(paper eqs 20/21 state this explicitly):
  clb = 0  -> iTRCA    (all sources kept)
  clb = 1  -> TRCA     (no source kept; only the subject-specific feature)
  clb in (0,1) -> SS-iTRCA

Every quantity is data driven (TRCA + CCA); no sine/cosine reference signal
appears anywhere in the paper's formulation, so none is used here.
"""
import itertools
import json
import os
import sys
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from core import (N_FREQ, N_BLOCK, SUB_WEIGHTS, CH_IDX9,  # noqa: E402
                  preprocess, trca_sq, cca_from_grams, itr_bits_per_min)
from scipy.linalg import eigh


def eigvec(s, q):
    try:
        _, v = eigh(s, q)
        return v[:, -1]
    except np.linalg.LinAlgError:
        _, v = np.linalg.eigh(np.linalg.pinv(q) @ s)
        return v[:, -1]


# --------------------------------------------------------------- source side
def source_trcs(epochs):
    """TRCs of one subject (eqs 6-7). epochs: (Nm,Nf,Nb,Nc,Ns) -> (Nm,Nf,Ns)."""
    nm, nf, nb, nc, ns = epochs.shape
    y = np.zeros((nm, nf, ns))
    for m in range(nm):
        for i in range(nf):
            s, q = trca_sq(epochs[m, i])[:2]
            y[m, i] = eigvec(s, q) @ epochs[m, i].mean(axis=0)
    return y


# --------------------------------------------------------------- target side
def target_model(epochs_t, sub):
    """Target TRCA / template from a training-block subset.

    Returns wTa (Nm,Nf,Nc), xTa (Nm,Nf,Ns) = TRC of the template (eq 11) and
    xbar (Nm,Nf,Nc,Ns) the individual template used in the CCA (eq 9).
    """
    nm, nf, nb, nc, ns = epochs_t.shape
    tr = epochs_t[:, :, sub]                              # (Nm,Nf,ntb,Nc,Ns)
    z = tr - tr.mean(axis=4, keepdims=True)
    pz = np.einsum("mihcn,mihkn->mihck", z, z)
    zsum = z.sum(axis=2)
    pzsum = pz.sum(axis=2)
    total = np.einsum("micn,mikn->mick", zsum, zsum)
    q_all = pzsum / (ns - 1.0)
    s_all = (total - pzsum) / (ns - 1.0)
    wTa = np.zeros((nm, nf, nc))
    xTa = np.zeros((nm, nf, ns))
    xbar = np.zeros((nm, nf, nc, ns))
    for m in range(nm):
        for i in range(nf):
            wi = eigvec(s_all[m, i], q_all[m, i])
            xb = tr[m, i].mean(axis=0)
            wTa[m, i] = wi
            xTa[m, i] = wi @ xb
            xbar[m, i] = xb
    return wTa, xTa, xbar


def similarity(xTa, y_all):
    """TRC similarity coefficient c (eq 15), its normalised form (eq 17) and
    the two different extremes the paper uses for them.

    eq (18) normalises by ``cmax_abs`` = max |c|, whereas the trigger of
    eq (16) is the *signed* comparison "c_i^n <= gamma for all n", i.e.
    ``cmax_signed`` = max c.  They are NOT interchangeable: ~43% of TRC
    correlations are negative here, so the two rules disagree on ~28% of
    (sub-band, stimulus) cells.

    xTa : (Nm,Nf,Ns);  y_all : (Nm,Nsub,Nf,Ns)
    -> c, ctilde (Nm,Nsub,Nf), cmax_abs, cmax_signed (Nm,Nf)
    """
    xc = xTa - xTa.mean(axis=2, keepdims=True)
    yc = y_all - y_all.mean(axis=3, keepdims=True)
    xn = np.linalg.norm(xc, axis=2)
    yn = np.linalg.norm(yc, axis=3)
    num = np.einsum("min,msin->msi", xc, yc)
    c = num / np.maximum(xn[:, None, :] * yn, 1e-15)
    cmax_abs = np.abs(c).max(axis=1)
    cmax_signed = c.max(axis=1)
    return c, np.abs(c) / np.maximum(cmax_abs[:, None, :], 1e-15), \
        cmax_abs, cmax_signed


def mask_for(ctilde, cmax_signed, gamma, clb):
    """Selection rule, eqs (16)/(19). (Nm,Nsub,Nf) boolean.

    Selection is disabled (all sources kept) unless some source correlates
    *positively* above gamma with the target, eq (16).
    """
    triggered = cmax_signed > gamma
    return np.where(triggered[:, None, :], ctilde > clb, True)


# --------------------------------------------------------------- test stage
def _pearson(a, b):
    ac = a - a.mean(axis=-1, keepdims=True)
    bc = b - b.mean(axis=-1, keepdims=True)
    na = np.linalg.norm(ac, axis=-1)
    nb = np.linalg.norm(bc, axis=-1)
    return (ac * bc).sum(axis=-1) / np.maximum(na * nb, 1e-15)


def evaluate_subject(epochs_t, y_sub, clbs, gamma, ntb, subset_mode="all"):
    """One target subject: accuracy and mean source count for each clb.

    epochs_t : (Nm,Nf,Nb,Nc,Ns)  target epochs
    y_sub    : (Nsub,Nm,Nf,Ns)   source TRCs (target excluded)
    """
    nm, nf, nb, nc, ns = epochs_t.shape
    nsub = y_sub.shape[0]
    Yt = np.transpose(y_sub, (1, 0, 2, 3))                # (Nm,Nsub,Nf,Ns)
    Yc = Yt - Yt.mean(axis=3, keepdims=True)
    grams = np.einsum("msin,mtin->msti", Yc, Yc)          # (Nm,Nsub,Nsub,Nf)
    acc = {c: [] for c in clbs}
    nsel = {c: [] for c in clbs}
    nsel_stim = {c: [] for c in clbs}

    for b in range(nb):
        rest = [h for h in range(nb) if h != b]
        subsets = list(itertools.combinations(rest, ntb))
        if subset_mode == "first":
            subsets = [subsets[0]]
        for sub in subsets:
            wTa, xTa, xbar = target_model(epochs_t, list(sub))
            _, ctilde, _, cmax_signed = similarity(xTa, Yt)
            test = epochs_t[:, :, b]                      # (Nm,Nf,Nc,Ns)

            # subject-specific feature rho2, eq (11) -- shared by all methods
            proj_ta = np.einsum("mtcn,mic->mtin", test, wTa)
            rho2 = _pearson(proj_ta, xTa[:, None, :, :])

            for clb in clbs:
                mask = mask_for(ctilde, cmax_signed, gamma, clb) if clb < 1 else \
                    np.zeros_like(ctilde, dtype=bool)
                nsel[clb].append(mask.sum(axis=1).mean())
                nsel_stim[clb].append(mask.sum(axis=1).mean(axis=0))
                rho1 = np.zeros_like(rho2)
                if mask.any():
                    for m in range(nm):
                        ycm = Yc[m]                   # (Nsub,Nf,Ns)
                        wgt = np.empty((nf, nc))
                        wsrc = np.empty((nf, ns))     # (w_i^GS)^T Y_i^S, eq (10)
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
                        # rho1 for all (trial, candidate) at once
                        proj = np.einsum("tcn,ic->tin", test[m], wgt)
                        pc = proj - proj.mean(axis=2, keepdims=True)
                        pn = np.linalg.norm(pc, axis=2)
                        wc = wsrc - wsrc.mean(axis=1, keepdims=True)
                        wn = np.linalg.norm(wc, axis=1)
                        rho1[m] = np.einsum("tin,in->ti", pc, wc) / \
                            np.maximum(pn * wn[None, :], 1e-15)
                    rho = np.sign(rho1) * rho1 ** 2 + np.sign(rho2) * rho2 ** 2
                else:
                    rho = np.sign(rho2) * rho2 ** 2       # TRCA, eq (21)
                feat = (rho * SUB_WEIGHTS[:, None, None]).sum(axis=0)
                pred = feat.argmax(axis=1)
                acc[clb].append(float((pred == np.arange(nf)).mean()))

    return {c: float(np.mean(acc[c])) for c in clbs}, \
           {c: float(np.mean(nsel[c])) for c in clbs}, \
           {c: np.mean(np.stack(nsel_stim[c]), axis=0) for c in clbs}


# ------------------------------------------------------------------ driver
def run_config(d, nc, ntb, clbs=(0.0, 0.9, 1.0), gamma=0.5, subjects=35,
               subset_mode="all", verbose=True):
    ch = CH_IDX9[:nc]
    t0 = time.time()
    ys = {}
    for s in range(1, subjects + 1):
        ys[s] = source_trcs(preprocess(s, ch, d))
    Y_all = np.stack([ys[s] for s in range(1, subjects + 1)])
    t_pre = time.time() - t0

    acc = {c: [] for c in clbs}
    nsel = {c: [] for c in clbs}
    nsel_stim = {c: [] for c in clbs}
    for t in range(1, subjects + 1):
        others = np.array([s for s in range(1, subjects + 1) if s != t])
        ep_t = preprocess(t, ch, d)
        a, n, ns_stim = evaluate_subject(ep_t, Y_all[others - 1], clbs, gamma,
                                         ntb, subset_mode)
        for c in clbs:
            acc[c].append(a[c])
            nsel[c].append(n[c])
            nsel_stim[c].append(ns_stim[c])
        if verbose:
            print("  S%-3d %s  (%.0fs)" % (
                t, " ".join("clb%.1f=%.4f" % (c, a[c]) for c in clbs),
                time.time() - t0), flush=True)

    out = dict(d=d, nc=nc, ntb=ntb, gamma=gamma, subset_mode=subset_mode,
               preprocess_seconds=t_pre, subjects=subjects)
    for c in clbs:
        arr = np.array(acc[c])
        out["clb%.3f" % c] = dict(
            acc_mean=float(arr.mean()), acc_sem=float(arr.std(ddof=1) / np.sqrt(len(arr))),
            acc_per_subject=arr.tolist(),
            acc_percent=float(arr.mean() * 100),
            itr=float(itr_bits_per_min(arr.mean(), d=d)),
            n_selected_mean=float(np.mean(nsel[c])),
            n_selected_per_subject=np.array(nsel[c]).tolist(),
            n_selected_per_stim=np.array(nsel_stim[c]).tolist(),
        )
    return out


def clbs_for(ntb):
    return (0.0, 0.9, 1.0)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--d", type=float, nargs="+", default=[1.0])
    p.add_argument("--nc", type=int, nargs="+", default=[9])
    p.add_argument("--ntb", type=int, nargs="+", default=[3])
    p.add_argument("--clbs", type=float, nargs="+", default=[0.0, 0.9, 1.0])
    p.add_argument("--gamma", type=float, default=0.5)
    p.add_argument("--subjects", type=int, default=35)
    p.add_argument("--subset-mode", default="all", choices=["all", "first"])
    p.add_argument("--out", default=None)
    a = p.parse_args()

    os.makedirs(os.path.join(HERE, "..", "results"), exist_ok=True)
    allres = []
    for d in a.d:
        for nc in a.nc:
            for ntb in a.ntb:
                print("=== d=%.1f nc=%d ntb=%d ===" % (d, nc, ntb), flush=True)
                r = run_config(d, nc, ntb, tuple(a.clbs), a.gamma, a.subjects,
                               a.subset_mode)
                allres.append(r)
                print("  -> " + "  ".join(
                    "clb%.1f acc=%.2f%% nsel=%.1f" %
                    (c, r["clb%.3f" % c]["acc_percent"],
                     r["clb%.3f" % c]["n_selected_mean"]) for c in a.clbs),
                    flush=True)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(allres, f, indent=1)
        print("wrote", a.out)
