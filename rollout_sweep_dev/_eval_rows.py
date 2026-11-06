"""How much does the equilibrium concentration move with the forcing, in the model and in the data?

The architecture predicts an equilibrium from the inputs and relaxes toward it, so the equilibrium is
supposed to be a function of the forcing. The regime experiment found it barely moving: three ice
regimes whose observed summer minima span 0.86 to 0.01 gave box equilibria of 0.926, 0.919 and 0.919.
The forcing those regimes differ by is small, 2.65 K in air temperature, so a small response is not
by itself wrong. The question is whether the SENSITIVITY is right.

The reanalysis has an equilibrium too, and it is measurable the same way the model's is: the
concentration at which the mean observed tendency crosses zero, at a given forcing. Binning the data
by forcing and then by concentration gives that curve, and running the model over the same rows gives
its own. Comparing how the two move as the forcing is swept is the test.

Cells are restricted to the interior, both gradient columns and the second difference small, so the
comparison matches the box configuration where the flatness was found, and so that the zero crossing
is a statement about local thermodynamics rather than about proximity to an edge.

Usage: equilibrium_sensitivity.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import rollout_lib as R
import central_pack_rollout as C
from box_emulation import GATES, SUF_LOW, SUF_HIGH, SUF_MID

torch.set_grad_enabled(False)
YEARS = [1990, 1991]
# The suffixes come from box_emulation, which reads them from the environment with the old
# literals as defaults. They used to be written out here as well, so this module silently
# pinned itself to whichever cache that one suffix maps to.
models, norms = C.load(SUF_LOW, SUF_HIGH, SUF_MID)
gate = GATES["north"]
# [2026-10-14] spatial_sigma MUST be 0. rollout_lib's default is 1.0, every training cache is built
# with 0.0, and every rollout script overrides it for that reason. Without this the component is fed a
# Gaussian-smoothed field it was never trained on: at SIC 0.15-0.80 the smoothed tendencies have
# std 0.08169 against the unsmoothed 0.12798 and the training cache's 0.12283, a factor of 1.57, which
# made the component look over-amplified by 1.55 when it is in fact calibrated.
_orig_blocked = R.build_blocked_fields_2deg
R.build_blocked_fields_2deg = (lambda f, h, **kw: _orig_blocked(f, h, **{**kw, "spatial_sigma": 0.0}))
ctx = R.make_rollout_context(R.collect_processed_files(YEARS),
                             R.PROCESSED_DIR / "processed_1989.nc", "north")
R.build_blocked_fields_2deg = _orig_blocked
B = ctx["blocked"]; vm = ctx["valid_mask_final"]
nb = B["siconc"].shape[0]
i_idx, j_idx = ctx["i_idx"], ctx["j_idx"]

# [2026-10-12] "tos" added. Every melt diagnostic built on these rows used a FIXED 271.35 for it,
# which is right under dense ice (the phase change pins it) but too cold where there is open water,
# and that biases the model toward growth exactly where the open water is.
cols = {k: [] for k in ("sic", "tend", "tas", "rsds", "rld", "tdps", "wind", "tos",
                        "g_ns", "g_we", "curv", "h3", "h2", "h1")}
for t in range(3, nb - 1):
    f = B["siconc"][t]
    lp = np.empty_like(f); lp[1:] = f[:-1]; lp[0] = np.nan
    ln = np.empty_like(f); ln[:-1] = f[1:]; ln[-1] = np.nan
    d = lambda x: np.where(np.isfinite(x), x - f, 0.0)
    g_ns = (d(lp) - d(ln)) / 2.0
    w_, e_ = d(np.roll(f, 1, 1)), d(np.roll(f, -1, 1))
    g_we = (w_ - e_) / 2.0
    curv = np.tanh(((w_ + e_) / 2.0) / 0.5)
    cols["sic"].append(f[i_idx, j_idx]); cols["tend"].append((B["siconc"][t + 1] - f)[i_idx, j_idx])
    for k, v in (("tas", "tas"), ("rsds", "rsds"), ("rld", "rld"), ("tdps", "tdps"),
                 ("wind", "sfcWind"), ("tos", "tos")):
        cols[k].append(B[v][t + 1][i_idx, j_idx])
    cols["g_ns"].append(g_ns[i_idx, j_idx]); cols["g_we"].append(g_we[i_idx, j_idx])
    cols["curv"].append(curv[i_idx, j_idx])
    for k, dt in (("h3", 3), ("h2", 2), ("h1", 1)):
        cols[k].append(B["siconc"][t - dt][i_idx, j_idx])
D = {k: np.concatenate(v).astype(np.float64) for k, v in cols.items()}
ok = np.all([np.isfinite(D[k]) for k in D], axis=0)
interior = ok & (np.abs(D["g_ns"]) < 0.02) & (np.abs(D["g_we"]) < 0.02) & (np.abs(D["curv"]) < 0.05)
print(f"samples {ok.sum():,}, interior {interior.sum():,}")


def model_tend(idx):
    n = len(idx)
    f32 = lambda k: D[k][idx].astype(np.float32)
    sv = f32("sic")
    cr = np.stack([D["g_ns"][idx], D["g_we"][idx], D["curv"][idx]], axis=1).astype(np.float32)
    hv = np.stack([D["h3"][idx], D["h2"][idx], D["h1"][idx]], axis=1).astype(np.float32)
    args = (sv, np.full(n, 271.35, np.float32), f32("rsds"), np.full(n, -30.0, np.float32),
            f32("wind"), f32("tas"), np.zeros(n, np.float32), np.zeros(n, np.float32),
            f32("rld"), f32("tdps"))
    sg = lambda x: 1.0 / (1.0 + np.exp(-x))
    gl = sg((gate[0] - sv) / gate[1]); gh = sg((sv - gate[2]) / gate[3])
    gm = np.clip(1 - gl - gh, 0, 1)
    td = {b: R._model_forward(models[b], R.build_rollout_inputs(*args, cr, norms[b]), sv,
                              np.full(n, 271.35, np.float32), hv) for b in models}
    return (gl * td["low"] + gm * td["mid"] + gh * td["high"]).astype(np.float64)


def zero_cross(centres, means, counts):
    good = counts >= 120
    c, m = centres[good], means[good]
    for i in range(len(c) - 1):
        if m[i] > 0 >= m[i + 1]:
            return c[i] + (c[i + 1] - c[i]) * m[i] / (m[i] - m[i + 1])
    return np.nan


