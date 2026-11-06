"""0-D box test, SUPPORT-AWARE. Supersedes box_emulation.py, which was invalid.

Why v1 was wrong. It fixed the forcing at the Arctic JFM pack mean and swept tas over 235-253 K at
SIC 0.55. But mid-concentration northern points are MELT-SEASON points -- the training tas at SIC
0.35-0.75 runs 256-275 K with a median of 271.6, because in deep winter the Arctic is at SIC 0.9+, not
0.55. The whole sweep sat 20-35 K below its own support, and it reported a reversed thermal response
that disappears entirely once the sweep is put back inside it. Five "controls" were run against that
artefact and none of them helped, because all five varied the wrong thing.

What this does instead. Every sweep point is placed at the MEDIAN FORCING OF ITS OWN SIC BIN, measured
from the training cache, and each swept variable moves only between that bin's 5th and 95th percentile.
Variables that co-vary with the swept one (tdps and rld with tas) follow the in-bin regression rather
than being held fixed. So the box stays on the joint distribution the model was fitted to.

What a box can and cannot establish here, which is the honest framing. A single gradient-free cell with
no history cannot test magnitudes: the model's tendency is a function of the full state, and a box
fixes most of that state at a value the data only supplies jointly. What it CAN test is the RESPONSE
DIRECTION to one boundary condition at a time, inside the support -- and the sign of a response is
exactly what a box model is for. The implied-thickness column is therefore reported as a diagnostic of
internal consistency across the sweep, not as a measurement of a physical thickness.

Usage: box_emulation_v2.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import rollout_lib as R
import central_pack_rollout as C
import sic_pipeline_lib as L
from box_emulation import f_open, GATES, T_F, PS, RHO_I, L_FUS, BLOCK_SECONDS, SUF_LOW, SUF_HIGH, SUF_MID

torch.set_grad_enabled(False)
CD = Path("cache_curv_tanh")
BINS = ((0.15, 0.35), (0.35, 0.55), (0.55, 0.75), (0.75, 0.95))


def load_support():
    tr, _, _, _ = L.make_pooled_loaders(
        str(CD / "north_train_midband_rld_tdps_gradbasis.npz"), str(CD / "south_train_midband_rld_tdps_gradbasis.npz"),
        str(CD / "north_val_midband_rld_tdps_gradbasis.npz"), str(CD / "south_val_midband_rld_tdps_gradbasis.npz"),
        feature_set="sic+sst+rsds+rld+sfcwind+tas+tdps", normalize_phys=True,
        batch_size=L.BATCH_SIZE, use_cross=True)
    fn = list(tr.feature_names); X = np.asarray(tr.X, dtype=np.float64)
    d = dict(tas=X[:, fn.index("tas")] * tr.std[5] + tr.mean[5],
             rsds=X[:, fn.index("rsds")] * tr.std[2] + tr.mean[2],
             wind=X[:, fn.index("sfcwind")] * tr.std[4] + tr.mean[4],
             rld=X[:, fn.index("rld")] * tr.rld_std + tr.rld_mean,
             tdps=X[:, fn.index("tdps")] * tr.tdps_std + tr.tdps_mean,
             # [2026-10-16] the ocean temperature each sample actually had. The box used to feed the
             # model T_F regardless, which conflates two different things: T_F is the assumed skin
             # temperature of OPEN WATER in the energy budget, while `tos` is the mixed layer
             # temperature the network was trained on. In the Arctic the two separate by up to 2.8 K
             # under warm air, and the network melts through `tos`.
             tos=np.asarray(tr.tos_raw, dtype=np.float64))
    ym, ys = tr.base_target_stats
    y = np.asarray(tr.y, dtype=np.float64); y = y * ys + ym if y.std() < 5 else y
    return (d, np.asarray(tr.sic_raw, dtype=np.float64), np.asarray(tr.hemi_label),
            np.asarray(tr.ocean), y)


D, SIC_R, HEMI, OC, Y = load_support()
# The suffixes come from box_emulation, which reads them from the environment with the old
# literals as defaults. They used to be written out here as well, so this module silently
# pinned itself to whichever cache that one suffix maps to.
models_n, norms_n = C.load(SUF_LOW, SUF_HIGH, SUF_MID)


def tend(sic_, G, models, norms, gate):
    sv = np.full(1, sic_, dtype=np.float32); hv = np.repeat(sv[:, None], 3, axis=1)
    z3 = np.zeros((1, 3), dtype=np.float32); g = lambda k: np.full(1, G[k], dtype=np.float32)
    args = (sv, g("tos"), g("rsds"), g("rls"), g("wind"), g("tas"),
            np.zeros(1, np.float32), np.zeros(1, np.float32), g("rld"), g("tdps"))
    sg = lambda x: 1.0 / (1.0 + np.exp(-x))
    gl = sg((gate[0] - sv) / gate[1])
    gh = sg((sv - gate[2]) / gate[3]) if gate[2] is not None else np.zeros_like(gl)
    gm = np.clip(1 - gl - gh, 0, 1)
    td = {b: R._model_forward(models[b], R.build_rollout_inputs(*args, z3, norms[b]), sv, g("tos"), hv)
          for b in models}
    return float((gl * td["low"] + gm * td["mid"] + (gh * td["high"] if "high" in td else 0.0)).mean())


def h_imp(sic_, dA, G):
    rate = dA / BLOCK_SECONDS
    Fo = f_open(G["tas"], G["tdps"], G["rld"], G["rsds"], G["wind"], PS)
    if rate <= 0 or Fo <= 0:
        return float("nan"), Fo
    return (1 - sic_) * Fo / (RHO_I * L_FUS * rate), Fo


print("NORTH, every point at its own SIC bin's median forcing, each sweep inside that bin's p5-p95")
for lo, hi in BINS:
    m = (OC > 0) & (HEMI == 0) & (SIC_R >= lo) & (SIC_R < hi)
    sc = (lo + hi) / 2
    base = {k: float(np.percentile(D[k][m], 50)) for k in ("rsds", "rld", "wind", "tas", "tdps")}
    base["tos"] = T_F; base["rls"] = -30.0
    s_td = np.polyfit(D["tas"][m], D["tdps"][m], 1)
    s_rld = np.polyfit(D["tas"][m], D["rld"][m], 1)
    q1, q2 = np.percentile(D["tas"][m], [33.3, 66.7])
    era_t = Y[m & (D["tas"] < q1)].mean() - Y[m & (D["tas"] >= q2)].mean()
    w1, w2 = np.percentile(D["wind"][m], [33.3, 66.7])
    era_w = Y[m & (D["wind"] >= w2)].mean() - Y[m & (D["wind"] < w1)].mean()
    print(f"\n  SIC bin [{lo:.2f},{hi:.2f}), n={m.sum():,}  median forcing: "
          + "  ".join(f"{k}={base[k]:.1f}" for k in ("tas", "rld", "rsds", "wind")))
    print(f"    {'tas':>7s} | {'F_open':>8s} | {'dSIC/5d':>9s} | {'h_imp [m]':>10s}")
    for p in (5, 25, 50, 75, 95):
        T = float(np.percentile(D["tas"][m], p))
        G = dict(base); G["tas"] = T
        G["tdps"] = float(np.polyval(s_td, T)); G["rld"] = float(np.polyval(s_rld, T))
        dA = tend(sc, G, models_n, norms_n, GATES["north"])
        h, Fo = h_imp(sc, dA, G)
        print(f"    {T:7.2f} | {Fo:8.1f} | {dA:+9.4f} | {h:10.3f}")
    Tl, Th = float(np.percentile(D["tas"][m], 5)), float(np.percentile(D["tas"][m], 95))
    gl_ = dict(base); gl_["tas"] = Tl; gl_["tdps"] = float(np.polyval(s_td, Tl)); gl_["rld"] = float(np.polyval(s_rld, Tl))
    gh_ = dict(base); gh_["tas"] = Th; gh_["tdps"] = float(np.polyval(s_td, Th)); gh_["rld"] = float(np.polyval(s_rld, Th))
    mdl = tend(sc, gl_, models_n, norms_n, GATES["north"]) - tend(sc, gh_, models_n, norms_n, GATES["north"])
    print(f"    direction, colder minus warmer:  model {mdl:+.4f}   ERA5 {era_t:+.4f}   (box says > 0)")
    print(f"    {'wind':>7s} | {'F_open':>8s} | {'dSIC/5d':>9s} | {'h_imp [m]':>10s}")
    for p in (5, 50, 95):
        V = float(np.percentile(D["wind"][m], p))
        G = dict(base); G["wind"] = V
        dA = tend(sc, G, models_n, norms_n, GATES["north"])
        h, Fo = h_imp(sc, dA, G)
        print(f"    {V:7.2f} | {Fo:8.1f} | {dA:+9.4f} | {h:10.3f}")
    Vl, Vh = float(np.percentile(D["wind"][m], 5)), float(np.percentile(D["wind"][m], 95))
    a = dict(base); a["wind"] = Vh; b = dict(base); b["wind"] = Vl
    mdl_w = tend(sc, a, models_n, norms_n, GATES["north"]) - tend(sc, b, models_n, norms_n, GATES["north"])
    print(f"    direction, windier minus calmer: model {mdl_w:+.4f}   ERA5 {era_w:+.4f}   (box says > 0)")
print("\nBOX_EMULATION_V2_DONE")
