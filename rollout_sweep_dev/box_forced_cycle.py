"""Forced freeze/melt/freeze cycle (client request, relayed 2026-10-07).

Setup. A spatially uniform cell under FIXED external forcing, autoregressive over many consecutive
5-day steps rather than the one-step response box_emulation.py measures. The client's "3x3 biperiodic
box" reduces exactly to this: with a spatially uniform initial condition, uniform forcing, and a
deterministic model with no noise, every cell in a periodic domain stays identical to every other
cell for all time, so the three neighbour columns stay at zero throughout (their own-value neighbours
equal the centre) and the problem is one-dimensional in time.

Why this is a genuine test of what was learned, not a restatement of the architecture. The model is
never told the calendar: there is no day-of-year, latitude or hemisphere input anywhere in its feature
set (see section 1 of the log). A switch from a multi-decade seasonal cycle to a single sustained
forcing level is therefore not something the model could special-case even if it wanted to -- whatever
trajectory comes out is assembled from the same physical-input-to-tendency mapping that every other
number in this project comes from.

Two forcing sets, each the MEAN of the coldest and warmest 5 per cent of the full northern training
population by air temperature (not conditioned on concentration, since the point of this test is to
hold an external climate state fixed while concentration runs through its whole range). This keeps
every number as an observed climatological mean rather than an invented one, at the cost of leaving
the model's training support once concentration moves away from where that forcing actually occurs in
the data -- a deep-winter cold snap at SIC 0 essentially never happens in ERA5. That is flagged in the
output, not hidden.

Sequence: cold start at SIC = 0 with a stationary history (no information about what came before
day 0), freeze forcing for 1 year, melt forcing for 1 year, freeze forcing for 1 year, 219 five-day
blocks in total.

Usage: box_forced_cycle.py
"""
import os as _os
from pathlib import Path as _Path
# Repository root, so this script runs from a clone. Set SIC_ROOT to override.
_ROOT = _Path(_os.environ.get("SIC_ROOT", _Path(__file__).resolve().parent.parent))
_FIG = _ROOT / "report_figures"
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import rollout_lib as R
import os
_WSETS = {"hw2curvtanh100": ("_hw2curvtanh100", "_hw2curvtanh100", None),
          # the SHIPPED package as of 2026-10-12: low unchanged, mid self-distilled, high
          # state-contraction at lambda 0.3. Matches ml_seaice_weights_v4_selfdist_* in Julia.
          "selfdist": ("_hw2curvtanh100", "_hw2curvtanh100statemonog0.3",
                       "_hw2curvtanh100selfdist1"),
          # selfdist with the LOW band retrained on a cache that keeps the SIC == 0 mass point
          "zeroprop": ("_hw2curvtanh100zeroprop", "_hw2curvtanh100statemonog0.3",
                       "_hw2curvtanh100selfdist1"),
          # [2026-10-15] narrow mid bands distilled from the 128-wide teacher
          "dist64": ("_hw2curvtanh100zeroprop", "_hw2curvtanh100statemonog0.3",
                     "_hw2curvtanh100selfdist1_dist64_l1"),
          "dist48": ("_hw2curvtanh100zeroprop", "_hw2curvtanh100statemonog0.3",
                     "_hw2curvtanh100selfdist1_dist48_l1"),
          # [2026-10-15] all three bands on the physical-Laplacian curvature column
          "lap": ("_hw2curvtanh100zeroprop_lap", "_hw2curvtanh100statemonog0.3_lap",
                  "_hw2curvtanh100selfdist1_dist64_l1_lap"),
          "lap_mtas": ("_hw2curvtanh100zeroprop_lap", "_hw2curvtanh100statemonog0.3_lap",
                       "_hw2curvtanh100selfdist1_dist64_l1_mtas1000_lap"),
          # the SHIPPED package: mtas100, not the mtas1000 sweep arm above, with the mid
          # band's student matching the teacher under masking over ("tas","tdps","rld").
          "lapmtas": ("_hw2curvtanh100zeroprop_lap", "_hw2curvtanh100statemonog0.3_lap",
                      "_hw2curvtanh100selfdist1_dist64_l1_mtas100_md_lap")}
DEADBAND = float(os.environ.get("DEADBAND", "0"))
WSET = os.environ.get("WSET", "hw2curvtanh100")
_LM, _HI, _MID = _WSETS[WSET]
_TAG = "" if WSET == "hw2curvtanh100" else f"_{WSET}"

import central_pack_rollout as C
from box_emulation import GATES, BLOCK_SECONDS
from box_emulation_v2 import load_support

torch.set_grad_enabled(False)
D, SIC_R, HEMI, OC, Y = load_support()
models, norms = C.load(_LM, _HI, _MID)
gate = GATES["north"]
m_north = (OC > 0) & (HEMI == 0)

FIELDS = ("tas", "rsds", "wind", "rld", "tdps", "tos")
t5, t95 = np.percentile(D["tas"][m_north], [5, 95])
# [2026-10-19] tos used to be pinned at 271.35 (freezing point) in BOTH phases, including melt --
# conditioned on nothing, unlike every other field here. Fixed to take tos's own mean over the same
# coldest/warmest 5% subset the rest of the forcing is drawn from.
FREEZE = {k: float(D[k][m_north & (D["tas"] <= t5)].mean()) for k in FIELDS}
MELT = {k: float(D[k][m_north & (D["tas"] >= t95)].mean()) for k in FIELDS}
for G in (FREEZE, MELT):
    G["rls"] = -30.0
print("forcing (mean of the coldest/warmest 5% of the north population by tas):")
print("  freeze: " + "  ".join(f"{k}={v:.2f}" for k, v in FREEZE.items()))
print("  melt:   " + "  ".join(f"{k}={v:.2f}" for k, v in MELT.items()))

TAU_TRAINED = {"low": 8.64, "mid_inc": 8.68, "mid_dec": 8.55, "high_inc": 8.43, "high_dec": 8.55}
BLOCKS_PER_YEAR = 73           # 365 / 5
PHASES = [("freeze", FREEZE, BLOCKS_PER_YEAR), ("melt", MELT, BLOCKS_PER_YEAR),
          ("freeze", FREEZE, BLOCKS_PER_YEAR)]


def step(sic, hist3, G):
    sv = np.full(1, sic, dtype=np.float32)
    hv = np.array([hist3], dtype=np.float32)
    z3 = np.zeros((1, 3), dtype=np.float32); g = lambda k: np.full(1, G[k], dtype=np.float32)
    args = (sv, g("tos"), g("rsds"), g("rls"), g("wind"), g("tas"),
            np.zeros(1, np.float32), np.zeros(1, np.float32), g("rld"), g("tdps"))
    sig = lambda x: 1.0 / (1.0 + np.exp(-x))
    gl = sig((gate[0] - sv) / gate[1]); gh = sig((sv - gate[2]) / gate[3]); gm = np.clip(1 - gl - gh, 0, 1)
    td = {b: R._model_forward(models[b], R.build_rollout_inputs(*args, z3, norms[b]), sv, g("tos"), hv)
          for b in models}
    tendency = (gl * td["low"] + gm * td["mid"] + gh * td["high"]).item()
    if DEADBAND > 0 and gl.item() > 0.5:
        # the same calibrated change gate the rollout applies. This cycle COLD STARTS at zero, so it
        # is the diagnostic most exposed to a gate that can now refuse to move, and it is the one
        # that shows whether the component can still begin forming ice.
        pcal = R._calibrate_change_prob(
            R._change_prob(models["low"], R.build_rollout_inputs(*args, z3, norms["low"]),
                           sv, g("tos"), hv))
        if float(pcal[0]) < DEADBAND:
            tendency = 0.0
    return float(np.clip(sic + tendency, 0.0, 1.0)), (gl.item(), gm.item(), gh.item())


sic = 0.0
hist3 = [0.0, 0.0, 0.0]
traj, phase_lab, bands = [sic], [], []
for name, G, nblk in PHASES:
    for _ in range(nblk):
        sic_new, w = step(sic, hist3, G)
        hist3 = [hist3[1], hist3[2], sic]
        sic = sic_new
        traj.append(sic); phase_lab.append(name); bands.append(w)
traj = np.array(traj)
t_days = np.arange(len(traj)) * 5.0

# ---- extract a rate / effective relaxation timescale for each phase ----
print(f"\n{'phase':>8s} {'start day':>9s} {'SIC start':>9s} {'SIC end':>8s} | {'initial rate':>13s} | "
      f"{'tau_eff [d]':>11s} | {'dominant band (final)':>22s}")
off = 1
for name, _, nblk in PHASES:
    seg = traj[off - 1:off + nblk]
    tgt = seg[-1]                                       # asymptotic value this phase is heading to
    y0 = seg[0]
    # initial rate: least-squares slope over the first 6 blocks (30 days), a direct "dSIC/dt"
    k0 = min(6, nblk)
    slope = np.polyfit(np.arange(k0) * 5.0, seg[:k0 + 1][:k0], 1)[0] if k0 >= 2 else float("nan")
    # effective exponential relaxation: log|seg - tgt| should be linear in t with slope -1/tau
    resid = seg[:-1] - tgt
    valid = np.abs(resid) > 1e-5
    if valid.sum() > 5:
        tt = np.arange(len(resid))[valid] * 5.0
        lr = np.log(np.abs(resid[valid]))
        A = np.vstack([tt, np.ones_like(tt)]).T
        slope_log, _ = np.linalg.lstsq(A, lr, rcond=None)[0]
        tau_eff = -1.0 / slope_log if slope_log < 0 else float("nan")
    else:
        tau_eff = float("nan")
    w_final = bands[off + nblk - 2]
    dom = {"low": w_final[0], "mid": w_final[1], "high": w_final[2]}
    dom_lab = max(dom, key=dom.get)
    print(f"  {name:>8s} {(off-1)*5:9d} {y0:9.4f} {seg[-1]:8.4f} | {slope:13.5f} | {tau_eff:11.2f} | "
          f"{dom_lab} ({dom[dom_lab]:.2f})")
    off += nblk

print(f"\n  trained relaxation timescales for comparison (days, tau_floor + softplus(raw param), dt=1 block):")
for k, v in TAU_TRAINED.items():
    print(f"    {k:>10s}: {v:.2f}")

np.savez("/tmp/box_forced_cycle.npz" if False else
         f"{_FIG}/box_forced_cycle{_TAG}{'_db' if DEADBAND>0 else ''}.npz",
         t_days=t_days, traj=traj, phase_lab=np.array(["start"] + phase_lab))
print("BOX_FORCED_CYCLE_DONE")


# ---------------------------------------------------------------------------------------------
# Sensitivity check: the "melt" forcing above is the mean of the warmest 5% of the WHOLE north
# population by tas, unconditioned on concentration. Once the box is near SIC 0.97 that specific
# combination (tas 275.5, rld 297.8) essentially never co-occurs with that much ice in the real
# data -- the warm extremes of the full population draw in points that are warm in part because
# they carry more cloud than what actually sits over dense pack. Rerun with forcing conditioned
# on the SIC~0.9 bin instead (same construction as box_emulation_v2/v3: median/regression within
# the bin, evaluated at its own 90th percentile tas) to see how much the result depends on that
# choice, holding everything else in the script identical.
# ---------------------------------------------------------------------------------------------
m90 = m_north & (SIC_R >= 0.83) & (SIC_R < 0.97)
s_td = np.polyfit(D["tas"][m90], D["tdps"][m90], 1)
s_rld = np.polyfit(D["tas"][m90], D["rld"][m90], 1)
# [2026-10-19] tos regressed the same way as tdps/rld within this bin, rather than pinned at
# freezing -- the same bug this whole fix addresses, in the one place this branch still had it.
s_tos = np.polyfit(D["tas"][m90], D["tos"][m90], 1)
T90 = float(np.percentile(D["tas"][m90], 90))
MELT_COND = dict(tas=T90, rsds=float(np.percentile(D["rsds"][m90], 90)),
                 wind=float(np.percentile(D["wind"][m90], 50)),
                 rld=float(np.polyval(s_rld, T90)), tdps=float(np.polyval(s_td, T90)),
                 tos=float(np.polyval(s_tos, T90)), rls=-30.0)
print("\nSENSITIVITY: same cycle, melt forcing conditioned on the SIC~0.9 bin instead")
print("  melt (bin-conditioned): " + "  ".join(f"{k}={v:.2f}" for k, v in MELT_COND.items()))
PHASES2 = [("freeze", FREEZE, BLOCKS_PER_YEAR), ("melt", MELT_COND, BLOCKS_PER_YEAR),
          ("freeze", FREEZE, BLOCKS_PER_YEAR)]
sic2, hist3b = 0.0, [0.0, 0.0, 0.0]
traj2, lab2 = [sic2], []
for name, G, nblk in PHASES2:
    for _ in range(nblk):
        sic2, _ = step(sic2, hist3b, G)
        hist3b = [hist3b[1], hist3b[2], traj2[-1]]
        traj2.append(sic2); lab2.append(name)
traj2 = np.array(traj2)
print(f"  trajectory: SIC after year 1 (freeze) {traj2[73]:.4f}, after year 2 (melt) {traj2[146]:.4f}, "
      f"after year 3 (freeze) {traj2[219]:.4f}")

np.savez(f"{_FIG}/box_forced_cycle{_TAG}{'_db' if DEADBAND>0 else ''}.npz",
         t_days=t_days, traj=traj, phase_lab=np.array(["start"] + phase_lab),
         traj2=traj2, forcing_freeze=FREEZE, forcing_melt=MELT, forcing_melt_cond=MELT_COND)

# ---------------------------------------------------------------------------------------------
# Figure: the two trajectories, phase boundaries marked.
# ---------------------------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(9.5, 4.2), facecolor="white")
for lo, hi, col in ((0, 365, "#dce8f2"), (365, 730, "#f6ddd3"), (730, 1095, "#dce8f2")):
    ax.axvspan(lo, hi, color=col, alpha=0.6, lw=0, zorder=0)
ax.plot(t_days, traj, lw=2.2, color="#1b4f72",
        label="melt forcing = warmest 5% of the WHOLE population")
ax.plot(t_days, traj2, lw=2.2, color="#b03a2e", ls="--",
        label="melt forcing = warmest decile of the SIC~0.9 bin")
for d_ in (365, 730):
    ax.axvline(d_, color="#888", lw=0.8, ls=":")
ax.text(180, 1.02, "freeze", ha="center", fontsize=9, color="#2c3e50")
ax.text(547, 1.02, "melt", ha="center", fontsize=9, color="#2c3e50")
ax.text(912, 1.02, "freeze", ha="center", fontsize=9, color="#2c3e50")
ax.set_xlabel("day (cold start at SIC = 0, no calendar input anywhere in the model)")
ax.set_ylabel("concentration")
ax.set_ylim(0, 1.05); ax.set_xlim(0, 1095)
ax.grid(alpha=0.25, lw=0.6)
ax.legend(fontsize=8.5, loc="center right")
ax.set_title("Forced freeze / melt / freeze cycle, spatially uniform box, no spatial or calendar input",
             fontsize=11)
fig.tight_layout()
fn = f"{_FIG}/box_forced_cycle{_TAG}{'_db' if DEADBAND>0 else ''}.png"
fig.savefig(fn, dpi=140, facecolor="white", bbox_inches="tight")
print(f"\nsaved {fn}")
print("BOX_FORCED_CYCLE_V2_DONE")
