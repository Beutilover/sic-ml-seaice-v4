import os
"""Can the model emulate a 0-D sea-ice box? (client request, relayed 2026-10-04)

The box. A single cell, horizontally uniform, so all three cross columns are zero and the SIC history
equals the current SIC -- a genuine steady state with no spatial information, driven only by prescribed
boundary conditions. The model's own output is then its ONE-dimensional answer to "given these
conditions, how should the concentration change", which is exactly what a box model computes.

The theory. With CONCENTRATION as the state variable rather than thickness, the standard 0-D closure is
LATERAL growth: the energy the open water loses freezes new ice of some assumed thickness, so

    F_open = eps*sig*T_f^4 - Rld - (1-alb_w)*Rsd
             + rho*cp*Cd*V*(T_f - T_air) + Lv*rho*Cd*V*(q_sat,water(T_f) - q_air)      [W/m2]
    dA/dt  = (1 - A) * F_open / (rho_ice * L_fusion * h_new)

F_open positive means the open water is losing heat and ice forms. This is the same closure
SpeedyWeather's own sea_ice.jl uses, with its freeze rate standing in for 1/(rho*L*h).

Rather than guess h_new and compare, the test INVERTS it and measures the thickness the model behaves
as if it had:

    h_implied = (1 - A) * F_open / (rho_ice * L_fusion * (dA/dt)_model)

That turns an arbitrary guess into a number with two falsifiable properties: h_implied should be
physically plausible for new ice (roughly 0.05-0.3 m) and, more importantly, it should be ROUGHLY
CONSTANT across conditions. Constancy is the real content of "the model emulates the box", because
dA/dt is then proportional to (1-A) and to F_open separately, which is the box's entire structure.
Three sweeps test those proportionalities one variable at a time.

Two forcing sets, as mutual controls: anchored on measured ERA5 polar pack means, where the model is
inside its training manifold, and purely analytic round numbers, where it may not be. A divergence
between them measures how much of the behaviour is learned dynamics and how much is extrapolation.

Usage: box_emulation.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import rollout_lib as R
import central_pack_rollout as C

torch.set_grad_enabled(False)
# The suffix this module loads at import time. It used to be a bare literal, which tied
# every importer -- _eval_rows and box_forced_cycle among them -- to whichever cache that
# one suffix maps to. The defaults reproduce the old behaviour exactly.
SUF = os.environ.get("SIC_BOX_SUFFIX", "_hw2curvtanh100")
SUF_LOW  = os.environ.get("SIC_BOX_SUFFIX_LOW",  SUF)
SUF_HIGH = os.environ.get("SIC_BOX_SUFFIX_HIGH", SUF)
SUF_MID  = os.environ.get("SIC_BOX_SUFFIX_MID") or None
BLOCK_SECONDS = 5 * 86400.0          # the model's tendency is dSIC per 5-day block (DT = 5.0 days)
RHO_I, L_FUS = 917.0, 3.34e5         # kg/m3, J/kg
RHO_A, CP, CD = None, 1004.0, 0.9e-3
RD, RV, SIG, EPS_LW = 287.0, 461.53, 5.670374419e-8, 0.98
LV, LS, T0, E0, EPSM = 2501.0e3, 2801.0e3, 273.15, 610.78, 0.622
ALB_W, T_F = 0.06, 271.35
esat_w = lambda T: E0 * np.exp(LV / RV * (1 / T0 - 1 / T))
AW, BW = 17.625, 243.04
e_from_td = lambda Td: 610.94 * np.exp(AW * (Td - T0) / ((Td - T0) + BW))
GATES = {"north": (0.20, 0.035, 0.65, 0.05), "south": (0.02, 0.025, None, None)}


def f_open(tas, tdps, rld, rsds, wind, ps):
    """Energy the OPEN WATER at the freezing point loses, W/m2, positive = freezing."""
    rho = ps / (RD * tas)
    K = rho * CD * wind
    q_a = EPSM * e_from_td(tdps) / ps
    q_s = EPSM * esat_w(T_F) / ps
    return (EPS_LW * SIG * T_F ** 4 - rld - (1 - ALB_W) * rsds
            + K * CP * (T_F - tas) + LV * K * (q_s - q_a))


def model_tendency(models, norms, gate, sic, F, n=1):
    """Blended dSIC per 5-day block for a uniform box at concentration `sic`."""
    sv = np.full(n, sic, dtype=np.float32)
    hv = np.repeat(sv[:, None], 3, axis=1)
    z3 = np.zeros((n, 3), dtype=np.float32)
    g = lambda k: np.full(n, F[k], dtype=np.float32)
    args = (sv, g("tos"), g("rsds"), g("rls"), g("wind"), g("tas"),
            np.zeros(n, np.float32), np.zeros(n, np.float32), g("rld"), g("tdps"))
    sig = lambda x: 1.0 / (1.0 + np.exp(-x))
    gl = sig((gate[0] - sv) / gate[1])
    gh = sig((sv - gate[2]) / gate[3]) if gate[2] is not None else np.zeros_like(gl)
    gm = np.clip(1 - gl - gh, 0, 1)
    td = {b: R._model_forward(models[b], R.build_rollout_inputs(*args, z3, norms[b]), sv, g("tos"), hv)
          for b in models}
    o = gl * td["low"] + gm * td["mid"] + (gh * td["high"] if "high" in td else 0.0)
    return float(o.mean())


def h_implied(sic, dA_block, F, ps):
    """Thickness the model behaves as if it had, m. NaN when the model is not freezing."""
    rate = dA_block / BLOCK_SECONDS
    Fo = f_open(F["tas"], F["tdps"], F["rld"], F["rsds"], F["wind"], ps)
    if rate <= 0 or Fo <= 0:
        return float("nan"), Fo
    return (1 - sic) * Fo / (RHO_I * L_FUS * rate), Fo


def era5_baseline(hemi, months):
    _o = R.build_blocked_fields_2deg
    R.build_blocked_fields_2deg = (lambda f, h, **kw: _o(f, h, **{**kw, "spatial_sigma": 0.0, "spatial_block": 1}))
    R.cross_diff_2d = lambda s2, lat_coarse=None: np.zeros(s2.shape + (3,), np.float32)
    ctx = R.make_rollout_context(R.collect_processed_files(C.YEARS),
                                 R.PROCESSED_DIR / f"processed_{C.SEED}.nc", hemi)
    R.build_blocked_fields_2deg = _o
    i_idx, j_idx, bl = ctx["i_idx"], ctx["j_idx"], ctx["blocked"]
    deep = np.abs(np.asarray(ctx["lat_coarse"])[i_idx]) >= 70
    blocks = [k for k in range(3, ctx["n_blocks"] - 1) if ((((k % 73) * 5) // 30) + 1) in months]
    g = lambda v: np.concatenate([bl[v][t + 1][i_idx, j_idx][deep] for t in blocks])
    sic = np.concatenate([bl["siconc"][t][i_idx, j_idx][deep] for t in blocks])
    k = sic > 0.15
    out = {"tas": g("tas")[k].mean(), "tdps": g("tdps")[k].mean(), "rld": g("rld")[k].mean(),
           "rsds": g("rsds")[k].mean(), "wind": g("sfcWind")[k].mean(), "rls": g("rls")[k].mean(),
           "tos": T_F}
    return {kk: float(v) for kk, v in out.items()}


SETS = {}
SETS["ERA5 Arctic winter (JFM)"] = ("north", era5_baseline("north", (1, 2, 3)))
SETS["ERA5 Antarctic winter (JAS)"] = ("south", era5_baseline("south", (7, 8, 9)))
SETS["analytic (round numbers)"] = ("north", dict(tas=250.0, tdps=247.3, rld=160.0, rsds=0.0,
                                                  wind=8.0, rls=-30.0, tos=T_F))
PS = 0.98e5

for name, (hemi, F) in SETS.items():
    models, norms = C.load(SUF_LOW, SUF_HIGH, SUF_MID)
    if hemi == "south":
        models = {b: m for b, m in models.items() if b != "high"}
        norms = {b: v for b, v in norms.items() if b != "high"}
    gate = GATES[hemi]
    print(f"\n{'='*94}\n{name}   ({hemi} bands)")
    print("  forcing: " + "  ".join(f"{k}={F[k]:.2f}" for k in ("tas", "tdps", "rld", "rsds", "wind")))
    print(f"  F_open at the baseline = {f_open(F['tas'],F['tdps'],F['rld'],F['rsds'],F['wind'],PS):.1f} W/m2"
          f"   (positive = open water freezing)")

    print(f"\n  SWEEP 1, concentration. Box says dA/dt is proportional to (1-A), so h_implied is flat.")
    print(f"    {'SIC':>5s} | {'dSIC/5d':>9s} | {'(1-A) factor':>12s} | {'h_implied [m]':>13s}")
    for sic in (0.10, 0.25, 0.40, 0.55, 0.70, 0.85, 0.95):
        dA = model_tendency(models, norms, gate, sic, F)
        h, _ = h_implied(sic, dA, F, PS)
        print(f"    {sic:5.2f} | {dA:+9.4f} | {1-sic:12.2f} | {h:13.4f}")

    print(f"\n  SWEEP 2, air temperature. Box says dA/dt is proportional to F_open, so h_implied is flat.")
    print(f"    {'tas':>7s} | {'F_open':>8s} | {'dSIC/5d':>9s} | {'h_implied [m]':>13s}")
    for dT in (-15, -10, -5, 0, +3):
        G = dict(F); G["tas"] = F["tas"] + dT; G["tdps"] = F["tdps"] + dT   # hold the depression fixed
        dA = model_tendency(models, norms, gate, 0.55, G)
        h, Fo = h_implied(0.55, dA, G, PS)
        print(f"    {G['tas']:7.2f} | {Fo:8.1f} | {dA:+9.4f} | {h:13.4f}")

    print(f"\n  SWEEP 3, wind. Enters F_open only through the turbulent terms.")
    print(f"    {'wind':>5s} | {'F_open':>8s} | {'dSIC/5d':>9s} | {'h_implied [m]':>13s}")
    for V in (2.0, 5.0, 8.0, 12.0, 16.0):
        G = dict(F); G["wind"] = V
        dA = model_tendency(models, norms, gate, 0.55, G)
        h, Fo = h_implied(0.55, dA, G, PS)
        print(f"    {V:5.1f} | {Fo:8.1f} | {dA:+9.4f} | {h:13.4f}")

print("\n  new ice is typically 0.05-0.30 m thick; a flat h_implied in that range means the model is")
print("  reproducing the box's structure, a varying one means it is not.")
print("\nBOX_EMULATION_DONE")
