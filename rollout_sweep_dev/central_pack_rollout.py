"""
Does variance matching keep the central Arctic pack in a free 1-degree offline rollout?

This is the metric that diagnosed the problem in the first place, so it is the one that decides it:
over >=85N ocean points in the melt-season blocks, what fraction of the model's cells fall below
SIC 0.5 (ERA5 never does; the shipped weights put 0.04% there, the widened high band 1.49%).
The shrinkage table says whether the dynamic range came back; this says whether the pack survives.

Only the HIGH band differs between the candidates -- low and mid are the same *_highwide2
checkpoints throughout -- and the gate is held fixed at the highwide gate for every candidate, so
the comparison isolates the high band. A proper per-weight-set gate sweep comes after a winner is
picked, not before.

True-gradient cross convention and 1-degree grid, matching the offline replay these numbers came
from.

Usage: central_pack_rollout.py [set_name ...]
"""
import os as _os
from pathlib import Path as _Path
# Repository root, so this script runs from a clone. Set SIC_ROOT to override.
_ROOT = _Path(_os.environ.get("SIC_ROOT", _Path(__file__).resolve().parent.parent))
_FIG = _ROOT / "report_figures"
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import torch
import rollout_lib as R
import sic_pipeline_lib as L

ROOT = _ROOT
YEARS, SEED = [1990, 1991], 1989
LAT_MIN = 85.0

GATE_HIGHWIDE = dict(gate_low_center=0.11, gate_width=0.055, gate_high_center=0.65, gate_width_high=0.05)
GATE_NOSMOOTH = dict(gate_low_center=0.24, gate_width=0.085, gate_high_center=0.65, gate_width_high=0.01)

# name -> (low/mid suffix, high suffix, gate), or (low, mid, high, gate) when the mid band differs
SETS = {
    # mid-band growth-shortfall variants: low and HIGH are untouched highwide2 (user, 2026-09-30:
    # the high band should be the one with none of these operations applied)
    # the curvature column + damping penalty retrained on the highwide2 recipe, with its OWN swept
    # gate (gate_sweep_banded_v4.py hw2curvdamp100 north, TRAIN years): (0.20, 0.035, 0.65, 0.05),
    # TRAIN CORR 0.97333 against highwide2's 0.97144. The low gate moved a lot (0.11 -> 0.20), which
    # is exactly why a gate may never be carried over between weight sets.
    # same recipe with the curvature column bounded by tanh(c/0.5); its swept gate came out
    # IDENTICAL to the raw column's, and so did every in-distribution metric
    # magnitude-target variants: MID band retrained, low and high reused from hw2curvtanh100
    # its own swept gate: the low WIDTH moved 0.035 -> 0.05 once the mid band's curvature response
    # was strengthened, which is why a gate is re-swept for every weight set
    # [2026-10-11] "hw2curvtanh100statemono": mid and high retrained with the GLOBAL
    # state-contraction penalty, mean(relu(d eq / d c - 1)^2), at lambda 1 and 0.3. It keeps the
    # CONTROL'S OWN GATE deliberately: its swept optimum was (0.19, 0.01) but the VAL surface is
    # flat in the low width (0.97366 at 0.010 against 0.97336 at 0.035), so the 0.0003 is not worth
    # a near-discontinuous low gate, and an identical gate makes the comparison a pure weights
    # comparison. At that identical gate the new weights beat the old on VAL 1989-90: CORR 0.97336
    # against 0.96970, RMSE 0.08643 against 0.09198, yearly-min error -0.0037 against -0.0053.
    # [2026-10-12] statemono with the mid band also trained for self-consistency under a masked
    # thermodynamic input. Gate carried over pending its own sweep.
    "hw2curvtanh100selfdist": ("_hw2curvtanh100", "_hw2curvtanh100selfdist1",
                               "_hw2curvtanh100statemonog0.3",
                               dict(gate_low_center=0.20, gate_width=0.035,
                                    gate_high_center=0.65, gate_width_high=0.05)),
    # [2026-10-13] the adopted mid/high with the LOW band retrained on a cache that keeps the
    # SIC == 0 mass point, carrying its OWN swept gate (gate_sweep_zeroprop_north.log).
    "hw2curvtanh100zeroprop": ("_hw2curvtanh100zeroprop", "_hw2curvtanh100selfdist1",
                               "_hw2curvtanh100statemonog0.3",
                               dict(gate_low_center=0.15, gate_width=0.005,
                                    gate_high_center=0.65, gate_width_high=0.05)),
    # [2026-10-14] the adopted package with the HIGH band retrained on BOTH hemispheres. Listed so
    # the southern high gate can be tested through the normal chain; not adopted.
    # the adopted package with a SOUTH-ONLY high band. The north is untouched by construction.
    "hw2curvtanh100dist64": ("_hw2curvtanh100zeroprop", "_hw2curvtanh100selfdist1_dist64_l1",
                             "_hw2curvtanh100statemonog0.3",
                             dict(gate_low_center=0.20, gate_width=0.035,
                                  gate_high_center=0.65, gate_width_high=0.05)),
    # [2026-10-15] the adopted dist64 package with the curvature column replaced by the PHYSICAL
    # Laplacian, ((W-C)+(E-C))/2 + cos^2(lat) ((N-C)+(S-C))/2, all three bands retrained on it. Gate
    # re-swept from the adopted point; see gate_adopted_lap_*.log.
    # [2026-10-19] the shipped low+change-gate package with BOTH trainable bands swapped to their
    # tos-masked versions, to test whether relaxing the online tos ceiling is safe once the bands no
    # longer over-react to an out-of-envelope ocean. Not yet shipped.
    "hw2curvtanh100lap_mtos": ("_hw2curvtanh100zeroprop_lap", "_hw2curvtanh100selfdist1_dist64_l1_mtos_lap",
                               "_hw2curvtanh100statemonog0.3_sd1_mtos_lap",
                               dict(gate_low_center=0.20, gate_width=0.035,
                                    gate_high_center=0.65, gate_width_high=0.05)),
    "hw2curvtanh100lap_mtosmid": ("_hw2curvtanh100zeroprop_lap", "_hw2curvtanh100selfdist1_dist64_l1_mtos_lap",
                                  "_hw2curvtanh100statemonog0.3_lap",
                                  dict(gate_low_center=0.20, gate_width=0.035,
                                       gate_high_center=0.65, gate_width_high=0.05)),
    "hw2curvtanh100lap_mtoshigh": ("_hw2curvtanh100zeroprop_lap", "_hw2curvtanh100selfdist1_dist64_l1_lap",
                                   "_hw2curvtanh100statemonog0.3_sd1_mtos_lap",
                                   dict(gate_low_center=0.20, gate_width=0.035,
                                        gate_high_center=0.65, gate_width_high=0.05)),
    "hw2curvtanh100lap": ("_hw2curvtanh100zeroprop_lap", "_hw2curvtanh100selfdist1_dist64_l1_lap",
                          "_hw2curvtanh100statemonog0.3_lap",
                          dict(gate_low_center=0.20, gate_width=0.035,
                               gate_high_center=0.65, gate_width_high=0.05)),
    # the SHIPPED package. "hw2curvtanh100lap" above is the pre-mtas lap set and is NOT what
    # ml_seaice_v4.jl points at; the shipped mid band carries the one-sided warming penalty at
    # lambda 100 and matches its teacher under masking over the teacher's own
    # ("tas","tdps","rld"). The gate was swept on the TRAIN years and kept at these values: no
    # better band-separated point exists.
    "hw2curvtanh100lapmtas": ("_hw2curvtanh100zeroprop_lap",
                              "_hw2curvtanh100selfdist1_dist64_l1_mtas100_md_lap",
                              "_hw2curvtanh100statemonog0.3_lap",
                              dict(gate_low_center=0.20, gate_width=0.035,
                                   gate_high_center=0.65, gate_width_high=0.05)),
    "hw2curvtanh100dist48": ("_hw2curvtanh100zeroprop", "_hw2curvtanh100selfdist1_dist48_l1",
                             "_hw2curvtanh100statemonog0.3",
                             dict(gate_low_center=0.20, gate_width=0.035,
                                  gate_high_center=0.65, gate_width_high=0.05)),
    "hw2curvtanh100h128": ("_hw2curvtanh100zeroprop", "_hw2curvtanh100selfdist1_h128",
                           "_hw2curvtanh100statemonog0.3",
                           dict(gate_low_center=0.20, gate_width=0.035,
                                gate_high_center=0.65, gate_width_high=0.05)),
    "hw2curvtanh100h192": ("_hw2curvtanh100zeroprop", "_hw2curvtanh100selfdist1_h192",
                           "_hw2curvtanh100statemonog0.3",
                           dict(gate_low_center=0.20, gate_width=0.035,
                                gate_high_center=0.65, gate_width_high=0.05)),
    "hw2curvtanh100ls0.1": ("_hw2curvtanh100zeroprop", "_hw2curvtanh100selfdist1_ls0.1",
                            "_hw2curvtanh100statemonog0.3",
                            dict(gate_low_center=0.20, gate_width=0.035,
                                 gate_high_center=0.65, gate_width_high=0.05)),
    "hw2curvtanh100ls0.2": ("_hw2curvtanh100zeroprop", "_hw2curvtanh100selfdist1_ls0.2",
                            "_hw2curvtanh100statemonog0.3",
                            dict(gate_low_center=0.20, gate_width=0.035,
                                 gate_high_center=0.65, gate_width_high=0.05)),
    "hw2curvtanh100mono100": ("_hw2curvtanh100zeroprop", "_hw2curvtanh100selfdist1_m100",
                              "_hw2curvtanh100statemonog0.3",
                              dict(gate_low_center=0.20, gate_width=0.035,
                                   gate_high_center=0.65, gate_width_high=0.05)),
    "hw2curvtanh100mono10": ("_hw2curvtanh100zeroprop", "_hw2curvtanh100selfdist1_m10",
                             "_hw2curvtanh100statemonog0.3",
                             dict(gate_low_center=0.20, gate_width=0.035,
                                  gate_high_center=0.65, gate_width_high=0.05)),
    "hw2curvtanh100southhigh": ("_hw2curvtanh100zeroprop", "_hw2curvtanh100selfdist1",
                                "_hw2curvtanh100southtaper",
                                dict(gate_low_center=0.20, gate_width=0.035,
                                     gate_high_center=0.65, gate_width_high=0.05)),
    "hw2curvtanh100pooledhigh": ("_hw2curvtanh100zeroprop", "_hw2curvtanh100selfdist1",
                                 "_hw2curvtanh100pooled",
                                 dict(gate_low_center=0.20, gate_width=0.035,
                                      gate_high_center=0.65, gate_width_high=0.05)),
    # No sharpened set is listed on purpose. Inference-time branch sharpening was evaluated through
    # the full chain on 2026-10-13 and CLOSED: the hedged combination is the squared-loss optimum, so
    # sharpening degrades accuracy by construction, the training-time form scores BELOW the baseline
    # at every temperature, and the deployed function would differ from the fitted one with nothing in
    # the objective sanctioning it. The high_temp plumbing below is kept for diagnostics only.
    "hw2curvtanh100statemono": ("_hw2curvtanh100", "_hw2curvtanh100statemonog1",
                                "_hw2curvtanh100statemonog0.3",
                                dict(gate_low_center=0.20, gate_width=0.035,
                                     gate_high_center=0.65, gate_width_high=0.05)),
    "hw2curvtanh_m0.3": ("_hw2curvtanh100", "_hw2curvtanh100m0.3", "_hw2curvtanh100",
                         dict(gate_low_center=0.20, gate_width=0.05,
                              gate_high_center=0.65, gate_width_high=0.05)),
    "hw2curvtanh_m1": ("_hw2curvtanh100", "_hw2curvtanh100m1", "_hw2curvtanh100",
                       dict(gate_low_center=0.20, gate_width=0.035,
                            gate_high_center=0.65, gate_width_high=0.05)),
    "monotas100": ("_hw2curvtanh100mt100", None, "_hw2curvtanh100mt100",
                   {"gate_low_center": 0.2, "gate_width": 0.035, "gate_high_center": 0.65, "gate_width_high": 0.05}),
    "monotas300": ("_hw2curvtanh100mt300", None, "_hw2curvtanh100mt300",
                   {"gate_low_center": 0.2, "gate_width": 0.035, "gate_high_center": 0.65, "gate_width_high": 0.05}),
    # [2026-10-08] MID band retrained on SIC [0.10, 1.00] instead of [0.10, 0.90], so it carries the
    # dense pack's data (820,921 northern samples against 198,987) while still being gated out above
    # 0.65. Low and high are the untouched hw2curvtanh100 checkpoints. Its own swept gate moved only
    # the low/mid boundary, 0.20/0.035 -> 0.14/0.06; the high gate did not move, as expected since
    # the high band did not change. Single-step CORR is strictly non-losing (+0.033 at SIC 0.30-0.50,
    # 0.000 above 0.75) but the free rollout is 0.0026 (north) and 0.0074 (south) WORSE, both inside
    # the ~0.01 seed-to-seed band, so this is unresolved rather than adopted.
    "hw2curvtanh_midwide": ("_hw2curvtanh100", "_hw2curvtanh100midwide", "_hw2curvtanh100",
                            dict(gate_low_center=0.14, gate_width=0.06,
                                 gate_high_center=0.65, gate_width_high=0.05)),
    "hw2curvtanh100": ("_hw2curvtanh100", None, "_hw2curvtanh100",
                       dict(gate_low_center=0.20, gate_width=0.035,
                            gate_high_center=0.65, gate_width_high=0.05)),
    "hw2curvdamp100": ("_hw2curvdamp100", None, "_hw2curvdamp100",
                       dict(gate_low_center=0.20, gate_width=0.035,
                            gate_high_center=0.65, gate_width_high=0.05)),
    # the 3-column curvature weights WITH the one-sided damping penalty; their own (nosmooth) gate
    "curvdamp100":   ("_curv_curvdamp100",   None, "_curv_curvdamp100",   GATE_NOSMOOTH),
    "curvdamp1000":  ("_curv_curvdamp1000",  None, "_curv_curvdamp1000",  GATE_NOSMOOTH),
    "mid_b0":   ("_highwide2", "_highwide2_mg0.01b0",   "_highwide2", GATE_HIGHWIDE),
    "mid_b0.1": ("_highwide2", "_highwide2_mg0.01b0.1", "_highwide2", GATE_HIGHWIDE),
    "mid_b0.3": ("_highwide2", "_highwide2_mg0.01b0.3", "_highwide2", GATE_HIGHWIDE),
    "shipped":   ("_nosmooth",  "_nosmooth",         GATE_NOSMOOTH),
    "highwide2": ("_highwide2", "_highwide2",        GATE_HIGHWIDE),
    "gr0.001":   ("_highwide2", "_highwide2_gr0.001b0.1", GATE_HIGHWIDE),
    "gr0.003":   ("_highwide2", "_highwide2_gr0.003b0.1", GATE_HIGHWIDE),
    "gr0.01":    ("_highwide2", "_highwide2_gr0.01b0.1",  GATE_HIGHWIDE),
    "vb0.03":    ("_highwide2", "_highwide2_vb0.03", GATE_HIGHWIDE),
    "vb0.1":     ("_highwide2", "_highwide2_vb0.1",  GATE_HIGHWIDE),
    "vb0.3":     ("_highwide2", "_highwide2_vb0.3",  GATE_HIGHWIDE),
    "vb1":       ("_highwide2", "_highwide2_vb1",    GATE_HIGHWIDE),
    "vb3":       ("_highwide2", "_highwide2_vb3",    GATE_HIGHWIDE),
}

# [2026-10-01] EXPLICIT suffix -> cache maps. Substring tests were wrong three separate times
# today: "highwide2" contains "highwide", "hw2curvtanh100" contains "curv" but is not the raw-curv
# set, and a mis-resolved high-band cache mis-normalises every input (it collapsed a free rollout to
# CORR 0.717 and inflated a measured gain to 1.58). A missing key now raises instead of guessing.
_LOWMID = {"_nosmooth": "cache_rld_ablation_nosmooth",
           "_highwide": "cache_rld_ablation_nosmooth",
           "_highwide2": "cache_rld_ablation_nosmooth",
           "_curv": "cache_rld_ablation_curv",
           "_curv_curvdamp100": "cache_rld_ablation_curv",
           "_curv_curvdamp1000": "cache_rld_ablation_curv",
           "_curv_curvdamp10000": "cache_rld_ablation_curv",
           "_hw2curvdamp100": "cache_rld_ablation_curv",
           "_hw2curvtanh100": "cache_curv_tanh",
           "_hw2curvtanh100mt100": "cache_curv_tanh", "_hw2curvtanh100mt300": "cache_curv_tanh",
           "_hw2curvtanh100mt2100": "cache_curv_tanh", "_hw2curvtanh100mt2300": "cache_curv_tanh",
           "_hw2curvtanh100m0.3": "cache_curv_tanh",
           "_hw2curvtanh100m1": "cache_curv_tanh",
           # [2026-10-13] the SIC loss-weight ablation arms (high band only; the low/mid pieces are
           # the adopted ones, so these map to the same low/mid cache as the adopted set)
           "_hw2curvtanh100sicwtaper": "cache_curv_tanh",
           "_hw2curvtanh100sicwflat": "cache_curv_tanh",
           # [2026-10-13] LOW band retrained on caches that keep the SIC == 0 mass point. These MUST
           # map to their own cache: the band's normalisation statistics come from it, and including
           # ~20% or ~86% zeros shifts the siconc mean and std.
           "_hw2curvtanh100zeroprop": "cache_curv_tanh_zeroprop",
           "_hw2curvtanh100zeroall": "cache_curv_tanh_zeroall",
           "_hw2curvtanh100pooled": "cache_curv_tanh",
           # [2026-10-14] mid band retrained with a lower radiation monotonicity weight
           # [2026-10-14] mid band retrained with label smoothing on the direction classifier
           # [2026-10-15] mid band at greater width. The transition zone's gap against a free tree is
           # mostly the DIRECTION decision and raising the classification weight makes it worse, so
           # the deficit is representational rather than a matter of loss balance.
           # [2026-10-15] narrow students distilled from the 128-wide teacher: the transition-zone
           # gain at the shipped parameter count, or below it.
           "_hw2curvtanh100selfdist1_dist64_l1": "cache_curv_tanh",
           "_hw2curvtanh100selfdist1_dist48_l1": "cache_curv_tanh",
           "_hw2curvtanh100selfdist1_h128": "cache_curv_tanh",
           "_hw2curvtanh100selfdist1_h192": "cache_curv_tanh",
           "_hw2curvtanh100selfdist1_ls0.1": "cache_curv_tanh",
           "_hw2curvtanh100selfdist1_ls0.2": "cache_curv_tanh",
           "_hw2curvtanh100selfdist1_m100": "cache_curv_tanh",
           "_hw2curvtanh100selfdist1_m10": "cache_curv_tanh",
           "_hw2curvtanh100southtaper": "cache_curv_tanh",
           "_hw2curvtanh100southflat": "cache_curv_tanh",
           # [2026-10-11] mid band retrained with the global state-contraction penalty (lambda 1);
           # same cache as the control mid band, only the weights differ
           "_hw2curvtanh100statemonog1": "cache_curv_tanh",
           # [2026-10-12] mid band with thermodynamic input masking on top of statemono lambda 1
           "_hw2curvtanh100robust0.5": "cache_curv_tanh",
           "_hw2curvtanh100robust0.15": "cache_curv_tanh",
           "_hw2curvtanh100selfdist1": "cache_curv_tanh",
           "_hw2curvtanh100selfdist3": "cache_curv_tanh",
           "_hw2curvtanh100robust0.3": "cache_curv_tanh",
           # load() evaluates LOWMID_CACHE_FOR for every band including "high", so the high band's
           # own suffix needs an entry here too even though only HIGH_CACHE_FOR is used for it
           "_hw2curvtanh100statemonog0.3": "cache_curv_tanh",
           "_hw2curvtanh100histbb0.3": "cache_curv_tanh",
           # [2026-10-08] mid band retrained on SIC [0.10, 1.00]; the low band files in this dir are
           # symlinks to cache_curv_tanh and the mid band files to cache_midhigh, so each band keeps
           # the normalisation of the data it was trained on
           "_hw2curvtanh100midwide": "cache_curv_tanh_midwide"}
_HIGH = {"_nosmooth": "cache_rld_ablation_nosmooth",
         "_highwide": "cache_high_wide",
         "_highwide2": "cache_high_wide",
         "_curv": "cache_rld_ablation_curv",
         "_curv_curvdamp100": "cache_rld_ablation_curv",
         "_curv_curvdamp1000": "cache_rld_ablation_curv",
         "_curv_curvdamp10000": "cache_rld_ablation_curv",
         "_hw2curvdamp100": "cache_high_wide_curv",
         "_hw2curvtanh100": "cache_high_wide_curv_tanh",
           "_hw2curvtanh100mt100": "cache_high_wide_curv_tanh",
           "_hw2curvtanh100mt2100": "cache_high_wide_curv_tanh",
           "_hw2curvtanh100mt2300": "cache_high_wide_curv_tanh",
           "_hw2curvtanh100mt300": "cache_high_wide_curv_tanh",
         "_hw2curvtanh100m0.3": "cache_high_wide_curv_tanh",
         # [2026-10-11] high band with the global state-contraction penalty (lambda 0.3)
         "_hw2curvtanh100statemonog0.3": "cache_high_wide_curv_tanh",
         "_hw2curvtanh100histbb0.3": "cache_high_wide_curv_tanh",
         # [2026-10-14] the high band trained on BOTH hemispheres. Its normalisation must come from
         # the POOLED statistics, so POOLED_HIGH below routes it to make_pooled_loaders.
         "_hw2curvtanh100pooled": "cache_high_wide_curv_tanh",
         "_hw2curvtanh100southtaper": "cache_high_wide_curv_tanh",
         "_hw2curvtanh100southflat": "cache_high_wide_curv_tanh",
         # [2026-10-13] the SIC loss-weight ablation arms, both normalised to mean weight 1
         "_hw2curvtanh100sicwtaper": "cache_high_wide_curv_tanh",
         "_hw2curvtanh100sicwflat": "cache_high_wide_curv_tanh",
         "_hw2curvtanh100m1": "cache_high_wide_curv_tanh"}


# [2026-10-15] the physical-Laplacian variant rebuilds ALL THREE bands into one pair of caches
# (cache_curv_lap for low+mid, cache_high_wide_curv_lap for high), so a trailing _lap routes there by
# rule instead of needing one literal entry per band suffix.
def LOWMID_CACHE_FOR(suffix):
    base = suffix.split("_mg")[0].split("_var")[0].split("_gr")[0].split("_vb")[0]
    if base.endswith("_lap"):
        return "cache_curv_lap"
    if base not in _LOWMID:
        raise KeyError(f"no low/mid cache registered for suffix {suffix!r} (base {base!r})")
    return _LOWMID[base]


def HIGH_CACHE_FOR(suffix):
    base = suffix.split("_mg")[0].split("_var")[0].split("_gr")[0].split("_vb")[0]
    if base.endswith("_lap"):
        return "cache_high_wide_curv_lap"
    if base not in _HIGH:
        raise KeyError(f"no high cache registered for suffix {suffix!r} (base {base!r})")
    return _HIGH[base]


BANDS = {"low":  (L.ChangeGateClassifier,    True,  "sic+sst+rsds+rld+sfcwind+tas+tdps", "unified_low_gradbasis_cross_sep_radrobust"),
         "mid":  (L.RelaxationMoEClassifier, True,  "sic+sst+rsds+rld+sfcwind+tas+tdps", "unified_mid_gradbasis_cross_sep_radrobust"),
         "high": (L.RelaxationMoEClassifier, False, "sic+sst+rsds+rld+sfcwind+tas",      "north_high_gradbasis_cross_sep_radrobust")}

_orig = R.build_blocked_fields_2deg
torch.set_grad_enabled(False)


def cross_truegrad(sic2d, lat_coarse=None):
    lp = np.empty_like(sic2d); lp[1:] = sic2d[:-1]; lp[0] = np.nan
    ln = np.empty_like(sic2d); ln[:-1] = sic2d[1:]; ln[-1] = np.nan
    d0 = lambda nb: np.where(np.isfinite(nb), nb - sic2d, 0.0).astype(np.float32)
    return 2.0 * np.stack([(d0(lp) - d0(ln)) / 2.0,
                           (d0(np.roll(sic2d, 1, 1)) - d0(np.roll(sic2d, -1, 1))) / 2.0], axis=-1)


def cross_tanh(sic2d, lat_coarse=None):
    """The SHIPPED three-column form on a ONE-DEGREE grid: the 2.0 rescales a one-degree neighbour
    difference to the two-degree arc the caches were built on. The curvature column is grid-relative
    and therefore NOT rescaled, only bounded by tanh(c / 0.5)."""
    lp = np.empty_like(sic2d); lp[1:] = sic2d[:-1]; lp[0] = np.nan
    ln = np.empty_like(sic2d); ln[:-1] = sic2d[1:]; ln[-1] = np.nan
    d0 = lambda nb: np.where(np.isfinite(nb), nb - sic2d, 0.0).astype(np.float32)
    n_d, s_d = d0(lp), d0(ln)
    w_d, e_d = d0(np.roll(sic2d, 1, 1)), d0(np.roll(sic2d, -1, 1))
    return np.stack([2.0 * (n_d - s_d) / 2.0, 2.0 * (w_d - e_d) / 2.0,
                     np.tanh(((w_d + e_d) / 2.0) / 0.5).astype(np.float32)], axis=-1)


def cross_lap(sic2d, lat_coarse=None):
    """[2026-10-15] Same, with the curvature column replaced by the PHYSICAL Laplacian on the lat-lon
    grid: ((W-C)+(E-C))/2 + cos^2(lat) ((N-C)+(S-C))/2. Dividing the true Laplacian through by its
    1/cos^2 zonal coefficient keeps the zonal term at 1, which is what the tanh scale of 0.5 and the
    column's whole trained range are calibrated to, so only the meridional term is new.

    Requires lat_coarse on every call: set R.CROSS_NEEDS_LAT = True wherever this is installed."""
    assert lat_coarse is not None, "cross_lap needs lat_coarse; set R.CROSS_NEEDS_LAT = True"
    lp = np.empty_like(sic2d); lp[1:] = sic2d[:-1]; lp[0] = np.nan
    ln = np.empty_like(sic2d); ln[:-1] = sic2d[1:]; ln[-1] = np.nan
    d0 = lambda nb: np.where(np.isfinite(nb), nb - sic2d, 0.0).astype(np.float32)
    n_d, s_d = d0(lp), d0(ln)
    w_d, e_d = d0(np.roll(sic2d, 1, 1)), d0(np.roll(sic2d, -1, 1))
    cl2 = (np.maximum(np.cos(np.deg2rad(np.asarray(lat_coarse, dtype=np.float64))), 1e-3) ** 2
           ).astype(np.float32)[:, None]
    curv = (w_d + e_d) / 2.0 + cl2 * (n_d + s_d) / 2.0
    return np.stack([2.0 * (n_d - s_d) / 2.0, 2.0 * (w_d - e_d) / 2.0,
                     np.tanh(curv / 0.5).astype(np.float32)], axis=-1)


def cross_for(*suffixes):
    """Pick the cross-column builder a set of checkpoints was TRAINED on. The curvature column is an
    input, so using the wrong one silently evaluates the model off its training distribution -- the
    same class of error as the 2026-10-13 smoothing bug. Deciding it from the suffixes rather than
    from a per-script flag means no caller can get it wrong by omission."""
    lap = [bool(x) and x.endswith("_lap") for x in suffixes if x is not None]
    if any(lap):
        assert all(lap), f"mixed curvature columns in one package: {suffixes}"
        R.CROSS_NEEDS_LAT = True
        return cross_lap
    return cross_tanh


POOLED_HIGH = {"_hw2curvtanh100pooled"}
# a high band fitted on SOUTHERN rows only. Its normalisation must come from the southern cache, not
# the northern one, so it gets its own set rather than sharing the pooled path.
SOUTH_ONLY_HIGH = {"_hw2curvtanh100southtaper", "_hw2curvtanh100southflat"}


def load(lowmid_suffix, high_suffix, mid_suffix=None):
    """The high band's cache follows its checkpoint: any *highwide* high band was trained on the
    widened [0.65,1] cache and must be normalised with that cache's statistics.

    high_temp sharpens the HIGH band's branch softmax at inference only, with no retraining and no
    change to any weight. It defaults to None, which leaves every existing caller byte-identical."""
    models, norms = {}, {}
    for b, (cls, pooled, fs, ck) in BANDS.items():
        suffix = (high_suffix if b == "high"
                  else (mid_suffix if (b == "mid" and mid_suffix) else lowmid_suffix))
        cache = ROOT / LOWMID_CACHE_FOR(suffix)
        if pooled:
            tr, _, _, _ = L.make_pooled_loaders(str(cache / f"north_train_{b}band_rld_tdps_gradbasis.npz"),
                                                str(cache / f"south_train_{b}band_rld_tdps_gradbasis.npz"),
                                                str(cache / f"north_val_{b}band_rld_tdps_gradbasis.npz"),
                                                str(cache / f"south_val_{b}band_rld_tdps_gradbasis.npz"),
                                                feature_set=fs, normalize_phys=True, batch_size=L.BATCH_SIZE, use_cross=True)
        else:
            hc = ROOT / HIGH_CACHE_FOR(suffix)
            if suffix in SOUTH_ONLY_HIGH:
                tr, _, _, _ = L.make_loaders(str(hc / "south_train_highband_rld_gradbasis.npz"),
                                             str(hc / "south_val_highband_rld_gradbasis.npz"),
                                             feature_set=fs, normalize_phys=True,
                                             batch_size=L.BATCH_SIZE, use_cross=True)
            elif suffix in POOLED_HIGH:
                tr, _, _, _ = L.make_pooled_loaders(
                    str(hc / "north_train_highband_rld_gradbasis.npz"),
                    str(hc / "south_train_highband_rld_gradbasis.npz"),
                    str(hc / "north_val_highband_rld_gradbasis.npz"),
                    str(hc / "south_val_highband_rld_gradbasis.npz"),
                    feature_set=fs, normalize_phys=True, batch_size=L.BATCH_SIZE, use_cross=True)
            else:
                tr, _, _, _ = L.make_loaders(str(hc / "north_train_highband_rld_gradbasis.npz"),
                                             str(hc / "north_val_highband_rld_gradbasis.npz"),
                                             feature_set=fs, normalize_phys=True, batch_size=L.BATCH_SIZE, use_cross=True)
        ck_path = R.CKPT_DIR / (ck + suffix) / "best.pt"
        _w = 64
        if ck_path.exists():
            _sd = torch.load(ck_path, map_location="cpu", weights_only=False)["model_state"]
            _w = int(_sd["backbone.0.weight"].shape[0])      # width from the checkpoint, not assumed
        kw = dict(in_dim=tr.X.shape[1], hidden=_w, depth=4, tau_floor=L.TAU_FLOOR, dt=L.DT,
                  history_h=tr.history_h, mom_hidden=16)
        m = (cls(**kw) if cls is L.ChangeGateClassifier else cls(dropout=0.0, **kw)).to(L.DEVICE)
        ckpt_dir = R.CKPT_DIR / (ck + suffix)
        if not (ckpt_dir / "best.pt").exists():
            raise FileNotFoundError(str(ckpt_dir / "best.pt"))
        m.load_state_dict(torch.load(ckpt_dir / "best.pt", map_location=L.DEVICE, weights_only=False)["model_state"])
        m.eval()
        # An inference-time branch-sharpening diagnostic lived here (a temperature on the
        # increase/decrease logit). It is not shipped: it changed no weights and was only
        # ever used to show that the marginal zone over-amplifies in both directions.
        models[b], norms[b] = m, dict(mean=tr.mean, std=tr.std, rld_mean=tr.rld_mean, rld_std=tr.rld_std,
                                      tdps_mean=tr.tdps_mean, tdps_std=tr.tdps_std, cross_mean=tr.cross_mean,
                                      cross_std=tr.cross_std, feature_set=fs, cross_cols=None)
    return models, norms


def main():
    wanted = sys.argv[1:] or list(SETS)

    R.cross_diff_2d = cross_truegrad
    R.build_blocked_fields_2deg = (lambda f, h, **kw: _orig(f, h, **{**kw, "spatial_sigma": 0.0, "spatial_block": 1}))
    ctx = R.make_rollout_context(R.collect_processed_files(YEARS), R.PROCESSED_DIR / f"processed_{SEED}.nc", "north")
    R.build_blocked_fields_2deg = _orig

    lat = np.asarray(ctx["lat_coarse"]); vm = ctx["valid_mask_final"]
    nb = ctx["n_blocks"]
    melt = [k for k in range(nb) if 7 <= (((k % 73) * 5) // 30) + 1 <= 10]     # Jul-Oct blocks
    cell = vm & (lat[:, None] >= LAT_MIN)
    print(f"context: {nb} blocks, {len(melt)} melt-season blocks, "
          f"{int(cell.sum())} ocean cells >= {LAT_MIN:.0f}N", flush=True)

    fields = {"ERA5": ctx["blocked"]["siconc"]}
    for name in wanted:
        spec = SETS[name]
        lm, mid_s, hi, gate = (spec[0], spec[1], spec[2], spec[3]) if len(spec) == 4 else (spec[0], None, spec[1], spec[2])
        # The cross builder belongs to the SET, not to the battery: a `_lap` package carries a
        # three-column operator and the pre-lap sets carry two, so the array has to be built the
        # way the set's own normalisation expects. cross_for reads that off the suffixes and sets
        # CROSS_NEEDS_LAT itself; the pre-lap assignment above is left in place so every earlier
        # battery stays reproducible.
        if any(x is not None and x.endswith("_lap") for x in (lm, mid_s, hi)):
            R.cross_diff_2d = cross_for(lm, mid_s, hi)
        try:
            models, norms = load(lm, hi, mid_s)
        except FileNotFoundError as e:
            print(f"[skip] {name}: missing {e}", flush=True)
            continue
        roll, corr, _ = R.run_rollout(ctx, 0, models, norms, "north", recalibrate=False, **gate)
        fields[name] = roll
        print(f"[done] {name}: rollout CORR {corr:.5f}", flush=True)

    print(f"\n>= {LAT_MIN:.0f}N, melt-season blocks (Jul-Oct), free 1-degree rollout")
    print(f"  {'field':>12s} | {'mean SIC':>8s} {'min SIC':>8s} | {'% < 0.5':>8s} {'% < 0.8':>8s} {'% < 0.9':>8s}")
    for name, F in fields.items():
        v = F[np.ix_(melt, np.ones(F.shape[1], bool), np.ones(F.shape[2], bool))][:, cell]
        v = v[np.isfinite(v)]
        print(f"  {name:>12s} | {v.mean():8.4f} {v.min():8.4f} | "
              f"{100*(v < 0.5).mean():8.3f} {100*(v < 0.8).mean():8.3f} {100*(v < 0.9).mean():8.3f}")

    print(f"\n>= {LAT_MIN:.0f}N, whole period (drift check)")
    print(f"  {'field':>12s} | {'mean SIC':>8s} {'% < 0.5':>8s}")
    for name, F in fields.items():
        v = F[:, cell]; v = v[np.isfinite(v)]
        print(f"  {name:>12s} | {v.mean():8.4f} {100*(v < 0.5).mean():8.3f}")

    print("\nCENTRAL_PACK_ROLLOUT_DONE")


if __name__ == "__main__":
    main()
