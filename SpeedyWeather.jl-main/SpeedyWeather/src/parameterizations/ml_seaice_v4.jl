"""
Gradient-basis-cross candidate ML sea-ice model (2026-08-26), added ALONGSIDE `MLSeaIce` (V1),
`MLSeaIceV2`, and `MLSeaIceV3` -- purely additive, does not modify, call into, or share any
weights/state with any of them. All four can be defined in the same SpeedyWeather session; only one
is ever attached to a given `PrimitiveWetModel` at a time via the `ocean` component slot.

Differs from `MLSeaIceV3` in exactly one way, driven by the underlying PyTorch work documented in
`hemisphere_unified_gradbasis_mono_variants.ipynb`: the two cross-neighbor input columns hold a
clean centered-difference gradient per axis, `[(N-S)/2, (W-E)/2]`, instead of V3's 4 raw redundant
`[N,S,W,E]` columns (N/S and W/E are each highly anti-correlated in the real data, r=-0.6 to -0.89 --
mostly two opposite-signed views of the same local gradient; V3's plain-4-column model was found to
under-use the real, data-supported N-S dominance). This was tested because V3's online output showed
a near-pole radial "spoke" streaking artifact; feeding the clean gradient directly (rather than
asking the network to discover it from the redundant raw columns) substantially reduced it in
offline rollout diagnostics: south (`d_dt=0`, no diffusion) rollout CORR 0.7386->0.9333, mean local
SIC anomaly 0.0926->0.0240; north (`d_dt=0`) CORR 0.9171->0.9670, anomaly 0.0154->0.0039. Two
earlier hypotheses (dropping N/S entirely; rescaling the raw E/W difference by the true spherical
1/cos(lat) factor) were tried first and both made the streaking WORSE -- see the notebook for the
full comparison table.

Blend-gate boundaries were independently re-swept for this representation (TRAIN-only two-stage
sweep, `d_dt=0` fixed throughout -- diffusion has NOT yet been swept for this variant, see below):
south `lowsic_gate_center_south=0.02`, `width=0.025` (V3 used center 0.05).
Two mono-penalty formulations (joint/sum vs separate) were compared at their own best-swept gate and
came out essentially tied on pure offline rollout metrics (CORR/RMSE/anomaly deltas all
<=0.0035/<=0.0007). Separate (`relu(grad_rsds)^2 + relu(grad_rld)^2`, guaranteeing
d(pred)/d(rsds)<=0 AND d(pred)/d(rld)<=0 individually) was adopted over joint/sum once online
radiation-COMPOSITION sensitivity became relevant, not just magnitude: since rsds/rld are fed as two
separate raw columns (not a pre-combined `radsum`), the model's response depends on the short/long
split, and SpeedyWeather's own online radiation has a real composition bias (2026-08-27 finding:
JJA >=60N rsds overestimated ~+150-190 W/m^2, rld underestimated ~-70-90 W/m^2, roughly similar raw
sum). A jointly-constrained model only guarantees the SUM's gradient is non-positive and could in
principle have a wrong-signed individual response to one flux compensated by the other; separate
constraints rule that out for both fluxes unconditionally -- the more defensible default now that
composition, not just magnitude, is known to matter for this online coupling.

**2026-09-14 update, "radiation-composition-robust" retrain (now the default weights)**: the
separate-penalty model above still under-melts online in both hemispheres' summers, traced to the
real ERA5 training distribution rarely/never covering the online GCM's own biased
`(rsds, rld)` composition (the same 2026-08-27 bias above). Retrained
(`run_gradbasis_cross_sep_radrobust.py`) with one added loss term: the same separate gradient-sign
mono-penalty, ALSO evaluated at randomly perturbed synthetic `(rsds, rld)` points
(`SYNTH_PERTURB_STD=1.5` training-std, Gaussian, independent per flux), not just real training points
-- a label-free constraint (a sign penalty needs no target value), so this adds no fabricated
supervision, only a wider domain over which the existing constraint is enforced. A synthetic-
composition sensitivity check confirmed the mechanism: evaluated at the online-biased point, the
mid-band model's shift effect on predicted tendency flips sign (plain: +0.00034, more growth/less
melt, reproducing the online bug's direction; radrobust: -0.00023, more melt, corrected). Online
(T63/15L, 3yr, no diffusion; this first check used the pre-2026-09-15 gate/timestep, isolating the
weight change alone, see the Gate paragraph below for what changed after) confirms a
melt-season-specific deepening of the summer SIC minimum in both hemispheres with winter maxima
essentially unchanged and no streaking regression. See `hemisphere_unified_gradbasis_radrobust_v4.
ipynb` for the full record (training log, offline VAL comparison, sensitivity check, held-out VAL
effect-verification, online comparison figure). Weights: `checkpoints_hemisphere_unified_rld_tdps/
{unified_low,unified_mid,north_high}_gradbasis_cross_sep_radrobust`, exported via
`export_v4_radrobust_weights.py`. The plain (non-synthetic-constrained) separate-penalty weights
remain on disk (`ml_seaice_weights_v4_unified_{low,mid}`, `ml_seaice_weights_v4_north_high`) but are
no longer this struct's default.

**Gate: a fresh TRAIN-only sweep for this checkpoint picked a different North gate
(`highsic_gate_center` 0.9->0.85, both North widths 0.05->0.025), now ADOPTED below, but only
together with a REQUIRED timestep change -- this is worth recording in detail, both the false start
and the resolution, because it is a real methodology gap, not just a routine gate update.** The swept
gate looked mildly WORSE on TRAIN/VAL single-step and held-out rollout metrics (CORR deltas <=0.005)
but was assumed adoptable since offline rollout showed no red flags. A first online 3yr T63 attempt
with that gate at the DEFAULT timestep (2026-09-14) diverged to NaN partway through year 2 (around
1990-09, the Arctic autumn refreeze onset), while the SAME weights with the OLD North gate ran 3 full
years clean at the same default timestep. Root-caused (not just observed, 2026-09-15): the SIC field
itself diverges measurably between the two gate configurations from month 1 (>60% of Arctic points
differing by more than 0.15 SIC by June of year 2), but this divergence alone does not explain the
NaN -- the gate's own sigmoid blend is unconditionally NaN-safe for any finite input and width, and
the dedicated high-SIC sub-model extrapolates smoothly, with no blowup, from its trained range
[0.9,1.0] down to 0.5. The actual trigger, found by checking the coupled atmosphere's own vorticity
field (not just surface temperature) at increasing time resolution: a fast, top-model-layer,
high-Southern-latitude numerical instability with NO visible precursor at daily or even 3-hourly
resolution, consistent with a classic CFL-type explicit-scheme failure. Quantitatively: this
component's online scripts use `Grid=FullGaussianGrid` (equal longitude-point count at every
latitude ring, required so the neighbor-gradient cross feature above has an EXACT neighbor at every
point, not a reduced-grid nearest-longitude approximation), whose zonal grid spacing shrinks with
`cos(latitude)`; combined with `PrimitiveWetModel`'s fixed, non-adaptive `Leapfrog` timestep (no CFL
logic anywhere in SpeedyWeather), the zonal Courant number at just the ALREADY-observed ~100 m/s
background wind speed is ~1.0 at 55 degrees latitude and rises above 1 at every higher latitude --
this whole online configuration runs with almost no CFL margin at high latitudes, independent of
which sea-ice gate is active; the new gate's compounding SIC drift merely perturbed the coupled
system enough, during the fastest-changing part of the seasonal cycle, to cross a threshold that was
already thin. **Confirmed directly**: re-running the identical new-gate checkpoint with the
atmospheric timestep HALVED (`Δt_at_T31=Minute(20)`, giving `Δt=600s` at T63 instead of the default
`1200s`) completed a full, clean 3-year run (2026-09-15), including the previously-fatal window, with
no other change. **The new gate is therefore adopted below, on the explicit condition that any
`PrimitiveWetModel` using it online must also be constructed with a halved timestep** (see
`weights_dir_mid`'s docstring and every gate field below for the exact kwarg) -- this is NOT
something `MLSeaIceV4` itself can enforce (the timestep is a `PrimitiveWetModel`-level
`time_stepping` setting, orthogonal to this component), so it must be set explicitly by whoever
constructs the model. Running this gate at the default timestep reproduces the NaN.

**2026-09-22: North gate REVERTED to `(lowsic 0.05, highsic 0.9, both widths 0.05)`, halved-timestep
requirement dropped pending online validation.** The TRAIN-only sweep that picked `0.85/0.025` above was
run with a high-band checkpoint that predated a val-target normalization fix in `make_loaders` (val
targets had been normalized with the val cache's own stats while callers un-normalized with train's,
distorting best-epoch selection; only the Arctic-only high band went through that code path). The
high band was retrained with the fix (best epoch 4 -> 7, VAL CORR 0.38328 -> 0.41420) and its exported
weights replaced in place. Re-running the same sweep with it picks `(0.05, 0.9, 0.05)` (Stage 2 CORR
0.97098 vs 0.96974 for `0.85/0.025`); pre-fix, `0.85/0.025` had won by a noise-level 0.00005.
**Validated online (2026-09-22)**: these defaults (fixed high band, gate `0.05/0.9/0.05`, DEFAULT
timestep 1200s, T63/15L FullGaussianGrid, no diffusion) ran 3 years with no NaN
(`run_3yr_t63_15layer_ml_v4radrobust_fixedhigh_oldgate_defaultdt_netcdf.jl`), so the halved-timestep
requirement no longer applies to the defaults. Against the previous default (0.85/0.025 at 600s):
land-masked zonal second difference south 0.00538 vs 0.00646 (cleanest of all runs compared), north
0.00611 vs 0.00541 with an unchanged grid-scale (k=61-96) power share (5.67% vs 5.63%) and no radial
spokes visible in the monthly maps; year-2 -> year-3 mean-SIC change north +0.043 vs +0.046, south
+0.009 vs +0.033. The `0.85/0.025` gate itself is unchanged in character: if selected again
explicitly, it still needs the halved timestep described above.

**2026-09-24: North gate changed to `(lowsic 0.08/0.07, highsic 0.83/0.11)`, the previous
`(0.05/0.05, 0.9/0.05)` is now the old default.** The 2026-09-22 sweep that produced the old values
tried only a handful of hand-picked combinations and tied the two North widths together. A full
coordinate search with the widths separated (`gate_sweep_full_v4.py`, then `gate_sweep_extended_v4.py`)
found a much better but degenerate optimum -- `(0.17/0.09, 0.70/0.2875)` blends all three sub-models
across nearly the whole SIC range, i.e. an ensemble average rather than the band decomposition this
model is built on, so it was rejected. Re-running the search under an explicit separation constraint
(mid keeps >= 0.95 of the weight at SIC 0.5, high >= 0.8 at SIC 1.0, low >= 0.7 at SIC 0.0,
`gate_sweep_banded_v4.py`) gives the gate adopted here; it hands low 0.002 and high 0.047 at SIC 0.5,
so the bands stay separate.

Offline (TRAIN-only objective, two 4-year free rollouts, unsmoothed inputs, adopted weights):
CORR 0.96551 / RMSE 0.09923 / mean yearly-min error +0.0589, against 0.95943 / 0.10759 / +0.0685 for
the old gate. Single held-out VAL check (1989-1990): 0.96423 / 0.10134 / +0.0460 against
0.95623 / 0.11155 / +0.0523.

Online (3yr T63/15L, default timestep, no diffusion,
`run_3yr_t63_15layer_ml_v4radrobust_bandedgate_netcdf.jl`): no NaN, and on the criteria that decide an
online run it is a wash rather than a gain -- grid-scale (k=61-96) power share 5.73% north / 2.42%
south against the old gate's 5.67% / 2.45%, i.e. no new grid-scale artefact; land-masked zonal second
difference 0.00719 north (old 0.00611) and 0.00589 south (old 0.00538); year-2 -> year-3 mean SIC
+0.056 north / +0.015 south against +0.043 / +0.009. Adopted on the offline gain, with the online
behaviour unchanged within the spread already seen between runs (user decision, 2026-09-24).

**2026-09-24, second change: the UNSMOOTHED-TRAINED weights and their own gate become the default.**
Weights `ml_seaice_weights_v4_nosmooth_*` (trained on `cache_rld_ablation_nosmooth`, spatial_sigma=0,
matching the raw per-point values this component actually feeds online), North gate
`(lowsic 0.24/0.085, highsic 0.65/0.01)`. The previous adopted weights and their gate
`(0.08/0.07, 0.83/0.11)` remain on disk as `ml_seaice_weights_v4_radrobust_*`.

The gate had to be re-swept for these weights, and that is the point: the gate decides how the three
sub-models are blended, so it is model-dependent. Optimised for the adopted weights the same
band-separated search gives `(0.08/0.07, 0.83/0.11)`; optimised for these it gives
`(0.24/0.085, 0.65/0.01)` -- a far lower, much steeper high-SIC boundary. Earlier online tests of these
weights used the adopted weights' gate and looked bad because of it.

Offline (free rollout, ERA5 forcing, unsmoothed inputs): held-out VAL 1989-1990 north CORR 0.97103 /
RMSE 0.09042 against 0.96423 / 0.10134 for the adopted weights at their own gate; TRAIN objective
0.97226 / 0.08824 against 0.96551 / 0.09923. South rollout CORR 0.944 against 0.909.

Online (3yr T63/15L, default timestep, no diffusion,
`run_3yr_t63_v4_forcingdump.jl nosmooth nsgate`): no NaN -- the steep high gate (width 0.01) raised a
CFL concern, since 0.025 once forced a halved timestep, but it ran clean at 1200s. Arctic summer melt
is the closest to observations of any online run here: ocean-masked >=60N minimum 0.4840 against
ERA5's 0.3014, where the adopted weights give 0.4965 at their own gate and 0.5000 at the older one, and
these same weights gave 0.6114 when run with the adopted weights' gate. Year-2 -> year-3 mean SIC
+0.036 north / +0.028 south, comparable to the other runs.

On the Antarctic zonal roughness, which earlier rounds treated as this weight set's disqualifying
defect: it is ~2x the adopted runs', but measuring ERA5 itself through the identical metric on the
identical T63 grid (`era5_zonal_roughness_on_t63.py`) shows BOTH are far too smooth -- south, all
SIC >= 0.02, ERA5 0.0446 against 0.0102 here and 0.0059 for the adopted weights; per bin ERA5 runs
6-12x the adopted weights and 3-6x these. "Smoother" was never the right objective; against
observations these weights are the closer of the two. (That the roughness is closer in MAGNITUDE does
not establish that it is in the right places -- a date-matched spatial comparison has not been done.)

**Diffusion has deliberately NOT been swept for this variant** (unlike V3, whose
`diffusion_coefficient`/`diffusion_coefficient_south` are genuine TRAIN-window-swept, adopted
values) -- the gate sweep above was run entirely with diffusion OFF to isolate and evaluate the
per-pixel model+gate behavior on its own merits first (matching the offline spoke-diagnostic
methodology). `apply_spectral_diffusion`/`apply_spectral_diffusion_south` therefore default to
`false` here; the coefficient fields are kept (for a future sweep) but are not meaningful defaults
and must not be read as "adopted" values the way V3's are.

Update cadence, running-mean forcing convention, `rld`-only (no `Rlu`) longwave accumulation,
shifted-notos TOS lag, SST floor at Arctic/Antarctic ML-branch points, and the spectral
(unconditionally-stable) diffusion pass mechanism are all identical to `MLSeaIceV3` -- see that
file's docstring for the underlying rationale, not re-derived here. `dewpoint_from_humidity`
(Magnus-Tetens dewpoint from lowest-level specific humidity) is reused directly from
`ml_seaice_v2.jl` rather than redefined here -- it is a stateless physics helper with no
version-specific dependencies.
"""

include("ml_seaice_model_v4.jl")
using .MLSeaIceModelV4: MLWeightsV4, MLWeightsChangeGateV4
using .MLSeaIceModelV4: load_ml_weights_v4, load_ml_weights_changegate_v4
using .MLSeaIceModelV4: ml_forward_v4, ml_normalize_features_v4, ml_forward_changegate_with_prob,
    flatten_weights_v4, ml_forward_flat_v4, ml_forward_flat_changegate_v4, ml_normalize_flat_v4!,
    MLWeightsFlatV4CPU, MLWeightsFlatChangeGateV4CPU

export MLSeaIceV4

@kwdef mutable struct MLSeaIceV4 <: AbstractSeaIce
    "[OPTION] latitude (degrees north) above which the ML model is used; below, falls back to the thermodynamic scheme"
    lat_limit::Float64 = 60.0

    "[OPTION] update interval, matching the training block length"
    update_every::Second = Day(5)

    "[OPTION] [2026-10-28] the mid band is now the ONE-SIDED WARMING PENALTY retrain of the
    distilled student below (the \"lapmtas\" package). Only this band changed: the low and high
    packages are byte-identical to the previous export, and the curvature column is unchanged, so
    `curvature_tanh_scale` and `curvature_laplacian` keep their values. Adopted on three clean 3-year
    T63/15L coupled runs (1989-1992, no diffusion), 37 of 37 steps each with ZERO NaN in SIC and SST:
    Δt=600s, Δt=1200s (the DEFAULT), and a snowball stress case at solar_constant=600. No timestep
    change is required -- the earlier 1200s divergence that had been cited as an adoption condition
    was an artifact of the 2026-10-27 script defect that fed this package the 4th-difference column
    instead of its own tanh Laplacian, and the re-run at 1200s is clean. No spokes: on the raw
    lat-lon field there is no meridional streaking, and the marginal-ice-zone gradient anisotropy
    <gx^2>/<gy^2> at 65-85N is 4.67 (600s) and 4.31 (1200s) against ERA5's 5.65, i.e. both runs are
    MORE isotropic than the reanalysis. Radial rays seen in orthographic polar renderings of T63
    output are a nearest-neighbour projection artifact of the 1.875 deg longitude spacing, not a
    property of the field. Known cost, accepted: the Arctic melts out far too far in late summer at
    BOTH resolutions -- ocean-masked 50-90N summer minimum 0.064 at T63 and 0.032 at T127, against
    ERA5's 0.196, with the two tracking each other all year (T127-T63 year mean -0.044). This is a
    property of the coupled system, not of T127, and not of this weight package: it is the
    central-pack summer deficit driven by SpeedyWeather's cloud-poor polar atmosphere, whose rld sits
    about 2 sigma below the mid band's training distribution through the melt season.

    [2026-10-15] the mid band is DISTILLED from a 128-wide teacher that is never
    deployed. Width is the only lever that closed any of the transition zone's gap: at SIC 0.30 to
    0.80 a free tree on the same inputs reaches CORR 0.8262 against the 64-wide band's 0.7115, and
    most of that gap is the DIRECTION decision, which a heavier classification weight makes WORSE.
    Widening to 128 closes part of it, so a narrow student was then trained to reproduce the wide
    teacher's predicted tendency alongside the reanalysis. The student carries EXACTLY the shipped
    parameter count, 13,574, and beats its own teacher: north VAL rollout 0.95960 to 0.96332, south
    0.93889 to 0.94227, and all four untouched-year windows improve, 2010 and 2017 in both
    hemispheres. Bistability is unchanged at 4 of 30, the coupled run is 37 of 37 steps NaN-free, and
    the grid-scale share falls in both hemispheres, north 5.751 to 5.651 per cent and south 2.472 to
    2.358. The low and high bands are byte-identical to the previous export.

    path to the exported weight/manifest directory for the UNIFIED mid-band (SIC 0.1-0.9)
    model -- serves BOTH hemispheres. Defaults to the 2026-09-14 radiation-composition-robust
    retrain (synthetic-perturbed sign-constraint); the plain separate-penalty weights remain on
    disk at ml_seaice_weights_v4_unified_mid but are no longer the default.
    Gate/timestep pairing: the shipped North gate (lowsic 0.20/0.035, highsic 0.65/0.05) does not
    need the halved timestep -- see the 2026-10-28 paragraph above, which re-established that at
    Δt=1200s for this weight package, and the struct docstring's 2026-09-22 paragraph for the
    history. If the old 0.85/0.025 gate is passed explicitly instead, the PrimitiveWetModel must be
    constructed with `time_stepping=Leapfrog(spectral_grid; Δt_at_T31=Minute(20))` (Δt=600s at T63);
    that gate at the default timestep reproduces a confirmed online NaN divergence."
    weights_dir_mid::String = joinpath(@__DIR__, "ml_seaice_weights_v4_lapmtas_unified_mid")

    "[OPTION] path to the exported weight/manifest directory for the Arctic-only dedicated high-SIC
    (0.9-1.0) model (no tdps, no Southern counterpart, seed=2024 adopted from a 5-seed sweep).
    Defaults to the 2026-09-14 radiation-composition-robust retrain, see weights_dir_mid; the files
    in this directory were replaced on 2026-09-22 by the val-target-normalization-fixed retrain of
    the same recipe (best epoch 4 -> 7). Pre-fix copy: archive_prefix_highband_20260922/weights/."
    weights_dir_highsic::String = joinpath(@__DIR__, "ml_seaice_weights_v4_lapmtas_north_high")

    "[OPTION] sigmoid gate center (SIC units) for blending the mid-band and high-SIC models (Arctic
    only). 0.65, from the band-separated search re-run for the unsmoothed-trained weights (2026-09-24) (see the struct docstring);
    0.9 from 2026-09-22 to 2026-09-24, and 0.85 from 2026-09-15 to 2026-09-22. Only that 0.85 value
    required the halved timestep (Δt=600s at T63): it is a steep gate and this configuration runs at
    Courant~1.0-1.7 at 55-70 degrees on FullGaussianGrid, so it had no CFL margin left. The gate here
    is much wider and ran 3 years clean at the default 1200s."
    highsic_gate_center::Float32 = 0.65f0

    "[OPTION] sigmoid gate width (SIC units) for blending the mid-band and high-SIC models (Arctic
    only). 0.01, from the same search. Steep, but it ran 3 years clean at the DEFAULT timestep; only the
    old 0.85/0.025 gate ever needed the halved one."
    highsic_gate_width::Float32 = 0.05f0

    "[OPTION] path to the exported weight/manifest directory for the UNIFIED low-SIC (0.0-0.1)
    change-gate model -- serves BOTH hemispheres. [2026-10-14] Defaults to the zeroprop retrain. The
    cache builder masked the band with a STRICT (sic > sic_band_lo); for a lower edge of exactly 0.0
    that deleted a mass point, so this band had 0 training rows at SIC == 0 while 20.2% of its
    deployment rows sit there and never change. Its change gate was therefore fitted where cells
    change 69.6% of the time and applied where they change 8.7%, emitting p_changed = 0.400 against a
    true 0.087 below SIC 0.02 -- 3.7x too much motion injected into open water. Rebuilt with the
    zeros kept at their deployment proportion of 20% (keeping ALL of them, 86.4% of the band, was
    worse everywhere). ONLY this band changed; mid and high are byte-identical to the selfdist
    export. Ice edge SIC < 0.02: RMSE -14.3%, bias +0.00535 -> +0.00157, spurious motion 3.7x -> 1.9x.
    Offline rollout at the IDENTICAL gate: north CORR 0.95867 -> 0.95960, RMSE 0.11004 -> 0.10883.
    3yr T63 coupled: 37/37 steps NaN-free, spokes inside the accepted spread (north grid-scale
    5.751% against selfdist's 5.743%, zonal 2nd-diff 0.00370 against 0.00380)."
    weights_dir_lowsic::String = joinpath(@__DIR__, "ml_seaice_weights_v4_lapmtas_unified_low")

    "[OPTION] sigmoid gate center (SIC units), Northern low-SIC boundary. 0.20, the hw2curvtanh100
    sweep that the state-contraction weights kept deliberately (0.24 was the 2026-09-24 nosmooth
    search, 0.08 the adopted weights before that). A gate belongs to the weight set it was swept
    for: these defaults and weights_dir_* must always move together.
    [2026-10-12] ADOPTED: the state-contraction weights WITH the mid band additionally trained for
    self-consistency under a masked thermodynamic input. Every sample gets one copy with one of
    (tas, tdps, rld) replaced by its training mean, trained to match the same model's full-input
    prediction, detached. Reason: erasing tas from the state-contraction mid band took its CORR from
    0.7413 to -0.2100, a sign flip, and the coupled atmosphere gives no guarantee of an unbiased tas
    (its rld sits 4 sigma low in the summer central pack). After the change, tas erased gives +0.6226
    and a +2 K tas bias costs 0.0259 instead of 0.1032. It also cut bistable rows under the warmest
    forcing from 4 of 15 to 2. Costs, all measured with ERA5's correct tas, which is not the regime
    the coupled model is in: mid single-step CORR 0.7413 -> 0.7300, 1 deg free rollout north
    0.9633 -> 0.9596 and south 0.9420 -> 0.9393. Three-year coupled T63: zero NaN, no spokes in
    either hemisphere, box cycle unchanged.
    The previous entry, kept because the gate and the curvature column were set by it:
    [2026-10-11] the state-contraction weights. The mid and high bands were retrained
    under the global penalty mean(relu(d eq / d c - 1)^2) on synthetic concentrations at the real
    forcings (mid lambda 1, high lambda 0.3); the low band is unchanged. Reason: the blended model
    was BISTABLE -- under a prescribed warmest-phase forcing with no feedback in the calculation it
    settled at 0.92 from above and 0.00 from below at 11 of 15 latitude rows, so the summer state was
    path dependent. Bounding d eq / d c below 1 everywhere makes the fixed point unique by
    construction; bistable rows fall from 12 of 15 to 4. Offline 1 deg free rollout north 0.9614 ->
    0.9633 and SOUTH 0.9310 -> 0.9420. Three-year coupled T63 A/B against the previous weights: zero
    NaN, north grid-scale share 5.687% -> 5.673% with the zonal second difference 16% lower, south
    within a ratio artefact and toward ERA5's known greater roughness."
    lowsic_gate_center::Float32 = 0.20f0

    "[OPTION] sigmoid gate width (SIC units), Northern low-SIC boundary. 0.085, from the same search (0.07 for the adopted weights, 0.05 before)."
    lowsic_gate_width::Float32 = 0.035f0

    "[OPTION] path to a flat Float32 binary with real ERA5-derived initial SIC + 3-block history,
    same layout as MLSeaIceV2's initial_sic_path -- reused directly (real observational initial
    state, independent of which model version consumes it)."
    initial_sic_path::String = joinpath(@__DIR__, "ml_seaice_weights", "initial_sic.bin")

    "[OPTION] SST floor [K] applied every timestep at Arctic/Antarctic (ML-branch) points -- same
    role/value as MLSeaIceV2.tos_floor."
    tos_floor::Float64 = 273.15 - 1.8

    "[OPTION] fallback thermodynamic scheme parameters (same defaults as ThermodynamicSeaIce), used
    for non-ML ocean points only"
    temp_freeze::Float64 = 273.15 - 1.8
    melt_rate::Float64 = 1.0e-6
    freeze_rate::Float64 = 0.1

    "[OPTION] reference angular width, in DEGREES, that the cross (gradient) feature is expressed
    over. 0 (the default) keeps the historical behaviour exactly: the raw difference between
    IMMEDIATELY ADJACENT cells, unscaled, which is what every adopted checkpoint was trained on and
    what every run to date used. A positive value instead rescales the adjacent-cell centred
    difference by (cross_ref_deg / grid spacing), turning it into an estimate of the SIC change per
    `cross_ref_deg` degrees of arc -- the same quantity regardless of the grid it is evaluated on,
    i.e. a true gradient in angular units. Set it to 2.0 to match the 2-degree training data.

    Why it exists (2026-09-24, client request): the client wants weights that transfer to higher
    resolution. Measured offline on a 1-degree grid with the shipped 2-degree weights, free rollout
    1990-1991 (`rollout_sweep_dev/res_transfer_rollout.py`): this rescaled \"true gradient\" form
    loses nothing -- coarsened back to 2 degrees it scores north CORR 0.97295 / RMSE 0.08806 against
    the native 2-degree run's 0.97126 / 0.09071 -- but it IS the roughest of the variants tested
    (zonal roughness 0.00562 against ERA5's 0.00346, where holding the stencil at a fixed 2-degree
    width gives 0.00408). Roughness is the standing online criterion, so a run with this enabled
    must be checked for spokes, not just for finishing.

    NOTE the scale is near 1 on the current grid: T63's FullGaussianGrid spacing is 1.875 degrees, so
    cross_ref_deg=2.0 only multiplies by 1.067. The setting does real work only at a resolution whose
    spacing differs appreciably from 2 degrees."
    # [2026-10-11] 2.0 by default, matching the three-year T63 run that validated the adopted
    # state-contraction weights. CAVEAT, measured 2026-09-26 and not superseded: at T63 this passes,
    # but at T127 it brings SPOKES back (Antarctic pack 8x ERA5, radial stripes), and T127 then needs
    # the halved timestep for atmospheric reasons anyway. Anyone running above T63 should check that
    # before trusting this default.
    cross_ref_deg::Float64 = 2.0

    "[OPTION] scale of the tanh that bounds the THIRD cross column when that column is the zonal
    CURVATURE (W+E)/2 - C rather than the 4th difference. 0 (the default) keeps the historical
    behaviour exactly: a three-column checkpoint gets the 4th-difference q4 column and a two-column
    one is untouched. A positive value selects the curvature column instead and feeds
    tanh(curvature / scale); set it to 0.5 to match `build_cache_curv_tanh.py`.

    Why the curvature (2026-10-01). For a zonal wave the antisymmetric (W-E)/2 is in QUADRATURE, so
    in the complex gain it enters as an imaginary term and can only SHIFT the wave or raise its
    amplitude -- the floor |1 + df/dsic| = 0.944 is reached regardless of how the model uses it. Only
    an IN-PHASE symmetric second difference can damp, and the model had no such input. Offline at
    1 degree, adding it with a one-sided sign penalty took the melt-season interior zonal amplitude
    at 80-84N from 3.18/3.61 of ERA5 at k7/k8 to 2.37/2.44, at 84-88N from 3.56/5.06 to 1.14/1.64,
    and RAISED free rollout CORR from 0.94507 to 0.96140.

    Why bounded. The column is deliberately NOT rescaled between grids -- a 2-cell checkerboard has
    per-cell curvature +-1 at any resolution -- so its magnitude is grid-relative and means nothing
    physical. Unbounded it is then used far outside its trained range: nothing in the training caches
    exceeds |curv| = 0.37 (std 0.015-0.056 by band) while the checkerboard the column exists to damp
    sits at 1.0. tanh leaves the trained range essentially linear (p99 compressed under 1%) and
    saturates the extrapolation at 0.964. Measured consequence: under a x4 stress test the RAW column
    produced visible grid-scale banding in the Antarctic (CORR 0.8278) where the bounded one did not
    (0.9318). Spokes are the first rejection criterion online, which is why the bounded form is the
    adopted one.

    Scaling. The GRADIENT columns carry the cross_ref_deg angular rescaling; this one does NOT, by
    design and not by omission. The gradient is a physical quantity -- advection has a physical speed
    -- so a fixed angular baseline is the meaningful one for it. The curvature is here to damp
    grid-scale structure, and a 2-cell checkerboard has per-cell curvature +-1 at ANY resolution, so
    the quantity is already grid-relative. The two conventions differ on purpose.

    It is also what was validated: offline at 1 degree, where a physical rescaling would be
    (2.0/1.0)^2 = 4, the coefficient sweep put the optimum at 1.0-1.5 and measured a real cost at 4
    (north CORR 0.96140 -> 0.91872, and with the UNBOUNDED column visible Antarctic banding at
    CORR 0.8278). Coefficient 1.0 is the setting the acceptance table was produced with."
    # [2026-10-11] 0.5 with the adopted state-contraction weights. This is NOT an independent
    # option: those weights were trained on the bounded zonal curvature tanh(c/0.5) as their third
    # cross column, and leaving this at 0.0 feeds them the 4th-difference q4 column instead --
    # silently, because both are three-column forms. It must move with weights_dir_*.
    curvature_tanh_scale::Float64 = 0.5

    "[OPTION] make the third cross column the PHYSICAL LAPLACIAN on the lat-lon grid rather than the
    zonal second difference alone: ((W-C)+(E-C))/2 + cos^2(lat) ((N-C)+(S-C))/2, still bounded by
    `curvature_tanh_scale`. The true Laplacian's zonal term carries 1/cos^2 of the meridional one at
    a fixed angular spacing, because two degrees of longitude is the shorter arc everywhere off the
    equator; dividing through by that keeps the zonal coefficient at 1, which is what the tanh scale
    and the whole trained range of this column are calibrated to, so only the meridional term is new.

    Like `curvature_tanh_scale` this is NOT an independent knob. It is part of the column's
    DEFINITION, so it must move with `weights_dir_*`: it is true only for weights trained on a
    `curvature_laplacian` cache (`ml_seaice_weights_v4_lap_*` and `ml_seaice_weights_v4_lapmtas_*`,
    which share that column) and false for every other package.
    Setting it false while `weights_dir_*` still points at one of those, or true while they
    point at any other, feeds the bands a column they were never fitted on, SILENTLY, because both
    forms are three-column.

    [2026-10-15] ADOPTED, together with the `_lap` weights, on the client's preference for an
    isotropic column. What the change buys and costs, all measured: the meridional Nyquist response
    falls from 0.8607 to 0.6321 in the north and 0.9516 to 0.7692 in the south while the zonal
    response does not degrade (0.6371 to 0.6082, 0.7774 to 0.7420), and the Antarctic's small
    amplitude meridional AMPLIFICATION at 1.0749 is removed; every offline window improves in both
    hemispheres; the coupled run is a wash on spokes (northern grid-scale share 5.635 against
    5.675 per cent) but 17 per cent smoother in the north, 6.15x smoother than ERA5 against the
    previous 5.08x, which is movement away from the reanalysis in a field that was already about
    five times too smooth. Stability is unchanged: 37 of 37 steps, NaN-free."
    curvature_laplacian::Bool = true
    """[2026-10-22] Resolution factor applied to the RESOLVED part of the curvature column only.

    The client requires the curvature column to carry a grid-scaling factor, as the gradient columns
    already do through `cross_ref_deg`. Applying one to the whole column is measured to be harmful:
    at 1 degree offline, a factor of 4 on the whole column takes the rolled-out field's zonal second
    difference from 0.76x the reanalysis to 3.53x it, and rollout correlation from 0.96679 to
    0.95261. The reason is that `curvature_tanh_scale` is implicitly anchored to the 2-cell mode,
    whose raw curvature is 1.25 at ANY resolution, so a resolution factor lifts real structure toward
    a ceiling that does not move and the column stops separating the two.

    This applies the factor only where it is physically meaningful. The same Laplacian is evaluated
    on a 2-cell stencil and normalised by 4 so it agrees with the 1-cell operator on resolved
    structure; being a second difference over 2 cells it is IDENTICALLY ZERO on the 2-cell mode. With
    q1 the usual operator and q2 that one,

        q = curvature_k_resolved * q2 + (q1 - q2)

    is bit-identical to q1 on the 2-cell mode, so the anchor is untouched, while resolved structure
    carries the full factor. Measured at 1 degree offline: factor 4 gives correlation 0.96816 and
    0.62x the reanalysis roughness, against 0.96679 and 0.76x with no factor at all.

    1.0 reproduces the current column exactly and is the default; nothing changes unless it is set.
    NOT yet confirmed in a coupled run at a resolution where the factor differs materially from 1."""
    curvature_k_resolved::Float32 = 1f0
    """[2026-10-22] Derive `curvature_k_resolved` from the grid in initialize!, as the AREA factor
    `cross_scale_ns * cross_scale_we`, i.e. the dx^-2 analogue of the gradient's `cross_ref_deg`
    rescaling and taken from the same source so the two cannot drift apart. At T63 that is about
    1.14, so enabling it there is close to a null test; at T127 it is about 4.5, which is where it
    has to prove itself. false leaves `curvature_k_resolved` exactly as set."""
    curvature_k_auto::Bool = false

    "[OPTION] threshold on the low band's CALIBRATED probability of change. Below it the cell does
    not change at all, wherever the low band holds more than half the blend.

    Why. The low band cannot emit \"no change\": its tendency is p_changed * (eq - c) * (1 - decay)
    with p_changed from a softmax and eq from a sigmoid, so an exactly zero tendency is not
    representable and every cell moves every block. The reanalysis holds 94.5 to 98.2 per cent of
    its open water at exactly zero from one block to the next and the component holds 0.0 per cent,
    which parks a third of the Arctic at 0.02 to 0.04 instead of 0 and biases the mean concentration
    up. Letting the change gate mean what it says recovers about 19 percentage points of that area.

    The threshold is applied to a CALIBRATED probability, because the raw head is overconfident
    against its own training label: it reports 0.62 where the truth is 0.29 and 0.021 where the
    truth is 0.0036. `changegate_calib_a` and `_b` are a Platt scaling fitted on training years
    1981-1988 over both hemispheres, after which 0.5 means what it says. Selected on the training
    years, where both stages put the optimum at exactly 0.5 with 0.4 and 0.6 worse on each.
    0 disables the whole mechanism and reproduces every run before 2026-10-16."
    changegate_threshold::Float32 = 0.5f0
    "[OPTION] Platt slope for the low band's change probability; see `changegate_threshold`."
    changegate_calib_a::Float32 = 1.14784f0
    "[OPTION] Platt intercept for the low band's change probability; see `changegate_threshold`."
    changegate_calib_b::Float32 = -1.33014f0

    "[OPTION] clamp the `tos` fed to the network to the TRAINING-SET envelope: a ceiling that
    tightens as SIC rises, because more ice means less open water able to warm. 0 disables it and
    reproduces every run before 2026-09-30. It does NOT touch the ocean state, only what the
    surrogate is asked about.

    Why (2026-09-30). In the training data the under-ice water is pinned at the freezing point by the
    phase change: over 620k north points with SIC >= 0.9 the tos has std 0.0003 K and never exceeds
    271.47. Coupled, it reaches 272.7 K on average in polar summer and 276.2 K at worst -- roughly
    5000 sigma outside anything the network was fit on, and the band it governs is exactly the
    central pack. Replaying the coupled run offline from its own recorded forcing (which reproduces
    the trajectory to corr 0.998) shows this single input explains most of the central-Arctic summer
    deficit: clamping it takes the 80-90N August mean from 0.50 to 0.70, and substituting ERA5's tos
    outright takes it to 0.82, while fixing the radiation instead changes little.

    The warm water is itself an artifact rather than a signal to respect: SpeedyWeather's slab ocean
    absorbs about 124 W/m2 under a 0.9 concentration pack (rsds 273 against ERA5's 143, ice albedo
    0.6 against a real 0.65-0.8) and melting ice removes NO heat from the 50 m mixed layer, whose
    heat capacity is 2.09e8 J/K/m2 -- the excess alone integrates to ~2.75 K over a melt season, the
    right order for the +1.2 K observed. So no explicit melt term is added for the clamped-off
    excess: that would amplify the artifact, and it would double count against a network whose
    training targets already contain the real ocean heat flux at freezing-point tos.

    Limitation, stated plainly: with the clamp the ice no longer responds to a genuinely warm ocean
    at high concentration. The training data gives no basis for such a response (zero variance), so
    nothing measurable is lost here, but this would matter in a warming-ocean experiment."
    # [2026-10-11] 1.0 by default: the justification above is a measurement, not a preference --
    # coupled under-ice tos reaches 272.7 K in polar summer and 276.2 K at worst against a training
    # distribution pinned at 271.47 with std 0.0003 K, and clamping takes the 80-90N August mean
    # from 0.50 to 0.70. It is also what the run that validated the adopted weights used.
    tos_ceiling_from_training::Float64 = 1.0

    "[OPTION] apply the unconditionally-stable spectral Laplacian diffusion pass at Arctic points.
    Defaults to FALSE for V4 -- diffusion has not yet been swept for the gradient-basis-cross
    representation (the gate sweep this model's defaults come from was run entirely with d_dt=0).
    Set true and provide a coefficient explicitly if/when a diffusion sweep is done."
    apply_spectral_diffusion::Bool = false

    "[OPTION] diffusion coefficient [m^2/s] for the Northern spectral Laplacian pass. NOT a
    swept/adopted value for V4 (carried over from V3 as a placeholder only) -- unused while
    apply_spectral_diffusion=false."
    diffusion_coefficient::Float64 = 34.7222222

    "[OPTION] minimum locally-diffused ocean weight, Northern pass -- same role as MLSeaIceV2's own."
    diffusion_min_weight::Float64 = 0.3

    # ======================================================================
    # Southern Hemisphere (Antarctic) options
    # ======================================================================

    "[OPTION] enable the Southern Hemisphere ML branch."
    enable_south::Bool = true

    "[OPTION] latitude (degrees, positive) below which (i.e. south of -lat_limit_south) the
    Southern ML branch is used -- same value as MLSeaIceV2 (55)."
    lat_limit_south::Float64 = 55.0

    "[OPTION] sigmoid gate center (SIC units), Southern low-SIC boundary. 0.02, unchanged between
    the 2026-08-26 (plain separate-penalty) and 2026-09-14 (radrobust) TRAIN-only offline sweeps
    (d_dt=0 fixed both times)."
    lowsic_gate_center_south::Float32 = 0.02f0

    "[OPTION] sigmoid gate width (SIC units), Southern low-SIC boundary."
    lowsic_gate_width_south::Float32 = 0.025f0

    "[OPTION] apply the spectral Laplacian diffusion pass at Antarctic points, own D (separate
    spectral transform from the Northern pass, see MLSeaIce for why). Defaults to FALSE for V4 --
    see apply_spectral_diffusion's docstring above."
    apply_spectral_diffusion_south::Bool = false

    "[OPTION] diffusion coefficient [m^2/s] for the Southern spectral Laplacian pass. NOT a
    swept/adopted value for V4 (carried over from V3 as a placeholder only) -- unused while
    apply_spectral_diffusion_south=false."
    diffusion_coefficient_south::Float64 = 2314.8148148

    "[OPTION] minimum locally-diffused ocean weight, Southern pass."
    diffusion_min_weight_south::Float64 = 0.3

    "[OPTION] path to a flat Float32 binary with real ERA5-derived initial Antarctic SIC + 3-block
    history, same layout as initial_sic_path."
    initial_sic_path_south::String = joinpath(@__DIR__, "ml_seaice_weights", "initial_sic_south.bin")

    # -- allocated in initialize!(vars, model) once the grid is known --
    weights_mid::Union{Nothing, MLWeightsV4} = nothing
    weights_highsic::Union{Nothing, MLWeightsV4} = nothing
    weights_lowsic::Union{Nothing, MLWeightsChangeGateV4} = nothing

    """[OPTION] Run the three bands through the allocation-free flat forward pass
    (`ml_forward_flat_v4`) instead of the clarity-first one. Same arithmetic, different storage:
    the backbone's H->H layers become one (H, H, 3) array, single-output heads become vectors,
    and the per-point working vectors become columns of preallocated scratch -- which is what
    makes the sweep expressible as a kernel. Measured on 20000 inputs across the full coupled
    envelope: allocations 5456 -> 0 bytes per point per band, 1.25x faster on the CPU, and
    max|original - flat| of 3.9e-7 (mid), 8.3e-7 (high), 8.6e-7 (low) -- NOT bit-identical,
    because `W' * h` goes through BLAS gemv whose blocked reduction order no scalar loop
    reproduces. Default false."""
    flat_forward::Bool = false

    # flat weights, built from the loaded ones in initialize! when flat_forward is on
    flat_mid::Union{Nothing, MLWeightsFlatV4CPU} = nothing
    flat_highsic::Union{Nothing, MLWeightsFlatV4CPU} = nothing
    flat_lowsic::Union{Nothing, MLWeightsFlatChangeGateV4CPU} = nothing
    """[OPTION] Number of scratch SLOTS the flat pass has. One slot per concurrent worker, not
    per grid point: a serial sweep needs exactly one and should use it, because a column per
    point scatters the working vectors over megabytes and costs more in cache misses than the
    allocation it saves (measured: the 5-day update went 0.10 -> 0.14 s per 6-day chunk when
    indexed by point). A kernel sets this to npoints so every thread owns a column; a threaded
    CPU sweep would set it to the thread count. 0 means npoints."""
    flat_nslots::Int = 1

    # scratch: (width, flat_nslots). One column per concurrent worker.
    flat_x_mid::Matrix{Float32} = zeros(Float32, 0, 0)
    flat_x_high::Matrix{Float32} = zeros(Float32, 0, 0)
    flat_x_low::Matrix{Float32} = zeros(Float32, 0, 0)
    flat_h1::Matrix{Float32} = zeros(Float32, 0, 0)
    flat_h2::Matrix{Float32} = zeros(Float32, 0, 0)
    flat_mb::Matrix{Float32} = zeros(Float32, 0, 0)

    """[OPTION] Run the 5-day update as KernelAbstractions kernels instead of the serial sweep, so
    it can execute on a GPU. Requires `flat_forward` (the kernels take each band's weights as ONE
    struct argument, which is the only way the launches fit inside Metal's ~29-argument budget) and
    `cross_from_snapshot` (without it the sweep's result depends on the visiting order, so no
    parallel form of it exists -- see ml_seaice_v4_kernels.jl). Both are asserted, not switched on
    silently. Measured at T63 size on Metal: 4.0x the serial path, with the CPU kernel path
    bit-identical to it."""
    kernel_sweep::Bool = false

    """[OPTION] Architecture the kernel sweep runs on, when it should NOT be the model's. The
    realistic configuration is exactly that: the model on the CPU and this component on the GPU,
    because the spectral transform is 45-260x SLOWER on Metal while this component is 4x faster, so
    moving the whole model would lose far more than the component gains. `nothing` means the
    model's own architecture. Only the sea-ice field, the sea surface temperature and the six cycle
    means cross the bus, once per 5-day update."""
    kernel_architecture::Any = nothing

    """Device buffers for `kernel_sweep`, built in `initialize!`. Typed `Any` deliberately: the
    element type is the device's array type, which differs per backend, and a type parameter on
    this struct collided with `@kwdef`. It is only ever read once per 5-day update, immediately
    passed into `ml_seaice_sweep_kernels!`, so the single dynamic dispatch is free and nothing
    inside the barrier is untyped."""
    kernel_buffers::Any = nothing
    is_arctic::Vector{Bool} = zeros(Bool, 0)
    is_antarctic::Vector{Bool} = zeros(Bool, 0)
    east_idx::Vector{Int} = zeros(Int, 0)
    west_idx::Vector{Int} = zeros(Int, 0)
    # +-2 cells along the latitude circle, needed only for the 4th-difference third cross column
    east2_idx::Vector{Int} = zeros(Int, 0)
    west2_idx::Vector{Int} = zeros(Int, 0)
    north2_idx::Vector{Int} = zeros(Int, 0)
    south2_idx::Vector{Int} = zeros(Int, 0)
    north_idx::Vector{Int} = zeros(Int, 0)
    south_idx::Vector{Int} = zeros(Int, 0)
    # cos^2(latitude) per point, the metric factor the physical Laplacian weights its meridional
    # term by; filled in initialize! and used only when curvature_laplacian is true
    coslat2::Vector{Float32} = zeros(Float32, 0)
    # multipliers applied to the two cross components, set in initialize! from cross_ref_deg and the
    # actual grid spacing; both stay 1 when cross_ref_deg == 0 (the default)
    cross_scale_ns::Float32 = 1f0
    cross_scale_we::Float32 = 1f0

    # running-mean accumulators for the CURRENT (not-yet-finalized) cycle
    accum_rsds::Vector{Float64} = zeros(0)
    accum_lw::Vector{Float64} = zeros(0)   # rld ALONE here (not net longwave, see docstring item 1)
    accum_wind::Vector{Float64} = zeros(0)
    accum_tas::Vector{Float64} = zeros(0)
    accum_tos::Vector{Float64} = zeros(0)
    accum_tdps::Vector{Float64} = zeros(0)
    accum_count::Int = 0

    tos_prev_mean::Vector{Float64} = zeros(0)
    sic_hist1::Vector{Float64} = zeros(0)
    sic_hist2::Vector{Float64} = zeros(0)
    sic_hist3::Vector{Float64} = zeros(0)
    initialized::Vector{Bool} = zeros(Bool, 0)

    """[OPTION] Precision the 5-day update is carried out in ON THE DEVICE, 64 or 32. Set by the
    generator from what the device actually supports (probed, see
    `architecture_supports_float64`): 64 on the CPU and CUDA, 32 only where there is no double
    precision at all, and then only because it was asked for -- requesting 64 on such a device
    is an error, not a silent downgrade.

    [2026-11-11] This option does NOT change the HOST accumulators, and an earlier version of
    this docstring wrongly said it did. The accumulator fields above are declared
    `Vector{Float64}`, so `setproperty!` converts anything assigned to them back to Float64 and
    `zeros(Float32, npoints)` silently became a Float64 array -- measured, not inferred. Nothing
    needs them narrowed: the per-timestep accumulation is host work either way, and the kernel
    sweep converts to Float32 once per 5-day cycle when it copies them to the device. What the
    option still does is refuse Float64 DEVICE arithmetic on a device that has none, which is
    the assertion it was added for."""
    accum_precision::Int = 64

    """Build the cross (gradient / curvature) columns from a SNAPSHOT of the sea-ice field taken
    before the 5-day update, instead of from the live array the same loop is writing into.

    This is unconditional and `initialize!` REFUSES a model built without it, because it is not a
    modelling option: with it off the sweep is wrong. The
    update loop assigns `ℵ[ij] = sic_next` and reads `ℵ[nidx]` for the neighbours in the same
    pass, so with `ij` ascending the WEST neighbour (ij-1, same ring) and the NORTH one (ring j-1,
    lower indices) are already updated while EAST (ij+1) and SOUTH (ring j+1) are not. The gradient
    columns `(W-E)/2`, `(N-S)/2` and the curvature are then formed from a mix of post- and
    pre-update values, with a fixed sign per axis, while the offline training cache builds all of
    them from one consistent snapshot -- so the component is fed a different operator than the one
    its weights were fitted to, and the discrepancy is of the order of the feature itself.

    The snapshot also makes the sweep a Jacobi update, i.e. order-independent, which is what allows
    the loop to be threaded or run as kernels at all; the kernel and GPU paths have always required
    it, so this change makes the serial path agree with them instead of diverging from them."""
    cross_from_snapshot::Bool = true

    """[OPTION] Print, for each of the first `cross_snapshot_report` updates, how large the
    sequential-sweep contamination of the cross columns actually is: the RMS of
    (live-array column - snapshot column) against the RMS of the column itself, per column and
    per hemisphere. Costs a second pass over the ice points on those updates only."""
    cross_snapshot_report::Int = 0

    # snapshot buffer, allocated in initialize! only when cross_from_snapshot or the report is on
    sic_snapshot::Vector{Float32} = zeros(Float32, 0)

    last_update::Base.RefValue{DateTime} = Ref(DEFAULT_DATE)
end

"""$(TYPEDSIGNATURES)
Does this architecture actually support Float64? Probed, not inferred from the backend's name:
allocate a Float64 array on the device and do one operation on it. CPU and CUDA pass; Metal has
no double precision at all and throws. The result is cached per architecture type, so the probe
runs once per session.

The component accumulates 720 timesteps of radiation and temperature per 5-day cycle, which is
why the accumulators are Float64 wherever that is available: a naive Float32 sum of 720 terms of
order 300 W/m2 carries a rounding walk of ~0.2 in a total of ~216000, i.e. ~3e-4 W/m2 in the
cycle mean. That is small but it is NOT zero, so the precision is a declared option that is
asserted against the device rather than silently chosen."""
function architecture_supports_float64(arch)
    get!(_F64_SUPPORT, typeof(arch)) do
        try
            # NOT on_architecture: that SILENTLY DOWNCASTS a Float64 array to the device's
            # precision (measured: on_architecture(GPU(), ones(Float64,4)) returns an
            # MtlVector{Float32}), so it reports success on a device with no double precision at
            # all. Build the device array type at Float64 and force a broadcast, which is what
            # actually has to compile.
            AT = Architectures.array_type(arch)
            a = AT{Float64}(undef, 4)
            fill!(a, 1.0)
            b = a .* 2.0
            isfinite(Float64(first(Array(b))))
        catch
            false
        end
    end
end
const _F64_SUPPORT = Dict{Any, Bool}()

"""$(TYPEDSIGNATURES)
The element type the 5-day cycle is accumulated in. Float64 unless the device cannot do it, in
which case the generator has already asserted that Float32 was asked for explicitly."""
accumulator_type(sea_ice_model::MLSeaIceV4) =
    sea_ice_model.accum_precision == 32 ? Float32 : Float64

"""$(TYPEDSIGNATURES)
Generator. The accumulator precision is chosen from what the architecture actually supports and
ASSERTED, never silently downgraded: pass `accum_precision = 64` and it errors on a device
without double precision instead of quietly switching to Float32 and changing every cycle mean.
Left unset, it takes Float64 where the device has it (CPU, CUDA) and Float32 where it does not
(Metal), and says so."""
function MLSeaIceV4(SG::SpectralGrid; accum_precision = nothing, kwargs...)
    # the device the SWEEP will run on, which is the component's own architecture when it has one
    # (model on the CPU, component on the GPU) and the model's otherwise
    arch = get(kwargs, :kernel_architecture, nothing)
    arch = arch === nothing ? SG.architecture : arch
    has_f64 = architecture_supports_float64(arch)
    A = if accum_precision === nothing
        has_f64 ? Float64 : Float32
    elseif accum_precision == 64
        has_f64 || error("MLSeaIceV4: accum_precision = 64 was requested but $(typeof(arch)) " *
                         "has no Float64 (probed by allocating one on the device). Metal is the usual " *
                         "case; CUDA and the CPU both support it. Pass accum_precision = 32 to accept " *
                         "Float32 accumulation of the 720-step cycle (~3e-4 W/m2 in the cycle mean), " *
                         "or run this on a device that has double precision.")
        Float64
    elseif accum_precision == 32
        Float32
    else
        error("MLSeaIceV4: accum_precision must be 32, 64 or nothing, got $(accum_precision)")
    end
    accum_precision === nothing && !has_f64 &&
        @info "MLSeaIceV4: $(typeof(arch)) has no Float64; the kernel sweep will run in Float32"

    # [2026-11-11] FOLLOW THE MODEL. When the model itself is running on a GPU, the serial sweep
    # and the per-timestep accumulation loop both index the model's fields one element at a time,
    # which on device arrays is scalar indexing -- CUDA.jl refuses it outright, so the component
    # does not merely run slowly there, it does not run at all. The kernel path is therefore the
    # DEFAULT whenever the model is on a GPU, together with the two options it requires, and the
    # user does not have to know any of this to run a parallel model. Nothing changes for a host
    # model: there the defaults stay exactly as they were, so every existing run is untouched.
    extra = NamedTuple()
    if SG.architecture isa Architectures.GPU && !haskey(kwargs, :kernel_sweep)
        extra = (;
            kernel_sweep = true,
            flat_forward = get(kwargs, :flat_forward, true),
            cross_from_snapshot = get(kwargs, :cross_from_snapshot, true),
        )
        @info "MLSeaIceV4: the model is on $(typeof(SG.architecture)), so the component follows it " *
              "-- kernel_sweep on, with flat_forward and cross_from_snapshot, which it requires"
    end
    return MLSeaIceV4(; accum_precision = (A === Float64 ? 64 : 32), kwargs..., extra...)
end


# Training-set tos ceiling as a function of SIC, derived from the TRAIN years 1980-1988 of
# cache_rld_ablation_nosmooth (the per-bin maximum, made monotone non-increasing in SIC). Bin k
# covers SIC < TOS_CEIL_EDGES[k]. Generated by the analysis in project memory, 2026-09-30.
const TOS_CEIL_EDGES = Float32[0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 1.01]
const TOS_CEIL_NORTH = Float32[280.216, 277.691, 277.691, 275.870, 275.247, 274.751, 274.127, 274.071, 273.835, 273.611, 273.377, 273.118, 272.492, 272.341, 272.193, 271.913, 271.594, 271.512, 271.469]
const TOS_CEIL_SOUTH = Float32[276.091, 274.963, 274.226, 273.598, 273.455, 273.146, 272.920, 272.683, 272.609, 272.519, 272.226, 272.090, 271.814, 271.803, 271.616, 271.565, 271.556, 271.512, 271.512]

@inline function tos_ceiling(sic::Float32, north::Bool)
    tbl = north ? TOS_CEIL_NORTH : TOS_CEIL_SOUTH
    @inbounds for k in eachindex(TOS_CEIL_EDGES)
        sic < TOS_CEIL_EDGES[k] && return tbl[k]
    end
    return tbl[end]
end

"""$(TYPEDSIGNATURES)
Read a weight package's `curvature.txt`, the declaration of the cross-feature curvature column the
package was TRAINED on. Returns `nothing` when the file is absent (packages exported before
2026-10-29), otherwise a NamedTuple with `cross_columns` and, when that is at least 3,
`curvature_tanh_scale` and `curvature_laplacian`."""
function read_curvature_declaration(dir::String)
    path = joinpath(dir, "curvature.txt")
    isfile(path) || return nothing
    kv = Dict{String, String}()
    for line in eachline(path)
        line = strip(line)
        (isempty(line) || startswith(line, "#")) && continue
        parts = split(line, '='; limit = 2)
        length(parts) == 2 || error("$(path): cannot parse line \"$(line)\"")
        kv[strip(parts[1])] = strip(parts[2])
    end
    haskey(kv, "cross_columns") || error("$(path): no cross_columns entry")
    ncross = parse(Int, kv["cross_columns"])
    ncross < 3 && return (; cross_columns = ncross)
    return (; cross_columns = ncross,
            curvature_tanh_scale = parse(Float64, kv["curvature_tanh_scale"]),
            curvature_laplacian = parse(Bool, kv["curvature_laplacian"]))
end

"""$(TYPEDSIGNATURES)
Check this model's curvature settings against what each weight package declares it was trained on,
and `error` on a mismatch.

The cross-feature curvature column is defined by `curvature_tanh_scale` together with
`curvature_laplacian`, and BOTH belong to the column, not to the run. Setting either one away from
what the weights were fitted to silently feeds every band a different operator -- silently, because
all the variants are three-column forms of the same shape. That failure has now happened three
times (2026-09-29 feature order, 2026-10-15 the Antarctic branch's q4, 2026-10-27 `lapmtas` missing
from a script's `curvature_tanh_scale` list, which cost three 3-year coupled runs and four hours of
T127). Each time the configuration lived in a caller's flag list, which drifted away from the
weights. The declaration now travels INSIDE the package, so a caller cannot get it wrong quietly.

A package with no `curvature.txt` is warned about once rather than rejected, so externally exported
weights still load."""
function check_curvature_declarations(sea_ice_model::MLSeaIceV4)
    for (band, dir) in (("mid", sea_ice_model.weights_dir_mid),
                        ("high", sea_ice_model.weights_dir_highsic),
                        ("low", sea_ice_model.weights_dir_lowsic))
        decl = read_curvature_declaration(dir)
        if decl === nothing
            @warn "MLSeaIceV4: $(band) band at $(dir) declares no curvature column " *
                  "(no curvature.txt); its settings cannot be checked against the weights."
            continue
        end
        decl.cross_columns < 3 && continue   # no curvature column is built for this package
        if decl.curvature_tanh_scale != sea_ice_model.curvature_tanh_scale ||
           decl.curvature_laplacian != sea_ice_model.curvature_laplacian
            error("MLSeaIceV4: the $(band) band's weights at $(dir) were trained on the curvature " *
                  "column (tanh_scale=$(decl.curvature_tanh_scale), " *
                  "laplacian=$(decl.curvature_laplacian)), but this model is configured for " *
                  "(tanh_scale=$(sea_ice_model.curvature_tanh_scale), " *
                  "laplacian=$(sea_ice_model.curvature_laplacian)). These define the column and " *
                  "must move with weights_dir_*; running on would feed the band an operator it was " *
                  "never fitted on.")
        end
    end
    return nothing
end

function initialize!(sea_ice_model::MLSeaIceV4, model::AbstractModel)
    check_curvature_declarations(sea_ice_model)
    sea_ice_model.weights_mid = load_ml_weights_v4(sea_ice_model.weights_dir_mid)
    sea_ice_model.weights_highsic = load_ml_weights_v4(sea_ice_model.weights_dir_highsic)
    sea_ice_model.weights_lowsic = load_ml_weights_changegate_v4(sea_ice_model.weights_dir_lowsic)
    return nothing
end

function initialize!(vars::Variables, sea_ice_model::MLSeaIceV4, model::PrimitiveEquation)
    ℵ = vars.prognostic.ocean.sea_ice_concentration
    grid = ℵ.grid
    npoints = length(ℵ)
    londs, latds = RingGrids.get_londlatds(grid)

    (; land_fraction) = model.land_sea_mask
    # [2026-11-11] `.data` and an explicit Array(): on a GPU model land_fraction is a Field over a
    # device array and these comprehensions index it point by point, which CUDA.jl refuses. Going
    # through the underlying array makes this one device-to-host copy instead of npoints of them --
    # `Array(::Field)` would fall back to the generic, scalar path. Everything initialize! builds
    # below is a host-side table, so the host copy is the right representation to work from.
    lf_host = Array(land_fraction.data)
    is_arctic = [(latds[ij] >= sea_ice_model.lat_limit) && (lf_host[ij] < 1) for ij in 1:npoints]
    is_antarctic = sea_ice_model.enable_south ?
        [(latds[ij] <= -sea_ice_model.lat_limit_south) && (lf_host[ij] < 1) for ij in 1:npoints] :
        zeros(Bool, npoints)

    rings = grid.rings
    ring_of = zeros(Int, npoints)
    for (j, r) in enumerate(rings), ij in r
        ring_of[ij] = j
    end

    east_idx = zeros(Int, npoints)
    west_idx = zeros(Int, npoints)
    east2_idx = zeros(Int, npoints)
    west2_idx = zeros(Int, npoints)
    north_idx = zeros(Int, npoints)
    south_idx = zeros(Int, npoints)
    north2_idx = zeros(Int, npoints)
    south2_idx = zeros(Int, npoints)

    for (j, r) in enumerate(rings)
        n = length(r)
        for (k, ij) in enumerate(r)
            east_idx[ij] = r[mod1(k + 1, n)]
            west_idx[ij] = r[mod1(k - 1, n)]
            east2_idx[ij] = r[mod1(k + 2, n)]
            west2_idx[ij] = r[mod1(k - 2, n)]
        end
    end

    function nearest_in_ring(ij, target_ring)
        (target_ring < 1 || target_ring > length(rings)) && return 0
        r = rings[target_ring]
        isempty(r) && return 0
        lon0 = londs[ij]
        best = r[1]; bestd = 361.0
        for jk in r
            d = abs(mod(londs[jk] - lon0 + 180, 360) - 180)
            if d < bestd
                bestd = d; best = jk
            end
        end
        return best
    end

    for ij in 1:npoints
        j = ring_of[ij]
        north_idx[ij] = nearest_in_ring(ij, j - 1)
        south_idx[ij] = nearest_in_ring(ij, j + 1)
        north2_idx[ij] = nearest_in_ring(ij, j - 2)
        south2_idx[ij] = nearest_in_ring(ij, j + 2)
    end

    # cross-feature rescaling (see cross_ref_deg's docstring). The meridional spacing is taken as the
    # mean gap between consecutive rings and the zonal spacing from the number of longitudes in the
    # equatorial ring; both are exact for the FullGaussianGrid this component requires anyway (the
    # gradient cross feature needs an exact neighbour at every point, see the type docstring).
    if sea_ice_model.cross_ref_deg > 0
        ring_lats = [latds[first(r)] for r in rings if !isempty(r)]
        dlat = length(ring_lats) > 1 ? abs(ring_lats[end] - ring_lats[1]) / (length(ring_lats) - 1) : 1.0
        dlon = 360.0 / maximum(length(r) for r in rings)
        sea_ice_model.cross_scale_ns = Float32(sea_ice_model.cross_ref_deg / dlat)
        sea_ice_model.cross_scale_we = Float32(sea_ice_model.cross_ref_deg / dlon)
        if sea_ice_model.curvature_k_auto
            sea_ice_model.curvature_k_resolved = sea_ice_model.cross_scale_ns * sea_ice_model.cross_scale_we
            @info "MLSeaIceV4: curvature_k_resolved set from the grid to " *
                  "$(round(sea_ice_model.curvature_k_resolved, digits=4)) (resolved part only)"
        end
        @info "MLSeaIceV4: cross feature rescaled to a $(sea_ice_model.cross_ref_deg) deg reference " *
              "(grid spacing $(round(dlat, digits=4)) x $(round(dlon, digits=4)) deg, " *
              "scale $(round(sea_ice_model.cross_scale_ns, digits=4)) x $(round(sea_ice_model.cross_scale_we, digits=4)))"
    end

    sea_ice_model.is_arctic = is_arctic
    sea_ice_model.is_antarctic = is_antarctic
    sea_ice_model.east_idx = east_idx
    sea_ice_model.west_idx = west_idx
    sea_ice_model.east2_idx = east2_idx
    sea_ice_model.west2_idx = west2_idx
    sea_ice_model.north_idx = north_idx
    sea_ice_model.south_idx = south_idx
    sea_ice_model.north2_idx = north2_idx
    sea_ice_model.south2_idx = south2_idx
    sea_ice_model.coslat2 = Float32[max(cosd(latds[ij]), 1f-3)^2 for ij in 1:npoints]
    if sea_ice_model.cross_from_snapshot || sea_ice_model.cross_snapshot_report > 0
        sea_ice_model.sic_snapshot = zeros(Float32, npoints)
    end

    if sea_ice_model.flat_forward
        fm = flatten_weights_v4(sea_ice_model.weights_mid)
        fh = flatten_weights_v4(sea_ice_model.weights_highsic)
        fl = flatten_weights_v4(sea_ice_model.weights_lowsic)
        # the three bands share the backbone width and the momentum width, which is what lets one
        # h1/h2/mb scratch serve all of them; assert it rather than assume it
        (fm.H == fh.H == fl.H) ||
            error("MLSeaIceV4: flat_forward needs one backbone width across bands, got " *
                  "mid $(fm.H), high $(fh.H), low $(fl.H)")
        (fm.Mm == fh.Mm == fl.Mm) ||
            error("MLSeaIceV4: flat_forward needs one momentum width across bands, got " *
                  "mid $(fm.Mm), high $(fh.Mm), low $(fl.Mm)")
        sea_ice_model.flat_mid = fm
        sea_ice_model.flat_highsic = fh
        sea_ice_model.flat_lowsic = fl
        nslots = sea_ice_model.flat_nslots <= 0 ? npoints : sea_ice_model.flat_nslots
        sea_ice_model.flat_x_mid = zeros(Float32, fm.nin, nslots)
        sea_ice_model.flat_x_high = zeros(Float32, fh.nin, nslots)
        sea_ice_model.flat_x_low = zeros(Float32, fl.nin, nslots)
        sea_ice_model.flat_h1 = zeros(Float32, fm.H, nslots)
        sea_ice_model.flat_h2 = zeros(Float32, fm.H, nslots)
        sea_ice_model.flat_mb = zeros(Float32, fm.Mm, nslots)
        @info "MLSeaIceV4: flat_forward on (nin mid/high/low = $(fm.nin)/$(fh.nin)/$(fl.nin), " *
              "H = $(fm.H), Mm = $(fm.Mm), $(nslots) slot(s), scratch " *
              "$(round((fm.nin + fh.nin + fl.nin + 2fm.H + fm.Mm) * nslots * 4 / 2^20, digits=3)) MB)"
    end

    # [2026-11-11] The accumulator FIELDS are declared Vector{Float64}, so these assignments are
    # Float64 whatever A is -- see the note on accum_precision. A is kept only for the NaN fills
    # below, where it is equally Float64, and the option's real job is the device-side assertion.
    A = accumulator_type(sea_ice_model)
    sea_ice_model.accum_rsds = zeros(A, npoints)
    sea_ice_model.accum_lw = zeros(A, npoints)
    sea_ice_model.accum_wind = zeros(A, npoints)
    sea_ice_model.accum_tas = zeros(A, npoints)
    sea_ice_model.accum_tos = zeros(A, npoints)
    sea_ice_model.accum_tdps = zeros(A, npoints)
    sea_ice_model.accum_count = 0

    sea_ice_model.tos_prev_mean = fill(A(NaN), npoints)
    sea_ice_model.sic_hist1 = fill(A(NaN), npoints)
    sea_ice_model.sic_hist2 = fill(A(NaN), npoints)
    sea_ice_model.sic_hist3 = fill(A(NaN), npoints)
    sea_ice_model.initialized = zeros(Bool, npoints)
    sea_ice_model.last_update[] = vars.prognostic.clock.time

    # [2026-11-11] The two blocks below write the concentration field and read the SST field point
    # by point. Staged through the underlying arrays for the same reason as the masks above, written
    # on the host, and pushed back once at the end. On the CPU this is two copies of npoints floats,
    # paid once per run; it is NOT inside any loop.
    sic_host = Array(ℵ.data)
    sst_host = Array(vars.prognostic.ocean.sea_surface_temperature.data)

    if !isempty(sea_ice_model.initial_sic_path) && isfile(sea_ice_model.initial_sic_path)
        raw = reinterpret(Float32, read(sea_ice_model.initial_sic_path))
        length(raw) == 4 * npoints || error("initial_sic.bin has $(length(raw)) values, expected $(4 * npoints)")
        hist3 = raw[1:npoints]; hist2 = raw[npoints+1:2npoints]
        hist1 = raw[2npoints+1:3npoints]; current = raw[3npoints+1:4npoints]
        for ij in 1:npoints
            if is_arctic[ij]
                sic_host[ij] = clamp(current[ij], 0f0, 1f0)
                sea_ice_model.sic_hist3[ij] = clamp(hist3[ij], 0f0, 1f0)
                sea_ice_model.sic_hist2[ij] = clamp(hist2[ij], 0f0, 1f0)
                sea_ice_model.sic_hist1[ij] = clamp(hist1[ij], 0f0, 1f0)
                sea_ice_model.tos_prev_mean[ij] = isfinite(sst_host[ij]) ? Float64(sst_host[ij]) : sea_ice_model.temp_freeze
                sea_ice_model.initialized[ij] = true
            end
        end
    end

    if sea_ice_model.enable_south && !isempty(sea_ice_model.initial_sic_path_south) && isfile(sea_ice_model.initial_sic_path_south)
        raw_s = reinterpret(Float32, read(sea_ice_model.initial_sic_path_south))
        length(raw_s) == 4 * npoints || error("initial_sic_south.bin has $(length(raw_s)) values, expected $(4 * npoints)")
        hist3_s = raw_s[1:npoints]; hist2_s = raw_s[npoints+1:2npoints]
        hist1_s = raw_s[2npoints+1:3npoints]; current_s = raw_s[3npoints+1:4npoints]
        for ij in 1:npoints
            if is_antarctic[ij]
                sic_host[ij] = clamp(current_s[ij], 0f0, 1f0)
                sea_ice_model.sic_hist3[ij] = clamp(hist3_s[ij], 0f0, 1f0)
                sea_ice_model.sic_hist2[ij] = clamp(hist2_s[ij], 0f0, 1f0)
                sea_ice_model.sic_hist1[ij] = clamp(hist1_s[ij], 0f0, 1f0)
                sea_ice_model.tos_prev_mean[ij] = isfinite(sst_host[ij]) ? Float64(sst_host[ij]) : sea_ice_model.temp_freeze
                sea_ice_model.initialized[ij] = true
            end
        end
    end

    # push the staged concentration back, once, after BOTH blocks have had their say. They touch
    # disjoint points (arctic vs antarctic) so the order between them does not matter, but the
    # field itself must end up holding what the host copy says.
    copyto!(ℵ.data, sic_host)

    # LAST: the device buffers copy the component's persistent state onto the device, and that
    # state is only final here -- the two initial_sic blocks above fill the history, the previous
    # tos and the initialised flags. Built any earlier (it was, briefly) the history arrays are
    # still length zero and the first kernel launch writes out of bounds.
    # The sequential sweep is refused for EVERY model, not only a parallel one: it builds the cross
    # columns from the field it is writing, so the component would be fed an operator no weight set
    # was fitted against -- measured at 32/75/90% sweep noise per column.
    sea_ice_model.cross_from_snapshot ||
        error("MLSeaIceV4: cross_from_snapshot = false is a sequential sweep that reads the " *
              "sea-ice field it is writing, feeding the networks cross columns they were never " *
              "fitted against. It is not supported.")
    if sea_ice_model.kernel_sweep
        sea_ice_model.flat_forward ||
            error("MLSeaIceV4: kernel_sweep needs flat_forward = true -- the kernels pass each " *
                  "band's weights as a single struct argument, which the clarity-first layout " *
                  "cannot do inside Metal's ~29-argument limit.")
        sea_ice_model.cross_snapshot_report == 0 ||
            error("MLSeaIceV4: cross_snapshot_report compares the live field against the snapshot, " *
                  "which the kernel sweep never forms. Run the report on the serial path.")
        # [2026-11-11] Both diffusion branches fill a weight and a filled field point by point and
        # then transform them. There is no device form of that here, and on a GPU model the loops
        # would be a scalar index of a device array, so say it at initialisation rather than let a
        # run reach the first 5-day update and fail there. Neither option is on in any adopted run.
        (!sea_ice_model.apply_spectral_diffusion && !sea_ice_model.apply_spectral_diffusion_south) ||
            error("MLSeaIceV4: apply_spectral_diffusion is not available on the kernel path -- it " *
                  "builds and indexes fields point by point. Run it on the serial path (CPU).")
        field_arch = model.spectral_grid.architecture
        arch = sea_ice_model.kernel_architecture === nothing ?
            field_arch : sea_ice_model.kernel_architecture
        sea_ice_model.kernel_buffers =
            ml_seaice_kernel_buffers(sea_ice_model, arch, field_arch, npoints)
        @info "MLSeaIceV4: kernel_sweep on ($(typeof(arch)), $(npoints) points, " *
              "accum_precision = $(sea_ice_model.accum_precision), fields " *
              "$(sea_ice_model.kernel_buffers.resident ? "resident (no transfer)" : "staged via the host"))"
    elseif model.spectral_grid.architecture isa Architectures.GPU
        error("MLSeaIceV4: the model is on $(typeof(model.spectral_grid.architecture)) but " *
              "kernel_sweep is off. Both the 5-day sweep and the per-timestep accumulation index " *
              "the model's fields element by element, which on device arrays is scalar indexing; " *
              "CUDA.jl refuses it and nothing would run. Leave kernel_sweep unset and it is turned " *
              "on for you, or pass kernel_architecture = CPU() to keep the component on the host.")
    end

    return nothing
end

# [2026-11-11] where the component's time actually goes, split between the per-timestep
# accumulation (every step, all ocean points) and the 5-day update (the forward passes). Plain
# accumulators, never read by the physics, so they cannot change a result. ML_TIME[3] counts the
# steps. Printed by the run script, which divides by its own chunk wall time.
const ML_TIME = zeros(Float64, 3)

function timestep!(vars::Variables, sea_ice_model::MLSeaIceV4, model::PrimitiveEquation)
    _t_entry = time_ns()
    ℵ = vars.prognostic.ocean.sea_ice_concentration
    haskey(vars.prognostic.ocean, :sea_surface_temperature) || return nothing
    sst = vars.prognostic.ocean.sea_surface_temperature

    rsds = vars.parameterizations.surface_shortwave_down
    Rld = vars.parameterizations.surface_longwave_down
    # NOTE: no Rlu (upward longwave) read here at all -- this is the entire point of V3. V2 reads
    # Rlu (SpeedyWeather's own crude SST-only gray-body approximation) to build net longwave; V3
    # uses downward-only longwave and has no dependence on that approximation whatsoever.
    wind10 = vars.parameterizations.surface_wind_speed
    tas = vars.parameterizations.surface_air_temperature
    pₛ_field = vars.parameterizations.surface_pressure

    npoints = length(ℵ)
    (; land_fraction) = model.land_sea_mask
    (; accum_rsds, accum_lw, accum_wind, accum_tas, accum_tos, accum_tdps, is_arctic, is_antarctic) = sea_ice_model
    tos_floor = Float32(sea_ice_model.tos_floor)

    # tdps: dewpoint diagnostic from lowest-model-level specific humidity, evaluated at surface
    # pressure (see ml_seaice_v2.jl's dewpoint_from_humidity docstring for the approximation this
    # makes -- reused directly here, not redefined).
    (; nlayers) = model.geometry
    have_humidity = haskey(vars.grid, :humidity)
    A_ = accumulator_type(sea_ice_model)

    # [2026-11-11] The loop below reads seven of the model's fields and writes one, one element at
    # a time. That is fine on the host and impossible on a device, so when the fields live where
    # the kernels run it is replaced by a launch over the same body. This is the path that matters
    # for following a parallel model: it runs every timestep, where the sweep runs every 5 days.
    B_acc = sea_ice_model.kernel_buffers
    accum_on_device = sea_ice_model.kernel_sweep && B_acc !== nothing && B_acc.resident
    if accum_on_device
        # a dry model still has to hand the kernel SOMETHING of the right shape: the humidity read
        # is dead code behind `have_humidity`, but it is 2-D indexing, so a 1-D stand-in would be a
        # compile-time hazard on a device rather than a harmless unused argument
        humid_field = have_humidity ?
            get_prognostic_step(vars.grid.humidity, model.time_stepping, sea_ice_model) :
            vars.grid.temperature
        ml_seaice_accumulate_kernel!(
            sea_ice_model, B_acc, sst, rsds, Rld, wind10, tas, land_fraction,
            humid_field, pₛ_field, nlayers, have_humidity,
        )
    end

    @inbounds for ij in (accum_on_device ? (1:0) : (1:npoints))
        if land_fraction[ij] < 1 && isfinite(sst[ij])
            if (is_arctic[ij] || is_antarctic[ij]) && sst[ij] < tos_floor
                sst[ij] = tos_floor
            end
            accum_rsds[ij] += A_(rsds[ij])
            accum_lw[ij] += A_(Rld[ij])
            accum_wind[ij] += A_(wind10[ij])
            accum_tas[ij] += A_(tas[ij])
            accum_tos[ij] += A_(sst[ij])
            if have_humidity
                q_ij = Float32(get_prognostic_step(vars.grid.humidity, model.time_stepping, sea_ice_model)[ij, nlayers])
                td = dewpoint_from_humidity(max(q_ij, 0f0), Float32(pₛ_field[ij]))
                accum_tdps[ij] += A_(td)
            end
        end
    end
    sea_ice_model.accum_count += 1
    ML_TIME[1] += (time_ns() - _t_entry) * 1e-9
    ML_TIME[3] += 1

    now = vars.prognostic.clock.time
    (now - sea_ice_model.last_update[]) < sea_ice_model.update_every && return nothing
    _t_update = time_ns()
    sea_ice_model.last_update[] = now

    n = A_(sea_ice_model.accum_count)
    w_mid = sea_ice_model.weights_mid
    w_high = sea_ice_model.weights_highsic
    w_low = sea_ice_model.weights_lowsic
    gate_center = sea_ice_model.highsic_gate_center
    gate_width = sea_ice_model.highsic_gate_width
    low_gate_center = sea_ice_model.lowsic_gate_center
    low_gate_width = sea_ice_model.lowsic_gate_width
    low_gate_center_south = sea_ice_model.lowsic_gate_center_south
    low_gate_width_south = sea_ice_model.lowsic_gate_width_south
    (; east_idx, west_idx, north_idx, south_idx, east2_idx, west2_idx, north2_idx, south2_idx,
       cross_scale_ns, cross_scale_we) = sea_ice_model
    curv_k_res = Float32(sea_ice_model.curvature_k_resolved)
    (; coslat2) = sea_ice_model
    # how many cross inputs the loaded weights expect; drives whether the 4th-difference column is built
    tos_ceil_on = sea_ice_model.tos_ceiling_from_training > 0
    n_cross = length(sea_ice_model.weights_mid.cross_mean)
    curv_tanh_scale = Float32(sea_ice_model.curvature_tanh_scale)
    curv_tanh_on = curv_tanh_scale > 0
    curv_lap_on = sea_ice_model.curvature_laplacian
    cg_thresh = sea_ice_model.changegate_threshold
    cg_a = sea_ice_model.changegate_calib_a
    cg_b = sea_ice_model.changegate_calib_b
    cg_on = cg_thresh > 0f0
    # Platt scaling of the low band's change probability, then the threshold. Clamped away from the
    # endpoints because the logit of an exactly saturated softmax is not finite.
    @inline function change_gate_open(p::Float32)
        pc = clamp(p, 1f-6, 1f0 - 1f-6)
        z = log(pc / (1f0 - pc))
        return 1f0 / (1f0 + exp(-(cg_a * z + cg_b))) >= cg_thresh
    end
    n_cross_s = n_cross
    (; tos_prev_mean, sic_hist1, sic_hist2, sic_hist3, initialized) = sea_ice_model
    flat_on = sea_ice_model.flat_forward
    fw_mid = sea_ice_model.flat_mid; fw_high = sea_ice_model.flat_highsic; fw_low = sea_ice_model.flat_lowsic
    fx_mid = sea_ice_model.flat_x_mid; fx_high = sea_ice_model.flat_x_high; fx_low = sea_ice_model.flat_x_low
    fh1 = sea_ice_model.flat_h1; fh2 = sea_ice_model.flat_h2; fmb = sea_ice_model.flat_mb
    # one slot while the sweep is serial; a kernel would pass ij here
    flat_many = flat_on && size(fh1, 2) > 1
    Δt_days = Float32(Dates.value(Second(sea_ice_model.update_every)) / 86400)

    # The sweep below writes ℵ[ij] while reading ℵ[nidx] for the neighbours, so without a
    # snapshot the cross columns see a mix of updated and not-yet-updated points (see
    # cross_from_snapshot). Taken once per update, so the cost is one copy every 5 days.
    snap_on = sea_ice_model.cross_from_snapshot || sea_ice_model.cross_snapshot_report > 0
    sic_snap = sea_ice_model.sic_snapshot
    # [2026-11-11] Skipped on the kernel path. seaice_prep_kernel! takes the snapshot itself, on the
    # device, so this host copy is redundant there -- and it is a point-by-point read of the sea-ice
    # field, which CUDA.jl refuses on a GPU model. Nothing below reads sic_snap when it is skipped:
    # the serial loop is an empty range and the report block needs cross_snapshot_report > 0, which
    # kernel_sweep rejects outright (see the preconditions in initialize!).
    if snap_on && !sea_ice_model.kernel_sweep
        @inbounds for ij in 1:npoints
            sic_snap[ij] = Float32(ℵ[ij])
        end
    end
    # when the option is off the field read for neighbours stays the live array, bit-for-bit
    cross_snap = sea_ice_model.cross_from_snapshot
    # report accumulators: sum of squares of (live - snapshot) and of the snapshot column itself,
    # per cross column, north then south
    rep_on = sea_ice_model.cross_snapshot_report > 0
    rep_d = zeros(Float64, 3, 2)
    rep_c = zeros(Float64, 3, 2)
    rep_n = zeros(Int, 2)

    # The kernel sweep replaces the loop below wholesale. Written as an empty range rather than an
    # if/else around 290 lines so the serial path stays textually untouched and keeps reproducing
    # earlier runs bit-for-bit.
    kernel_done = false
    if sea_ice_model.kernel_sweep
        ml_seaice_sweep_kernels!(
            sea_ice_model, sea_ice_model.kernel_buffers, ℵ, sst, land_fraction, n, have_humidity,
            Δt_days * 86400,
        )
        kernel_done = true
    end

    for ij in (kernel_done ? (1:0) : (1:npoints))
        if is_arctic[ij] && isfinite(sst[ij])
            sic_now = Float32(ℵ[ij])
            rsds_mean = Float32(accum_rsds[ij] / n)
            lw_mean = Float32(accum_lw[ij] / n)
            wind_mean = Float32(accum_wind[ij] / n)
            tas_mean = Float32(accum_tas[ij] / n)
            tdps_mean = have_humidity ? Float32(accum_tdps[ij] / n) : 0f0
            tos_mean_this_cycle = Float32(accum_tos[ij] / n)

            if !initialized[ij]
                sic_hist1[ij] = sic_now; sic_hist2[ij] = sic_now; sic_hist3[ij] = sic_now
                tos_prev_mean[ij] = tos_mean_this_cycle
                initialized[ij] = true
            end

            @inline function neighbor_diff(nidx::Int)
                (nidx == 0 || land_fraction[nidx] >= 1) && return 0f0
                return cross_snap ? (sic_snap[nidx] - sic_now) : Float32(ℵ[nidx] - sic_now)
            end
            # gradient-basis cross: [(N-S)/2, (W-E)/2] -- a clean centered-difference gradient per
            # axis, matching build_cache_v2.py's gradient_basis_cross option / cross_diff_2d_gradbasis
            # in the offline diagnostic scripts. north_idx/south_idx/west_idx/east_idx here are the
            # SAME index arrays V3 uses for its raw N/S/W/E cross4 -- only the combination differs.
            n_d = neighbor_diff(north_idx[ij]); s_d = neighbor_diff(south_idx[ij])
            w_d = neighbor_diff(west_idx[ij]); e_d = neighbor_diff(east_idx[ij])
            cross2 = (cross_scale_ns * (n_d - s_d) / 2f0, cross_scale_we * (w_d - e_d) / 2f0)
            # [2026-09-29] Optional THIRD cross column: the normalized 4th zonal difference
            #     q = (f_W2 - 4 f_W + 6 f_C - 4 f_E + f_E2) / 16
            # Appended only when the loaded weights actually expect three cross inputs, so two-column
            # checkpoints are completely unaffected (the call sites already slice cross to
            # length(w.cross_mean)). It exists because (W-E)/2 ANNIHILATES the 2-cell zonal mode -- a
            # checkerboard has W == E at every point -- leaving the model structurally blind to the
            # artifact that shows up as radial spokes at T127. NOT rescaled at any resolution: this
            # operator is the identity on the 2-cell mode, so the quantity is already
            # resolution-invariant, and |q| <= 0.5 holds for any SIC field because the coefficients
            # have absolute sum 1. A land or missing neighbour contributes the centre value, so it
            # drops out of the stencil exactly as it does in the training cache.
            if n_cross >= 3 && curv_tanh_on
                # zonal curvature (W+E)/2 - C, bounded. neighbor_diff already returns 0 for a land or
                # missing neighbour, which is exactly what build_cache_v2.py's diff_or_zero does, so
                # the stencil degrades at a coast the same way it did in training.
                curv = (w_d + e_d) / 2f0
                if curv_lap_on
                    curv += coslat2[ij] * (n_d + s_d) / 2f0
                end
                if curv_k_res != 1f0
                    # the same operator on a 2-cell stencil, normalised by 4 so it matches on
                    # resolved structure and is identically zero on the 2-cell mode
                    w2_d = neighbor_diff(west2_idx[ij]); e2_d = neighbor_diff(east2_idx[ij])
                    curv2 = (w2_d + e2_d) / 2f0
                    if curv_lap_on
                        n2_d = neighbor_diff(north2_idx[ij]); s2_d = neighbor_diff(south2_idx[ij])
                        curv2 += coslat2[ij] * (n2_d + s2_d) / 2f0
                    end
                    curv2 /= 4f0
                    curv = curv_k_res * curv2 + (curv - curv2)
                end
                cross2 = (cross2[1], cross2[2], tanh(curv / curv_tanh_scale))
            elseif n_cross >= 3
                @inline function neighbor_val(nidx::Int)
                    (nidx == 0 || land_fraction[nidx] >= 1) && return sic_now
                    return cross_snap ? sic_snap[nidx] : Float32(ℵ[nidx])
                end
                q4 = (neighbor_val(west2_idx[ij]) - 4f0 * neighbor_val(west_idx[ij]) + 6f0 * sic_now
                      - 4f0 * neighbor_val(east_idx[ij]) + neighbor_val(east2_idx[ij])) / 16f0
                cross2 = (cross2[1], cross2[2], q4)
            end

            if rep_on
                @inline function nd_live(nidx::Int)
                    (nidx == 0 || land_fraction[nidx] >= 1) && return 0f0
                    return Float32(ℵ[nidx] - sic_now)
                end
                @inline function nd_snap(nidx::Int)
                    (nidx == 0 || land_fraction[nidx] >= 1) && return 0f0
                    return sic_snap[nidx] - sic_now
                end
                nl = nd_live(north_idx[ij]); sl = nd_live(south_idx[ij])
                wl = nd_live(west_idx[ij]);  el = nd_live(east_idx[ij])
                ns = nd_snap(north_idx[ij]); ss = nd_snap(south_idx[ij])
                ws = nd_snap(west_idx[ij]);  es = nd_snap(east_idx[ij])
                cvl = (wl + el) / 2f0; cvs = (ws + es) / 2f0
                if curv_lap_on
                    cvl += coslat2[ij] * (nl + sl) / 2f0
                    cvs += coslat2[ij] * (ns + ss) / 2f0
                end
                if curv_tanh_on
                    cvl = tanh(cvl / curv_tanh_scale); cvs = tanh(cvs / curv_tanh_scale)
                end
                live = (cross_scale_ns * (nl - sl) / 2f0, cross_scale_we * (wl - el) / 2f0, cvl)
                snap = (cross_scale_ns * (ns - ss) / 2f0, cross_scale_we * (ws - es) / 2f0, cvs)
                for c in 1:3
                    rep_d[c, 1] += (live[c] - snap[c])^2
                    rep_c[c, 1] += snap[c]^2
                end
                rep_n[1] += 1
            end

            hist3 = (Float32(sic_hist3[ij]), Float32(sic_hist2[ij]), Float32(sic_hist1[ij]))

            tos_in = Float32(tos_prev_mean[ij])
            if tos_ceil_on
                tos_in = min(tos_in, tos_ceiling(sic_now, true))
            end
            c1 = cross2[1]; c2 = cross2[2]
            c3 = length(cross2) >= 3 ? cross2[3] : 0f0
            local tendency_mid::Float32, tendency_high::Float32, tendency_low::Float32, p_change::Float32
            if flat_on
                slot = flat_many ? ij : 1
                ml_normalize_flat_v4!(fx_mid, slot, fw_mid, sic_now, tos_in, rsds_mean, lw_mean, wind_mean, tas_mean, tdps_mean, c1, c2, c3)
                tendency_mid = ml_forward_flat_v4(fw_mid, fx_mid, fh1, fh2, fmb, slot, sic_now, hist3)
                ml_normalize_flat_v4!(fx_high, slot, fw_high, sic_now, tos_in, rsds_mean, lw_mean, wind_mean, tas_mean, tdps_mean, c1, c2, c3)
                tendency_high = ml_forward_flat_v4(fw_high, fx_high, fh1, fh2, fmb, slot, sic_now, hist3)
                ml_normalize_flat_v4!(fx_low, slot, fw_low, sic_now, tos_in, rsds_mean, lw_mean, wind_mean, tas_mean, tdps_mean, c1, c2, c3)
                tendency_low, p_change = ml_forward_flat_changegate_v4(fw_low, fx_low, fh1, fh2, fmb, slot, sic_now, hist3)
            else
                x_mid = ml_normalize_features_v4(w_mid, sic_now, tos_in, rsds_mean, lw_mean, wind_mean, tas_mean, tdps_mean, cross2[1:length(w_mid.cross_mean)])
                tendency_mid = ml_forward_v4(w_mid, x_mid, sic_now, hist3)

                x_high = ml_normalize_features_v4(w_high, sic_now, tos_in, rsds_mean, lw_mean, wind_mean, tas_mean, tdps_mean, cross2[1:length(w_high.cross_mean)])
                tendency_high = ml_forward_v4(w_high, x_high, sic_now, hist3)

                x_low = ml_normalize_features_v4(w_low, sic_now, tos_in, rsds_mean, lw_mean, wind_mean, tas_mean, tdps_mean, cross2[1:length(w_low.cross_mean)])
                tendency_low, p_change = ml_forward_changegate_with_prob(w_low, x_low, sic_now, hist3)
            end

            gate_high = 1f0 / (1f0 + exp(-(sic_now - gate_center) / gate_width))
            gate_low = 1f0 / (1f0 + exp(-(low_gate_center - sic_now) / low_gate_width))
            gate_mid = 1f0 - gate_low - gate_high
            tendency = gate_low * tendency_low + gate_mid * tendency_mid + gate_high * tendency_high
            if cg_on && gate_low > 0.5f0 && !change_gate_open(p_change)
                tendency = 0f0
            end

            sic_next = clamp(sic_now + tendency, 0f0, 1f0)

            sic_hist3[ij] = sic_hist2[ij]; sic_hist2[ij] = sic_hist1[ij]; sic_hist1[ij] = sic_now
            tos_prev_mean[ij] = tos_mean_this_cycle
            ℵ[ij] = sic_next
        elseif is_antarctic[ij] && isfinite(sst[ij])
            sic_now = Float32(ℵ[ij])
            rsds_mean = Float32(accum_rsds[ij] / n)
            lw_mean = Float32(accum_lw[ij] / n)
            wind_mean = Float32(accum_wind[ij] / n)
            tas_mean = Float32(accum_tas[ij] / n)
            tdps_mean = have_humidity ? Float32(accum_tdps[ij] / n) : 0f0
            tos_mean_this_cycle = Float32(accum_tos[ij] / n)

            if !initialized[ij]
                sic_hist1[ij] = sic_now; sic_hist2[ij] = sic_now; sic_hist3[ij] = sic_now
                tos_prev_mean[ij] = tos_mean_this_cycle
                initialized[ij] = true
            end

            @inline function neighbor_diff_s(nidx::Int)
                (nidx == 0 || land_fraction[nidx] >= 1) && return 0f0
                return cross_snap ? (sic_snap[nidx] - sic_now) : Float32(ℵ[nidx] - sic_now)
            end
            n_d_s = neighbor_diff_s(north_idx[ij]); s_d_s = neighbor_diff_s(south_idx[ij])
            w_d_s = neighbor_diff_s(west_idx[ij]); e_d_s = neighbor_diff_s(east_idx[ij])
            cross2_s = (cross_scale_ns * (n_d_s - s_d_s) / 2f0, cross_scale_we * (w_d_s - e_d_s) / 2f0)
            # [2026-10-15 FIX] this block used to build q4_s UNCONDITIONALLY, with no curv_tanh_on
            # test, while the Arctic branch above has built the bounded curvature since the column
            # was adopted on 2026-10-01. Every southern online run from then until now therefore fed
            # the mid and low bands a column ANTI-CORRELATED with the one they were trained on
            # (measured on 1990 Antarctic ERA5 at 2 degrees: r = -0.9241) and four times too small
            # (std 0.0369 against 0.1506), then normalised it with the curvature column's own mean
            # and standard deviation. It is the same class of error as the 2026-09-29 slot bugs: a
            # three-column form silently standing in for a different three-column form. The branch
            # now mirrors the north exactly, q4 included as the fallback.
            if n_cross_s >= 3 && curv_tanh_on
                curv_s = (w_d_s + e_d_s) / 2f0
                if curv_lap_on
                    curv_s += coslat2[ij] * (n_d_s + s_d_s) / 2f0
                end
                cross2_s = (cross2_s[1], cross2_s[2], tanh(curv_s / curv_tanh_scale))
            elseif n_cross_s >= 3
                @inline function neighbor_val_s(nidx::Int)
                    (nidx == 0 || land_fraction[nidx] >= 1) && return sic_now
                    return cross_snap ? sic_snap[nidx] : Float32(ℵ[nidx])
                end
                q4_s = (neighbor_val_s(west2_idx[ij]) - 4f0 * neighbor_val_s(west_idx[ij]) + 6f0 * sic_now
                        - 4f0 * neighbor_val_s(east_idx[ij]) + neighbor_val_s(east2_idx[ij])) / 16f0
                cross2_s = (cross2_s[1], cross2_s[2], q4_s)
            end

            hist3 = (Float32(sic_hist3[ij]), Float32(sic_hist2[ij]), Float32(sic_hist1[ij]))

            # SAME unified weights as the Arctic branch above -- this is the point of "unified"
            tos_in_s = Float32(tos_prev_mean[ij])
            if tos_ceil_on
                tos_in_s = min(tos_in_s, tos_ceiling(sic_now, false))
            end
            c1s = cross2_s[1]; c2s = cross2_s[2]
            c3s = length(cross2_s) >= 3 ? cross2_s[3] : 0f0
            local tendency_mid_s::Float32, tendency_low_s::Float32, p_change_s::Float32
            if flat_on
                slot = flat_many ? ij : 1
                ml_normalize_flat_v4!(fx_mid, slot, fw_mid, sic_now, tos_in_s, rsds_mean, lw_mean, wind_mean, tas_mean, tdps_mean, c1s, c2s, c3s)
                tendency_mid_s = ml_forward_flat_v4(fw_mid, fx_mid, fh1, fh2, fmb, slot, sic_now, hist3)
                ml_normalize_flat_v4!(fx_low, slot, fw_low, sic_now, tos_in_s, rsds_mean, lw_mean, wind_mean, tas_mean, tdps_mean, c1s, c2s, c3s)
                tendency_low_s, p_change_s = ml_forward_flat_changegate_v4(fw_low, fx_low, fh1, fh2, fmb, slot, sic_now, hist3)
            else
                x_mid_s = ml_normalize_features_v4(w_mid, sic_now, tos_in_s, rsds_mean, lw_mean, wind_mean, tas_mean, tdps_mean, cross2_s[1:length(w_mid.cross_mean)])
                tendency_mid_s = ml_forward_v4(w_mid, x_mid_s, sic_now, hist3)

                x_low_s = ml_normalize_features_v4(w_low, sic_now, tos_in_s, rsds_mean, lw_mean, wind_mean, tas_mean, tdps_mean, cross2_s[1:length(w_low.cross_mean)])
                tendency_low_s, p_change_s = ml_forward_changegate_with_prob(w_low, x_low_s, sic_now, hist3)
            end

            gate_low_s = 1f0 / (1f0 + exp(-(low_gate_center_south - sic_now) / low_gate_width_south))
            gate_mid_s = 1f0 - gate_low_s
            tendency_s = gate_low_s * tendency_low_s + gate_mid_s * tendency_mid_s
            if cg_on && gate_low_s > 0.5f0 && !change_gate_open(p_change_s)
                tendency_s = 0f0
            end

            sic_next_s = clamp(sic_now + tendency_s, 0f0, 1f0)

            sic_hist3[ij] = sic_hist2[ij]; sic_hist2[ij] = sic_hist1[ij]; sic_hist1[ij] = sic_now
            tos_prev_mean[ij] = tos_mean_this_cycle
            ℵ[ij] = sic_next_s
        elseif land_fraction[ij] < 1 && isfinite(sst[ij])
            sst_mean = Float32(accum_tos[ij] / n)
            dT = sst_mean - sea_ice_model.temp_freeze
            F = -sea_ice_model.melt_rate * max(dT, 0) - sea_ice_model.freeze_rate / (Δt_days * 86400) * min(dT, 0)
            sst[ij] = max(sst[ij], sea_ice_model.temp_freeze)
            ℵ[ij] = clamp(ℵ[ij] + Δt_days * 86400 * F, 0, 1)
        end
    end

    if sea_ice_model.apply_spectral_diffusion
        S = model.spectral_transform
        radius = model.planet.radius
        D = Float32(sea_ice_model.diffusion_coefficient)
        Δt_seconds = Float32(Dates.value(Second(sea_ice_model.update_every)))

        weight_field = copy(ℵ)
        filled_field = copy(ℵ)
        for ij in 1:npoints
            wij = 1f0 - Float32(land_fraction[ij])
            weight_field[ij] = wij
            filled_field[ij] = ℵ[ij] * wij
        end

        coeffs_filled = transform(filled_field, S)
        coeffs_weight = transform(weight_field, S)
        (; l_indices) = coeffs_filled.spectrum
        for lm in eachindex(coeffs_filled)
            l = l_indices[lm]
            decay = exp(-D * l * (l + 1) / radius^2 * Δt_seconds)
            coeffs_filled[lm] *= decay
            coeffs_weight[lm] *= decay
        end
        filled_diffused = transform(coeffs_filled, S)
        weight_diffused = transform(coeffs_weight, S)

        min_weight = Float32(sea_ice_model.diffusion_min_weight)
        for ij in 1:npoints
            if is_arctic[ij]
                wd = weight_diffused[ij]
                if wd > min_weight
                    ℵ[ij] = clamp(filled_diffused[ij] / wd, 0f0, 1f0)
                end
            end
        end
    end

    if sea_ice_model.enable_south && sea_ice_model.apply_spectral_diffusion_south
        S = model.spectral_transform
        radius = model.planet.radius
        D_s = Float32(sea_ice_model.diffusion_coefficient_south)
        Δt_seconds = Float32(Dates.value(Second(sea_ice_model.update_every)))

        weight_field_s = copy(ℵ)
        filled_field_s = copy(ℵ)
        for ij in 1:npoints
            wij = 1f0 - Float32(land_fraction[ij])
            weight_field_s[ij] = wij
            filled_field_s[ij] = ℵ[ij] * wij
        end

        coeffs_filled_s = transform(filled_field_s, S)
        coeffs_weight_s = transform(weight_field_s, S)
        (; l_indices) = coeffs_filled_s.spectrum
        for lm in eachindex(coeffs_filled_s)
            l = l_indices[lm]
            decay_s = exp(-D_s * l * (l + 1) / radius^2 * Δt_seconds)
            coeffs_filled_s[lm] *= decay_s
            coeffs_weight_s[lm] *= decay_s
        end
        filled_diffused_s = transform(coeffs_filled_s, S)
        weight_diffused_s = transform(coeffs_weight_s, S)

        min_weight_s = Float32(sea_ice_model.diffusion_min_weight_south)
        for ij in 1:npoints
            if is_antarctic[ij]
                wd_s = weight_diffused_s[ij]
                if wd_s > min_weight_s
                    ℵ[ij] = clamp(filled_diffused_s[ij] / wd_s, 0f0, 1f0)
                end
            end
        end
    end

    if rep_on && rep_n[1] > 0
        np_ = rep_n[1]
        @info "MLSeaIceV4 cross-column sequential-sweep contamination, Arctic, $(np_) ice points, " *
              "update $(now): " *
              join(["col$(c) RMS(live-snap)=$(round(sqrt(rep_d[c, 1] / np_), sigdigits=4)) " *
                    "RMS(col)=$(round(sqrt(rep_c[c, 1] / np_), sigdigits=4)) " *
                    "ratio=$(round(sqrt(rep_d[c, 1] / max(rep_c[c, 1], 1e-30)), sigdigits=4))"
                    for c in 1:3], " | ")
        sea_ice_model.cross_snapshot_report -= 1
    end

    fill!(accum_rsds, 0.0); fill!(accum_lw, 0.0); fill!(accum_wind, 0.0)
    fill!(accum_tas, 0.0); fill!(accum_tos, 0.0); fill!(accum_tdps, 0.0)
    # and the device copies, when they are the ones being accumulated into
    if sea_ice_model.kernel_sweep && sea_ice_model.kernel_buffers !== nothing &&
            sea_ice_model.kernel_buffers.resident
        ml_seaice_reset_device_accumulators!(sea_ice_model.kernel_buffers)
    end
    sea_ice_model.accum_count = 0
    ML_TIME[2] += (time_ns() - _t_update) * 1e-9

    return nothing
end
