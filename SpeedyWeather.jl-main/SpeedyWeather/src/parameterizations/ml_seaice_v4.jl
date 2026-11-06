"""
Gradient-based sea-ice concentration (SIC) parameterisation
for coupled atmospheric model integration.

Implements regime-specific neural SIC tendency prediction using
ChangeGateClassifier and RelaxationMoEClassifier architectures,
with smooth blending between low-, mid-, and high-SIC regimes.

Spatial features
----------------
Local spatial information is represented by centered-difference
cross-neighbor gradients:

    grad_ns = (N - S) / 2
    grad_we = (W - E) / 2

These features provide compact directional information without
requiring separate raw values for all four neighboring cells.

Model inputs
------------
Raw atmospheric and sea-ice inputs:

    [siconc, tos, rsds, rld, sfcwind, tas]

followed by the two spatial gradients and an optional dew-point
temperature (tdps) feature.

The low- and mid-SIC models include tdps, while the high-SIC
model uses the reduced feature set.

Physical-response regularisation
--------------------------------
The exported neural models are trained with physically motivated
loss penalties, including radiative-response monotonicity.

Separate shortwave and longwave response penalties are evaluated
at observed and synthetically perturbed forcing states to improve
robustness to changes in radiation composition.

These are training-time soft constraints, not exact guarantees
on the deployed neural operator.

Regime blending
---------------
The three models are combined through smooth SIC-dependent gates.
Gate parameters are calibrated for the corresponding checkpoint
set and should remain consistent with the exported weights.

Online evaluation
-----------------
The component evaluates SIC tendencies using atmospheric forcing,
local SIC history, and spatial features constructed from the
host model state.

The default configuration uses checkpoints trained with
unsmoothed spatial inputs, consistent with the per-grid-point
features supplied during online integration.

Optional spectral diffusion is available but disabled by default.
Its coefficients are not treated as calibrated parameters for
the default checkpoint configuration.

The implementation supports coupled evaluation in SpeedyWeather
using the model's prescribed update cadence and forcing
conventions.

"""

include("ml_seaice_model_v4.jl")
using .MLSeaIceModelV4: MLWeightsV4, MLWeightsChangeGateV4
using .MLSeaIceModelV4: load_ml_weights_v4, load_ml_weights_changegate_v4
using .MLSeaIceModelV4: ml_forward_v4, ml_normalize_features_v4, ml_forward_changegate_with_prob

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

    last_update::Base.RefValue{DateTime} = Ref(DEFAULT_DATE)
end

MLSeaIceV4(::SpectralGrid; kwargs...) = MLSeaIceV4(; kwargs...)


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
    is_arctic = [(latds[ij] >= sea_ice_model.lat_limit) && (land_fraction[ij] < 1) for ij in 1:npoints]
    is_antarctic = sea_ice_model.enable_south ?
        [(latds[ij] <= -sea_ice_model.lat_limit_south) && (land_fraction[ij] < 1) for ij in 1:npoints] :
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

    sea_ice_model.accum_rsds = zeros(npoints)
    sea_ice_model.accum_lw = zeros(npoints)
    sea_ice_model.accum_wind = zeros(npoints)
    sea_ice_model.accum_tas = zeros(npoints)
    sea_ice_model.accum_tos = zeros(npoints)
    sea_ice_model.accum_tdps = zeros(npoints)
    sea_ice_model.accum_count = 0

    sea_ice_model.tos_prev_mean = fill(NaN, npoints)
    sea_ice_model.sic_hist1 = fill(NaN, npoints)
    sea_ice_model.sic_hist2 = fill(NaN, npoints)
    sea_ice_model.sic_hist3 = fill(NaN, npoints)
    sea_ice_model.initialized = zeros(Bool, npoints)
    sea_ice_model.last_update[] = vars.prognostic.clock.time

    if !isempty(sea_ice_model.initial_sic_path) && isfile(sea_ice_model.initial_sic_path)
        raw = reinterpret(Float32, read(sea_ice_model.initial_sic_path))
        length(raw) == 4 * npoints || error("initial_sic.bin has $(length(raw)) values, expected $(4 * npoints)")
        hist3 = raw[1:npoints]; hist2 = raw[npoints+1:2npoints]
        hist1 = raw[2npoints+1:3npoints]; current = raw[3npoints+1:4npoints]
        sst0 = vars.prognostic.ocean.sea_surface_temperature
        for ij in 1:npoints
            if is_arctic[ij]
                ℵ[ij] = clamp(current[ij], 0f0, 1f0)
                sea_ice_model.sic_hist3[ij] = clamp(hist3[ij], 0f0, 1f0)
                sea_ice_model.sic_hist2[ij] = clamp(hist2[ij], 0f0, 1f0)
                sea_ice_model.sic_hist1[ij] = clamp(hist1[ij], 0f0, 1f0)
                sea_ice_model.tos_prev_mean[ij] = isfinite(sst0[ij]) ? Float64(sst0[ij]) : sea_ice_model.temp_freeze
                sea_ice_model.initialized[ij] = true
            end
        end
    end

    if sea_ice_model.enable_south && !isempty(sea_ice_model.initial_sic_path_south) && isfile(sea_ice_model.initial_sic_path_south)
        raw_s = reinterpret(Float32, read(sea_ice_model.initial_sic_path_south))
        length(raw_s) == 4 * npoints || error("initial_sic_south.bin has $(length(raw_s)) values, expected $(4 * npoints)")
        hist3_s = raw_s[1:npoints]; hist2_s = raw_s[npoints+1:2npoints]
        hist1_s = raw_s[2npoints+1:3npoints]; current_s = raw_s[3npoints+1:4npoints]
        sst0 = vars.prognostic.ocean.sea_surface_temperature
        for ij in 1:npoints
            if is_antarctic[ij]
                ℵ[ij] = clamp(current_s[ij], 0f0, 1f0)
                sea_ice_model.sic_hist3[ij] = clamp(hist3_s[ij], 0f0, 1f0)
                sea_ice_model.sic_hist2[ij] = clamp(hist2_s[ij], 0f0, 1f0)
                sea_ice_model.sic_hist1[ij] = clamp(hist1_s[ij], 0f0, 1f0)
                sea_ice_model.tos_prev_mean[ij] = isfinite(sst0[ij]) ? Float64(sst0[ij]) : sea_ice_model.temp_freeze
                sea_ice_model.initialized[ij] = true
            end
        end
    end

    return nothing
end

function timestep!(vars::Variables, sea_ice_model::MLSeaIceV4, model::PrimitiveEquation)
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

    @inbounds for ij in 1:npoints
        if land_fraction[ij] < 1 && isfinite(sst[ij])
            if (is_arctic[ij] || is_antarctic[ij]) && sst[ij] < tos_floor
                sst[ij] = tos_floor
            end
            accum_rsds[ij] += Float64(rsds[ij])
            accum_lw[ij] += Float64(Rld[ij])
            accum_wind[ij] += Float64(wind10[ij])
            accum_tas[ij] += Float64(tas[ij])
            accum_tos[ij] += Float64(sst[ij])
            if have_humidity
                q_ij = Float32(get_prognostic_step(vars.grid.humidity, model.time_stepping, sea_ice_model)[ij, nlayers])
                td = dewpoint_from_humidity(max(q_ij, 0f0), Float32(pₛ_field[ij]))
                accum_tdps[ij] += Float64(td)
            end
        end
    end
    sea_ice_model.accum_count += 1

    now = vars.prognostic.clock.time
    (now - sea_ice_model.last_update[]) < sea_ice_model.update_every && return nothing
    sea_ice_model.last_update[] = now

    n = Float64(sea_ice_model.accum_count)
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
    Δt_days = Float32(Dates.value(Second(sea_ice_model.update_every)) / 86400)

    for ij in 1:npoints
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
                return Float32(ℵ[nidx] - sic_now)
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
                    return Float32(ℵ[nidx])
                end
                q4 = (neighbor_val(west2_idx[ij]) - 4f0 * neighbor_val(west_idx[ij]) + 6f0 * sic_now
                      - 4f0 * neighbor_val(east_idx[ij]) + neighbor_val(east2_idx[ij])) / 16f0
                cross2 = (cross2[1], cross2[2], q4)
            end

            hist3 = (Float32(sic_hist3[ij]), Float32(sic_hist2[ij]), Float32(sic_hist1[ij]))

            tos_in = Float32(tos_prev_mean[ij])
            if tos_ceil_on
                tos_in = min(tos_in, tos_ceiling(sic_now, true))
            end
            x_mid = ml_normalize_features_v4(w_mid, sic_now, tos_in, rsds_mean, lw_mean, wind_mean, tas_mean, tdps_mean, cross2[1:length(w_mid.cross_mean)])
            tendency_mid = ml_forward_v4(w_mid, x_mid, sic_now, hist3)

            x_high = ml_normalize_features_v4(w_high, sic_now, tos_in, rsds_mean, lw_mean, wind_mean, tas_mean, tdps_mean, cross2[1:length(w_high.cross_mean)])
            tendency_high = ml_forward_v4(w_high, x_high, sic_now, hist3)

            x_low = ml_normalize_features_v4(w_low, sic_now, tos_in, rsds_mean, lw_mean, wind_mean, tas_mean, tdps_mean, cross2[1:length(w_low.cross_mean)])
            tendency_low, p_change = ml_forward_changegate_with_prob(w_low, x_low, sic_now, hist3)

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
                return Float32(ℵ[nidx] - sic_now)
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
                    return Float32(ℵ[nidx])
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
            x_mid_s = ml_normalize_features_v4(w_mid, sic_now, tos_in_s, rsds_mean, lw_mean, wind_mean, tas_mean, tdps_mean, cross2_s[1:length(w_mid.cross_mean)])
            tendency_mid_s = ml_forward_v4(w_mid, x_mid_s, sic_now, hist3)

            x_low_s = ml_normalize_features_v4(w_low, sic_now, tos_in_s, rsds_mean, lw_mean, wind_mean, tas_mean, tdps_mean, cross2_s[1:length(w_low.cross_mean)])
            tendency_low_s, p_change_s = ml_forward_changegate_with_prob(w_low, x_low_s, sic_now, hist3)

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

    fill!(accum_rsds, 0.0); fill!(accum_lw, 0.0); fill!(accum_wind, 0.0)
    fill!(accum_tas, 0.0); fill!(accum_tos, 0.0); fill!(accum_tdps, 0.0)
    sea_ice_model.accum_count = 0

    return nothing
end
