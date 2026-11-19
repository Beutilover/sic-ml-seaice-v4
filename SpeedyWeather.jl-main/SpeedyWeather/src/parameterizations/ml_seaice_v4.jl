"""
Machine-learned sea-ice concentration parameterisation for SpeedyWeather.jl.

The component updates sea-ice concentration every five days using atmospheric
and slab-ocean forcing accumulated over the update interval. Separate neural
models represent low, intermediate and high SIC regimes, combined through
smooth concentration-dependent gates. The high-SIC branch is Arctic-specific.

Inputs include radiation, surface meteorology, ocean temperature, recent SIC
history, neighbouring SIC gradients and a bounded curvature/Laplacian feature.
Feature conventions, normalisation and curvature settings must match those
stored with the trained weight packages.

The component supports both hemispheres, optional spectral diffusion, bounded
forcing inputs, calibrated change probabilities, and CPU or GPU inference.
Neighbour-derived features use a pre-update SIC snapshot so that updates are
independent of traversal order. GPU inference uses flat weights and preallocated
scratch buffers, with the atmospheric model permitted to remain on the CPU.

The default weight directories and gate parameters specify the current
configuration. Other weight sets may require corresponding feature and gate
settings; package declarations are checked when available.
"""

include("ml_seaice_model_v4.jl")
using .MLSeaIceModelV4: MLWeightsV4, MLWeightsChangeGateV4
using .MLSeaIceModelV4: load_ml_weights_v4, load_ml_weights_changegate_v4
using .MLSeaIceModelV4: ml_forward_v4, ml_normalize_features_v4, ml_forward_changegate_with_prob,
    flatten_weights_v4, ml_forward_flat_v4, ml_forward_flat_changegate_v4, ml_normalize_flat_v4!,
    MLWeightsFlatV4CPU, MLWeightsFlatChangeGateV4CPU

export MLSeaIceV4

@kwdef mutable struct MLSeaIceV4 <: AbstractSeaIce
    "Northern latitude threshold (degrees north) for the neural sea-ice branch."
    lat_limit::Float64 = 60.0

    "Interval between neural sea-ice updates; matches the training target interval."
    update_every::Second = Day(5)

    "Exported weights for the intermediate-SIC model shared by both hemispheres."
    weights_dir_mid::String = joinpath(@__DIR__, "ml_seaice_weights_v4_wttos_unified_mid")

    "Exported weights for the Arctic high-SIC model, including its ocean-temperature feature declaration."
    weights_dir_highsic::String = joinpath(@__DIR__, "ml_seaice_weights_v4_wttos_north_high")

    "Centre of the sigmoid transition between intermediate and high SIC in the Arctic."
    highsic_gate_center::Float32 = 0.65f0

    "Width of the Arctic intermediate-to-high SIC blending transition."
    highsic_gate_width::Float32 = 0.05f0

    "Exported weights for the low-SIC change-gated model shared by both hemispheres."
    weights_dir_lowsic::String = joinpath(@__DIR__, "ml_seaice_weights_v4_wttos_unified_low")

    "Centre of the Northern low-to-intermediate SIC blending transition."
    lowsic_gate_center::Float32 = 0.21f0

    "Width of the Northern low-to-intermediate SIC blending transition."
    lowsic_gate_width::Float32 = 0.02f0

    "Initial Northern SIC and three previous update states, stored as Float32 binary data."
    initial_sic_path::String = joinpath(@__DIR__, "ml_seaice_weights", "initial_sic.bin")

    "Minimum sea surface temperature (K) imposed at neural sea-ice points."
    tos_floor::Float64 = 273.15 - 1.8

    "Freezing temperature (K) for the thermodynamic fallback."
    temp_freeze::Float64 = 273.15 - 1.8
    melt_rate::Float64 = 1.0e-6
    freeze_rate::Float64 = 0.1

    "Reference angular width (degrees) for SIC gradient features. A positive value rescales differences to this angular spacing; the default matches the training convention."
    cross_ref_deg::Float64 = 2.0

    "Tanh scale for the bounded curvature/Laplacian feature. This setting must match the feature representation used to train the selected weights."
    curvature_tanh_scale::Float64 = 0.5

    "Use the latitude-metric-weighted Laplacian instead of zonal curvature; must match the weight package."
    curvature_laplacian::Bool = true
    "Scale factor applied to resolved curvature while preserving the two-cell grid-scale component."
    curvature_k_resolved::Float32 = 1f0
    "Compute the resolved-curvature factor from the grid-spacing scale factors during initialisation."
    curvature_k_auto::Bool = false

    "Threshold on calibrated low-SIC change probability; below it the concentration remains unchanged."
    changegate_threshold::Float32 = 0.5f0
    "Platt-calibration slope for the low-SIC change probability."
    changegate_calib_a::Float32 = 1.14784f0
    "Platt-calibration intercept for the low-SIC change probability."
    changegate_calib_b::Float32 = -1.33014f0

    "Enable the SIC-dependent upper bound on ocean-temperature input estimated from training data. Only the network input is constrained; the ocean state is unchanged."
    tos_ceiling_from_training::Float64 = 1.0

    "Enable the optional Northern spectral Laplacian diffusion pass."
    apply_spectral_diffusion::Bool = false

    "Northern spectral diffusion coefficient (m^2/s), used only when diffusion is enabled."
    diffusion_coefficient::Float64 = 34.7222222

    "Minimum local ocean weight for Northern diffusion."
    diffusion_min_weight::Float64 = 0.3

    # ======================================================================
    # Southern Hemisphere (Antarctic) options
    # ======================================================================

    "Enable the Southern Hemisphere neural sea-ice branch."
    enable_south::Bool = true

    "Absolute Southern latitude threshold (degrees) for the neural sea-ice branch."
    lat_limit_south::Float64 = 55.0

    "Centre of the Southern low-to-intermediate SIC blending transition."
    lowsic_gate_center_south::Float32 = 0.04f0

    "Width of the Southern low-to-intermediate SIC blending transition."
    lowsic_gate_width_south::Float32 = 0.005f0

    "Enable the optional Southern spectral Laplacian diffusion pass."
    apply_spectral_diffusion_south::Bool = false

    "Southern spectral diffusion coefficient (m^2/s), used only when diffusion is enabled."
    diffusion_coefficient_south::Float64 = 2314.8148148

    "Minimum local ocean weight for Southern diffusion."
    diffusion_min_weight_south::Float64 = 0.3

    "Initial Southern SIC and three previous update states in Float32 binary format."
    initial_sic_path_south::String = joinpath(@__DIR__, "ml_seaice_weights", "initial_sic_south.bin")

    # -- allocated in initialize!(vars, model) once the grid is known --
    weights_mid::Union{Nothing, MLWeightsV4} = nothing
    weights_highsic::Union{Nothing, MLWeightsV4} = nothing
    weights_lowsic::Union{Nothing, MLWeightsChangeGateV4} = nothing

    "Use allocation-free inference with flat weights and caller-owned scratch arrays."
    flat_forward::Bool = false

    # flat weights, built from the loaded ones in initialize! when flat_forward is on
    flat_mid::Union{Nothing, MLWeightsFlatV4CPU} = nothing
    flat_highsic::Union{Nothing, MLWeightsFlatV4CPU} = nothing
    flat_lowsic::Union{Nothing, MLWeightsFlatChangeGateV4CPU} = nothing
    "Number of scratch columns available to concurrent inference workers; zero selects one per grid point."
    flat_nslots::Int = 1

    # scratch: (width, flat_nslots). One column per concurrent worker.
    flat_x_mid::Matrix{Float32} = zeros(Float32, 0, 0)
    flat_x_high::Matrix{Float32} = zeros(Float32, 0, 0)
    flat_x_low::Matrix{Float32} = zeros(Float32, 0, 0)
    flat_h1::Matrix{Float32} = zeros(Float32, 0, 0)
    flat_h2::Matrix{Float32} = zeros(Float32, 0, 0)
    flat_mb::Matrix{Float32} = zeros(Float32, 0, 0)

    "Evaluate the five-day update using KernelAbstractions kernels; requires flat weights and snapshot features."
    kernel_sweep::Bool = false

    "Optional device architecture for the sea-ice kernels, independent of the atmospheric model."
    kernel_architecture::Any = nothing

    "Device buffers allocated during initialisation for kernel-based updates."
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

    "Requested floating-point precision for the update architecture: 32 or 64 bits."
    accum_precision::Int = 64

    "Compute neighbour gradients and curvature from the SIC field before the update, ensuring order-independent evaluation."
    cross_from_snapshot::Bool = true

    "Number of initial updates for which live-versus-snapshot feature differences are reported."
    cross_snapshot_report::Int = 0

    # snapshot buffer, allocated in initialize! only when cross_from_snapshot or the report is on
    sic_snapshot::Vector{Float32} = zeros(Float32, 0)

    last_update::Base.RefValue{DateTime} = Ref(DEFAULT_DATE)
end

"""$(TYPEDSIGNATURES)
Probe whether the architecture supports Float64 arithmetic using an actual device operation. Cache the result by architecture type."""
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
Return the configured floating-point type for accumulation."""
accumulator_type(sea_ice_model::MLSeaIceV4) =
    sea_ice_model.accum_precision == 32 ? Float32 : Float64

"""$(TYPEDSIGNATURES)
Construct a sea-ice component with architecture-compatible precision and inference settings. GPU-hosted models use the kernel path by default."""
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

    # FOLLOW THE MODEL. When the model itself is running on a GPU, the serial sweep
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
    # Numerical settings and array layouts follow the model configuration.
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
Read the curvature-feature specification bundled with a weight package. Return nothing if the declaration is absent."""
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
Check that curvature-feature settings match each weight package declaration, preventing incompatible inputs."""
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
    # `.data` and an explicit Array(): on a GPU model land_fraction is a Field over a
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

    # The accumulator FIELDS are declared Vector{Float64}, so these assignments are
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

    # The two blocks below write the concentration field and read the SST field point
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
        # Both diffusion branches fill a weight and a filled field point by point and
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

# where the component's time actually goes, split between the per-timestep
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

    # The loop below reads seven of the model's fields and writes one, one element at
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
    # Skipped on the kernel path. seaice_prep_kernel! takes the snapshot itself, on the
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
            # Optional THIRD cross column: the normalized 4th zonal difference
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
            # this block used to build q4_s UNCONDITIONALLY, with no curv_tanh_on
            # test, while the Arctic branch above has built the bounded curvature since the column
    # Neighbour features are evaluated consistently with the trained input representation.
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
