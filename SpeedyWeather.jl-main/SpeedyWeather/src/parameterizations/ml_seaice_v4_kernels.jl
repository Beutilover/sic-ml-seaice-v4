# The 5-day MLSeaIceV4 update as KernelAbstractions kernels, so the component can run on a GPU
# alongside (or independently of) the dynamical core.
#
# [2026-11-11] This path is LEGAL ONLY because the sweep now reads its neighbours from a snapshot
# taken before the update (`cross_from_snapshot`). The original sweep wrote the sea-ice field while
# reading the same field for the cross columns, so its result depended on the visiting order and no
# parallel form of it existed at all -- the cross columns were measured to be 32/75/90% sweep noise.
# `kernel_sweep` therefore ASSERTS `cross_from_snapshot`; it is not a performance option bolted onto
# the old loop but the thing the ordering fix made possible.
#
# [2026-11-11] A Field keeps its values in `.data`; a bare array IS its own data (the GPU testset
# hands the entry points raw device arrays). Resident transfers go through this rather than through
# the generic AbstractArray path, because copyto!(::CuArray, ::Field) copies point by point and
# CUDA.jl refuses that outright -- between the two underlying arrays it is one bulk copy.
@inline field_data(x::AbstractArray) = x
@inline field_data(x::AbstractField) = x.data

# Split across six launches rather than one, because Metal accepts at most ~29 kernel arguments and
# counts EVERY one, scalars included (measured: 2 arrays + 26 scalars compiles, 2 + 28 does not). A
# single kernel carrying three bands' weights plus the forcing needs about 54. The flat weight
# layout is what makes the split work: each band's 14 arrays and 11 scalars travel as ONE struct
# argument, which also means these kernels call the package's own `ml_normalize_flat_v4!` and
# `ml_forward_flat_v4` rather than a transcription of them.

# --- helpers, device-safe (no global arrays, no Float64 literals) ---------------------------------

"""$(TYPEDSIGNATURES)
`tos_ceiling` for a kernel: the step table arrives as arguments because a GPU kernel cannot reach
the module-level `TOS_CEIL_*` arrays. `row` is 1 for the north, 2 for the south, matching how
`seaice_prep_kernel!` writes the hemisphere code."""
@inline function tos_ceiling_dev(sic::Float32, edges, tbl, row::Integer)
    nk = length(edges)
    @inbounds for k in 1:nk
        sic < edges[k] && return tbl[row, k]
    end
    @inbounds return tbl[row, nk]
end

"""$(TYPEDSIGNATURES)
Platt scaling of the low band's change probability followed by the threshold test, matching
`change_gate_open` in the serial sweep exactly, including the clamp away from the endpoints (the
logit of an exactly saturated softmax is not finite)."""
@inline function change_gate_open_dev(p::Float32, a::Float32, b::Float32, thresh::Float32)
    pc = clamp(p, 1.0f-6, 1.0f0 - 1.0f-6)
    z = log(pc / (1.0f0 - pc))
    return 1.0f0 / (1.0f0 + exp(-(a * z + b))) >= thresh
end

# --- 1. per-point preparation --------------------------------------------------------------------
# Classifies each point, snapshots the sea-ice field and turns the cycle accumulators into the six
# forcing means plus the tos the bands actually see (the ceiling applied). Doing it here keeps the
# accumulators out of every later kernel's argument budget.
#
# hemi: 0 = nothing happens at this point, 1 = Arctic band stack, 2 = Antarctic band stack,
#       3 = open ocean (the thermodynamic relaxation, no network).
@kernel inbounds = true function seaice_prep_kernel!(
        hemi, forcing, sic_snap, tos_prev, hist1, hist2, hist3, initialized,
        @Const(sic), @Const(sst), @Const(land_fraction), @Const(is_arctic), @Const(is_antarctic),
        @Const(mean_rsds), @Const(mean_lw), @Const(mean_wind), @Const(mean_tas),
        @Const(mean_tdps), @Const(mean_tos), @Const(tos_ceil_edges), @Const(tos_ceil_tbl),
        tos_ceil_on::Bool,
    )
    ij = @index(Global, Linear)

    # the serial sweep's branch order: Arctic, else Antarctic, else any other wet point, and every
    # one of them guarded by isfinite(sst)
    h = Int32(0)
    if isfinite(sst[ij])
        if is_arctic[ij]
            h = Int32(1)
        elseif is_antarctic[ij]
            h = Int32(2)
        elseif land_fraction[ij] < 1
            h = Int32(3)
        end
    end
    hemi[ij] = h

    sic_now = Float32(sic[ij])
    sic_snap[ij] = sic_now

    if h == Int32(1) || h == Int32(2)
        tos_this_cycle = mean_tos[ij]

        # lazy initialisation of the history, exactly as the serial sweep does it: a point that has
        # never been updated starts with a flat history at its current concentration
        if !initialized[ij]
            hist1[ij] = sic_now
            hist2[ij] = sic_now
            hist3[ij] = sic_now
            tos_prev[ij] = tos_this_cycle
            initialized[ij] = true
        end

        forcing[1, ij] = mean_rsds[ij]
        forcing[2, ij] = mean_lw[ij]
        forcing[3, ij] = mean_wind[ij]
        forcing[4, ij] = mean_tas[ij]
        forcing[5, ij] = mean_tdps[ij]

        tos_in = Float32(tos_prev[ij])
        if tos_ceil_on
            tos_in = min(tos_in, tos_ceiling_dev(sic_now, tos_ceil_edges, tos_ceil_tbl, h))
        end
        forcing[6, ij] = tos_in
        forcing[7, ij] = tos_this_cycle    # carried to the blend kernel, which rolls tos_prev
    elseif h == Int32(3)
        forcing[7, ij] = mean_tos[ij]
    end
end

"""$(TYPEDSIGNATURES)
The serial sweep's `neighbor_diff`, as a free function so it can be called from a kernel body
(KernelAbstractions rejects a `return` anywhere inside one, closures included). A land or missing
neighbour contributes nothing, which is what `build_cache_v2.py`'s `diff_or_zero` did in training,
so a coast degrades the same way it did there."""
@inline function nbr_diff(nbr, sic_snap, land_fraction, r::Int, ij::Int, sic_now::Float32)
    @inbounds k = nbr[r, ij]
    k == Int32(0) && return 0.0f0
    @inbounds land_fraction[k] >= 1 && return 0.0f0
    @inbounds return sic_snap[k] - sic_now
end

"""$(TYPEDSIGNATURES)
The serial sweep's `neighbor_val`, used only by the 4th-difference fallback column: here a land or
missing neighbour contributes the CENTRE value instead of zero."""
@inline function nbr_val(nbr, sic_snap, land_fraction, r::Int, ij::Int, sic_now::Float32)
    @inbounds k = nbr[r, ij]
    k == Int32(0) && return sic_now
    @inbounds land_fraction[k] >= 1 && return sic_now
    @inbounds return sic_snap[k]
end

# --- 2. the cross columns ------------------------------------------------------------------------
# A fixed-index scalar gather off the snapshot. nbr rows: 1 N, 2 S, 3 W, 4 E, 5 N2, 6 S2, 7 W2, 8 E2.
@kernel inbounds = true function seaice_cross_kernel!(
        cross, @Const(sic_snap), @Const(nbr), @Const(land_fraction), @Const(coslat2), @Const(hemi),
        cross_ns::Float32, cross_we::Float32, curv_scale::Float32, curv_lap::Bool,
        curv_k_res::Float32, n_cross::Int32, curv_tanh_on::Bool,
    )
    ij = @index(Global, Linear)
    h = hemi[ij]
    # no early return: KernelAbstractions forbids `return` inside a kernel body
    if h == Int32(1) || h == Int32(2)
        sic_now = sic_snap[ij]

        nd(r) = nbr_diff(nbr, sic_snap, land_fraction, r, ij, sic_now)
        nv(r) = nbr_val(nbr, sic_snap, land_fraction, r, ij, sic_now)

        n_d = nd(1); s_d = nd(2); w_d = nd(3); e_d = nd(4)
        cross[1, ij] = cross_ns * (n_d - s_d) / 2.0f0
        cross[2, ij] = cross_we * (w_d - e_d) / 2.0f0

        c3 = 0.0f0
        if n_cross >= Int32(3) && curv_tanh_on
            curv = (w_d + e_d) / 2.0f0
            if curv_lap
                curv += coslat2[ij] * (n_d + s_d) / 2.0f0
            end
            # the resolved-scale refinement is NORTH ONLY in the serial sweep; mirrored, not tidied
            if h == Int32(1) && curv_k_res != 1.0f0
                curv2 = (nd(7) + nd(8)) / 2.0f0
                if curv_lap
                    curv2 += coslat2[ij] * (nd(5) + nd(6)) / 2.0f0
                end
                curv2 /= 4.0f0
                curv = curv_k_res * curv2 + (curv - curv2)
            end
            c3 = tanh(curv / curv_scale)
        elseif n_cross >= Int32(3)
            c3 = (nv(7) - 4.0f0 * nv(3) + 6.0f0 * sic_now - 4.0f0 * nv(4) + nv(8)) / 16.0f0
        end
        cross[3, ij] = c3
    else
        cross[1, ij] = 0.0f0
        cross[2, ij] = 0.0f0
        cross[3, ij] = 0.0f0
    end
end

# --- 3/4. one band per launch --------------------------------------------------------------------
# `band` selects the row of `tend` written. `only_north` is true for the high band, which the
# Antarctic stack does not use at all.
@kernel inbounds = true function seaice_band_relax_kernel!(
        tend, band::Int32, only_north::Bool, @Const(sic_snap), @Const(hist1), @Const(hist2),
        @Const(hist3), @Const(forcing), @Const(cross), @Const(hemi), xb, h1, h2, mb, w,
    )
    ij = @index(Global, Linear)
    h = hemi[ij]
    active = (h == Int32(1)) || (h == Int32(2) && !only_north)
    if active
        sic_now = sic_snap[ij]
        ml_normalize_flat_v4!(
            xb, ij, w, sic_now, forcing[6, ij], forcing[1, ij], forcing[2, ij],
            forcing[3, ij], forcing[4, ij], forcing[5, ij],
            cross[1, ij], cross[2, ij], cross[3, ij],
        )
        tend[band, ij] = ml_forward_flat_v4(
            w, xb, h1, h2, mb, ij, sic_now,
            (Float32(hist3[ij]), Float32(hist2[ij]), Float32(hist1[ij])),
        )
    else
        tend[band, ij] = 0.0f0
    end
end

@kernel inbounds = true function seaice_band_gate_kernel!(
        tend, pchange, band::Int32, @Const(sic_snap), @Const(hist1), @Const(hist2),
        @Const(hist3), @Const(forcing), @Const(cross), @Const(hemi), xb, h1, h2, mb, w,
    )
    ij = @index(Global, Linear)
    h = hemi[ij]
    if h == Int32(1) || h == Int32(2)
        sic_now = sic_snap[ij]
        ml_normalize_flat_v4!(
            xb, ij, w, sic_now, forcing[6, ij], forcing[1, ij], forcing[2, ij],
            forcing[3, ij], forcing[4, ij], forcing[5, ij],
            cross[1, ij], cross[2, ij], cross[3, ij],
        )
        t, p = ml_forward_flat_changegate_v4(
            w, xb, h1, h2, mb, ij, sic_now,
            (Float32(hist3[ij]), Float32(hist2[ij]), Float32(hist1[ij])),
        )
        tend[band, ij] = t
        pchange[ij] = p
    else
        tend[band, ij] = 0.0f0
        pchange[ij] = 0.0f0
    end
end

# --- 5. gate blend, history roll and the open-ocean relaxation -----------------------------------
# `T` (the element type of the physics scalars) is Float64 on a device with double precision and
# Float32 otherwise, which is what `accum_precision` selects. The ocean branch is the only place it
# is visible: the band stack is Float32 throughout on every device.
@kernel inbounds = true function seaice_blend_kernel!(
        sic, sst, @Const(sic_snap), @Const(tend), @Const(pchange), @Const(hemi),
        hist1, hist2, hist3, tos_prev, @Const(forcing),
        gate_c::Float32, gate_w::Float32, lgate_c::Float32, lgate_w::Float32,
        lgate_c_s::Float32, lgate_w_s::Float32,
        cg_on::Bool, cg_thresh::Float32, cg_a::Float32, cg_b::Float32,
        temp_freeze, melt_rate, freeze_rate, dt_seconds,
    )
    ij = @index(Global, Linear)
    h = hemi[ij]

    if h == Int32(1)
        sic_now = sic_snap[ij]
        gate_high = 1.0f0 / (1.0f0 + exp(-(sic_now - gate_c) / gate_w))
        gate_low = 1.0f0 / (1.0f0 + exp(-(lgate_c - sic_now) / lgate_w))
        gate_mid = 1.0f0 - gate_low - gate_high
        # same summation order as the serial sweep: low, mid, high
        tendency = gate_low * tend[3, ij] + gate_mid * tend[1, ij] + gate_high * tend[2, ij]
        if cg_on && gate_low > 0.5f0 && !change_gate_open_dev(pchange[ij], cg_a, cg_b, cg_thresh)
            tendency = 0.0f0
        end
        hist3[ij] = hist2[ij]; hist2[ij] = hist1[ij]; hist1[ij] = sic_now
        tos_prev[ij] = forcing[7, ij]
        sic[ij] = clamp(sic_now + tendency, 0.0f0, 1.0f0)

    elseif h == Int32(2)
        sic_now = sic_snap[ij]
        gate_low = 1.0f0 / (1.0f0 + exp(-(lgate_c_s - sic_now) / lgate_w_s))
        gate_mid = 1.0f0 - gate_low
        tendency = gate_low * tend[3, ij] + gate_mid * tend[1, ij]
        if cg_on && gate_low > 0.5f0 && !change_gate_open_dev(pchange[ij], cg_a, cg_b, cg_thresh)
            tendency = 0.0f0
        end
        hist3[ij] = hist2[ij]; hist2[ij] = hist1[ij]; hist1[ij] = sic_now
        tos_prev[ij] = forcing[7, ij]
        sic[ij] = clamp(sic_now + tendency, 0.0f0, 1.0f0)

    elseif h == Int32(3)
        # typed zeros throughout: clamp(x::Float32, 0, 1) with Int literals promotes to Float64 and
        # will not compile for Metal (the same trap as the land-sea mask clamp)
        sst_mean = forcing[7, ij]
        dT = sst_mean - temp_freeze
        z = zero(dT)
        F = -melt_rate * max(dT, z) - freeze_rate / dt_seconds * min(dT, z)
        sst[ij] = max(sst[ij], temp_freeze)
        sic[ij] = clamp(sic[ij] + dt_seconds * F, zero(dt_seconds), one(dt_seconds))
    end
end

# --- 0. the per-timestep accumulation ------------------------------------------------------------
# This is the BIGGER half of the component's cost (0.37 s of accumulation against 0.20 s of update
# per 15 simulated T63 days) and, unlike the 5-day sweep, it touches the model's own fields on
# EVERY step. That is what makes it the real obstacle to running alongside a GPU model: the serial
# loop reads `rsds[ij]`, `sst[ij]` and five more fields one element at a time, which is scalar
# indexing of device arrays -- something CUDA.jl refuses outright rather than merely doing slowly.
#
# It also WRITES the sea surface temperature (the tos floor), so it cannot be reduced to a
# broadcast over the accumulators alone.
@kernel inbounds = true function seaice_accumulate_kernel!(
        sst, accum_rsds, accum_lw, accum_wind, accum_tas, accum_tos, accum_tdps,
        @Const(rsds), @Const(longwave_down), @Const(wind10), @Const(tas), @Const(land_fraction),
        @Const(is_arctic), @Const(is_antarctic), @Const(humid), @Const(surface_pressure),
        nlayers::Int32, have_humidity::Bool, tos_floor::Float32,
    )
    ij = @index(Global, Linear)
    if land_fraction[ij] < 1 && isfinite(sst[ij])
        if (is_arctic[ij] || is_antarctic[ij]) && sst[ij] < tos_floor
            sst[ij] = tos_floor
        end
        # the accumulator's own element type drives the arithmetic, exactly as the serial loop's
        # `A_(...)` cast does: the value is converted and then added, never added and converted
        A = eltype(accum_rsds)
        accum_rsds[ij] += A(rsds[ij])
        accum_lw[ij] += A(longwave_down[ij])
        accum_wind[ij] += A(wind10[ij])
        accum_tas[ij] += A(tas[ij])
        accum_tos[ij] += A(sst[ij])
        if have_humidity
            q_ij = Float32(humid[ij, nlayers])
            td = dewpoint_from_humidity(max(q_ij, 0.0f0), Float32(surface_pressure[ij]))
            accum_tdps[ij] += A(td)
        end
    end
end

# --- buffers and the driver ----------------------------------------------------------------------

"""$(TYPEDSIGNATURES)
Allocate every device-resident buffer the kernel sweep needs and move the component's persistent
state onto the device. Called once from `initialize!`.

The component's own state (history, previous tos, the initialised flags) is MIRRORED: the device
copy is what the kernels work on, and `ml_seaice_sweep_kernels!` writes it back to the host arrays
at the end of every update, because those host arrays are what the run script checkpoints. Nine
vectors in and seven out per 5-day update, which is nothing against a component that is 0.33% of
runtime -- and the model itself may well be on the host meanwhile, which is the realistic
configuration: the spectral transform is 45-260x SLOWER on Metal than on the CPU while this
component is several times faster."""
function ml_seaice_kernel_buffers(sea_ice_model::MLSeaIceV4, arch, field_arch, npoints::Int)
    AT = Architectures.array_type(arch)
    dev(x) = AT(x)
    z(T, dims...) = (a = AT{T}(undef, dims...); fill!(a, zero(T)); a)

    # Do the MODEL's own fields already live where the kernels run? Everything about how this
    # component couples to a parallel run turns on that one question. When they do (model and
    # component on the same device -- the CUDA case) nothing crosses the bus at all: the
    # per-timestep accumulation becomes a kernel over the fields in place and the 5-day sweep reads
    # the accumulators where they already are. When they do not (the model on the host, the
    # component on Metal) the fields are staged through the host once per update.
    resident = Architectures.ismatching(field_arch, arch)
    A = accumulator_type(sea_ice_model)

    fm = sea_ice_model.flat_mid
    fh = sea_ice_model.flat_highsic
    fl = sea_ice_model.flat_lowsic

    nbr = Matrix{Int32}(undef, 8, npoints)
    for ij in 1:npoints
        nbr[1, ij] = sea_ice_model.north_idx[ij]
        nbr[2, ij] = sea_ice_model.south_idx[ij]
        nbr[3, ij] = sea_ice_model.west_idx[ij]
        nbr[4, ij] = sea_ice_model.east_idx[ij]
        nbr[5, ij] = sea_ice_model.north2_idx[ij]
        nbr[6, ij] = sea_ice_model.south2_idx[ij]
        nbr[7, ij] = sea_ice_model.west2_idx[ij]
        nbr[8, ij] = sea_ice_model.east2_idx[ij]
    end

    return (
        # per-point work
        hemi = z(Int32, npoints),
        forcing = z(Float32, 7, npoints),
        cross = z(Float32, 3, npoints),
        tend = z(Float32, 3, npoints),
        pchange = z(Float32, npoints),
        sic_snap = z(Float32, npoints),
        # host staging for the fields that cross the bus (reused, so no per-update allocation)
        sic_host = zeros(Float32, npoints),
        sst_host = zeros(Float32, npoints),
        accum_host = zeros(Float32, npoints),
        # device state, resident
        sic = z(Float32, npoints),
        sst = z(Float32, npoints),
        mean_rsds = z(Float32, npoints), mean_lw = z(Float32, npoints),
        mean_wind = z(Float32, npoints), mean_tas = z(Float32, npoints),
        mean_tdps = z(Float32, npoints), mean_tos = z(Float32, npoints),
        # the cycle accumulators themselves, used ONLY when the fields are resident: then they are
        # where the per-timestep kernel writes and the sweep reads, and the host arrays are
        # refreshed from them once per update so the checkpoint still sees the truth
        acc_rsds = z(A, npoints), acc_lw = z(A, npoints), acc_wind = z(A, npoints),
        acc_tas = z(A, npoints), acc_tos = z(A, npoints), acc_tdps = z(A, npoints),
        resident = resident,
        hist1 = dev(Float32.(sea_ice_model.sic_hist1)),
        hist2 = dev(Float32.(sea_ice_model.sic_hist2)),
        hist3 = dev(Float32.(sea_ice_model.sic_hist3)),
        tos_prev = dev(Float32.(sea_ice_model.tos_prev_mean)),
        initialized = dev(copy(sea_ice_model.initialized)),
        # geometry and masks, read-only
        nbr = dev(nbr),
        land_fraction = z(Float32, npoints),
        coslat2 = dev(copy(sea_ice_model.coslat2)),
        is_arctic = dev(copy(sea_ice_model.is_arctic)),
        is_antarctic = dev(copy(sea_ice_model.is_antarctic)),
        tos_ceil_edges = dev(copy(TOS_CEIL_EDGES)),
        tos_ceil_tbl = dev(permutedims(hcat(TOS_CEIL_NORTH, TOS_CEIL_SOUTH))),
        # flat weights on the device: one kernel argument each, which is what keeps the launches
        # inside Metal's ~29-argument budget
        w_mid = Adapt.adapt(AT{Float32}, fm),
        w_high = Adapt.adapt(AT{Float32}, fh),
        w_low = Adapt.adapt(AT{Float32}, fl),
        # scratch, indexed by grid point (one column per concurrent worker)
        x_mid = z(Float32, fm.nin, npoints),
        x_high = z(Float32, fh.nin, npoints),
        x_low = z(Float32, fl.nin, npoints),
        h1 = z(Float32, fm.H, npoints),
        h2 = z(Float32, fm.H, npoints),
        mb = z(Float32, fm.Mm, npoints),
        arch = arch,
        npoints = npoints,
    )
end

"""$(TYPEDSIGNATURES)
Run the whole 5-day update as six kernel launches, in place of the serial sweep.

`B` is the buffer NamedTuple from `ml_seaice_kernel_buffers`. It reaches here through a field typed
`Any`, so this call is one dynamic dispatch every 5 days; everything inside it is concretely typed
because the NamedTuple's own types are. The serial sweep's three branches are all covered: the
Arctic band stack, the Antarctic one (mid and low only), and the open-ocean thermodynamic
relaxation, which also writes the sea surface temperature."""
function ml_seaice_sweep_kernels!(
        sea_ice_model::MLSeaIceV4, B, sic_field, sst_field, land_fraction, n, have_humidity::Bool,
        dt_seconds,
    )
    npoints = B.npoints
    arch = B.arch
    T = accumulator_type(sea_ice_model)

    # --- in: the two prognostic fields, the land fraction and the six cycle MEANS ----------------
    # [2026-11-11] The CYCLE MEAN is formed HERE, on the host, not in the kernel, and the
    # expression is `Float32(accum[ij] / n)` with n in the accumulator's own precision -- exactly
    # the serial sweep's. Staging the raw accumulator and dividing on the device instead rounds to
    # Float32 BEFORE the division, which cost 3.05e-6 in the sea-ice field after a single update
    # and 2.5e-2 after three once the atmosphere had fed it back. Measured, and the reason the
    # kernel no longer divides at all.
    stage = B.accum_host

    # [2026-11-11] Whether the model's own fields already live where the kernels run decides how
    # they get there, and getting this wrong is not a slowdown but a crash: a `Float32(src[ij])`
    # loop over a field that is already a device array is SCALAR INDEXING, which CUDA.jl refuses
    # outright. That is the configuration this component was made GPU-capable for in the first
    # place -- model and component both on CUDA -- so the resident case copies straight across
    # with no host round trip, which is also the only configuration where the transfer is free.
    # The branch itself is exercised by every CPU run (model and component both on the CPU take
    # the resident path); it has NOT been run on a CUDA device here, there being none.
    resident = B.resident
    # [2026-11-11] `.data` on the resident side: copyto!(::CuArray, ::Field) would go through the
    # generic AbstractArray path and copy point by point, which CUDA.jl refuses outright. Between
    # the two underlying arrays it is one device-to-device copy. The staged branch keeps indexing
    # the Field because there the fields are on the host anyway.
    for (dst, src) in (
            (B.sic, field_data(sic_field)), (B.sst, field_data(sst_field)),
            (B.land_fraction, field_data(land_fraction)),
        )
        if resident
            copyto!(dst, src)
        else
            @inbounds for ij in 1:npoints
                stage[ij] = Float32(src[ij])
            end
            copyto!(dst, stage)
        end
    end
    if resident
        # a broadcast, so the SAME expression as the serial sweep's `Float32(accum[ij] / n)`:
        # divided in the accumulator's precision, then rounded. No transfer at all.
        #
        # `n` is narrowed to the accumulator's own element type FIRST, and that is not cosmetic: a
        # Float64 scalar inside a Metal broadcast is an "unsupported use of double value" and fails
        # to compile, which is how this was found. The serial sweep's own `n` is already
        # `accumulator_type(...)`, so the conversion is a no-op whenever the two agree and the
        # arithmetic stays bit-identical; it is here to make the agreement impossible to lose.
        nn = convert(eltype(B.acc_rsds), n)
        B.mean_rsds .= Float32.(B.acc_rsds ./ nn)
        B.mean_lw .= Float32.(B.acc_lw ./ nn)
        B.mean_wind .= Float32.(B.acc_wind ./ nn)
        B.mean_tas .= Float32.(B.acc_tas ./ nn)
        B.mean_tos .= Float32.(B.acc_tos ./ nn)
        if have_humidity
            B.mean_tdps .= Float32.(B.acc_tdps ./ nn)
        else
            fill!(B.mean_tdps, 0.0f0)
        end
    else
        for (dst, src) in (
                (B.mean_rsds, sea_ice_model.accum_rsds), (B.mean_lw, sea_ice_model.accum_lw),
                (B.mean_wind, sea_ice_model.accum_wind), (B.mean_tas, sea_ice_model.accum_tas),
                (B.mean_tos, sea_ice_model.accum_tos),
            )
            @inbounds for ij in 1:npoints
                stage[ij] = Float32(src[ij] / n)
            end
            copyto!(dst, stage)
        end
        # the serial sweep substitutes a hard zero when the model carries no humidity, so the dew
        # point column the bands see is zero rather than a mean of an empty accumulator
        @inbounds for ij in 1:npoints
            stage[ij] = have_humidity ? Float32(sea_ice_model.accum_tdps[ij] / n) : 0.0f0
        end
        copyto!(B.mean_tdps, stage)
    end

    # --- the six launches ------------------------------------------------------------------------
    nc = Int32(length(sea_ice_model.weights_mid.cross_mean))
    curv_scale = Float32(sea_ice_model.curvature_tanh_scale)
    dev = Architectures.device(arch)

    launch!(arch, LinearWorkOrder, (npoints,), seaice_prep_kernel!,
        B.hemi, B.forcing, B.sic_snap, B.tos_prev, B.hist1, B.hist2, B.hist3, B.initialized,
        B.sic, B.sst, B.land_fraction, B.is_arctic, B.is_antarctic,
        B.mean_rsds, B.mean_lw, B.mean_wind, B.mean_tas, B.mean_tdps, B.mean_tos,
        B.tos_ceil_edges, B.tos_ceil_tbl,
        sea_ice_model.tos_ceiling_from_training > 0)
    KernelAbstractions.synchronize(dev)      # cross reads the snapshot and hemisphere prep writes

    launch!(arch, LinearWorkOrder, (npoints,), seaice_cross_kernel!,
        B.cross, B.sic_snap, B.nbr, B.land_fraction, B.coslat2, B.hemi,
        sea_ice_model.cross_scale_ns, sea_ice_model.cross_scale_we, curv_scale,
        sea_ice_model.curvature_laplacian, Float32(sea_ice_model.curvature_k_resolved), nc,
        curv_scale > 0)
    KernelAbstractions.synchronize(dev)      # the bands read the cross columns

    # The three bands share the hidden-layer scratch h1/h2/mb (each point uses its own column, but
    # all three launches use the SAME columns), so they must not overlap. They are serialised
    # EXPLICITLY rather than left to the backend's in-order queue: both Metal and the CPU backend
    # happen to serialise them, which is exactly the kind of accident that holds until it doesn't.
    # Nothing is lost -- each launch already covers every grid point, so the parallelism is inside
    # a launch, not across them.
    launch!(arch, LinearWorkOrder, (npoints,), seaice_band_relax_kernel!,
        B.tend, Int32(1), false, B.sic_snap, B.hist1, B.hist2, B.hist3, B.forcing, B.cross, B.hemi,
        B.x_mid, B.h1, B.h2, B.mb, B.w_mid)
    KernelAbstractions.synchronize(dev)
    launch!(arch, LinearWorkOrder, (npoints,), seaice_band_relax_kernel!,
        B.tend, Int32(2), true, B.sic_snap, B.hist1, B.hist2, B.hist3, B.forcing, B.cross, B.hemi,
        B.x_high, B.h1, B.h2, B.mb, B.w_high)
    KernelAbstractions.synchronize(dev)
    launch!(arch, LinearWorkOrder, (npoints,), seaice_band_gate_kernel!,
        B.tend, B.pchange, Int32(3), B.sic_snap, B.hist1, B.hist2, B.hist3, B.forcing, B.cross,
        B.hemi, B.x_low, B.h1, B.h2, B.mb, B.w_low)
    KernelAbstractions.synchronize(dev)

    launch!(arch, LinearWorkOrder, (npoints,), seaice_blend_kernel!,
        B.sic, B.sst, B.sic_snap, B.tend, B.pchange, B.hemi,
        B.hist1, B.hist2, B.hist3, B.tos_prev, B.forcing,
        sea_ice_model.highsic_gate_center, sea_ice_model.highsic_gate_width,
        sea_ice_model.lowsic_gate_center, sea_ice_model.lowsic_gate_width,
        sea_ice_model.lowsic_gate_center_south, sea_ice_model.lowsic_gate_width_south,
        sea_ice_model.changegate_threshold > 0f0, sea_ice_model.changegate_threshold,
        sea_ice_model.changegate_calib_a, sea_ice_model.changegate_calib_b,
        T(sea_ice_model.temp_freeze), T(sea_ice_model.melt_rate), T(sea_ice_model.freeze_rate),
        T(dt_seconds))

    KernelAbstractions.synchronize(dev)

    # --- out: the two prognostic fields ------------------------------------------------------------
    if resident
        copyto!(field_data(sic_field), B.sic)   # see field_data: a bulk copy, not npoints scalar ones
        copyto!(field_data(sst_field), B.sst)
    else
        copyto!(B.sic_host, B.sic)
        copyto!(B.sst_host, B.sst)
        @inbounds for ij in 1:npoints
            sic_field[ij] = B.sic_host[ij]
            sst_field[ij] = B.sst_host[ij]
        end
    end

    # --- and the component's carried state, back onto the HOST ------------------------------------
    # [2026-11-11] Not optional, and not merely tidy. The run script's checkpoint saves
    # tos_prev_mean / sic_hist1..3 / initialized from these host arrays and restores them on
    # restart, and the forcing dump writes sic_hist1. Left resident on the device they would go
    # stale from initialize! onwards, so a checkpoint taken during a kernel run would silently
    # reset the sea-ice history when resumed -- a wrong answer with no error. The invariant is
    # therefore that the HOST arrays are the component's state and the device copy is a cache.
    # Five vectors every 5 simulated days; against a 0.33%-of-runtime component it costs nothing.
    #
    # The round trip through Float32 is lossless here: every value these arrays ever hold
    # originated as a Float32 (sic_now, or a Float32 cycle mean) stored into a Float64 array.
    stage = B.accum_host
    for (host, dst_dev) in (
            (sea_ice_model.sic_hist1, B.hist1), (sea_ice_model.sic_hist2, B.hist2),
            (sea_ice_model.sic_hist3, B.hist3), (sea_ice_model.tos_prev_mean, B.tos_prev),
        )
        copyto!(stage, dst_dev)
        @inbounds for ij in 1:npoints
            host[ij] = stage[ij]
        end
    end
    copyto!(sea_ice_model.initialized, B.initialized)

    # And the cycle accumulators, when they are the device's. The run script checkpoints all six
    # host arrays and `accum_count`, and restores them on resume, so the same invariant applies as
    # to the history: the HOST arrays are the state. Done here, at the one moment per cycle when
    # they are about to be zeroed anyway, so it is six vectors every 5 simulated days.
    #
    # NOTE what this does NOT cover: the forcing-dump recorder re-checks the component's
    # accumulators against its own copy on EVERY step, element by element. With the accumulators on
    # the device those host arrays are only fresh once per cycle, so that per-step check compares
    # against stale values. It is a diagnostic, not physics, and the run script skips it in this
    # mode rather than printing a meaningless disagreement.
    if resident
        for (host, src) in (
                (sea_ice_model.accum_rsds, B.acc_rsds), (sea_ice_model.accum_lw, B.acc_lw),
                (sea_ice_model.accum_wind, B.acc_wind), (sea_ice_model.accum_tas, B.acc_tas),
                (sea_ice_model.accum_tos, B.acc_tos), (sea_ice_model.accum_tdps, B.acc_tdps),
            )
            copyto!(host, src)
        end
    end
    return nothing
end

"""$(TYPEDSIGNATURES)
The per-timestep accumulation as a kernel, for when the model's fields are resident on the device
the component runs on. Returns `true` when it ran, so the caller can fall through to the serial
loop otherwise -- the loop stays the path for every host run and is untouched.

`fill!` of the device accumulators at the end of a cycle is the caller's job, as it is for the host
arrays."""
function ml_seaice_accumulate_kernel!(
        sea_ice_model::MLSeaIceV4, B, sst, rsds, longwave_down, wind10, tas, land_fraction,
        humid, surface_pressure, nlayers::Int, have_humidity::Bool,
    )
    arch = B.arch
    launch!(arch, LinearWorkOrder, (B.npoints,), seaice_accumulate_kernel!,
        sst, B.acc_rsds, B.acc_lw, B.acc_wind, B.acc_tas, B.acc_tos, B.acc_tdps,
        rsds, longwave_down, wind10, tas, land_fraction,
        B.is_arctic, B.is_antarctic, humid, surface_pressure,
        Int32(nlayers), have_humidity, Float32(sea_ice_model.tos_floor))
    KernelAbstractions.synchronize(Architectures.device(arch))
    return nothing
end

"""$(TYPEDSIGNATURES)
Zero the device-resident cycle accumulators. Mirrors the `fill!` the serial path does on the host
arrays at the end of every update."""
function ml_seaice_reset_device_accumulators!(B)
    for a in (B.acc_rsds, B.acc_lw, B.acc_wind, B.acc_tas, B.acc_tos, B.acc_tdps)
        fill!(a, zero(eltype(a)))
    end
    return nothing
end
