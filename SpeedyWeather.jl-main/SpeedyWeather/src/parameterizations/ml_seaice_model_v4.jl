"""
Native Julia re-implementation of the gradient-basis-cross candidate checkpoints (2026-08-26, see
`hemisphere_unified_gradbasis_mono_variants.ipynb`): identical `ChangeGateClassifier`/
`RelaxationMoEClassifier` architecture and rld-based radiation input to `ml_seaice_model_v3.jl`,
differing only in what the cross-neighbor columns hold -- 2 columns `[(N-S)/2, (W-E)/2]` (a clean
centered-difference gradient per axis) here, in place of V3's 4 raw redundant `[N,S,W,E]` columns.
This was found to substantially reduce the near-pole "spoke" streaking artifact in offline rollout
diagnostics (south d_dt=0: CORR 0.7386->0.9333, mean local-anomaly 0.0926->0.0240; north d_dt=0:
CORR 0.9171->0.9670, anomaly 0.0154->0.0039 -- see the notebook). The joint (sum) mono-penalty
formulation is used (tied with a separate-penalty variant after independent gate re-sweeping, see
the same notebook), so weight semantics otherwise match V3's `unified_low_gradbasis_cross`/
`unified_mid_gradbasis_cross`/`north_high_gradbasis_cross` checkpoints exactly.

This is a purely ADDITIVE module: it does not modify, import from, or get imported by
`ml_seaice_model.jl` / `ml_seaice_model_v2.jl` / `ml_seaice_model_v3.jl` in any way. V1/V2/V3
weights/behavior are untouched -- this duplicates rather than shares V3's (structurally identical,
cross-width-agnostic) normalize/forward code, matching this project's established convention of
keeping each ML sea-ice version fully independent.

Model recap (identical math to `MLSeaIceModelV3`, only the cross-neighbor columns' physical meaning
differs):
    h = backbone(x)                                   # 4x(Linear+GELU), width 64
    diffs = hist[2:end] .- hist[1:end-1] (append sic_raw as last point)
    mom = momentum_mlp(diffs)                          # Linear+GELU, width 16
    combined = [h; mom]                                # 80-d
    class_logits = class_head(combined)
    ... (inc/dec heads or single changed_head, exactly as in MLSeaIceModelV3)

Raw input feature order (before normalization): [siconc, tos, rsds, rld, sfcwind, tas], then 2
cross-neighbor gradient columns [(N-S)/2, (W-E)/2], then OPTIONALLY tdps (2m dewpoint) as the final
column -- present for the unified low/mid bands, absent for north_high (same convention as V3).
"""
module MLSeaIceModelV4

import Adapt

export MLWeightsV4, MLWeightsChangeGateV4
export load_ml_weights_v4, load_ml_weights_changegate_v4, ml_forward_changegate_with_prob
export ml_forward_v4, ml_normalize_features_v4

@inline function erf_approx(x::Float32)
    a1 = 0.254829592f0; a2 = -0.284496736f0; a3 = 1.421413741f0
    a4 = -1.453152027f0; a5 = 1.061405429f0; p = 0.3275911f0
    s = sign(x)
    ax = abs(x)
    t = 1f0 / (1f0 + p * ax)
    y = 1f0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-ax * ax)
    return s * y
end

@inline gelu(x::Float32) = 0.5f0 * x * (1f0 + erf_approx(x / sqrt(2f0)))
@inline gelu(x::AbstractArray{Float32}) = gelu.(x)

@inline sigmoid(x::Float32) = 1f0 / (1f0 + exp(-x))
@inline softplus(x::Float32) = log1p(exp(-abs(x))) + max(x, 0f0)

# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------
struct MLWeightsV4
    W::Vector{Matrix{Float32}}
    b::Vector{Vector{Float32}}
    Wm::Matrix{Float32}
    bm::Vector{Float32}
    Wc::Matrix{Float32}; bc::Vector{Float32}   # class_head
    Wi::Matrix{Float32}; bi::Vector{Float32}   # inc_head
    Wd::Matrix{Float32}; bd::Vector{Float32}   # dec_head
    tau_inc_param::Float32
    tau_dec_param::Float32
    tau_floor::Float32
    dt::Float32
    # normalization: base cols [siconc, tos, rsds, sfcwind, tas] use norm_mean/norm_std (still a
    # 6-slot array for layout compatibility with the exporter -- slot 4, "rls", is UNUSED here and
    # must NOT be read for rld; see rld_mean/rld_std below, a separate pair, matching how the
    # Python training pipeline actually normalizes rld -- own stats, not the base-6-array's rls slot).
    norm_mean::Vector{Float32}
    norm_std::Vector{Float32}
    cross_mean::Vector{Float32}
    cross_std::Vector{Float32}
    rld_mean::Float32
    rld_std::Float32
    has_tdps::Bool
    tdps_mean::Float32
    tdps_std::Float32
end

struct MLWeightsChangeGateV4
    W::Vector{Matrix{Float32}}
    b::Vector{Vector{Float32}}
    Wm::Matrix{Float32}
    bm::Vector{Float32}
    Wc::Matrix{Float32}; bc::Vector{Float32}     # class_head: [unchanged, changed]
    Wch::Matrix{Float32}; bch::Vector{Float32}   # changed_head
    tau_changed_param::Float32
    tau_floor::Float32
    dt::Float32
    norm_mean::Vector{Float32}
    norm_std::Vector{Float32}
    cross_mean::Vector{Float32}
    cross_std::Vector{Float32}
    rld_mean::Float32
    rld_std::Float32
    has_tdps::Bool
    tdps_mean::Float32
    tdps_std::Float32
end

function _read_bin(dir::String, fname::AbstractString, shape)
    path = joinpath(dir, fname)
    data = reinterpret(Float32, read(path))
    if shape == "0"
        return Float32(data[1])
    end
    dims = Tuple(parse(Int, s) for s in split(shape, ","))
    if length(dims) == 1
        return Vector{Float32}(data)
    else
        out_f, in_f = dims  # PyTorch (out,in) row-major
        return reshape(Vector{Float32}(data), (in_f, out_f))
    end
end

function _read_manifest(dir::String)
    manifest = readlines(joinpath(dir, "manifest.txt"))
    tensors = Dict{String, Any}()
    for line in manifest
        isempty(strip(line)) && continue
        name, shape, fname = split(line, "\t")
        tensors[name] = _read_bin(dir, fname, shape)
    end
    return tensors
end

function load_ml_weights_v4(dir::String; tau_floor::Float32=1.0f0, dt::Float32=5.0f0)
    tensors = _read_manifest(dir)
    W = [tensors["backbone.0.weight"], tensors["backbone.2.weight"],
         tensors["backbone.4.weight"], tensors["backbone.6.weight"]]
    b = [tensors["backbone.0.bias"], tensors["backbone.2.bias"],
         tensors["backbone.4.bias"], tensors["backbone.6.bias"]]
    has_tdps = haskey(tensors, "norm.tdps_mean")
    tdps_mean = has_tdps ? tensors["norm.tdps_mean"] : 0f0
    tdps_std = has_tdps ? tensors["norm.tdps_std"] : 1f0

    return MLWeightsV4(
        W, b,
        tensors["momentum_mlp.0.weight"], tensors["momentum_mlp.0.bias"],
        tensors["class_head.weight"], tensors["class_head.bias"],
        tensors["inc_head.weight"], tensors["inc_head.bias"],
        tensors["dec_head.weight"], tensors["dec_head.bias"],
        tensors["tau_inc_param"][1], tensors["tau_dec_param"][1],
        tau_floor, dt,
        tensors["norm.mean"], tensors["norm.std"],
        tensors["norm.cross_mean"], tensors["norm.cross_std"],
        tensors["norm.rld_mean"], tensors["norm.rld_std"],
        has_tdps, Float32(tdps_mean), Float32(tdps_std),
    )
end

function load_ml_weights_changegate_v4(dir::String; tau_floor::Float32=1.0f0, dt::Float32=5.0f0)
    tensors = _read_manifest(dir)
    W = [tensors["backbone.0.weight"], tensors["backbone.2.weight"],
         tensors["backbone.4.weight"], tensors["backbone.6.weight"]]
    b = [tensors["backbone.0.bias"], tensors["backbone.2.bias"],
         tensors["backbone.4.bias"], tensors["backbone.6.bias"]]
    has_tdps = haskey(tensors, "norm.tdps_mean")
    tdps_mean = has_tdps ? tensors["norm.tdps_mean"] : 0f0
    tdps_std = has_tdps ? tensors["norm.tdps_std"] : 1f0

    return MLWeightsChangeGateV4(
        W, b,
        tensors["momentum_mlp.0.weight"], tensors["momentum_mlp.0.bias"],
        tensors["class_head.weight"], tensors["class_head.bias"],
        tensors["changed_head.weight"], tensors["changed_head.bias"],
        tensors["tau_changed_param"][1],
        tau_floor, dt,
        tensors["norm.mean"], tensors["norm.std"],
        tensors["norm.cross_mean"], tensors["norm.cross_std"],
        tensors["norm.rld_mean"], tensors["norm.rld_std"],
        has_tdps, Float32(tdps_mean), Float32(tdps_std),
    )
end

# ---------------------------------------------------------------------------
# Feature normalization -- mirrors build_feature_matrix_from_raw for
# feature_set="sic+sst+rsds+rld+sfcwind+tas" (+ N cross-neighbor cols) (+ tdps if has_tdps),
# matching the shifted-notos convention (tos NOT shifted, other forcing IS -- handled by the
# caller passing already-shifted values in, same as ml_seaice_v2.jl does).
# raw order in: [siconc, tos, rsds, rld, sfcwind, tas]  (NO doy_sin/doy_cos)
#
# CRITICAL: rld is normalized with its OWN rld_mean/rld_std (a separate scalar pair), NOT with
# slot 4 of the base m/s arrays -- that slot structurally holds RLS (net longwave) statistics in
# the exporter's base-6-column convention, regardless of which features were actually trained on.
# Passing rld through m[4]/s[4] silently applies the wrong (and wildly different-scale, ~-48 mean
# vs rld's real ~277) normalization -- this exact bug caused a real online run's Arctic sea ice to
# collapse to near-zero within months; caught by comparing online Rld values directly against the
# exported norm stats. See export_v3_weights.py's pooled_norm_low_mid docstring for the full story.
# ---------------------------------------------------------------------------
function _normalize_base(m, s, rld_mean, rld_std, siconc, tos, rsds, rld, sfcwind, tas)
    f_sic = (siconc - m[1]) / max(s[1], 1f-6)
    f_tos = (tos - m[2]) / max(s[2], 1f-6)
    f_rsds = (rsds - m[3]) / max(s[3], 1f-6)
    f_rld = (rld - rld_mean) / max(rld_std, 1f-6)
    f_wind = (sfcwind - m[5]) / max(s[5], 1f-6)
    f_tas = (tas - m[6]) / max(s[6], 1f-6)
    return Float32[f_sic, f_tos, f_rsds, f_rld, f_wind, f_tas]
end

function ml_normalize_features_v4(w::MLWeightsV4, siconc, tos, rsds, rld, sfcwind, tas, tdps, cross::NTuple{N,Float32}) where N
    base = _normalize_base(w.norm_mean, w.norm_std, w.rld_mean, w.rld_std, siconc, tos, rsds, rld, sfcwind, tas)
    cross_norm = ntuple(i -> (cross[i] - w.cross_mean[i]) / max(w.cross_std[i], 1f-6), N)
    # [2026-09-29 BUG FIX] tdps goes BEFORE the cross columns, not after.
    # sic_pipeline_lib.build_feature_matrix_from_raw emits the base features in the order of the
    # feature-set string -- "sic+sst+rsds+rld+sfcwind+tas+tdps" -- and the cross columns are
    # concatenated AFTER that, so the layout the weights were trained on is
    #     [sic, tos, rsds, rld, sfcwind, tas, tdps, cross...]
    # This function used to append tdps after the cross columns instead, i.e. it fed the network
    #     [sic, tos, rsds, rld, sfcwind, tas, cross..., tdps]
    # so for every band carrying tdps (low and mid, i.e. everything below the high gate) the last
    # three inputs were cycled: the tdps weights received cross_ns, the cross_ns weights received
    # cross_we, and the cross_we weights received tdps. The high band has no tdps and was therefore
    # always correct, which is why online runs still produced plausible dense pack and the fault
    # stayed hidden. Measured on the shipped unified_mid weights, the tendency for a marginal-zone
    # input was 0.1944 with the old order against 0.0557 with this one, which matches PyTorch to
    # 7 significant figures. EVERY online result produced before this date is affected; offline
    # training and evaluation never were, because both sides there call the same Python builder.
    # Regression test: test/parameterizations/ml_seaice_v4_feature_order.jl
    if w.has_tdps
        f_tdps = (tdps - w.tdps_mean) / max(w.tdps_std, 1f-6)
        return vcat(base, Float32[f_tdps], collect(Float32, cross_norm))
    end
    return vcat(base, collect(Float32, cross_norm))
end

function ml_normalize_features_v4(w::MLWeightsChangeGateV4, siconc, tos, rsds, rld, sfcwind, tas, tdps, cross::NTuple{N,Float32}) where N
    base = _normalize_base(w.norm_mean, w.norm_std, w.rld_mean, w.rld_std, siconc, tos, rsds, rld, sfcwind, tas)
    cross_norm = ntuple(i -> (cross[i] - w.cross_mean[i]) / max(w.cross_std[i], 1f-6), N)
    # [2026-09-29 BUG FIX] tdps goes BEFORE the cross columns, not after.
    # sic_pipeline_lib.build_feature_matrix_from_raw emits the base features in the order of the
    # feature-set string -- "sic+sst+rsds+rld+sfcwind+tas+tdps" -- and the cross columns are
    # concatenated AFTER that, so the layout the weights were trained on is
    #     [sic, tos, rsds, rld, sfcwind, tas, tdps, cross...]
    # This function used to append tdps after the cross columns instead, i.e. it fed the network
    #     [sic, tos, rsds, rld, sfcwind, tas, cross..., tdps]
    # so for every band carrying tdps (low and mid, i.e. everything below the high gate) the last
    # three inputs were cycled: the tdps weights received cross_ns, the cross_ns weights received
    # cross_we, and the cross_we weights received tdps. The high band has no tdps and was therefore
    # always correct, which is why online runs still produced plausible dense pack and the fault
    # stayed hidden. Measured on the shipped unified_mid weights, the tendency for a marginal-zone
    # input was 0.1944 with the old order against 0.0557 with this one, which matches PyTorch to
    # 7 significant figures. EVERY online result produced before this date is affected; offline
    # training and evaluation never were, because both sides there call the same Python builder.
    # Regression test: test/parameterizations/ml_seaice_v4_feature_order.jl
    if w.has_tdps
        f_tdps = (tdps - w.tdps_mean) / max(w.tdps_std, 1f-6)
        return vcat(base, Float32[f_tdps], collect(Float32, cross_norm))
    end
    return vcat(base, collect(Float32, cross_norm))
end

# ---------------------------------------------------------------------------
# Forward pass -- identical math to MLSeaIceModelV2, just under a new name/type to keep this
# module fully independent. hist3 = [sic(t-3), sic(t-2), sic(t-1)] raw (not normalized).
# ---------------------------------------------------------------------------
function ml_forward_v4(w::MLWeightsChangeGateV4, x::Vector{Float32}, sic_raw::Float32, hist3::NTuple{3,Float32})
    h = x
    for l in 1:4
        h = gelu(w.W[l]' * h .+ w.b[l])
    end
    full_seq = Float32[hist3[1], hist3[2], hist3[3], sic_raw]
    diffs = Float32[full_seq[2]-full_seq[1], full_seq[3]-full_seq[2], full_seq[4]-full_seq[3]]
    mom = gelu(w.Wm' * diffs .+ w.bm)

    combined = vcat(h, mom)

    class_logits = w.Wc' * combined .+ w.bc
    m = max(class_logits[1], class_logits[2])
    e1 = exp(class_logits[1] - m); e2 = exp(class_logits[2] - m)
    p_changed = e2 / (e1 + e2)

    changed_logit = (w.Wch' * combined .+ w.bch)[1]
    sic_eq_changed = sigmoid(changed_logit)

    tau_changed = w.tau_floor + softplus(w.tau_changed_param)
    decay_changed = exp(-w.dt / tau_changed)

    return p_changed * (sic_eq_changed - sic_raw) * (1f0 - decay_changed)
end

"""
    ml_forward_changegate_with_prob(w, x, sic_raw, hist3)

The same forward pass, returning the change probability alongside the tendency. The caller needs it
to apply the calibrated change-gate threshold, which cannot be done inside this function because the
threshold only applies where the low band actually holds the blend.
"""
function ml_forward_changegate_with_prob(w::MLWeightsChangeGateV4, x::Vector{Float32},
                                         sic_raw::Float32, hist3::NTuple{3,Float32})
    h = x
    for l in 1:4
        h = gelu(w.W[l]' * h .+ w.b[l])
    end
    full_seq = Float32[hist3[1], hist3[2], hist3[3], sic_raw]
    diffs = Float32[full_seq[2]-full_seq[1], full_seq[3]-full_seq[2], full_seq[4]-full_seq[3]]
    mom = gelu(w.Wm' * diffs .+ w.bm)
    combined = vcat(h, mom)
    class_logits = w.Wc' * combined .+ w.bc
    m = max(class_logits[1], class_logits[2])
    e1 = exp(class_logits[1] - m); e2 = exp(class_logits[2] - m)
    p_changed = e2 / (e1 + e2)
    changed_logit = (w.Wch' * combined .+ w.bch)[1]
    sic_eq_changed = sigmoid(changed_logit)
    tau_changed = w.tau_floor + softplus(w.tau_changed_param)
    decay_changed = exp(-w.dt / tau_changed)
    return p_changed * (sic_eq_changed - sic_raw) * (1f0 - decay_changed), p_changed
end

function ml_forward_v4(w::MLWeightsV4, x::Vector{Float32}, sic_raw::Float32, hist3::NTuple{3,Float32})
    h = x
    for l in 1:4
        h = gelu(w.W[l]' * h .+ w.b[l])
    end
    full_seq = Float32[hist3[1], hist3[2], hist3[3], sic_raw]
    diffs = Float32[full_seq[2]-full_seq[1], full_seq[3]-full_seq[2], full_seq[4]-full_seq[3]]
    mom = gelu(w.Wm' * diffs .+ w.bm)

    combined = vcat(h, mom)

    class_logits = w.Wc' * combined .+ w.bc
    m = max(class_logits[1], class_logits[2])
    e1 = exp(class_logits[1] - m); e2 = exp(class_logits[2] - m)
    p_inc = e1 / (e1 + e2)
    p_dec = e2 / (e1 + e2)

    inc_logit = (w.Wi' * combined .+ w.bi)[1]
    dec_logit = (w.Wd' * combined .+ w.bd)[1]
    sic_eq_inc = sic_raw + (1f0 - sic_raw) * sigmoid(inc_logit)
    sic_eq_dec = sic_raw * sigmoid(dec_logit)

    tau_inc = w.tau_floor + softplus(w.tau_inc_param)
    tau_dec = w.tau_floor + softplus(w.tau_dec_param)
    decay_inc = exp(-w.dt / tau_inc)
    decay_dec = exp(-w.dt / tau_dec)

    tendency_inc = (sic_eq_inc - sic_raw) * (1f0 - decay_inc)
    tendency_dec = (sic_eq_dec - sic_raw) * (1f0 - decay_dec)

    return p_inc * tendency_inc + p_dec * tendency_dec
end

# ===========================================================================================
# [2026-11-11] GPU-ready flat weights and an allocation-free forward pass.
#
# The forward pass above is written for clarity and allocates about ten times per grid point per
# band: `h = gelu(w.W[l]' * h .+ w.b[l])` builds a fresh Vector on every one of the four layers,
# plus two `Float32[...]` literals, a `vcat` and three head matvecs. It also stores the backbone
# as a `Vector{Matrix{Float32}}` -- an array of arrays, which cannot be moved to a device as it
# stands. Neither is a problem on the CPU (the whole component is 0.33% of runtime), but both
# stop the 5-day update from being a kernel.
#
# This section keeps the SAME arithmetic and reorganises the storage:
#   * the backbone's three H->H layers become one (H, H, 3) array and the first one stays (nin, H),
#     so every field is a plain dense array,
#   * single-output heads become vectors with a scalar bias, and
#   * the per-point working vectors become columns of caller-owned scratch, so the pass itself
#     allocates nothing and can be called from inside a kernel.
#
# The ONLY numerical difference from the original is the summation order: `W' * h` goes through
# BLAS gemv, which blocks and vectorises the reduction, while the loops here accumulate i = 1..n
# in order. The flatten/forward agreement against the original is measured, not assumed -- see
# test/parameterizations/ml_seaice_v4_flat.jl.
# ===========================================================================================

export MLWeightsFlatV4, MLWeightsFlatChangeGateV4, flatten_weights_v4
export ml_forward_flat_v4, ml_forward_flat_changegate_v4, ml_normalize_flat_v4!
export MLWeightsFlatV4CPU, MLWeightsFlatChangeGateV4CPU

struct MLWeightsFlatV4{V <: AbstractVector{Float32}, M <: AbstractMatrix{Float32}, A3 <: AbstractArray{Float32, 3}}
    W1::M       # (nin, H)   first backbone layer
    b1::V       # (H,)
    Wh::A3      # (H, H, 3)  backbone layers 2..4
    bh::M       # (H, 3)
    Wm::M       # (3, Mm)    momentum MLP
    bm::V       # (Mm,)
    Wc::M       # (H+Mm, 2)  class head
    bc::V       # (2,)
    Wi::V       # (H+Mm,)    inc head, single output
    bi::Float32
    Wd::V       # (H+Mm,)    dec head, single output
    bd::Float32
    tau_inc_param::Float32
    tau_dec_param::Float32
    tau_floor::Float32
    dt::Float32
    norm_mean::V
    norm_std::V
    cross_mean::V
    cross_std::V
    rld_mean::Float32
    rld_std::Float32
    has_tdps::Bool
    tdps_mean::Float32
    tdps_std::Float32
    nin::Int
    H::Int
    Mm::Int
    ncross::Int
end

struct MLWeightsFlatChangeGateV4{V <: AbstractVector{Float32}, M <: AbstractMatrix{Float32}, A3 <: AbstractArray{Float32, 3}}
    W1::M
    b1::V
    Wh::A3
    bh::M
    Wm::M
    bm::V
    Wc::M
    bc::V
    Wch::V      # changed head, single output
    bch::Float32
    tau_changed_param::Float32
    tau_floor::Float32
    dt::Float32
    norm_mean::V
    norm_std::V
    cross_mean::V
    cross_std::V
    rld_mean::Float32
    rld_std::Float32
    has_tdps::Bool
    tdps_mean::Float32
    tdps_std::Float32
    nin::Int
    H::Int
    Mm::Int
    ncross::Int
end

# the backbone's H->H layers are stacked; every matrix is already stored (in, out) by _read_bin
function _stack_backbone(W, b)
    H = size(W[2], 1)
    size(W[2], 2) == H || error("flatten_weights_v4: backbone layer 2 is $(size(W[2])), not square")
    Wh = Array{Float32, 3}(undef, H, H, 3)
    bh = Matrix{Float32}(undef, H, 3)
    for l in 1:3
        size(W[l + 1]) == (H, H) || error("flatten_weights_v4: backbone layer $(l+1) is $(size(W[l+1])), expected ($H, $H)")
        @views Wh[:, :, l] .= W[l + 1]
        @views bh[:, l] .= b[l + 1]
    end
    return Wh, bh
end

"""
Reorganise loaded weights into the flat, device-transferable layout. Values are copied, never
recomputed, so the flat weights hold exactly the same numbers as the originals."""
function flatten_weights_v4(w::MLWeightsV4)
    Wh, bh = _stack_backbone(w.W, w.b)
    nin, H = size(w.W[1])
    Mm = length(w.bm)
    return MLWeightsFlatV4(
        copy(w.W[1]), copy(w.b[1]), Wh, bh,
        copy(w.Wm), copy(w.bm),
        copy(w.Wc), copy(w.bc),
        vec(copy(w.Wi)), w.bi[1],
        vec(copy(w.Wd)), w.bd[1],
        w.tau_inc_param, w.tau_dec_param, w.tau_floor, w.dt,
        copy(w.norm_mean), copy(w.norm_std), copy(w.cross_mean), copy(w.cross_std),
        w.rld_mean, w.rld_std, w.has_tdps, w.tdps_mean, w.tdps_std,
        nin, H, Mm, length(w.cross_mean),
    )
end

function flatten_weights_v4(w::MLWeightsChangeGateV4)
    Wh, bh = _stack_backbone(w.W, w.b)
    nin, H = size(w.W[1])
    Mm = length(w.bm)
    return MLWeightsFlatChangeGateV4(
        copy(w.W[1]), copy(w.b[1]), Wh, bh,
        copy(w.Wm), copy(w.bm),
        copy(w.Wc), copy(w.bc),
        vec(copy(w.Wch)), w.bch[1],
        w.tau_changed_param, w.tau_floor, w.dt,
        copy(w.norm_mean), copy(w.norm_std), copy(w.cross_mean), copy(w.cross_std),
        w.rld_mean, w.rld_std, w.has_tdps, w.tdps_mean, w.tdps_std,
        nin, H, Mm, length(w.cross_mean),
    )
end

# Concrete instantiations for host arrays. The component stores these so that every field access
# inside the forward pass is statically typed: with the fields declared `Any` the pass is a chain
# of dynamic dispatches and runs SLOWER than the allocating version it replaces (measured: the
# 5-day update went 0.09 -> 0.14 s per 6-day chunk). A device build instantiates the same structs
# on device array types and is passed to the kernel as an argument rather than through a field.
# Adapt lets the whole weight struct be moved to a device in one call, which is how it reaches a
# kernel: every field is either a dense array or an immutable scalar.
Adapt.@adapt_structure MLWeightsFlatV4
Adapt.@adapt_structure MLWeightsFlatChangeGateV4

const MLWeightsFlatV4CPU = MLWeightsFlatV4{Vector{Float32}, Matrix{Float32}, Array{Float32, 3}}
const MLWeightsFlatChangeGateV4CPU = MLWeightsFlatChangeGateV4{Vector{Float32}, Matrix{Float32}, Array{Float32, 3}}

# --- normalisation, written straight into a scratch column -----------------------------------
# Feature layout, as the training pipeline emits it (see the 2026-09-29 note above):
#     [sic, tos, rsds, rld, sfcwind, tas, (tdps), cross...]
@inline function ml_normalize_flat_v4!(xb, ij, w, siconc, tos, rsds, rld, sfcwind, tas, tdps,
                                       c1::Float32, c2::Float32, c3::Float32)
    m = w.norm_mean; s = w.norm_std
    xb[1, ij] = (siconc - m[1]) / max(s[1], 1f-6)
    xb[2, ij] = (tos - m[2]) / max(s[2], 1f-6)
    xb[3, ij] = (rsds - m[3]) / max(s[3], 1f-6)
    xb[4, ij] = (rld - w.rld_mean) / max(w.rld_std, 1f-6)
    xb[5, ij] = (sfcwind - m[5]) / max(s[5], 1f-6)
    xb[6, ij] = (tas - m[6]) / max(s[6], 1f-6)
    k = 6
    if w.has_tdps
        k += 1
        xb[k, ij] = (tdps - w.tdps_mean) / max(w.tdps_std, 1f-6)
    end
    nc = w.ncross
    cm = w.cross_mean; cs = w.cross_std
    nc >= 1 && (xb[k + 1, ij] = (c1 - cm[1]) / max(cs[1], 1f-6))
    nc >= 2 && (xb[k + 2, ij] = (c2 - cm[2]) / max(cs[2], 1f-6))
    nc >= 3 && (xb[k + 3, ij] = (c3 - cm[3]) / max(cs[3], 1f-6))
    return nothing
end

# --- the shared trunk: backbone + momentum, leaving the 80-d combined vector in (h2, mb) -----
@inline function _flat_trunk!(xb, h1, h2, mb, ij, w, sic_raw::Float32, hist3::NTuple{3, Float32})
    H = w.H
    nin = w.nin
    W1 = w.W1; b1 = w.b1; Wh = w.Wh; bh = w.bh
    @inbounds for j in 1:H
        acc = b1[j]
        @simd for i in 1:nin
            acc += W1[i, j] * xb[i, ij]
        end
        h1[j, ij] = gelu(acc)
    end
    # layers 2..4, ping-ponging h1 -> h2 -> h1 -> h2, so the result is in h2
    @inbounds for l in 1:3
        if isodd(l)
            for j in 1:H
                acc = bh[j, l]
                @simd for i in 1:H
                    acc += Wh[i, j, l] * h1[i, ij]
                end
                h2[j, ij] = gelu(acc)
            end
        else
            for j in 1:H
                acc = bh[j, l]
                @simd for i in 1:H
                    acc += Wh[i, j, l] * h2[i, ij]
                end
                h1[j, ij] = gelu(acc)
            end
        end
    end
    d1 = hist3[2] - hist3[1]
    d2 = hist3[3] - hist3[2]
    d3 = sic_raw - hist3[3]
    Wm = w.Wm; bm = w.bm
    @inbounds for j in 1:w.Mm
        mb[j, ij] = gelu(bm[j] + Wm[1, j] * d1 + Wm[2, j] * d2 + Wm[3, j] * d3)
    end
    return nothing
end

# combined[k] = k <= H ? h2[k] : mb[k-H]
@inline function _flat_head(Wcol, bias::Float32, h2, mb, ij, H::Int, Mm::Int)
    acc = bias
    @inbounds @simd for k in 1:H
        acc += Wcol[k] * h2[k, ij]
    end
    @inbounds @simd for k in 1:Mm
        acc += Wcol[H + k] * mb[k, ij]
    end
    return acc
end

@inline function _flat_class(Wc, bc, h2, mb, ij, H::Int, Mm::Int)
    @inbounds l1 = bc[1]; @inbounds l2 = bc[2]
    @inbounds @simd for k in 1:H
        hk = h2[k, ij]
        l1 += Wc[k, 1] * hk
        l2 += Wc[k, 2] * hk
    end
    @inbounds @simd for k in 1:Mm
        mk = mb[k, ij]
        l1 += Wc[H + k, 1] * mk
        l2 += Wc[H + k, 2] * mk
    end
    return l1, l2
end

"""
Allocation-free relaxation (inc/dec) forward pass. Identical arithmetic to `ml_forward_v4`; the
working vectors are columns `ij` of the caller's scratch."""
@inline function ml_forward_flat_v4(w::MLWeightsFlatV4, xb, h1, h2, mb, ij,
                                    sic_raw::Float32, hist3::NTuple{3, Float32})
    _flat_trunk!(xb, h1, h2, mb, ij, w, sic_raw, hist3)
    H = w.H; Mm = w.Mm
    l1, l2 = _flat_class(w.Wc, w.bc, h2, mb, ij, H, Mm)
    m = max(l1, l2)
    e1 = exp(l1 - m); e2 = exp(l2 - m)
    p_inc = e1 / (e1 + e2)
    p_dec = e2 / (e1 + e2)

    inc_logit = _flat_head(w.Wi, w.bi, h2, mb, ij, H, Mm)
    dec_logit = _flat_head(w.Wd, w.bd, h2, mb, ij, H, Mm)
    sic_eq_inc = sic_raw + (1f0 - sic_raw) * sigmoid(inc_logit)
    sic_eq_dec = sic_raw * sigmoid(dec_logit)

    tau_inc = w.tau_floor + softplus(w.tau_inc_param)
    tau_dec = w.tau_floor + softplus(w.tau_dec_param)
    decay_inc = exp(-w.dt / tau_inc)
    decay_dec = exp(-w.dt / tau_dec)

    tendency_inc = (sic_eq_inc - sic_raw) * (1f0 - decay_inc)
    tendency_dec = (sic_eq_dec - sic_raw) * (1f0 - decay_dec)
    return p_inc * tendency_inc + p_dec * tendency_dec
end

"""
Allocation-free change-gate forward pass, returning (tendency, p_changed) exactly as
`ml_forward_changegate_with_prob` does."""
@inline function ml_forward_flat_changegate_v4(w::MLWeightsFlatChangeGateV4, xb, h1, h2, mb, ij,
                                               sic_raw::Float32, hist3::NTuple{3, Float32})
    _flat_trunk!(xb, h1, h2, mb, ij, w, sic_raw, hist3)
    H = w.H; Mm = w.Mm
    l1, l2 = _flat_class(w.Wc, w.bc, h2, mb, ij, H, Mm)
    m = max(l1, l2)
    e1 = exp(l1 - m); e2 = exp(l2 - m)
    p_changed = e2 / (e1 + e2)

    changed_logit = _flat_head(w.Wch, w.bch, h2, mb, ij, H, Mm)
    sic_eq_changed = sigmoid(changed_logit)
    tau_changed = w.tau_floor + softplus(w.tau_changed_param)
    decay_changed = exp(-w.dt / tau_changed)
    return p_changed * (sic_eq_changed - sic_raw) * (1f0 - decay_changed), p_changed
end

end # module
