"""
Native Julia implementation of the gradient-based sea-ice
concentration (SIC) parameterisation.

Implements the ChangeGateClassifier and RelaxationMoEClassifier
architectures for regime-specific SIC tendency prediction.

The neural architecture combines:
- A four-layer GELU backbone with hidden width 64.
- A momentum encoder operating on successive SIC-history differences,
  with hidden width 16.
- Classification and tendency-prediction heads.

The backbone and momentum representations are concatenated into
an 80-dimensional latent representation before prediction.

Spatial information is represented by two centered-difference
cross-neighbor gradients:
    [(N - S) / 2, (W - E) / 2]

Raw input feature order:
    [siconc, tos, rsds, rld, sfcwind, tas,
     grad_ns, grad_we, (optional) tdps]

The dew-point temperature (tdps) feature is included for the
low- and mid-SIC models and omitted for the high-SIC model.

Provides feature normalization, neural-network inference,
and regime-specific SIC tendency evaluation.
"""
module MLSeaIceModelV4

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

end # module
