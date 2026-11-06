"""
Rollout/blend-gate/diffusion sweep machinery for the rld-based hemisphere-unified low/mid + Arctic-
only high model (checkpoints_hemisphere_unified_rld_tdps/). Adapted from Modeltry24_arctic_server.
ipynb's field-level rollout code, generalized for:
  - split rsds+rld (not combined radsum) and per-band feature sets (low/mid include tdps, high does
    not) via sic_pipeline_lib.py's build_feature_matrix_from_raw (already handles this).
  - both hemispheres: north gets the full 3-way (low/mid/high) blend, south gets a 2-way (low/mid
    only) blend -- no dedicated south high-SIC expert exists (established project convention).
Normalization stats are read directly from the same L.make_pooled_loaders/L.make_loaders calls used
at training time (not reimplemented/hardcoded), guaranteeing exact consistency with training.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import sic_pipeline_lib as L

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "processed"
CACHE_DIR = Path(__file__).resolve().parent.parent / "cache_rld_ablation"
CKPT_DIR = Path(__file__).resolve().parent.parent / "checkpoints_hemisphere_unified_rld_tdps"

SPATIAL_BLOCK = 2
WINDOW_DAYS = 5
SPATIAL_SIGMA = 1.0
R_EARTH_KM = 6371.0

LAT_SLICE = {"north": (140, 180), "south": (0, 40)}
LAT_LIMIT = {"north": 60, "south": -60}

RAW_VAR_NAMES = ["siconc", "tos", "rsds", "rls", "sfcWind", "tas", "rld", "tdps"]


def collect_processed_files(years):
    return [PROCESSED_DIR / f"processed_{y}.nc" for y in years if (PROCESSED_DIR / f"processed_{y}.nc").exists()]


def build_blocked_fields_2deg(files, hemisphere, spatial_block=SPATIAL_BLOCK, spatial_sigma=SPATIAL_SIGMA,
                               spatial_min_valid_weight=0.3, window_days=WINDOW_DAYS):
    lat_slice_lo, lat_slice_hi = LAT_SLICE[hemisphere]
    files = sorted(Path(f) for f in files)
    smoothed = {}; lsm_static = None; lat = None
    for vname in RAW_VAR_NAMES:
        parts = []
        for path in files:
            with xr.open_dataset(path) as ds:
                arr = ds[vname].values[:, lat_slice_lo:lat_slice_hi, :].astype(np.float32)
                parts.append(arr)
                if lat is None: lat = ds["latitude"].values[lat_slice_lo:lat_slice_hi].astype(np.float32)
                if lsm_static is None:
                    lsm_val = ds["lsm"].values.astype(np.float32)
                    if lsm_val.ndim == 3: lsm_val = lsm_val[0]
                    lsm_static = lsm_val[lat_slice_lo:lat_slice_hi, :]
        raw = np.concatenate(parts, axis=0)

        def _tumbling_mean_forward(a):
            T = a.shape[0]; nb = T // window_days
            trimmed = a[: nb * window_days]
            finite = np.isfinite(trimmed)
            filled = np.where(finite, trimmed, 0.0).astype(np.float32)
            shape = (nb, window_days) + a.shape[1:]
            fb = filled.reshape(shape); ncount = (~finite).reshape(shape).sum(axis=1)
            out = (fb.sum(axis=1) / window_days).astype(np.float32)
            out[ncount > 0] = np.nan
            return out

        def _spatial_block_average_strict(a):
            T, nlat, nlon = a.shape
            nlat_c, nlon_c = nlat // spatial_block, nlon // spatial_block
            trimmed = a[:, : nlat_c * spatial_block, : nlon_c * spatial_block]
            finite = np.isfinite(trimmed)
            filled = np.where(finite, trimmed, 0.0).astype(np.float32)
            shape = (T, nlat_c, spatial_block, nlon_c, spatial_block)
            fb = filled.reshape(shape); ncount = (~finite).reshape(shape).sum(axis=(2, 4))
            out = (fb.sum(axis=(2, 4)) / (spatial_block * spatial_block)).astype(np.float32)
            out[ncount > 0] = np.nan
            return out

        def _spatial_gaussian_smooth_coarse(a):
            if spatial_sigma == 0: return a
            finite = np.isfinite(a)
            filled = np.where(finite, a, 0.0).astype(np.float32)
            weight = finite.astype(np.float32)
            sig = (0, spatial_sigma, spatial_sigma); mode = ("nearest", "nearest", "wrap")
            sf = gaussian_filter(filled, sigma=sig, mode=mode)
            sw = gaussian_filter(weight, sigma=sig, mode=mode)
            with np.errstate(invalid="ignore", divide="ignore"):
                out = sf / sw
            return np.where(sw >= spatial_min_valid_weight, out, np.nan).astype(np.float32)

        spatial_coarse = _spatial_block_average_strict(raw)
        spatial_coarse_smooth = _spatial_gaussian_smooth_coarse(spatial_coarse)
        smoothed[vname] = _tumbling_mean_forward(spatial_coarse_smooth)

    def _spatial_block_average_plain(a2d):
        nlat, nlon = a2d.shape
        nlat_c, nlon_c = nlat // spatial_block, nlon // spatial_block
        trimmed = a2d[: nlat_c * spatial_block, : nlon_c * spatial_block]
        shape = (nlat_c, spatial_block, nlon_c, spatial_block)
        return trimmed.reshape(shape).mean(axis=(1, 3)).astype(np.float32)

    lat_coarse = lat.reshape(-1, spatial_block).mean(axis=1).astype(np.float32)
    lsm_coarse = _spatial_block_average_plain(lsm_static)
    sin_parts, cos_parts = [], []
    for path in files:
        with xr.open_dataset(path) as ds:
            doy = ds["time"].dt.dayofyear.values.astype(np.float32)
            sin_parts.append(np.sin(2.0 * np.pi * doy / 366.0).astype(np.float32))
            cos_parts.append(np.cos(2.0 * np.pi * doy / 366.0).astype(np.float32))
    doy_sin = np.concatenate(sin_parts, axis=0); doy_cos = np.concatenate(cos_parts, axis=0)

    def _tf(a):
        T = a.shape[0]; nb = T // window_days
        trimmed = a[: nb * window_days]
        return trimmed.reshape(nb, window_days).mean(axis=1).astype(np.float32)

    sin_s = _tf(doy_sin); cos_s = _tf(doy_cos)
    ocean_frac = np.clip(1.0 - np.nan_to_num(lsm_coarse, nan=1.0), 0.0, 1.0).astype(np.float32)
    return {"blocked": smoothed, "lat_coarse": lat_coarse, "ocean_frac": ocean_frac, "sin_s": sin_s, "cos_s": cos_s}


def make_rollout_context(field_files, history_seed_file, hemisphere):
    f_main = build_blocked_fields_2deg(field_files, hemisphere)
    blocked = f_main["blocked"]
    f_hist = build_blocked_fields_2deg([history_seed_file], hemisphere)
    hist_seed = f_hist["blocked"]["siconc"][-3:]
    lat_coarse = f_main["lat_coarse"]; ocean_frac = f_main["ocean_frac"]
    sin_b, cos_b = f_main["sin_s"], f_main["cos_s"]
    n_blocks, nlat, nlon = blocked["siconc"].shape
    lat_limit = LAT_LIMIT[hemisphere]
    hemi_mask = (lat_coarse >= lat_limit) if hemisphere == "north" else (lat_coarse <= lat_limit)
    static_mask = np.broadcast_to(hemi_mask[:, None], ocean_frac.shape) & (ocean_frac > 0)
    finite_all = np.ones((nlat, nlon), dtype=bool)
    for v in RAW_VAR_NAMES: finite_all &= np.all(np.isfinite(blocked[v]), axis=0)
    finite_hist = np.all(np.isfinite(hist_seed), axis=0)
    valid_mask_final = static_mask & finite_all & finite_hist
    valid_idx = np.flatnonzero(valid_mask_final.ravel())
    i_idx, j_idx = np.unravel_index(valid_idx, valid_mask_final.shape)
    return dict(blocked=blocked, hist_seed=hist_seed, lat_coarse=lat_coarse, sin_b=sin_b, cos_b=cos_b,
                n_blocks=n_blocks, nlat=nlat, nlon=nlon, valid_mask_final=valid_mask_final,
                i_idx=i_idx, j_idx=j_idx, n_valid=len(valid_idx), ocean_frac=ocean_frac)


# [2026-10-15] a cross builder whose columns DEPEND ON LATITUDE (the physical Laplacian, whose
# zonal/meridional ratio is cos^2(lat)) needs lat_coarse on every call. `spherical_gradient_cross`
# cannot carry that: it means something else, "this checkpoint was trained on a *_sphgrad cache",
# and setting it would also rescale the E/W gradients. A script installing a lat-dependent builder
# sets this flag instead; unset leaves every existing caller byte-identical.
CROSS_NEEDS_LAT = False


def cross_diff_2d(sic2d, lat_coarse=None):
    """lat_coarse: if given, applies the same spherical-gradient E/W correction as build_cache_v2.
    py's spherical_gradient_cross option (divide by cos(lat) -- meridional/N-S spacing is constant
    on this grid, no correction needed there). Must match whether the checkpoint being used was
    trained on *_sphgrad caches (pass lat_coarse) or the original caches (leave None, default)."""
    latprev = np.empty_like(sic2d); latprev[1:] = sic2d[:-1]; latprev[0] = np.nan
    latnext = np.empty_like(sic2d); latnext[:-1] = sic2d[1:]; latnext[-1] = np.nan
    lonprev = np.roll(sic2d, 1, axis=1); lonnext = np.roll(sic2d, -1, axis=1)
    def d_or_0(neigh):
        d = neigh - sic2d
        return np.where(np.isfinite(neigh), d, 0.0).astype(np.float32)
    lat_d = [d_or_0(latprev), d_or_0(latnext)]
    lon_d = [d_or_0(lonprev), d_or_0(lonnext)]
    if lat_coarse is not None:
        coslat = np.maximum(np.cos(np.deg2rad(lat_coarse)), 1e-3).astype(np.float32)[:, None]
        lon_d = [lon_d[0] / coslat, lon_d[1] / coslat]
    return np.stack([lat_d[0], lat_d[1], lon_d[0], lon_d[1]], axis=-1)


def laplacian_physical(sic2d, lat_coarse, spatial_block=SPATIAL_BLOCK):
    dy_km = R_EARTH_KM * np.deg2rad(spatial_block)
    dx_km_per_lat = R_EARTH_KM * np.cos(np.deg2rad(lat_coarse)) * np.deg2rad(spatial_block)
    dx2 = (dx_km_per_lat ** 2)[:, None]; dy2 = dy_km ** 2
    latprev = np.empty_like(sic2d); latprev[1:] = sic2d[:-1]; latprev[0] = np.nan
    latnext = np.empty_like(sic2d); latnext[:-1] = sic2d[1:]; latnext[-1] = np.nan
    lonprev = np.roll(sic2d, 1, axis=1); lonnext = np.roll(sic2d, -1, axis=1)
    def ns_term(neigh):
        d = neigh - sic2d
        return np.where(np.isfinite(neigh) & np.isfinite(sic2d), d / dy2, 0.0)
    def ew_term(neigh):
        d = neigh - sic2d
        return np.where(np.isfinite(neigh) & np.isfinite(sic2d), d / dx2, 0.0)
    return ns_term(latprev) + ns_term(latnext) + ew_term(lonprev) + ew_term(lonnext)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


BAND_ARCH = {
    "low": dict(model_cls="changegate", feature_set="sic+sst+rsds+rld+sfcwind+tas+tdps", pooled=True,
                ckpt="unified_low_rld_mono"),
    "mid": dict(model_cls="moe", feature_set="sic+sst+rsds+rld+sfcwind+tas+tdps", pooled=True,
                ckpt="unified_mid_rld_mono"),
    "high": dict(model_cls="moe", feature_set="sic+sst+rsds+rld+sfcwind+tas", pooled=False,
                 ckpt="north_high_rld_mono"),
}


def load_band_model_for_rollout(band_name, ckpt_name=None, cross_cols=None):
    """Loads the frozen checkpoint + recomputes norm stats via the SAME L.make_pooled_loaders /
    L.make_loaders calls used at training time (not reimplemented), guaranteeing exact consistency.
    ckpt_name: override the default BAND_ARCH checkpoint dir name (e.g. for the no-N/S-cross
    ablation checkpoints). cross_cols: passed straight through to make_pooled_loaders/make_loaders
    -- must match what the checkpoint was actually trained with, e.g. [2,3] for east/west only."""
    arch = BAND_ARCH[band_name]
    feature_set = arch["feature_set"]
    ckpt_name = ckpt_name or arch["ckpt"]
    if arch["pooled"]:
        train_ds, _, _, _ = L.make_pooled_loaders(
            str(CACHE_DIR / f"north_train_{band_name}band_rld_tdps.npz"),
            str(CACHE_DIR / f"south_train_{band_name}band_rld_tdps.npz"),
            str(CACHE_DIR / f"north_val_{band_name}band_rld_tdps.npz"),
            str(CACHE_DIR / f"south_val_{band_name}band_rld_tdps.npz"),
            feature_set=feature_set, normalize_phys=True, batch_size=L.BATCH_SIZE, use_cross=True, cross_cols=cross_cols,
        )
    else:
        train_ds, _, _, _ = L.make_loaders(
            str(CACHE_DIR / f"north_train_{band_name}band_rld.npz"),
            str(CACHE_DIR / f"north_val_{band_name}band_rld.npz"),
            feature_set=feature_set, normalize_phys=True, batch_size=L.BATCH_SIZE, use_cross=True, cross_cols=cross_cols,
        )
    in_dim = train_ds.X.shape[1]
    if arch["model_cls"] == "changegate":
        model = L.ChangeGateClassifier(in_dim=in_dim, hidden=64, depth=4, tau_floor=L.TAU_FLOOR, dt=L.DT,
                                        history_h=train_ds.history_h, mom_hidden=16).to(L.DEVICE)
    else:
        model = L.RelaxationMoEClassifier(in_dim=in_dim, hidden=64, depth=4, dropout=0.0, tau_floor=L.TAU_FLOOR,
                                           dt=L.DT, history_h=train_ds.history_h, mom_hidden=16).to(L.DEVICE)
    ckpt = torch.load(CKPT_DIR / ckpt_name / "best.pt", map_location=L.DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state"]); model.eval()
    norm = dict(mean=train_ds.mean, std=train_ds.std, rld_mean=train_ds.rld_mean, rld_std=train_ds.rld_std,
                tdps_mean=train_ds.tdps_mean, tdps_std=train_ds.tdps_std,
                cross_mean=train_ds.cross_mean, cross_std=train_ds.cross_std, feature_set=feature_set,
                cross_cols=cross_cols)
    return model, norm


def build_rollout_inputs(sic_v, tos_v, rsds_v, rls_v, wind_v, tas_v, sin_v, cos_v, rld_v, tdps_v, cross_v, norm):
    if norm.get("cross_cols") is not None:
        cross_v = cross_v[:, norm["cross_cols"]]
    X_raw = np.stack([sic_v, tos_v, rsds_v, rls_v, wind_v, tas_v, sin_v, cos_v, rld_v, tdps_v], axis=1).astype(np.float32)
    X_norm, _ = L.build_feature_matrix_from_raw(
        X_raw, feature_set=norm["feature_set"], normalize_phys=True, mean=norm["mean"], std=norm["std"],
        rld_mean=norm["rld_mean"], rld_std=norm["rld_std"], tdps_mean=norm["tdps_mean"], tdps_std=norm["tdps_std"],
    )
    cross_norm = (cross_v - norm["cross_mean"]) / norm["cross_std"]
    return np.concatenate([X_norm, cross_norm], axis=1).astype(np.float32)


def _model_forward(model, X, sic_v, tos_v, hist_v):
    Xt = torch.from_numpy(X).to(L.DEVICE)
    sic_t = torch.from_numpy(sic_v).to(L.DEVICE); tos_t = torch.from_numpy(tos_v).to(L.DEVICE)
    hist_t = torch.from_numpy(hist_v).to(L.DEVICE)
    out = model(Xt, sic_t, tos_t, hist_t)
    tendency = out[0]
    return tendency.cpu().numpy()


# Platt scaling of the low band's change probability, fitted by
# rollout_sweep_dev/calibrate_change_gate.py on training years 1981-1988, both hemispheres,
# 861k cell blocks, against the band's own training label |dSIC| >= 0.005.
CHANGE_CALIB_A = 1.14784
CHANGE_CALIB_B = -1.33014


def _calibrate_change_prob(p):
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1.0 - np.clip(p, 1e-6, 1 - 1e-6)))
    return 1.0 / (1.0 + np.exp(-(CHANGE_CALIB_A * z + CHANGE_CALIB_B)))


def _change_prob(model, X, sic_v, tos_v, hist_v):
    """the low band's probability that the cell changes at all, from its own classification head"""
    Xt = torch.from_numpy(X).to(L.DEVICE)
    out = model(Xt, torch.from_numpy(sic_v).to(L.DEVICE), torch.from_numpy(tos_v).to(L.DEVICE),
                torch.from_numpy(hist_v).to(L.DEVICE))
    logits = out[3]
    return torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()


def run_rollout(ctx, d_dt, models, norms, hemisphere, recalibrate,
                 gate_low_center, gate_width, gate_high_center=None, spherical_gradient_cross=False,
                 gate_width_high=None, post_step=None, low_deadband=0.0):
    """3-way (north, gate_high_center given) or 2-way (south, gate_high_center=None) blend rollout.
    spherical_gradient_cross: must be True if the models were trained on *_sphgrad caches (E/W
    cross-neighbor features divided by cos(lat)) -- see build_cache_v2.py / cross_diff_2d.
    post_step: optional callable applied to each newly produced 2D field before it is carried into
    the next step, e.g. an explicit numerical filter. None (default) leaves every existing call
    bit-for-bit unchanged. Added 2026-09-29 to test zonal hyperdiffusion as an alternative to the
    curvature input column, since the 2-cell mode is in the null space of the cross feature and
    cannot be learned from the current inputs at all.
    gate_width_high: optional separate width for the mid/high gate (the online MLSeaIceV4 has separate
    lowsic_gate_width / highsic_gate_width fields); None (default) uses gate_width for both, i.e. the
    behaviour of every earlier call is unchanged."""
    blocked = ctx["blocked"]; hist_seed = ctx["hist_seed"]; lat_coarse = ctx["lat_coarse"]
    sin_b = ctx["sin_b"]; cos_b = ctx["cos_b"]; n_blocks = ctx["n_blocks"]; nlat = ctx["nlat"]; nlon = ctx["nlon"]
    i_idx = ctx["i_idx"]; j_idx = ctx["j_idx"]; n_valid = ctx["n_valid"]
    valid_mask_final = ctx["valid_mask_final"]
    recal_block = (n_blocks // 2) if recalibrate else n_blocks
    three_way = gate_high_center is not None

    roll = [None] * n_blocks
    roll[0] = blocked["siconc"][0].copy()
    with torch.no_grad():
        for t_block in range(0, n_blocks - 1):
            if t_block == recal_block:
                roll[recal_block] = blocked["siconc"][recal_block].copy()
            sic_now = roll[t_block]
            tos_now = blocked["tos"][t_block]
            rsds_next = blocked["rsds"][t_block + 1]; rls_next = blocked["rls"][t_block + 1]
            rld_next = blocked["rld"][t_block + 1]; tdps_next = blocked["tdps"][t_block + 1]
            wind_next = blocked["sfcWind"][t_block + 1]; tas_next = blocked["tas"][t_block + 1]
            sin_next, cos_next = sin_b[t_block + 1], cos_b[t_block + 1]

            sic_v = sic_now[i_idx, j_idx]; tos_v = tos_now[i_idx, j_idx]
            rsds_v = rsds_next[i_idx, j_idx]; rls_v = rls_next[i_idx, j_idx]
            rld_v = rld_next[i_idx, j_idx]; tdps_v = tdps_next[i_idx, j_idx]
            wind_v = wind_next[i_idx, j_idx]; tas_v = tas_next[i_idx, j_idx]
            sin_v = np.full(n_valid, sin_next, dtype=np.float32); cos_v = np.full(n_valid, cos_next, dtype=np.float32)
            cross_v = cross_diff_2d(sic_now, lat_coarse if (spherical_gradient_cross or CROSS_NEEDS_LAT)
                                    else None)[i_idx, j_idx, :]

            if t_block < recal_block:
                if t_block == 0: h3, h2, h1 = hist_seed[0], hist_seed[1], hist_seed[2]
                elif t_block == 1: h3, h2, h1 = hist_seed[1], hist_seed[2], roll[0]
                elif t_block == 2: h3, h2, h1 = hist_seed[2], roll[0], roll[1]
                else: h3, h2, h1 = roll[t_block-3], roll[t_block-2], roll[t_block-1]
            else:
                rel = t_block - recal_block
                if rel == 0: h3, h2, h1 = blocked["siconc"][recal_block-3], blocked["siconc"][recal_block-2], blocked["siconc"][recal_block-1]
                elif rel == 1: h3, h2, h1 = blocked["siconc"][recal_block-2], blocked["siconc"][recal_block-1], roll[recal_block]
                elif rel == 2: h3, h2, h1 = blocked["siconc"][recal_block-1], roll[recal_block], roll[recal_block+1]
                else: h3, h2, h1 = roll[t_block-3], roll[t_block-2], roll[t_block-1]
            hist_v = np.stack([h3[i_idx, j_idx], h2[i_idx, j_idx], h1[i_idx, j_idx]], axis=1).astype(np.float32)

            X_low = build_rollout_inputs(sic_v, tos_v, rsds_v, rls_v, wind_v, tas_v, sin_v, cos_v, rld_v, tdps_v, cross_v, norms["low"])
            X_mid = build_rollout_inputs(sic_v, tos_v, rsds_v, rls_v, wind_v, tas_v, sin_v, cos_v, rld_v, tdps_v, cross_v, norms["mid"])
            tendency_low = _model_forward(models["low"], X_low, sic_v, tos_v, hist_v)
            tendency_mid = _model_forward(models["mid"], X_mid, sic_v, tos_v, hist_v)

            if three_way:
                X_high = build_rollout_inputs(sic_v, tos_v, rsds_v, rls_v, wind_v, tas_v, sin_v, cos_v, rld_v, tdps_v, cross_v, norms["high"])
                tendency_high = _model_forward(models["high"], X_high, sic_v, tos_v, hist_v)
                gate_high = sigmoid((sic_v - gate_high_center) / (gate_width if gate_width_high is None else gate_width_high)).astype(np.float32)
                gate_low = sigmoid((gate_low_center - sic_v) / gate_width).astype(np.float32)
                gate_mid = 1.0 - gate_low - gate_high
                tendency = gate_low * tendency_low + gate_mid * tendency_mid + gate_high * tendency_high
            else:
                gate_low = sigmoid((gate_low_center - sic_v) / gate_width).astype(np.float32)
                gate_mid = 1.0 - gate_low
                tendency = gate_low * tendency_low + gate_mid * tendency_mid

            if low_deadband > 0:
                # [2026-10-16] the threshold is applied to a CALIBRATED probability. The raw head is
                # overconfident against its own training label, |dSIC| >= 0.005: it reports 0.62
                # where the truth is 0.29 and 0.021 where the truth is 0.0036. A threshold on the
                # raw number therefore has no interpretation, which is why the first version of this
                # needed 0.75 to work. The Platt map below was fitted on the eight TRAINING years
                # over both hemispheres, on the reanalysis state, and it tracks the truth to within
                # a few per cent in every probability bin. With it, 0.5 means what it says: the cell
                # moves only if it is more likely than not to move.
                # [2026-10-16] The component cannot emit "no change". Its low band gives
                # tendency = p_changed * (eq - c) * (1 - decay) with p_changed from a softmax and eq
                # from a sigmoid, so an exactly zero tendency is not representable and every cell
                # moves every block. The reanalysis holds 94.5 to 98.2 per cent of its open water at
                # exactly zero from one block to the next; the component holds 0.0 per cent, which
                # leaves a third of the Arctic parked at 0.02 to 0.04 instead of 0.
                #
                # This is the change gate doing what it is named for: where it reports a probability
                # of change below the threshold, the cell does not change. It overrides the blend
                # because that is the gate's purpose, and it acts only where the low band is the
                # band in charge. Inference only, no weight is touched, and 0 disables it.
                pc = _change_prob(models["low"], X_low, sic_v, tos_v, hist_v)
                pc = _calibrate_change_prob(pc)
                tendency = np.where((pc < low_deadband) & (gate_low > 0.5), 0.0, tendency).astype(np.float32)

            if d_dt > 0:
                L_full = laplacian_physical(sic_now, lat_coarse)
                tendency = tendency + d_dt * L_full[i_idx, j_idx]

            sic_next_v = np.clip(sic_v + tendency, 0.0, 1.0)
            next_field = np.full((nlat, nlon), np.nan, dtype=np.float32)
            next_field[i_idx, j_idx] = sic_next_v
            if post_step is not None:
                next_field = post_step(next_field)
            roll[t_block + 1] = next_field

    roll = np.stack(roll)
    true_sic = np.where(valid_mask_final[None, :, :], blocked["siconc"], np.nan)
    pred_v = roll[:, i_idx, j_idx]; true_v = true_sic[:, i_idx, j_idx]
    err = pred_v - true_v
    rmse = float(np.sqrt(np.nanmean(err ** 2)))
    corr = float(np.corrcoef(pred_v.ravel(), true_v.ravel())[0, 1])
    return roll, corr, rmse
