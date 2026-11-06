"""
Training-cache builder for the sea-ice concentration (SIC)
machine-learning parameterisation.

Constructs spatially augmented training caches from preprocessed
ERA5 NetCDF files in processed/*.nc, including local spatial
features and SIC history.

Supports downward longwave radiation (rld) when available,
appending it as raw feature column 8 in accordance with the
input layout defined in sic_pipeline_lib.py.

Provides reusable cache-generation utilities for model
training and validation.
"""
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter
from tqdm import tqdm


def build_cache(
    files,
    out_path,
    hemisphere="north",
    lat_limit=60,
    window_days=5,
    spatial_sigma=1.0,
    spatial_min_valid_weight=0.3,
    sic_band_lo=0.1,
    sic_band_hi=0.9,
    # [2026-10-13] The lower bound is STRICT, which is a harmless boundary convention for the mid
    # (0.1,0.9) and high (0.9,1.0) bands but deletes a MASS POINT for the low band, whose lower edge
    # is exactly 0.0: every permanently open-ocean cell is dropped. Measured consequence: the low
    # band's training set contains 0 cells at SIC == 0, while 20.2% of its deployment rows are
    # exactly 0 and never change, so its change gate is fitted on a population that changes 69.6% of
    # the time and applied to one that changes 8.7% of the time. Both options default to the old
    # behaviour, so every existing cache still rebuilds bit-for-bit.
    band_lo_inclusive=False,
    # cap the rows sitting exactly AT sic_band_lo to this fraction of each timestep's kept rows, so
    # the mass point can be included at its deployment proportion instead of swamping the band
    zero_cap_frac=None,
    rollout_k=3,
    history_h=3,
    anchor_stride_days=None,
    samples_per_day=5120,
    sampling="uniform",
    edge_eps=0.05,
    edge_sharpness=4.0,
    tau_raw=1e-3,
    seed=0,
    spatial_block=2,
    lat_slice_lo=140,
    lat_slice_hi=180,
    include_rld=True,
    include_tdps=True,
    spherical_gradient_cross=False,
    gradient_basis_cross=False,
    cross_offset_cells=1,
    curvature_cross=False,
    curvature_tanh_scale=0.0,
    # [2026-10-15] ISOTROPIC curvature, the standard five-point Laplacian mean
    # ((N-C)+(S-C)+(W-C)+(E-C))/4 in place of the zonal-only ((W-C)+(E-C))/2.
    #
    # The zonal-only choice was argued from which mode is EXCITED: the artefact is radial, so only
    # the zonal mode appears, and meridian convergence makes zonal cells the small ones. That part is
    # still true. What was never checked is whether the component DAMPS a meridional mode when one is
    # present, and it does not: measured on the shipped weights, the zonal Nyquist checkerboard decays
    # to 0.65-0.79 while the meridional one passes at 0.89-0.95 and in the south at small amplitude it
    # GROWS at 1.0128. The blindness the original comment called symmetric was only half corrected.
    #
    # SCALE. For a pure ZONAL checkerboard the four-point mean reads HALF what the two-point zonal
    # form reads, because the two meridional differences are zero and are still averaged in. Halving
    # the tanh scale alongside this option therefore leaves the zonal response exactly as it is today
    # while adding the meridional one: a zonal checkerboard still maps to 0.964.
    curvature_isotropic=False,
    # [2026-10-15] curvature_laplacian: the PHYSICAL Laplacian, not the index-space one.
    #
    # On a latitude-longitude grid the two second differences are not on the same footing, because
    # two degrees of longitude is a shorter arc than two degrees of latitude everywhere off the
    # equator. The physical operator weights the zonal term by 1/cos^2(lat), which at the latitudes
    # the ice occupies is 4.0 at 60 degrees, 8.5 at 70 and 33 at 80. So an EQUAL-weight four-point
    # mean is the anisotropic choice here, and the zonal-only column is the physical Laplacian to
    # leading order. The same arithmetic in reverse: a zonal 2-cell mode is 445 km at 60 degrees and
    # 154 km at 80, while a meridional 2-cell mode is 890 km at every latitude, so the two are not
    # the same structure and only the zonal one is genuinely grid-scale where the ice is.
    #
    # This option adds the meridional term at its CORRECT relative weight, cos^2(lat), so the column
    # is the physical Laplacian rather than an approximation to it:
    #     ((W-C)+(E-C))/2  +  cos^2(lat) * ((N-C)+(S-C))/2
    # Normalised on the zonal coefficient, so a zonal checkerboard still reads 1.0 and the existing
    # tanh scale of 0.5 carries over unchanged. The meridional contribution is 12 per cent at 70
    # degrees and 3 per cent at 80, which is what the geometry says it should be.
    curvature_laplacian=False,
    q4_cross=False,
):
    files = sorted(Path(f) for f in files)
    if len(files) == 0:
        raise ValueError("No files provided.")

    hemisphere = hemisphere.lower()
    window_days = int(window_days)
    rollout_k = int(rollout_k)
    history_h = int(history_h)
    # [2026-09-20] anchor_stride_days: spacing between consecutive training anchors, in DAYS.
    # None or == window_days reproduces the original non-overlapping ("tumbling") behaviour exactly.
    # A smaller stride builds the block means on a 1-day sliding window instead, so anchors can start
    # on any day while ONE STEP still spans window_days -- the target stays a window_days change and
    # history/future stay at +-window_days, i.e. the physical meaning of y is unchanged. Motivated by
    # the unsmoothed cache's severe, SIC-dependent sample loss in the marginal ice zone (south
    # [0.10,0.30) -71%, [0.05,0.10) -79%, [0.00,0.02) -96%): overlapping anchors multiply the scarce
    # marginal-zone samples without fabricating any data or altering band definitions.
    # NOTE: this affects TRAINING CACHES ONLY. Rollout/inference uses rollout_lib.py's own separate
    # tumbling-mean path and stays strictly non-overlapping.
    anchor_stride_days = int(window_days if anchor_stride_days is None else anchor_stride_days)
    if anchor_stride_days < 1 or anchor_stride_days > window_days:
        raise ValueError("anchor_stride_days must be in [1, window_days].")
    _sliding = anchor_stride_days != window_days
    block_stride = 1 if _sliding else window_days       # days between consecutive block indices
    step = window_days // block_stride                  # block indices spanned by one window_days step
    astride = anchor_stride_days // block_stride        # block indices between consecutive anchors
    spatial_block = int(spatial_block)
    if not (0.0 <= sic_band_lo < sic_band_hi <= 1.0):
        raise ValueError("Require 0 <= sic_band_lo < sic_band_hi <= 1.")

    def tumbling_mean_forward(arr):
        T = arr.shape[0]
        n_blocks = T // window_days
        trimmed = arr[: n_blocks * window_days]
        finite = np.isfinite(trimmed)
        filled = np.where(finite, trimmed, 0.0).astype(np.float32)
        shape = (n_blocks, window_days) + arr.shape[1:]
        filled_blocks = filled.reshape(shape)
        nan_count = (~finite).reshape(shape).sum(axis=1)
        out = (filled_blocks.sum(axis=1) / window_days).astype(np.float32)
        out[nan_count > 0] = np.nan
        return out

    def sliding_mean_forward(arr):
        """1-day-stride running mean over window_days, NaN-propagating like the tumbling version
        (a window containing any NaN becomes NaN). Returns T - window_days + 1 blocks."""
        T = arr.shape[0]
        finite = np.isfinite(arr)
        filled = np.where(finite, arr, 0.0).astype(np.float64)
        csum = np.concatenate([np.zeros((1,) + arr.shape[1:], dtype=np.float64),
                               np.cumsum(filled, axis=0)], axis=0)
        cnan = np.concatenate([np.zeros((1,) + arr.shape[1:], dtype=np.int64),
                               np.cumsum((~finite).astype(np.int64), axis=0)], axis=0)
        out = ((csum[window_days:] - csum[:-window_days]) / window_days).astype(np.float32)
        out[(cnan[window_days:] - cnan[:-window_days]) > 0] = np.nan
        return out

    def block_mean_forward(arr):
        return sliding_mean_forward(arr) if _sliding else tumbling_mean_forward(arr)

    def spatial_block_average_strict(arr):
        T, nlat, nlon = arr.shape
        nlat_c = nlat // spatial_block
        nlon_c = nlon // spatial_block
        trimmed = arr[:, : nlat_c * spatial_block, : nlon_c * spatial_block]
        finite = np.isfinite(trimmed)
        filled = np.where(finite, trimmed, 0.0).astype(np.float32)
        shape = (T, nlat_c, spatial_block, nlon_c, spatial_block)
        filled_blocks = filled.reshape(shape)
        nan_count = (~finite).reshape(shape).sum(axis=(2, 4))
        out = (filled_blocks.sum(axis=(2, 4)) / (spatial_block * spatial_block)).astype(np.float32)
        out[nan_count > 0] = np.nan
        return out

    def spatial_block_average_plain(arr2d):
        nlat, nlon = arr2d.shape
        nlat_c = nlat // spatial_block
        nlon_c = nlon // spatial_block
        trimmed = arr2d[: nlat_c * spatial_block, : nlon_c * spatial_block]
        shape = (nlat_c, spatial_block, nlon_c, spatial_block)
        return trimmed.reshape(shape).mean(axis=(1, 3)).astype(np.float32)

    def spatial_gaussian_smooth_coarse(arr):
        if spatial_sigma == 0:
            return arr
        finite = np.isfinite(arr)
        filled = np.where(finite, arr, 0.0).astype(np.float32)
        weight = finite.astype(np.float32)
        sig = (0, spatial_sigma, spatial_sigma)
        mode = ("nearest", "nearest", "wrap")
        smoothed_filled = gaussian_filter(filled, sigma=sig, mode=mode)
        smoothed_weight = gaussian_filter(weight, sigma=sig, mode=mode)
        with np.errstate(invalid="ignore", divide="ignore"):
            out = smoothed_filled / smoothed_weight
        out = np.where(smoothed_weight >= spatial_min_valid_weight, out, np.nan)
        return out.astype(np.float32)

    var_names = ["siconc", "tos", "rsds", "rls", "sfcWind", "tas"]
    if include_rld:
        var_names = var_names + ["rld"]
    if include_tdps:
        var_names = var_names + ["tdps"]
    smoothed = {}
    lsm_static = None
    lat = None
    T = None

    for vname in tqdm(var_names, desc=f"loading+coarsening+smoothing(2deg) -> {Path(out_path).name}"):
        parts = []
        for path in files:
            with xr.open_dataset(path) as ds:
                arr = ds[vname].values[:, lat_slice_lo:lat_slice_hi, :].astype(np.float32)
                parts.append(arr)
                if lat is None:
                    lat = ds["latitude"].values[lat_slice_lo:lat_slice_hi].astype(np.float32)
                if lsm_static is None:
                    lsm_val = ds["lsm"].values.astype(np.float32)
                    if lsm_val.ndim == 3:
                        lsm_val = lsm_val[0]
                    lsm_static = lsm_val[lat_slice_lo:lat_slice_hi, :]
        raw = np.concatenate(parts, axis=0)
        del parts
        if T is None:
            T = raw.shape[0]
        elif raw.shape[0] != T:
            raise RuntimeError(f"Inconsistent time length for {vname}: {raw.shape[0]} vs {T}")
        if T < window_days * (rollout_k + history_h + 1):
            raise RuntimeError(
                f"Not enough days ({T}) for {window_days}-day tumbling blocks, a {rollout_k}-step "
                f"rollout, and a {history_h}-block history window."
            )
        spatial_coarse = spatial_block_average_strict(raw)
        del raw
        spatial_coarse_smooth = spatial_gaussian_smooth_coarse(spatial_coarse)
        del spatial_coarse
        smoothed[vname] = block_mean_forward(spatial_coarse_smooth)
        del spatial_coarse_smooth

    lat_coarse = lat.reshape(-1, spatial_block).mean(axis=1).astype(np.float32)
    lsm_coarse = spatial_block_average_plain(lsm_static)

    sin_parts, cos_parts = [], []
    for path in files:
        with xr.open_dataset(path) as ds:
            doy = ds["time"].dt.dayofyear.values.astype(np.float32)
            sin_parts.append(np.sin(2.0 * np.pi * doy / 366.0).astype(np.float32))
            cos_parts.append(np.cos(2.0 * np.pi * doy / 366.0).astype(np.float32))
    doy_sin = np.concatenate(sin_parts, axis=0)
    doy_cos = np.concatenate(cos_parts, axis=0)
    sin_s = block_mean_forward(doy_sin)
    cos_s = block_mean_forward(doy_cos)

    siconc_s = smoothed["siconc"]
    tos_s = smoothed["tos"]
    rsds_s = smoothed["rsds"]
    rls_s = smoothed["rls"]
    wind_s = smoothed["sfcWind"]
    tas_s = smoothed["tas"]
    rld_s = smoothed["rld"] if include_rld else None
    tdps_s = smoothed["tdps"] if include_tdps else None

    n_days_s = siconc_s.shape[0]
    anchor_lo = history_h * step
    anchor_hi = n_days_s - rollout_k * step
    print(
        f"building cache -> {Path(out_path).name}: {T} raw days -> {n_days_s} blocks "
        f"({window_days} days/block) -> anchor range [{anchor_lo}, {anchor_hi}); "
        f"grid coarsened to {siconc_s.shape[1]}x{siconc_s.shape[2]} "
        f"({spatial_block}x{spatial_block} degree cells) + Gaussian sigma={spatial_sigma}; "
        f"rollout_k={rollout_k}, history_h={history_h}, anchor_stride_days={anchor_stride_days}"
        f" (sliding={_sliding}); keeping {sic_band_lo}<SIC<{sic_band_hi} at anchor; "
        f"include_rld={include_rld}; include_tdps={include_tdps}; spherical_gradient_cross={spherical_gradient_cross}; "
        f"gradient_basis_cross={gradient_basis_cross}"
    )

    ocean_frac = np.clip(1.0 - np.nan_to_num(lsm_coarse, nan=1.0), 0.0, 1.0).astype(np.float32)

    if hemisphere in ("north", "arctic"):
        hemi_1d = lat_coarse >= lat_limit
    elif hemisphere in ("south", "antarctic"):
        hemi_1d = lat_coarse <= -lat_limit
    else:
        raise ValueError("hemisphere must be 'north'/'arctic' or 'south'/'antarctic'")

    static_mask = np.broadcast_to(hemi_1d[:, None], ocean_frac.shape) & (ocean_frac > 0)

    rng = np.random.default_rng(seed)
    X_list, y_list, ocean_list, X_future_list, sic_hist_list, cross_list = [], [], [], [], [], []
    n_dropped_outside_band = 0

    n_phys = 6 + int(include_rld) + int(include_tdps)

    for t in tqdm(range(anchor_lo, anchor_hi, astride), desc="sampling anchor blocks"):
        sic = siconc_s[t]
        sic_next = siconc_s[t + step]
        tos_t = tos_s[t]
        rsds_t = rsds_s[t]
        rls_t = rls_s[t]
        wind_t = wind_s[t]
        tas_t = tas_s[t]
        rld_t = rld_s[t] if include_rld else None
        tdps_t = tdps_s[t] if include_tdps else None

        finite_all = (
            static_mask
            & np.isfinite(sic)
            & np.isfinite(tos_t)
            & np.isfinite(rsds_t)
            & np.isfinite(rls_t)
            & np.isfinite(wind_t)
            & np.isfinite(tas_t)
            & np.isfinite(sic_next)
        )
        if include_rld:
            finite_all = finite_all & np.isfinite(rld_t)
        if include_tdps:
            finite_all = finite_all & np.isfinite(tdps_t)
        band_mask = ((sic >= sic_band_lo) if band_lo_inclusive else (sic > sic_band_lo)) & (sic < sic_band_hi)

        finite_future = np.ones_like(finite_all)
        for k in range(1, rollout_k + 1):
            fk = (
                np.isfinite(siconc_s[t + k * step])
                & np.isfinite(tos_s[t + k * step])
                & np.isfinite(rsds_s[t + k * step])
                & np.isfinite(rls_s[t + k * step])
                & np.isfinite(wind_s[t + k * step])
                & np.isfinite(tas_s[t + k * step])
            )
            if include_rld:
                fk = fk & np.isfinite(rld_s[t + k * step])
            if include_tdps:
                fk = fk & np.isfinite(tdps_s[t + k * step])
            finite_future &= fk

        finite_hist = np.ones_like(finite_all)
        for k in range(1, history_h + 1):
            finite_hist &= np.isfinite(siconc_s[t - k * step])

        valid = finite_all & band_mask & finite_future & finite_hist
        n_dropped_outside_band += int((finite_all & finite_future & finite_hist & ~band_mask).sum())

        flat = np.flatnonzero(valid)
        if flat.size == 0:
            continue

        if zero_cap_frac is not None and band_lo_inclusive:
            at_lo = sic.ravel()[flat] <= sic_band_lo
            n_above = int((~at_lo).sum())
            f = float(zero_cap_frac)
            target = int(round(f / (1.0 - f) * n_above)) if f < 1.0 else at_lo.sum()
            if int(at_lo.sum()) > target:
                keep_lo = rng.choice(flat[at_lo], size=target, replace=False)
                flat = np.sort(np.concatenate([flat[~at_lo], keep_lo]))

        if samples_per_day is None or samples_per_day <= 0 or samples_per_day >= flat.size:
            chosen = flat
        else:
            k = int(samples_per_day)
            if sampling == "edge":
                y_raw_candidates = (sic_next - sic).ravel()[flat]
                w = edge_eps + edge_sharpness * (np.abs(y_raw_candidates) / (tau_raw + 1e-12)) ** 1.5
                w = np.asarray(w, dtype=np.float64)
                w = np.maximum(w, 1e-12)
                w = w / w.sum()
                chosen = rng.choice(flat, size=k, replace=(flat.size < k), p=w)
            else:
                chosen = rng.choice(flat, size=k, replace=(flat.size < k))

        i, j = np.unravel_index(chosen, sic.shape)

        # [2026-09-24] cross_offset_cells: how many COARSE CELLS away the neighbours sit. The
        # default 1 reproduces the historical behaviour bit-for-bit. Its purpose is to make the
        # cross feature's differencing baseline a fixed ANGULAR width instead of "one grid cell",
        # so the same feature (and hence the same weights) can be served on a different grid:
        # at spatial_block=2 one cell is 2 deg so offset 1 gives a 2 deg baseline, at
        # spatial_block=1 offset 2 gives the same 2 deg baseline. This mirrors what the online
        # MLSeaIceV4Interp already does (grad_offset_deg=2.0, a fixed angular offset sampled by
        # interpolation rather than by grid index). NOTE: no division by the baseline is applied
        # -- with the baseline held at the same number of degrees the raw centred differences are
        # already in the same per-2-deg units on both grids, so existing normalization statistics
        # transfer unchanged; dividing would only rescale by a constant the normalization absorbs.
        oc = int(cross_offset_cells)
        if oc < 1:
            raise ValueError("cross_offset_cells must be >= 1")
        sic_latprev = np.empty_like(sic); sic_latprev[oc:] = sic[:-oc]; sic_latprev[:oc] = sic[:oc]
        sic_latnext = np.empty_like(sic); sic_latnext[:-oc] = sic[oc:]; sic_latnext[-oc:] = sic[-oc:]
        sic_lonprev = np.roll(sic, oc, axis=1)
        sic_lonnext = np.roll(sic, -oc, axis=1)

        center_vals = sic[i, j]

        def diff_or_zero(neighbor_grid):
            vals = neighbor_grid[i, j]
            d = vals - center_vals
            return np.where(np.isfinite(vals), d, 0.0).astype(np.float32)

        lat_diff = diff_or_zero(sic_latprev), diff_or_zero(sic_latnext)
        lon_diff = diff_or_zero(sic_lonprev), diff_or_zero(sic_lonnext)
        if spherical_gradient_cross:
            # true spherical-gradient components: meridional (N/S) distance R*dphi is constant
            # everywhere on this grid (no correction needed beyond a shared constant that per-
            # feature normalization absorbs anyway); zonal (E/W) distance R*cos(lat)*dlambda
            # shrinks toward the pole (meridian convergence), so a raw same-grid-step E/W SIC
            # difference represents an increasingly larger true gradient there. Since dphi=dlambda
            # on this grid, the two differ only by this 1/cos(lat) factor -- dividing E/W by
            # cos(lat) converts the raw grid-index difference into the same physical units as the
            # (unscaled) N/S difference. See project memory: the client's hypothesis that the
            # online "spoke" artifact traces to the model's disproportionate N/S vs E/W input
            # sensitivity (measured ~2.6-2.8x), itself plausibly a consequence of feeding the model
            # a raw grid-index difference for E/W that understates the true zonal gradient near
            # the pole relative to the meridional one.
            coslat = np.maximum(np.cos(np.deg2rad(lat_coarse[i])), 1e-3).astype(np.float32)
            lon_diff = (lon_diff[0] / coslat, lon_diff[1] / coslat)

        if gradient_basis_cross:
            # N and S (and separately W and E) are highly anti-correlated in the real data (r=-0.6
            # to -0.89 measured directly) -- each pair is largely two redundant, opposite-signed
            # views of the SAME local axis gradient. Feeding the raw 4-column [N,S,W,E] leaves the
            # network to discover the (N-S)/(W-E) contrast itself from two separately-normalized,
            # collinear inputs -- measured to actually fail at this (model's own directional
            # sensitivity ratio ~1.0 vs the data's true ~2-60x N-S-over-W-E ratio, see project
            # memory). Precomputing the clean centered-difference gradient per axis removes the
            # redundancy and hands the network the same quantity a proper finite-difference gradient
            # estimator would use directly: grad_NS = (N-S)/2 ~ d(SIC)/d(lat) direction,
            # grad_WE = (W-E)/2 ~ d(SIC)/d(lon) direction (further divided by cos(lat) above if
            # spherical_gradient_cross is also set, applied before this combination). 2 output
            # columns instead of 4.
            grad_ns = (lat_diff[0] - lat_diff[1]) / 2.0
            grad_we = (lon_diff[0] - lon_diff[1]) / 2.0
            if q4_cross:
                # [2026-09-29] Third cross column = the NORMALIZED 4th zonal difference
                #     q = (f_-2 - 4 f_-1 + 6 f_0 - 4 f_+1 + f_+2) / 16
                # in place of the plain curvature (W+E)/2 - C. Four properties make it the better
                # channel for detecting the grid-scale mode the centred difference is blind to, all
                # verified numerically:
                #  1. it is the IDENTITY on the 2-cell mode at any resolution, so nothing has to be
                #     rescaled when the weights move to a finer grid -- the quantity it measures is
                #     resolution-invariant. The curvature needed (2/dx)^2, and applying that at 1
                #     degree put the input 87 sigma out and collapsed the rollout.
                #  2. its coefficients have |sum| = 1, so |q| <= 0.5 for any SIC field on any grid.
                #     A HARD bound, which means synthetic constraint points drawn from [-0.5, 0.5]
                #     cover every input the model can ever receive on this channel.
                #  3. it is exactly zero on any linear slope, so a monotone ice edge does not excite
                #     it and the |curv| > |grad| extremum mask the curvature version needed is no
                #     longer necessary.
                #  4. its response falls off as sin^4 rather than sin^2: measured on ERA5, only 6.2%
                #     of its variance comes from wavelengths above 3.5 cells, against 22.6% for the
                #     curvature -- 3.6x less contamination by structure the model should keep.
                # Known limitation: a physical isolated one-cell feature (a polynya at 1 degree) is
                # only 1.33x weaker through this operator than a full checkerboard of the same SIC
                # contrast. No local operator separates those -- telling them apart needs a wider
                # stencil or time information -- so some real small-scale structure will be damped.
                c = sic[i, j]
                def at(shift):
                    return np.roll(sic, -shift, axis=1)[i, j]
                vals = [at(-2), at(-1), c, at(1), at(2)]
                w = [1.0, -4.0, 6.0, -4.0, 1.0]
                q = sum(wi * np.where(np.isfinite(v), v, c) for wi, v in zip(w, vals)) / 16.0
                cross_diff = np.stack([grad_ns, grad_we, q.astype(np.float32)], axis=1)
            elif curvature_cross:
                # [2026-09-29] The centred difference (W-E)/2 ANNIHILATES the 2-cell zonal mode: an
                # ice/no-ice/ice/no-ice alternation along a latitude circle has W == E at every
                # point, so the feature reads exactly 0 there. The model therefore cannot perceive
                # that mode and cannot damp it, and in the coupled T127 run it grows into the visible
                # radial "spoke" pattern (measured: at the 1% zonally-roughest Antarctic points the
                # seen component averages 0.0139 while the unseen (W+E)/2 - C averages 0.7212, and
                # the Nyquist wavenumber carries 1.79x the power of its neighbours).
                # Adding the zonal curvature restores exactly the component the gradient basis threw
                # away when it replaced the raw [N,S,W,E] columns, WITHOUT reintroducing their
                # collinearity: {gradient, curvature} is an orthogonal decomposition of the same
                # information, whereas raw N and S are two anti-correlated views of one axis (which
                # the model measurably failed to combine, see the gradient_basis_cross comment).
                # It is also the Laplacian a diffusion operator uses, so it gives the network the
                # means to learn that a grid-scale checkerboard should relax.
                # Only the ZONAL curvature is added: the blindness is symmetric in the two axes, but
                # only the zonal mode is actually excited (the artifact is radial, i.e. alternating
                # along longitude), and meridian convergence makes zonal cells the small ones. A
                # meridional column would add distribution shift for no demonstrated need.
                if curvature_laplacian:
                    _cl2 = np.maximum(np.cos(np.deg2rad(lat_coarse[i])), 1e-3).astype(np.float32) ** 2
                    curv_we = ((lon_diff[0] + lon_diff[1]) / 2.0
                               + _cl2 * (lat_diff[0] + lat_diff[1]) / 2.0)
                elif curvature_isotropic:
                    curv_we = (lat_diff[0] + lat_diff[1] + lon_diff[0] + lon_diff[1]) / 4.0
                else:
                    curv_we = (lon_diff[0] + lon_diff[1]) / 2.0
                if curvature_tanh_scale > 0:
                    # [2026-10-01] Bound the curvature column with tanh(c / s).
                    #
                    # The gradient column is rescaled to a fixed 2-degree arc because advection is
                    # physical and has a physical speed. The curvature column is NOT rescaled,
                    # because what it damps is grid-scale structure and a 2-cell checkerboard has
                    # per-cell curvature +-1 at ANY resolution. Being grid-relative, its magnitude
                    # carries no physical meaning across grids, so there is no reason to leave it
                    # unbounded -- and a measured reason not to: in the training caches the column
                    # never exceeds |c| = 0.37 (std 0.015-0.056, p99.9 0.18-0.29) because ERA5 at
                    # 2 degrees carries no grid-scale structure, while the checkerboard the column
                    # exists to damp sits at |c| = 1. The linear column is therefore extrapolated
                    # about 3x beyond anything it was trained on, which is where the unconstrained
                    # curvature weights learned to AMPLIFY the Nyquist mode (growth 1.40).
                    #
                    # tanh keeps the trained range almost linear and saturates the extrapolation:
                    # at s = 0.5, p99 of the training data is compressed by under 1%, the largest
                    # training value by 15%, and a checkerboard maps to 0.964. The column is then
                    # bounded at every resolution by construction.
                    curv_we = np.tanh(curv_we / curvature_tanh_scale).astype(np.float32)
                cross_diff = np.stack([grad_ns, grad_we, curv_we], axis=1)
            else:
                cross_diff = np.stack([grad_ns, grad_we], axis=1)
        else:
            cross_diff = np.stack([lat_diff[0], lat_diff[1], lon_diff[0], lon_diff[1]], axis=1)

        # column order MUST match sic_pipeline_lib.build_feature_matrix_from_raw's fixed indices:
        # [sic(0),tos(1),rsds(2),rls(3),wind(4),tas(5),doy_sin(6),doy_cos(7),rld(8),tdps(9)] -- rld/
        # tdps appended AFTER doy_sin/doy_cos, in that order, so the original 8-column production
        # layout is preserved exactly (existing caches/checkpoints depend on those first 8 indices
        # staying put).
        x_time = np.tile(np.array([sin_s[t], cos_s[t]], dtype=np.float32)[None, :], (len(chosen), 1))
        x_phys_base = np.stack(
            [sic[i, j], tos_t[i, j], rsds_t[i, j], rls_t[i, j], wind_t[i, j], tas_t[i, j]], axis=1
        ).astype(np.float32)
        parts = [x_phys_base, x_time]
        if include_rld:
            parts.append(rld_t[i, j][:, None].astype(np.float32))
        if include_tdps:
            parts.append(tdps_t[i, j][:, None].astype(np.float32))
        x = np.concatenate(parts, axis=1).astype(np.float32)
        y = (sic_next[i, j] - sic[i, j]).astype(np.float32)
        ocean = ocean_frac[i, j].astype(np.float32)

        x_future = np.empty((len(chosen), rollout_k, n_phys + 2), dtype=np.float32)
        for k in range(1, rollout_k + 1):
            x_time_k = np.tile(
                np.array([sin_s[t + k * step], cos_s[t + k * step]], dtype=np.float32)[None, :], (len(chosen), 1)
            )
            x_phys_base_k = np.stack(
                [
                    siconc_s[t + k * step][i, j], tos_s[t + k * step][i, j], rsds_s[t + k * step][i, j],
                    rls_s[t + k * step][i, j], wind_s[t + k * step][i, j], tas_s[t + k * step][i, j],
                ],
                axis=1,
            ).astype(np.float32)
            parts_k = [x_phys_base_k, x_time_k]
            if include_rld:
                parts_k.append(rld_s[t + k * step][i, j][:, None].astype(np.float32))
            if include_tdps:
                parts_k.append(tdps_s[t + k * step][i, j][:, None].astype(np.float32))
            x_future[:, k - 1, :] = np.concatenate(parts_k, axis=1)

        sic_hist = np.empty((len(chosen), history_h), dtype=np.float32)
        for k in range(1, history_h + 1):
            sic_hist[:, history_h - k] = siconc_s[t - k * step][i, j]

        X_list.append(x)
        y_list.append(y)
        ocean_list.append(ocean)
        X_future_list.append(x_future)
        sic_hist_list.append(sic_hist)
        cross_list.append(cross_diff)

    if len(X_list) == 0:
        raise RuntimeError("No samples were collected. Check hemisphere/lat_limit/masks/band cut.")

    X = np.concatenate(X_list, axis=0).astype(np.float32)
    y = np.concatenate(y_list, axis=0).astype(np.float32)
    ocean = np.concatenate(ocean_list, axis=0).astype(np.float32)
    X_future = np.concatenate(X_future_list, axis=0).astype(np.float32)
    sic_hist = np.concatenate(sic_hist_list, axis=0).astype(np.float32)
    sic_cross_diff = np.concatenate(cross_list, axis=0).astype(np.float32)

    print(f"pixel-blocks dropped by the SIC<={sic_band_lo} or SIC>={sic_band_hi} cut: {n_dropped_outside_band}")

    y_mean = y.mean().astype(np.float32)
    y_std = y.std().astype(np.float32)
    y_std = np.maximum(y_std, 1e-6)
    y = ((y - y_mean) / y_std).astype(np.float32)

    mean = X[:, :6].mean(axis=0).astype(np.float32)
    std = X[:, :6].std(axis=0).astype(np.float32)
    std = np.maximum(std, 1e-6)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        X=X, y=y, ocean=ocean, X_future=X_future, sic_hist=sic_hist, sic_cross_diff=sic_cross_diff,
        mean=mean, std=std, y_mean=y_mean, y_std=y_std,
        hemisphere=np.array(hemisphere), lat_limit=np.array(lat_limit),
        samples_per_day=np.array(-1 if samples_per_day is None else samples_per_day),
        window_days=np.array(window_days), spatial_block=np.array(spatial_block), spatial_sigma=np.array(spatial_sigma),
        rollout_k=np.array(rollout_k), history_h=np.array(history_h),
        anchor_stride_days=np.array(anchor_stride_days),
        sic_band_lo=np.array(sic_band_lo), sic_band_hi=np.array(sic_band_hi),
        include_rld=np.array(include_rld), include_tdps=np.array(include_tdps),
        spherical_gradient_cross=np.array(spherical_gradient_cross),
        gradient_basis_cross=np.array(gradient_basis_cross),
        curvature_cross=np.array(curvature_cross),
        curvature_tanh_scale=np.array(curvature_tanh_scale),
        curvature_isotropic=np.array(curvature_isotropic),
        curvature_laplacian=np.array(curvature_laplacian),
        q4_cross=np.array(q4_cross),
        cross_offset_cells=np.array(cross_offset_cells),
        cross_offset_deg=np.array(cross_offset_cells * spatial_block * 1.0),
    )

    print(f"saved cache -> {out_path.resolve()}")
    print("X shape:", X.shape, "y shape:", y.shape, "X_future shape:", X_future.shape,
          "sic_hist shape:", sic_hist.shape, "sic_cross_diff shape:", sic_cross_diff.shape)
    return out_path
