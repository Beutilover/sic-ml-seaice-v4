"""
Shared library for the sea-ice concentration (SIC) parameterisation framework.

Provides neural-network architectures, feature construction, dataset and preprocessing utilities, training-cache generation, and baseline training routines.

Supports atmospheric and thermodynamic inputs, including downward longwave radiation (`rld`), with consistent feature ordering and normalisation across training and evaluation.

The raw input layout is:

`[sic, tos, rsds, rls, wind, tas, doy_sin, doy_cos, rld]`

The module serves as the common implementation layer for SIC model training, evaluation, and deployment preparation.
"""
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

SEED = 42
EPOCHS = 40
BATCH_SIZE = 6000
LR = 1e-3
WEIGHT_DECAY = 1e-4
TAU_FLOOR = 1.0
DT = 5.0
HISTORY_H = 3


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# Feature resolution / normalization
# ---------------------------------------------------------------------------
FEATURE_ORDER = ["siconc", "tos", "rsds", "rls", "sfcwind", "tas", "doy_sin", "doy_cos", "rld", "tdps"]
FEATURE_TO_COL = {name: i for i, name in enumerate(FEATURE_ORDER)}


def resolve_feature_set(feature_set):
    if isinstance(feature_set, str):
        key = feature_set.lower().strip()
        feats = [x.strip() for x in key.split("+") if x.strip()]
    else:
        feats = [str(x).lower().strip() for x in feature_set]
    out = []
    seen = set()
    for f in feats:
        if f in ("sic", "siconc"):
            f = "siconc"
        if f in ("sst", "tos"):
            f = "tos"
        if f in ("wind", "sfcwind", "speed"):
            f = "sfcwind"
        if f in ("t2m", "tas"):
            f = "tas"
        if f not in FEATURE_TO_COL and f not in ("doy_frac", "radsum"):
            raise ValueError(f"Unknown feature {f!r}")
        if f not in seen:
            out.append(f)
            seen.add(f)
    return out


def build_feature_matrix_from_raw(X_raw, feature_set="all", normalize_phys=True, mean=None, std=None,
                                   radsum_mean=None, radsum_std=None, rld_mean=None, rld_std=None,
                                   tdps_mean=None, tdps_std=None):
    feats = resolve_feature_set(feature_set)
    cols = []
    for f in feats:
        if f == "siconc":
            col = X_raw[:, 0]
            if normalize_phys and mean is not None: col = (col - mean[0]) / max(std[0], 1e-6)
        elif f == "tos":
            col = X_raw[:, 1]
            if normalize_phys and mean is not None: col = (col - mean[1]) / max(std[1], 1e-6)
        elif f == "rsds":
            col = X_raw[:, 2]
            if normalize_phys and mean is not None: col = (col - mean[2]) / max(std[2], 1e-6)
        elif f == "rls":
            col = X_raw[:, 3]
            if normalize_phys and mean is not None: col = (col - mean[3]) / max(std[3], 1e-6)
        elif f == "radsum":
            col = X_raw[:, 2] + X_raw[:, 3]
            if normalize_phys and radsum_mean is not None: col = (col - radsum_mean) / max(radsum_std, 1e-6)
        elif f == "sfcwind":
            col = X_raw[:, 4]
            if normalize_phys and mean is not None: col = (col - mean[4]) / max(std[4], 1e-6)
        elif f == "tas":
            col = X_raw[:, 5]
            if normalize_phys and mean is not None: col = (col - mean[5]) / max(std[5], 1e-6)
        elif f == "doy_sin":
            col = X_raw[:, 6]
        elif f == "doy_cos":
            col = X_raw[:, 7]
        elif f == "doy_frac":
            phase = np.mod(np.arctan2(X_raw[:, 6], X_raw[:, 7]), 2.0 * np.pi)
            col = phase / (2.0 * np.pi)
        elif f == "rld":
            # rld, when present, always immediately follows doy_cos (raw col 8) regardless of
            # whether tdps also follows it -- see build_cache_v2.py's concat order.
            col = X_raw[:, 8]
            if normalize_phys and rld_mean is not None: col = (col - rld_mean) / max(rld_std, 1e-6)
        elif f == "tdps":
            # tdps, when present, is always the LAST raw column -- col 9 if rld is also present
            # (rld then tdps), col 8 if not (tdps alone right after doy_cos). Do not hardcode 9:
            # a tdps-without-rld cache (e.g. an rls-based control built for a fair rld-vs-rls mono
            # comparison) has tdps at col 8, and hardcoding 9 silently reads into rld's column /
            # out of bounds instead.
            col = X_raw[:, X_raw.shape[1] - 1]
            if normalize_phys and tdps_mean is not None: col = (col - tdps_mean) / max(tdps_std, 1e-6)
        else:
            raise ValueError(f"Unknown feature {f!r}")
        cols.append(np.asarray(col, dtype=np.float32))
    return np.stack(cols, axis=1).astype(np.float32), feats


# ---------------------------------------------------------------------------
# Dataset chain (verbatim from Modeltry22, extended for optional rld normalization)
# ---------------------------------------------------------------------------
class RelaxationRolloutMultiHistDataset(Dataset):
    def __init__(self, cache_path, feature_set="all", normalize_phys=True, indices=None, override_stats=None):
        cache = np.load(cache_path, allow_pickle=True)
        X_raw = cache["X"].astype(np.float32)
        y_cache = cache["y"].astype(np.float32)
        ocean = cache["ocean"].astype(np.float32)
        X_future_raw = cache["X_future"].astype(np.float32)
        sic_hist_raw = cache["sic_hist"].astype(np.float32)

        if indices is not None:
            X_raw, y_cache, ocean = X_raw[indices], y_cache[indices], ocean[indices]
            X_future_raw, sic_hist_raw = X_future_raw[indices], sic_hist_raw[indices]

        self.base_target_stats = (float(cache["y_mean"]), float(cache["y_std"]))
        self.mean = cache["mean"].astype(np.float32)
        self.std = cache["std"].astype(np.float32)
        # [2026-09-24] override_stats may now also carry "mean"/"std" (the 6 base physical columns)
        # and "base_target_stats", matching what PooledRelaxationDataset has always accepted. Needed
        # because those must be applied BEFORE build_feature_matrix_from_raw runs below -- assigning
        # ds.mean after construction (as make_loaders does) does not rebuild the already-normalized
        # X. Purely additive: absent keys leave the cache's own statistics in place, so every
        # existing caller is bit-for-bit unchanged. Motivated by the cross-resolution transfer test,
        # where a 1-degree evaluation cache must be served with the 2-degree training statistics.
        if override_stats is not None:
            if override_stats.get("mean") is not None:
                self.mean = np.asarray(override_stats["mean"], dtype=np.float32)
                self.std = np.asarray(override_stats["std"], dtype=np.float32)
            if override_stats.get("base_target_stats") is not None:
                cache_target_stats = self.base_target_stats
                self.base_target_stats = override_stats["base_target_stats"]
            else:
                cache_target_stats = None
        else:
            cache_target_stats = None
        self.sic_raw = X_raw[:, 0].astype(np.float32)
        self.tos_raw = X_raw[:, 1].astype(np.float32)
        self.ocean = ocean.astype(np.float32)
        self.X_future_raw = X_future_raw
        self.sic_hist = sic_hist_raw
        self.history_h = sic_hist_raw.shape[1]

        feats = resolve_feature_set(feature_set)
        # Read the explicit flags the cache was built with -- do NOT infer rld/tdps presence from
        # column count alone: a 9-column cache is ambiguous (could be "+rld, no tdps" like a
        # highband cache, or "no rld, +tdps" like an rls-based control built for a fair mono
        # comparison), and inferring wrong silently reads tdps data as if it were rld or vice versa.
        has_rld = bool(cache["include_rld"]) if "include_rld" in cache else X_raw.shape[1] > 8
        has_tdps = bool(cache["include_tdps"]) if "include_tdps" in cache else X_raw.shape[1] > 9
        tdps_col = X_raw.shape[1] - 1  # tdps, when present, is always the last raw column

        if override_stats is not None:
            self.radsum_mean = override_stats.get("radsum_mean")
            self.radsum_std = override_stats.get("radsum_std")
            self.rld_mean = override_stats.get("rld_mean")
            self.rld_std = override_stats.get("rld_std")
            self.tdps_mean = override_stats.get("tdps_mean")
            self.tdps_std = override_stats.get("tdps_std")
        else:
            self.radsum_mean = None; self.radsum_std = None
            if "radsum" in feats:
                radsum_all = (X_raw[:, 2] + X_raw[:, 3]).astype(np.float32)
                self.radsum_mean = float(radsum_all.mean())
                self.radsum_std = float(max(radsum_all.std(), 1e-6))
            self.rld_mean = None; self.rld_std = None
            if "rld" in feats and has_rld:
                rld_all = X_raw[:, 8].astype(np.float32)
                self.rld_mean = float(rld_all.mean())
                self.rld_std = float(max(rld_all.std(), 1e-6))
            self.tdps_mean = None; self.tdps_std = None
            if "tdps" in feats and has_tdps:
                # matches the established convention from hemisphere_unification_with_tdps_test.ipynb:
                # tdps norm stats computed from the SHIFTED (block t+1) values specifically, unlike
                # every other base feature's stats (which come from X_raw / current-block) -- this
                # is what the ShiftedForcingNoTos overwrite below actually feeds the model at train
                # time, so the stats should match that distribution, not the raw/current-block one.
                tdps_shifted = X_future_raw[:, 0, tdps_col].astype(np.float32)
                self.tdps_mean = float(tdps_shifted.mean())
                self.tdps_std = float(max(tdps_shifted.std(), 1e-6))

        self.X, self.feature_names = build_feature_matrix_from_raw(
            X_raw, feature_set=feature_set, normalize_phys=normalize_phys,
            mean=self.mean, std=self.std, radsum_mean=self.radsum_mean, radsum_std=self.radsum_std,
            rld_mean=self.rld_mean, rld_std=self.rld_std, tdps_mean=self.tdps_mean, tdps_std=self.tdps_std,
        )
        self.y = y_cache.astype(np.float32)
        if cache_target_stats is not None:
            # y is stored normalized by the CACHE's own (y_mean, y_std); convert back to physical
            # units and re-normalize with the overriding (training) statistics, exactly as
            # make_loaders does after the fact. See the 2026-09-18 note there.
            c_mean, c_std = cache_target_stats
            o_mean, o_std = self.base_target_stats
            self.y = ((self.y * c_std + c_mean - o_mean) / o_std).astype(np.float32)

        K = X_future_raw.shape[1]
        n_feat = len(self.feature_names)
        X_future_norm = np.zeros((X_future_raw.shape[0], K, n_feat), dtype=np.float32)
        for k in range(K):
            X_future_norm[:, k, :], _ = build_feature_matrix_from_raw(
                X_future_raw[:, k, :], feature_set=feature_set, normalize_phys=normalize_phys,
                mean=self.mean, std=self.std, radsum_mean=self.radsum_mean, radsum_std=self.radsum_std,
                rld_mean=self.rld_mean, rld_std=self.rld_std, tdps_mean=self.tdps_mean, tdps_std=self.tdps_std,
            )
        self.X_future = X_future_norm

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return (torch.from_numpy(self.X[idx]).float(), torch.tensor(self.y[idx], dtype=torch.float32),
                torch.tensor(self.ocean[idx], dtype=torch.float32), torch.tensor(self.sic_raw[idx], dtype=torch.float32),
                torch.tensor(self.tos_raw[idx], dtype=torch.float32), torch.from_numpy(self.sic_hist[idx]).float(),
                torch.from_numpy(self.X_future[idx]).float(), torch.from_numpy(self.X_future_raw[idx]).float())


CROSS_COL_NAMES = ["cross_latprev", "cross_latnext", "cross_lonprev", "cross_lonnext"]  # [N, S, W, E]
# a gradient-basis cache stores 2 columns [(N-S)/2, (W-E)/2], or 3 with the zonal curvature
# (W+E)/2 - C appended (build_cache_v2.py's curvature_cross). Names are labels only -- nothing
# downstream indexes cross columns by name -- but they should not read as raw neighbour columns.
CROSS_BASIS_NAMES = {2: ["cross_grad_ns", "cross_grad_we"],
                     3: ["cross_grad_ns", "cross_grad_we", "cross_curv_we"]}


class RelaxationRolloutMultiHistCrossDataset(RelaxationRolloutMultiHistDataset):
    def __init__(self, cache_path, feature_set="all", normalize_phys=True, indices=None, use_cross=True,
                 cross_cols=None, override_stats=None, override_cross=None):
        """cross_cols: optional list of indices into the cache's 4-wide sic_cross_diff array
        ([0]=latprev/N, [1]=latnext/S, [2]=lonprev/W, [3]=lonnext/E) to select a SUBSET of cross-
        neighbor directions -- e.g. cross_cols=[2,3] keeps only east/west, dropping north/south.
        None (default) keeps all 4, matching every prior experiment's behavior unchanged. Added to
        test the client's hypothesis that north/south cross-neighbor inputs specifically cause the
        online "spoke" streaking pattern (meridional neighbors become highly distorted/redundant
        near the pole on a lat/lon grid) -- see project memory."""
        super().__init__(cache_path, feature_set=feature_set, normalize_phys=normalize_phys, indices=indices, override_stats=override_stats)
        self.use_cross = bool(use_cross)
        self.cross_cols = cross_cols
        cache = np.load(cache_path, allow_pickle=True)
        cross_raw = cache["sic_cross_diff"].astype(np.float32)
        if cross_cols is not None:
            cross_raw = cross_raw[:, cross_cols]
        if indices is not None:
            cross_raw = cross_raw[indices]
        if override_cross is not None:
            self.cross_mean = override_cross["mean"]
            self.cross_std = override_cross["std"]
        else:
            self.cross_mean = cross_raw.mean(axis=0).astype(np.float32)
            self.cross_std = np.maximum(cross_raw.std(axis=0), 1e-6).astype(np.float32)
        cross_norm = ((cross_raw - self.cross_mean) / self.cross_std).astype(np.float32)
        if self.use_cross:
            self.X = np.concatenate([self.X, cross_norm], axis=1).astype(np.float32)
            n_cross = cross_norm.shape[1]
            zeros_future = np.zeros((self.X_future.shape[0], self.X_future.shape[1], n_cross), dtype=np.float32)
            self.X_future = np.concatenate([self.X_future, zeros_future], axis=2).astype(np.float32)
            names = ([CROSS_COL_NAMES[i] for i in cross_cols] if cross_cols is not None
                     else CROSS_BASIS_NAMES.get(n_cross, CROSS_COL_NAMES[:n_cross]))
            self.feature_names = self.feature_names + names


class RelaxationRolloutMultiHistCrossShiftedForcingNoTosDataset(RelaxationRolloutMultiHistCrossDataset):
    """Overwrites every feature column except siconc/tos with the block-(t+1) ("shifted forcing")
    value -- this is the convention every production checkpoint is actually trained under (see
    project memory: this was the root cause of a previously long-open CORR discrepancy)."""
    def __init__(self, cache_path, feature_set="all", normalize_phys=True, indices=None, use_cross=True,
                 cross_cols=None, override_stats=None, override_cross=None):
        super().__init__(cache_path, feature_set=feature_set, normalize_phys=normalize_phys, indices=indices,
                          use_cross=use_cross, cross_cols=cross_cols, override_stats=override_stats, override_cross=override_cross)
        future0_raw = self.X_future_raw[:, 0, :]
        future0_norm, base_feat_names = build_feature_matrix_from_raw(
            future0_raw, feature_set=feature_set, normalize_phys=normalize_phys,
            mean=self.mean, std=self.std, radsum_mean=self.radsum_mean, radsum_std=self.radsum_std,
            rld_mean=self.rld_mean, rld_std=self.rld_std, tdps_mean=self.tdps_mean, tdps_std=self.tdps_std,
        )
        sic_idx = base_feat_names.index("siconc")
        tos_idx = base_feat_names.index("tos")
        for i in range(len(base_feat_names)):
            if i == sic_idx or i == tos_idx:
                continue
            self.X[:, i] = future0_norm[:, i]


def make_loaders(train_cache_path, val_cache_path, feature_set="all", normalize_phys=True, batch_size=BATCH_SIZE, use_cross=True, cross_cols=None):
    train_ds = RelaxationRolloutMultiHistCrossShiftedForcingNoTosDataset(train_cache_path, feature_set=feature_set, normalize_phys=normalize_phys, use_cross=use_cross, cross_cols=cross_cols)
    override_stats = dict(radsum_mean=train_ds.radsum_mean, radsum_std=train_ds.radsum_std, rld_mean=train_ds.rld_mean, rld_std=train_ds.rld_std,
                           tdps_mean=train_ds.tdps_mean, tdps_std=train_ds.tdps_std)
    override_cross = dict(mean=train_ds.cross_mean, std=train_ds.cross_std)
    val_ds = RelaxationRolloutMultiHistCrossShiftedForcingNoTosDataset(val_cache_path, feature_set=feature_set, normalize_phys=normalize_phys, use_cross=use_cross, cross_cols=cross_cols, override_stats=override_stats, override_cross=override_cross)
    # val must also use TRAIN mean/std (not its own) -- production convention (see project memory)
    val_ds.mean = train_ds.mean; val_ds.std = train_ds.std
    # ...and the same must hold for the TARGET normalization (2026-09-18 fix). Previously only the
    # input stats above were overridden, so val's `y` stayed normalized by the VAL cache's own
    # (y_mean, y_std) while callers un-normalize with `train_ds.base_target_stats` -- an affine
    # distortion of the val targets (measured: train/val y_std ratio 0.90-1.06 across the caches in
    # this project, so a few % on absolute val RMSE, no effect on CORR, but it also biased the
    # val-loss used for best-epoch selection). `self.y` must be converted back to physical units
    # with the VAL cache's own stats first, then re-normalized with TRAIN's -- overriding
    # base_target_stats alone would only relabel the same inconsistency. The pooled path
    # (PooledRelaxationDataset) already did this correctly; only this single-dataset path (used by
    # the north-only high band) was affected.
    val_own_mean, val_own_std = val_ds.base_target_stats
    train_mean, train_std = train_ds.base_target_stats
    y_phys = val_ds.y.astype(np.float32) * val_own_std + val_own_mean
    val_ds.y = ((y_phys - train_mean) / train_std).astype(np.float32)
    val_ds.base_target_stats = train_ds.base_target_stats
    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=pin)
    return train_ds, val_ds, train_loader, val_loader


class PooledRelaxationDataset(Dataset):
    """Pools north+south raw samples with POOLED normalization stats (not per-hemisphere) -- the
    actual mechanism behind the adopted "hemisphere-unified" low/mid models (see project memory:
    hemisphere_unification_with_tdps_test.ipynb's PooledDatasetWithTdps, reconstructed here in the
    same persistent-library style as the rest of this file). Applies the same ShiftedForcingNoTos
    overwrite as the single-hemisphere dataset. Tracks `hemi_label` (0=north, 1=south) so a trained
    model can be evaluated separately per hemisphere afterward."""

    def __init__(self, north_cache_path, south_cache_path, feature_set="all", normalize_phys=True,
                 use_cross=True, cross_cols=None, override_stats=None, override_cross=None):
        """cross_cols: see RelaxationRolloutMultiHistCrossDataset's docstring -- optional subset of
        the 4 cross-neighbor directions ([0]=N,[1]=S,[2]=W,[3]=E), e.g. [2,3] for east/west only."""
        nc = np.load(north_cache_path, allow_pickle=True)
        sc = np.load(south_cache_path, allow_pickle=True)

        X_raw = np.concatenate([nc["X"].astype(np.float32), sc["X"].astype(np.float32)], axis=0)
        X_future_raw = np.concatenate([nc["X_future"].astype(np.float32), sc["X_future"].astype(np.float32)], axis=0)
        y_n = nc["y"].astype(np.float32) * float(nc["y_std"]) + float(nc["y_mean"])
        y_s = sc["y"].astype(np.float32) * float(sc["y_std"]) + float(sc["y_mean"])
        y_phys = np.concatenate([y_n, y_s], axis=0)
        ocean = np.concatenate([nc["ocean"].astype(np.float32), sc["ocean"].astype(np.float32)], axis=0)
        sic_hist = np.concatenate([nc["sic_hist"].astype(np.float32), sc["sic_hist"].astype(np.float32)], axis=0)
        cross_raw = np.concatenate([nc["sic_cross_diff"].astype(np.float32), sc["sic_cross_diff"].astype(np.float32)], axis=0)
        if cross_cols is not None:
            cross_raw = cross_raw[:, cross_cols]
        self.cross_cols = cross_cols
        self.hemi_label = np.array([0] * len(nc["X"]) + [1] * len(sc["X"]))

        self.sic_raw = X_raw[:, 0].astype(np.float32)
        self.tos_raw = X_raw[:, 1].astype(np.float32)
        self.ocean = ocean
        self.X_future_raw = X_future_raw
        self.sic_hist = sic_hist
        self.history_h = sic_hist.shape[1]

        feats = resolve_feature_set(feature_set)
        # See RelaxationRolloutMultiHistDataset's identical comment: a 9-column cache is ambiguous
        # (+rld-no-tdps vs no-rld-+tdps) -- read the explicit flags, don't infer from column count.
        # Both caches must agree (pooling north+south of mismatched flags would silently misalign
        # columns), so assert rather than silently pick one.
        has_rld_n = bool(nc["include_rld"]) if "include_rld" in nc else X_raw.shape[1] > 8
        has_rld_s = bool(sc["include_rld"]) if "include_rld" in sc else X_raw.shape[1] > 8
        has_tdps_n = bool(nc["include_tdps"]) if "include_tdps" in nc else X_raw.shape[1] > 9
        has_tdps_s = bool(sc["include_tdps"]) if "include_tdps" in sc else X_raw.shape[1] > 9
        assert has_rld_n == has_rld_s and has_tdps_n == has_tdps_s, \
            "north/south cache include_rld/include_tdps flags disagree -- cannot pool mismatched column layouts"
        has_rld, has_tdps = has_rld_n, has_tdps_n
        tdps_col = X_raw.shape[1] - 1  # tdps, when present, is always the last raw column

        if override_stats is not None:
            self.mean = override_stats["mean"]; self.std = override_stats["std"]
            self.base_target_stats = override_stats["base_target_stats"]
            self.radsum_mean = override_stats["radsum_mean"]; self.radsum_std = override_stats["radsum_std"]
            self.rld_mean = override_stats["rld_mean"]; self.rld_std = override_stats["rld_std"]
            self.tdps_mean = override_stats["tdps_mean"]; self.tdps_std = override_stats["tdps_std"]
        else:
            self.mean = X_raw[:, :6].mean(axis=0).astype(np.float32)
            self.std = np.maximum(X_raw[:, :6].std(axis=0), 1e-6).astype(np.float32)
            self.base_target_stats = (float(y_phys.mean()), float(max(y_phys.std(), 1e-6)))
            self.radsum_mean = self.radsum_std = None
            if "radsum" in feats:
                radsum_all = (X_raw[:, 2] + X_raw[:, 3]).astype(np.float32)
                self.radsum_mean = float(radsum_all.mean()); self.radsum_std = float(max(radsum_all.std(), 1e-6))
            self.rld_mean = self.rld_std = None
            if "rld" in feats and has_rld:
                rld_all = X_raw[:, 8].astype(np.float32)
                self.rld_mean = float(rld_all.mean()); self.rld_std = float(max(rld_all.std(), 1e-6))
            self.tdps_mean = self.tdps_std = None
            if "tdps" in feats and has_tdps:
                tdps_shifted = X_future_raw[:, 0, tdps_col].astype(np.float32)
                self.tdps_mean = float(tdps_shifted.mean()); self.tdps_std = float(max(tdps_shifted.std(), 1e-6))

        self.X, self.feature_names = build_feature_matrix_from_raw(
            X_raw, feature_set=feature_set, normalize_phys=normalize_phys,
            mean=self.mean, std=self.std, radsum_mean=self.radsum_mean, radsum_std=self.radsum_std,
            rld_mean=self.rld_mean, rld_std=self.rld_std, tdps_mean=self.tdps_mean, tdps_std=self.tdps_std,
        )

        if override_cross is None:
            self.cross_mean = cross_raw.mean(axis=0).astype(np.float32)
            self.cross_std = np.maximum(cross_raw.std(axis=0), 1e-6).astype(np.float32)
        else:
            self.cross_mean = override_cross["mean"]; self.cross_std = override_cross["std"]
        self.use_cross = bool(use_cross)
        if self.use_cross:
            cross_norm = ((cross_raw - self.cross_mean) / self.cross_std).astype(np.float32)
            self.X = np.concatenate([self.X, cross_norm], axis=1).astype(np.float32)
            names = ([CROSS_COL_NAMES[i] for i in cross_cols] if cross_cols is not None
                     else CROSS_BASIS_NAMES.get(cross_norm.shape[1], CROSS_COL_NAMES[:cross_norm.shape[1]]))
            self.feature_names = self.feature_names + names

        # build normalized X_future (K rollout steps) the same way as the single-hemisphere dataset,
        # using the SAME pooled stats -- needed for the K-step rollout loss term in run_epoch
        base_feat_names = None
        K = X_future_raw.shape[1]
        X_future_norm = np.zeros((X_future_raw.shape[0], K, self.X.shape[1]), dtype=np.float32)
        for k in range(K):
            fk_norm, base_feat_names = build_feature_matrix_from_raw(
                X_future_raw[:, k, :], feature_set=feature_set, normalize_phys=normalize_phys,
                mean=self.mean, std=self.std, radsum_mean=self.radsum_mean, radsum_std=self.radsum_std,
                rld_mean=self.rld_mean, rld_std=self.rld_std, tdps_mean=self.tdps_mean, tdps_std=self.tdps_std,
            )
            if self.use_cross:
                # derive from the actual stored cross_mean width, not a hardcoded fallback -- robust
                # to any cross-feature-count variant (4 raw directions, a cross_cols subset, or a
                # transformed basis like the 2-column gradient (N-S)/(W-E) representation)
                n_cross = len(self.cross_mean)
                zeros_cross = np.zeros((fk_norm.shape[0], n_cross), dtype=np.float32)
                fk_norm = np.concatenate([fk_norm, zeros_cross], axis=1)
            X_future_norm[:, k, :] = fk_norm
        self.X_future = X_future_norm

        # shifted-forcing-notos overwrite (same convention as the single-hemisphere dataset):
        # every column except siconc/tos gets the block-(t+1) value instead of the current-block one
        sic_idx = base_feat_names.index("siconc"); tos_idx = base_feat_names.index("tos")
        for i in range(len(base_feat_names)):
            if i in (sic_idx, tos_idx):
                continue
            self.X[:, i] = self.X_future[:, 0, i]

        out_mean, out_std = self.base_target_stats
        self.y = ((y_phys - out_mean) / out_std).astype(np.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return (torch.from_numpy(self.X[idx]).float(), torch.tensor(self.y[idx], dtype=torch.float32),
                torch.tensor(self.ocean[idx], dtype=torch.float32), torch.tensor(self.sic_raw[idx], dtype=torch.float32),
                torch.tensor(self.tos_raw[idx], dtype=torch.float32), torch.from_numpy(self.sic_hist[idx]).float(),
                torch.from_numpy(self.X_future[idx]).float(), torch.from_numpy(self.X_future_raw[idx]).float())


def make_pooled_loaders(north_train, south_train, north_val, south_val, feature_set="all",
                         normalize_phys=True, batch_size=BATCH_SIZE, use_cross=True, cross_cols=None):
    train_ds = PooledRelaxationDataset(north_train, south_train, feature_set=feature_set, normalize_phys=normalize_phys, use_cross=use_cross, cross_cols=cross_cols)
    override_stats = dict(mean=train_ds.mean, std=train_ds.std, base_target_stats=train_ds.base_target_stats,
                           radsum_mean=train_ds.radsum_mean, radsum_std=train_ds.radsum_std,
                           rld_mean=train_ds.rld_mean, rld_std=train_ds.rld_std,
                           tdps_mean=train_ds.tdps_mean, tdps_std=train_ds.tdps_std)
    override_cross = dict(mean=train_ds.cross_mean, std=train_ds.cross_std)
    val_ds = PooledRelaxationDataset(north_val, south_val, feature_set=feature_set, normalize_phys=normalize_phys,
                                      use_cross=use_cross, cross_cols=cross_cols, override_stats=override_stats, override_cross=override_cross)
    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=pin)
    return train_ds, val_ds, train_loader, val_loader


# ---------------------------------------------------------------------------
# Models (verbatim from Modeltry22)
# ---------------------------------------------------------------------------
class RelaxationMoEClassifier(nn.Module):
    def __init__(self, in_dim, hidden=64, depth=4, dropout=0.0, tau_floor=1.0, dt=1.0, history_h=3, mom_hidden=16):
        super().__init__()
        layers = []
        d = int(in_dim)
        for _ in range(int(depth)):
            layers.append(nn.Linear(d, int(hidden))); layers.append(nn.GELU())
            if dropout > 0: layers.append(nn.Dropout(float(dropout)))
            d = int(hidden)
        self.backbone = nn.Sequential(*layers)
        self.momentum_mlp = nn.Sequential(nn.Linear(int(history_h), int(mom_hidden)), nn.GELU())
        combined_dim = d + int(mom_hidden)
        self.class_head = nn.Linear(combined_dim, 2)
        self.inc_head = nn.Linear(combined_dim, 1)
        self.dec_head = nn.Linear(combined_dim, 1)
        self.tau_inc_param = nn.Parameter(torch.zeros(1))
        self.tau_dec_param = nn.Parameter(torch.zeros(1))
        self.tau_floor = float(tau_floor)
        self.dt = float(dt)
        self.history_h = int(history_h)

    def forward(self, x, sic_raw, tos_raw, sic_hist):
        h = self.backbone(x)
        full_seq = torch.cat([sic_hist, sic_raw.unsqueeze(-1)], dim=-1)
        diffs = full_seq[:, 1:] - full_seq[:, :-1]
        mom_feat = self.momentum_mlp(diffs)
        combined = torch.cat([h, mom_feat], dim=-1)
        class_logits = self.class_head(combined)
        p = torch.softmax(class_logits, dim=-1)
        p_inc, p_dec = p[:, 0], p[:, 1]
        sic_eq_inc = sic_raw + (1.0 - sic_raw) * torch.sigmoid(self.inc_head(combined).squeeze(-1))
        sic_eq_dec = sic_raw * torch.sigmoid(self.dec_head(combined).squeeze(-1))
        tau_inc = self.tau_floor + F.softplus(self.tau_inc_param)
        tau_dec = self.tau_floor + F.softplus(self.tau_dec_param)
        decay_inc = torch.exp(-self.dt / tau_inc)
        decay_dec = torch.exp(-self.dt / tau_dec)
        tendency_inc = (sic_eq_inc - sic_raw) * (1.0 - decay_inc)
        tendency_dec = (sic_eq_dec - sic_raw) * (1.0 - decay_dec)
        tendency = p_inc * tendency_inc + p_dec * tendency_dec
        sic_eq = p_inc * sic_eq_inc + p_dec * sic_eq_dec
        tau = p_inc * tau_inc + p_dec * tau_dec
        return tendency, sic_eq, tau, class_logits


class ChangeGateClassifier(nn.Module):
    def __init__(self, in_dim, hidden=64, depth=4, tau_floor=1.0, dt=1.0, history_h=3, mom_hidden=16):
        super().__init__()
        layers = []
        d = int(in_dim)
        for _ in range(int(depth)):
            layers.append(nn.Linear(d, int(hidden))); layers.append(nn.GELU())
            d = int(hidden)
        self.backbone = nn.Sequential(*layers)
        self.momentum_mlp = nn.Sequential(nn.Linear(int(history_h), int(mom_hidden)), nn.GELU())
        combined_dim = d + int(mom_hidden)
        self.class_head = nn.Linear(combined_dim, 2)
        self.changed_head = nn.Linear(combined_dim, 1)
        self.tau_changed_param = nn.Parameter(torch.zeros(1))
        self.tau_floor = float(tau_floor); self.dt = float(dt); self.history_h = int(history_h)

    def forward(self, x, sic_raw, tos_raw, sic_hist):
        h = self.backbone(x)
        full_seq = torch.cat([sic_hist, sic_raw.unsqueeze(-1)], dim=-1)
        diffs = full_seq[:, 1:] - full_seq[:, :-1]
        mom_feat = self.momentum_mlp(diffs)
        combined = torch.cat([h, mom_feat], dim=-1)
        class_logits = self.class_head(combined)
        p = torch.softmax(class_logits, dim=-1)
        p_unchanged, p_changed = p[:, 0], p[:, 1]
        sic_eq_changed = torch.sigmoid(self.changed_head(combined).squeeze(-1))
        tau_changed = self.tau_floor + F.softplus(self.tau_changed_param)
        decay = torch.exp(-self.dt / tau_changed)
        tendency_changed = (sic_eq_changed - sic_raw) * (1.0 - decay)
        tendency = p_changed * tendency_changed
        return tendency, None, None, class_logits


BAND_SPECS = {
    ("north", "low"): dict(model_type="changegate", sic_band_lo=0.0, sic_band_hi=0.1),
    ("north", "mid"): dict(model_type="relaxationmoe", sic_band_lo=0.1, sic_band_hi=0.9),
    ("north", "high"): dict(model_type="relaxationmoe", sic_band_lo=0.9, sic_band_hi=1.0),
    ("south", "low"): dict(model_type="changegate", sic_band_lo=0.0, sic_band_hi=0.1),
    ("south", "mid"): dict(model_type="relaxationmoe", sic_band_lo=0.1, sic_band_hi=0.9),
}


# ---------------------------------------------------------------------------
# Plain (no monotonicity penalty) batch loss / epoch / predictions -- for a controlled A/B
# comparison between two feature sets (e.g. rls vs rld) under identical everything else.
#
# model_type-aware: verified against Modeltry22/23's actual training code (not assumed) that
# ChangeGateClassifier (low band) uses GENUINELY DIFFERENT classification-target semantics and
# SIC-weighting from RelaxationMoEClassifier (mid/high band) -- treating them identically was a
# real bug caught when a from-scratch low-band retrain collapsed to CORR~0.19 instead of the
# expected ~0.7 (a real production low-band checkpoint loaded fine, isolating the bug to training,
# not the model/dataset code). Specifically:
#   - RelaxationMoEClassifier: class target is SIGN of the true tendency (inc=0/dec=1), only
#     samples with |true tendency| >= binary_exclude_tau participate in the classification loss.
#     sic_weight = floor + scale*sic*(1-sic) for mid; floor + scale*(1-sic)/(1-0.9) for high
#     (asymmetric taper toward the high-SIC boundary).
#   - ChangeGateClassifier: class target is MAGNITUDE-threshold (unchanged=0/changed=1 by whether
#     |true tendency| >= binary_exclude_tau), ALL valid samples participate (no exclusion) --
#     this is a fundamentally different classification problem, not a relabeling of the same one.
#     sic_weight = floor + scale*sic/0.1 (asymmetric taper toward the shared 0.1 mid-band boundary).
# ---------------------------------------------------------------------------
def batch_loss_plain(model, x, y_target, ocean_frac, sic_raw, tos_raw, sic_hist0, x_future, x_future_raw,
                      base_target_stats=(0.0, 1.0), model_type="relaxationmoe",
                      sic_weight_scale=10.0, sic_weight_floor=1.0,
                      rollout_weight_decay=0.5, binary_exclude_tau=0.02, lambda_class=0.1):
    valid = ocean_frac > 0
    if valid.sum() == 0:
        zero = x.sum() * 0.0
        return zero, zero.detach(), zero.detach(), valid, None, None

    mean, std = base_target_stats
    K = x_future.shape[1]

    if model_type == "changegate":
        def sic_weight_of(sic):
            return sic_weight_floor + sic_weight_scale * sic / 0.1
    elif model_type == "relaxationmoe_high":
        def sic_weight_of(sic):
            return sic_weight_floor + sic_weight_scale * (1.0 - sic) / (1.0 - 0.9)
    else:
        def sic_weight_of(sic):
            return sic_weight_floor + sic_weight_scale * sic * (1.0 - sic)

    pred_0, _, _, class_logits_0 = model(x, sic_raw, tos_raw, sic_hist0)
    true_0 = y_target * std + mean
    w0 = sic_weight_of(sic_raw)
    loss_0 = torch.mean(w0[valid] * (pred_0[valid] - true_0[valid]) ** 2)

    if model_type == "changegate":
        # unchanged(0)/changed(1) by magnitude -- ALL valid samples participate
        class_valid = valid
        class_labels = (true_0.abs() >= binary_exclude_tau).long()
    else:
        # inc(0)/dec(1) by sign -- only samples with non-trivial true tendency participate
        class_valid = valid & (true_0.abs() >= binary_exclude_tau)
        class_labels = (true_0 < 0).long()

    if class_valid.sum() > 0:
        class_loss = F.cross_entropy(class_logits_0[class_valid], class_labels[class_valid])
    else:
        class_loss = x.sum() * 0.0

    total_loss = loss_0
    total_w = 1.0
    sic_hist_cur = torch.cat([sic_hist0[:, 1:], sic_raw.unsqueeze(-1)], dim=-1)
    for j in range(K - 1):
        x_j = x_future[:, j, :]
        sic_j = x_future_raw[:, j, 0]
        tos_j = x_future_raw[:, j, 1]
        sic_next_true_j = x_future_raw[:, j + 1, 0]
        pred_j, _, _, _ = model(x_j, sic_j, tos_j, sic_hist_cur)
        true_j = sic_next_true_j - sic_j
        w_j = sic_weight_of(sic_j)
        rw = rollout_weight_decay ** (j + 1)
        loss_j = torch.mean(w_j[valid] * (pred_j[valid] - true_j[valid]) ** 2)
        total_loss = total_loss + rw * loss_j
        total_w += rw
        sic_hist_cur = torch.cat([sic_hist_cur[:, 1:], sic_j.unsqueeze(-1)], dim=-1)

    reg_loss_final = total_loss / total_w
    loss = reg_loss_final + lambda_class * class_loss

    return loss, reg_loss_final.detach(), class_loss.detach(), valid.detach(), pred_0.detach(), true_0.detach()


def run_epoch(model, loader, optimizer, device, base_target_stats, model_type="relaxationmoe",
              sic_weight_scale=10.0, sic_weight_floor=1.0, rollout_weight_decay=0.5,
              binary_exclude_tau=0.02, lambda_class=0.1):
    """model_type in {"relaxationmoe", "relaxationmoe_high", "changegate"} -- selects the matching
    classification-target semantics and SIC-weighting formula, see batch_loss_plain's docstring."""
    train_mode = optimizer is not None
    model.train(mode=train_mode)
    total_loss = 0.0
    total_valid = 0
    for batch in loader:
        x, y_target, ocean_frac, sic_raw, tos_raw, sic_hist0, x_future, x_future_raw = batch
        x = x.to(device=device, dtype=torch.float32)
        y_target = y_target.to(device=device, dtype=torch.float32)
        ocean_frac = ocean_frac.to(device=device, dtype=torch.float32)
        sic_raw = sic_raw.to(device=device, dtype=torch.float32)
        tos_raw = tos_raw.to(device=device, dtype=torch.float32)
        sic_hist0 = sic_hist0.to(device=device, dtype=torch.float32)
        x_future = x_future.to(device=device, dtype=torch.float32)
        x_future_raw = x_future_raw.to(device=device, dtype=torch.float32)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)
        loss, reg_loss, class_loss, valid, pred_0, true_0 = batch_loss_plain(
            model, x, y_target, ocean_frac, sic_raw, tos_raw, sic_hist0, x_future, x_future_raw,
            base_target_stats=base_target_stats, model_type=model_type,
            sic_weight_scale=sic_weight_scale, sic_weight_floor=sic_weight_floor,
            rollout_weight_decay=rollout_weight_decay, binary_exclude_tau=binary_exclude_tau, lambda_class=lambda_class,
        )
        if train_mode:
            loss.backward()
            optimizer.step()
        n_valid = int(valid.sum().item())
        if n_valid == 0:
            continue
        total_loss += loss.item() * n_valid
        total_valid += n_valid
    return total_loss / max(total_valid, 1)


@torch.no_grad()
def collect_predictions(model, loader, device, base_target_stats):
    model.eval()
    mean, std = base_target_stats
    preds, trues = [], []
    for batch in loader:
        x, y_target, ocean_frac, sic_raw, tos_raw, sic_hist0, x_future, x_future_raw = batch
        x = x.to(device=device, dtype=torch.float32)
        sic_raw = sic_raw.to(device=device, dtype=torch.float32)
        tos_raw = tos_raw.to(device=device, dtype=torch.float32)
        sic_hist0 = sic_hist0.to(device=device, dtype=torch.float32)
        ocean_frac = ocean_frac.to(device=device, dtype=torch.float32)
        pred_raw, _, _, _ = model(x, sic_raw, tos_raw, sic_hist0)
        true_raw = y_target.to(device=device, dtype=torch.float32) * std + mean
        valid = ocean_frac > 0
        preds.append(pred_raw[valid].cpu().numpy())
        trues.append(true_raw[valid].cpu().numpy())
    return np.concatenate(preds), np.concatenate(trues)


def write_curvature_declaration(out_dir, cross_columns, curvature_tanh_scale=None,
                                curvature_laplacian=None):
    """Declare, alongside an exported weight package, the cross-feature curvature column it was
    TRAINED on.

    MLSeaIceV4.initialize! reads this file and refuses to run when the model's own
    curvature_tanh_scale / curvature_laplacian disagree with it. Call this from every export script
    right after writing a band's weights. The column's definition used to live only in the Julia run
    scripts' flag lists, which drifted away from the weights three times (2026-09-29, 2026-10-15,
    2026-10-27); the 2026-10-27 case silently fed the network a different operator for every coupled
    run of that package and invalidated three 3-year runs.

    cross_columns < 3 means the package has no curvature column, and the two settings are omitted.
    The regression test ml_seaice_v4_curvature_declaration.jl asserts EVERY shipped package has this
    file, so a new export that forgets the call fails the test suite rather than failing silently.
    """
    from pathlib import Path

    lines = [
        "# Declares the cross-feature curvature column these weights were TRAINED on.",
        "# MLSeaIceV4.initialize! validates its own settings against this file and errors on a",
        "# mismatch, so the column's definition can no longer drift away from the weights that",
        "# were fitted to it. Written by sic_pipeline_lib.write_curvature_declaration.",
        f"cross_columns={int(cross_columns)}",
    ]
    if int(cross_columns) >= 3:
        if curvature_tanh_scale is None or curvature_laplacian is None:
            raise ValueError("a 3-column package must declare curvature_tanh_scale and "
                             "curvature_laplacian")
        lines += [f"curvature_tanh_scale={float(curvature_tanh_scale)}",
                  f"curvature_laplacian={str(bool(curvature_laplacian)).lower()}"]
    (Path(out_dir) / "curvature.txt").write_text("\n".join(lines) + "\n")
