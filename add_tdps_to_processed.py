"""
Dew-point temperature augmentation for processed ERA5 datasets.

Adds 2 m dew-point temperature (tdps) to existing yearly NetCDF
files without modifying other variables.

Uses the preprocessing utilities in processthedata_updated.py
to preserve consistent coordinate orientation, data types,
and processing conventions, including latitude alignment.

Writes updates through temporary files to prevent partial
or corrupted outputs.

Usage:
    python add_tdps_to_processed.py 2016 2017

"""
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import xarray as xr

import processthedata_updated as P

RAW = Path("2m_dp_tdps")
OUT = Path("processed")


def main(years):
    for year in years:
        dst = OUT / f"processed_{year}.nc"
        src = sorted(RAW.glob(f"tdps_day_ERA5_1x1_{year}01-{year}12.nc"))
        if not src:
            print(f"{year}: no raw dew point file in {RAW}, skipped"); continue
        with xr.open_dataset(dst) as d:
            if "tdps" in d.data_vars:
                print(f"{year}: already has tdps, skipped"); continue
            have = sorted(d.data_vars)
            ref_lat = np.asarray(d["latitude"]); ref_time = np.asarray(d["time"])
            base = d.load()

        raw = P.standardize_dataset(xr.open_dataset(src[0]).load())
        raw = P.mask_fill_values(raw)
        raw = P.cast_numeric_coords_to_float32(raw)
        raw = P.cast_numeric_data_vars_to_float32(raw)
        raw = raw[["tdps"]]
        raw = raw.assign_coords(time=raw["time"].astype("datetime64[D]").astype("datetime64[ns]"))

        # the checks that matter: same grid, same calendar, and the latitude actually ascending
        assert np.array_equal(np.asarray(raw["latitude"]), ref_lat), f"{year}: latitude grid differs"
        assert np.array_equal(np.asarray(raw["longitude"]), np.asarray(base["longitude"])), f"{year}: longitude differs"
        assert float(ref_lat[0]) < float(ref_lat[-1]), f"{year}: reference latitude is not ascending"
        miss = np.setdiff1d(ref_time, np.asarray(raw["time"]))
        assert miss.size == 0, f"{year}: {miss.size} timesteps missing from the dew point file"

        merged = base.assign(tdps=raw["tdps"].sel(time=ref_time))
        v = np.asarray(merged["tdps"])
        fin = np.isfinite(v)
        print(f"{year}: tdps {v.shape}  finite {100*fin.mean():.1f}%  "
              f"range {np.nanmin(v):.1f}-{np.nanmax(v):.1f} K  "
              f"tas-tdps mean {np.nanmean(np.asarray(merged['tas'])[fin] - v[fin]):.2f} K")
        assert np.nanmin(v) > 150 and np.nanmax(v) < 330, f"{year}: dew point out of physical range"

        tmp = tempfile.NamedTemporaryFile(suffix=".nc", dir=OUT, delete=False).name
        merged.to_netcdf(tmp, encoding=P.dataset_encoding(merged))
        with xr.open_dataset(tmp) as chk:
            assert sorted(chk.data_vars) == sorted(have + ["tdps"]), f"{year}: variable set changed"
            for q in have:                       # the other eight must be untouched
                a, b = np.asarray(base[q]), np.asarray(chk[q])
                assert a.shape == b.shape and np.array_equal(np.nan_to_num(a, nan=-9e9),
                                                             np.nan_to_num(b, nan=-9e9)), f"{year}: {q} changed"
        shutil.move(tmp, dst)
        print(f"  -> {dst} now has {len(have)+1} variables, the other {len(have)} bit-identical")


if __name__ == "__main__":
    main([int(a) for a in (sys.argv[1:] or ["2016", "2017"])])
