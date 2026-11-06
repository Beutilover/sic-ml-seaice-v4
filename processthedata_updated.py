from __future__ import annotations

from pathlib import Path
import re
import argparse
import warnings

import numpy as np
import xarray as xr


# Configuration

BASE_DIR = Path(".")  # Root directory containing the ERA5 input folders.
OUT_DIR = BASE_DIR / "processed"

# Map input folders to standard variable names.
# Dew-point source files may use descending latitude; coordinates are sorted below.
# Radiative accumulations require the source-specific scaling handled below.
YEAR_FOLDER_TO_VAR = {
    "Downward_SW_rsds": "rsds",
    "Net_downard_LW_rls": "rls",
    "SST_tos": "tos",
    "SIC_siconc": "siconc",
    "2m_tem_tas": "tas",
    "10m_windu_uas": "uas",
    "10m_windv_vas": "vas",
    "2m_dp_tdps": "tdps",
    "Downward_LW_rlds": "rld",
}

# Static fields are shared across years.
STATIC_FOLDER_TO_VAR = {
    "LSM": "lsm",
}

# Exclude auxiliary bounds variables when identifying the main field.
AUX_DATA_VARS = {"time_bnds", "lat_bnds", "lon_bnds", "bnds"}


# Dataset utilities

def standardize_dataset(ds: xr.Dataset) -> xr.Dataset:
    rename = {}

    if "valid_time" in ds.coords or "valid_time" in ds.dims:
        rename["valid_time"] = "time"
    if "lat" in ds.coords or "lat" in ds.dims:
        rename["lat"] = "latitude"
    if "lon" in ds.coords or "lon" in ds.dims:
        rename["lon"] = "longitude"

    if rename:
        ds = ds.rename(rename)

    if "number" in ds.coords:
        ds = ds.drop_vars("number")

    # Keep latitude ascending for downstream NumPy indexing.
    if "latitude" in ds.coords and ds["latitude"].size > 1:
        if float(ds["latitude"].values[0]) > float(ds["latitude"].values[-1]):
            ds = ds.sortby("latitude")

    # Normalize timestamps to daily resolution.
    if "time" in ds.coords and np.issubdtype(ds["time"].dtype, np.datetime64):
        ds = ds.assign_coords(
            time=ds["time"].astype("datetime64[D]").astype("datetime64[ns]")
        )

    return ds

def mask_fill_values(ds: xr.Dataset) -> xr.Dataset:
    for v in ds.data_vars:
        if np.issubdtype(ds[v].dtype, np.floating):
            ds[v] = ds[v].where(np.abs(ds[v]) < 1e20)
    return ds

def pick_main_data_var(ds: xr.Dataset) -> str:
    """Select the main data variable, excluding auxiliary bounds fields."""
    candidates = [v for v in ds.data_vars if v not in AUX_DATA_VARS]

    if len(candidates) == 1:
        return candidates[0]

    if len(candidates) == 0 and len(ds.data_vars) == 1:
        return next(iter(ds.data_vars))

    raise ValueError(f"Cannot identify the main data variable; available variables: {list(ds.data_vars)}")


def infer_seconds_per_step(ds: xr.Dataset) -> float:
    """Infer the time-step duration in seconds from the time coordinate."""
    time_name = None
    for cand in ("time", "valid_time"):
        if cand in ds.coords:
            time_name = cand
            break

    if time_name is None:
        return 86400.0

    t = ds[time_name].values
    if t.size < 2:
        return 86400.0

    diffs = np.diff(t[: min(t.size, 8)]).astype("timedelta64[s]").astype(np.int64)
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return 86400.0

    return float(np.median(diffs))


def units_looks_like_energy_per_area(units: str | None) -> bool:
    """Recognize accumulated energy-per-area units (J m^-2)."""
    if units is None:
        return False
    s = re.sub(r"\s+", "", str(units)).lower()
    return s in {
        "jm**-2",
        "jm^-2",
        "jm-2",
        "j/m^2",
        "jm-2",
        "j m**-2",
        "j m^-2",
    }


def cast_numeric_coords_to_float32(ds: xr.Dataset) -> xr.Dataset:
    """Cast floating-point coordinates to float32; preserve other types."""
    for cname in list(ds.coords):
        arr = ds[cname]
        if np.issubdtype(arr.dtype, np.floating):
            ds[cname] = arr.astype("float32")
    return ds


def cast_numeric_data_vars_to_float32(ds: xr.Dataset) -> xr.Dataset:
    """Cast floating-point data variables to float32; preserve other types."""
    for vname in list(ds.data_vars):
        arr = ds[vname]
        if np.issubdtype(arr.dtype, np.floating):
            ds[vname] = arr.astype("float32")
    return ds


def convert_jm2_to_wm2(da: xr.DataArray, dt_seconds: float) -> xr.DataArray:
    """Convert accumulated energy per area to mean flux (W m^-2)."""
    attrs = dict(da.attrs)
    if units_looks_like_energy_per_area(attrs.get("units", "")):
        da = (da / np.float32(dt_seconds)).astype("float32")
        attrs["units"] = "W m**-2"
        da.attrs = attrs
        return da
    return da.astype("float32")


def open_year_files(files: list[Path]) -> xr.Dataset:
    """Open and standardize one or more files for a variable and year."""
    if len(files) == 1:
        ds = xr.open_dataset(files[0], decode_times=True)
    else:
        ds = xr.open_mfdataset(files, combine="by_coords", decode_times=True)

    ds = standardize_dataset(ds)
    ds = ds.load()
    return ds


def dataset_encoding(ds: xr.Dataset) -> dict:
    """Configure compressed float32 NetCDF output for floating variables."""
    encoding: dict = {}
    for name, arr in ds.variables.items():
        if np.issubdtype(arr.dtype, np.floating):
            encoding[name] = {
                "dtype": "float32",
                "zlib": True,
                "complevel": 1,
                "shuffle": True,
            }
    return encoding


def process_one_variable_file(
    ds: xr.Dataset,
    target_var: str,
    year: int | None = None,
    drop_time: bool = False,
) -> xr.Dataset:
    """Standardize coordinates, variable names, units, year selection and dtypes."""
    ds = standardize_dataset(ds)

    main_var = pick_main_data_var(ds)
    if main_var != target_var:
        ds = ds.rename({main_var: target_var})

    # Restrict the dataset to the requested year.
    if year is not None and "time" in ds.coords and np.issubdtype(ds["time"].dtype, np.datetime64):
        ds = ds.where(ds["time"].dt.year == year, drop=True)

    # Convert accumulated radiative energy to flux units when appropriate.
    if target_var in ds.data_vars and "time" in ds.coords:
        dt_seconds = infer_seconds_per_step(ds)
        ds[target_var] = convert_jm2_to_wm2(ds[target_var], dt_seconds)

    # Apply the scaling used for hourly accumulated downward radiation in the daily-statistics source.
    if target_var in ("ssrd", "rsds", "strd", "rld"):
        ds[target_var] = ds[target_var]*24
    # Flux sign conventions
# ERA5 and GCM radiation: downward positive.
# Surface turbulent heat fluxes: sign adjusted below when present.
    if target_var in ("hfss", "hfls"):
        ds[target_var] = -ds[target_var]

    # Remove the time dimension from static fields.
    if drop_time and "time" in ds.dims:
        ds = ds.isel(time=0, drop=True)

    for v in ["time_bnds", "valid_time", "forecast_reference_time"]:
        if v in ds.variables or v in ds.coords:
            try:
                ds = ds.drop_vars(v)
            except Exception:
                pass

    ds = cast_numeric_coords_to_float32(ds)
    ds = cast_numeric_data_vars_to_float32(ds)
    return ds


# Yearly processing

def process_year(year: int, base_dir: Path = BASE_DIR, out_dir: Path = OUT_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    pieces: list[xr.Dataset] = []

    # Load year-specific fields.
    for folder_name, target_var in YEAR_FOLDER_TO_VAR.items():
        folder = base_dir / folder_name
        if not folder.exists():
            warnings.warn(f"Directory not found: {folder}")
            continue

        # Match complete Jan–Dec filename ranges to avoid accidentally loading another year.
        files = sorted(folder.rglob(f"*_{year}01-{year}12.nc"))
        if not files:
            warnings.warn(f"{folder} contains no NetCDF file for year {year}")
            continue

        ds = open_year_files(files)
        ds = process_one_variable_file(ds, target_var, year=year, drop_time=False)
        pieces.append(ds)

    # Load static fields.
    for folder_name, target_var in STATIC_FOLDER_TO_VAR.items():
        folder = base_dir / folder_name
        if not folder.exists():
            warnings.warn(f"Directory not found: {folder}")
            continue

        files = sorted(folder.rglob("*.nc"))
        if not files:
            warnings.warn(f"{folder} contains no NetCDF files")
            continue

        if len(files) > 1:
            warnings.warn(f"{folder} contains multiple files; using the first: {files[0].name}")

        ds = open_year_files([files[0]])
        ds = process_one_variable_file(ds, target_var, year=None, drop_time=True)
        pieces.append(ds)

    if not pieces:
        raise RuntimeError(f"No processable data files found for year {year}.")

    # Merge the available fields.
    merged = xr.merge(
        pieces,
        compat="override",
        join="outer",
        combine_attrs="override",
    )

    merged = merged.sortby("time")
    merged = merged.groupby("time").first()

    # Store the magnitude of 10 m wind rather than its vector components.
    if "uas" in merged.data_vars and "vas" in merged.data_vars:
        speed = np.sqrt(merged["uas"] ** 2 + merged["vas"] ** 2).astype("float32")
        speed.attrs = {"units": "m s**-1", "long_name": "10m wind speed"}
        merged["sfcWind"] = speed
        merged = merged.drop_vars(["uas", "vas"])

    # Broadcast the static land-sea mask over the time axis.
    if "lsm" in merged.data_vars and "time" in merged.coords:
        if "time" not in merged["lsm"].dims:
            merged["lsm"] = merged["lsm"].expand_dims(time=merged.time)

    # Normalize numeric dtypes after merging.
    merged = standardize_dataset(merged)
    merged = cast_numeric_coords_to_float32(merged)
    merged = cast_numeric_data_vars_to_float32(merged)

    # Ensure chronological ordering.
    if "time" in merged.coords:
        merged = merged.sortby("time")

    merged = mask_fill_values(merged)

    out_path = out_dir / f"processed_{year}.nc"
    merged.to_netcdf(out_path, encoding=dataset_encoding(merged))
    return out_path


# Command-line entry point

def main():
    parser = argparse.ArgumentParser(
        description="Merge ERA5 files for one year into a float32 NetCDF."
    )
    parser.add_argument("year", type=int, help="e.g. 1980")
    parser.add_argument(
        "--base-dir",
        type=str,
        default=str(BASE_DIR),
        help="ERA5 root directory",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(OUT_DIR),
        help="output directory",
    )
    args = parser.parse_args()

    out_path = process_year(
        year=args.year,
        base_dir=Path(args.base_dir),
        out_dir=Path(args.out_dir),
    )
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
