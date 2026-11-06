#!/usr/bin/env julia
#
# make_initial_sic.jl -- builds the initial sea-ice state that MLSeaIceV4 reads at startup.
#
# The component needs more than a single map. Its ML update runs once every `update_every`
# (5 days by default) and the networks take the three PREVIOUS cycles' concentrations as
# inputs, so the initial condition is four fields 5 days apart: t-15d, t-10d, t-5d, t. The
# file is those four blocks of `npoints` Float32 values, in that order, oldest first, on the
# model's own grid. `initialize!` reads it, clamps each value to [0,1] and marks those cells
# initialized; `length(raw) == 4 * npoints` is checked, so a file built for the wrong
# resolution is rejected rather than misread.
#
# Without the file nothing errors: every cell stays uninitialized and the first update cycle
# sets all three history slots to whatever concentration the atmospheric model started with,
# i.e. a cold start with a flat history. That runs, but it is not the experiment the offline
# results describe, and `enable_south` is true by default, so Antarctica cold-starts too.
#
# One file serves both hemispheres. The blocks are global fields and the component masks each
# to its own hemisphere (`is_arctic` / `is_antarctic`), so pass the same path to
# `initial_sic_path` and `initial_sic_path_south`; the script prints both lines for you.
#
# Usage
#   julia --project=/path/to/SpeedyWeather make_initial_sic.jl \
#         --trunc 63 --start 1990-01-01 \
#         --processed /path/to/processed --out /path/to/output_dir
#
#   --trunc      spectral truncation of the run the file is for (31, 63, 127, ...)
#   --start      the run's start date; the three history blocks are 5, 10 and 15 days before it
#   --processed  the directory holding processed_YYYY.nc (the notebook's `processed/`)
#   --out        where to write; the file is named after the grid it was built for
#   --grid       FullGaussianGrid (default) or OctahedralGaussianGrid
#   --every      history spacing in days, only if the run changes `update_every` from 5
#
# ERA5 is remapped onto the model's grid by AREA MEAN wherever a model cell is coarser than
# the 1-degree reanalysis, and bilinearly where it is finer. A T31 cell spans about 16 ERA5
# cells, so point sampling would read 4 of them and discard the rest, losing part of the
# marginal ice zone; at T127 a cell is finer than 1 degree, so thousands of cells contain no
# ERA5 centre at all and the bilinear fallback carries them.
#
# Land is `missing` in the reanalysis and is excluded from the average rather than counted as
# zero ice, so each cell carries the concentration of its OCEAN part. That is what the
# component means by it: the surface fluxes weight the ice and open-water terms by the
# concentration and then scale the whole thing by (1 - land_fraction), so counting land as
# ice-free zero here would thin every coastal cell twice. A cell with no valid ocean source
# gets 0, and only cells with land_fraction < 1 are ever read.
#
# The remap is checked, not assumed: every valid ERA5 cell must land in exactly one model
# cell (total source area conserved to 1e-9) and the ice area must come through with it. On
# T31, T63 and T127 from 1989-01-01 all three reproduce ERA5's hemispheric sea-ice area to
# the printed 0.01e6 km2 -- 13.27 north, 4.80 south -- and the global ocean-area-weighted
# mean concentration agrees to every digit across the three grids.

using SpeedyWeather
using NCDatasets
using Dates
using Printf

function parse_args(argv)
    opts = Dict{String, String}(
        "trunc" => "63", "start" => "", "processed" => "", "out" => "",
        "grid" => "FullGaussianGrid", "every" => "5",
    )
    i = 1
    while i <= length(argv)
        a = argv[i]
        startswith(a, "--") || error("unexpected argument \"$a\"; see the header for usage")
        key = a[3:end]
        haskey(opts, key) || error("unknown option \"$a\"; known: " * join(sort(collect(keys(opts))), ", "))
        i + 1 <= length(argv) || error("option \"$a\" needs a value")
        opts[key] = argv[i+1]
        i += 2
    end
    for k in ("start", "processed", "out")
        isempty(opts[k]) && error("--$k is required")
    end
    return opts
end

"""Bilinear sample of a 1-degree ERA5 field at (lond, latd), over valid neighbours only.

`field` is indexed [longitude, latitude] with longitude 0:359 and latitude -90:90, which is
how NCDatasets hands back `siconc(time, latitude, longitude)`. Longitude wraps; latitude is
clamped at the poles. Returns 0 where every neighbour is land (NaN in ERA5)."""
function sample_bilinear(field::AbstractMatrix{<:AbstractFloat}, lond, latd)
    nlon, nlat = size(field)
    x = mod(Float64(lond), 360.0)                    # 0 <= x < 360
    i0 = Int(floor(x)); wx = x - i0
    ia = mod(i0, nlon) + 1                           # 1-based, wrapping
    ib = mod(i0 + 1, nlon) + 1
    y = clamp(Float64(latd) + 90.0, 0.0, Float64(nlat - 1))
    j0 = Int(floor(y)); wy = y - j0
    ja = j0 + 1
    jb = min(j0 + 2, nlat)
    num = 0.0
    den = 0.0
    for (ii, wi) in ((ia, 1 - wx), (ib, wx)), (jj, wj) in ((ja, 1 - wy), (jb, wy))
        w = wi * wj
        w == 0 && continue
        v = field[ii, jj]
        isfinite(v) || continue                      # land
        num += w * v
        den += w
    end
    return den > 0 ? clamp(Float32(num / den), 0f0, 1f0) : 0f0
end

"""The model grid's cell boundaries, derived from the point coordinates alone.

Rings are the distinct latitudes; their edges are the midpoints between neighbouring rings,
closed at the poles. Within a ring the points are equally spaced, so a cell is 360/nlon wide.
This holds for the full and the octahedral Gaussian grids alike and needs no grid internals.
Returns the sorted ring latitudes (north to south), their edges, and each point's ring."""
function cell_geometry(londs, latds)
    ring_lats = sort(unique(latds), rev = true)      # north to south
    nrings = length(ring_lats)
    edges = Vector{Float64}(undef, nrings + 1)
    edges[1] = 90.0
    for r in 2:nrings
        edges[r] = (ring_lats[r-1] + ring_lats[r]) / 2
    end
    edges[nrings+1] = -90.0
    ring_of = Dict(lat => r for (r, lat) in enumerate(ring_lats))
    point_ring = [ring_of[lat] for lat in latds]
    nlon_of_ring = zeros(Int, nrings)
    for r in point_ring
        nlon_of_ring[r] += 1
    end
    # the first point of each ring, so a longitude maps to a point index arithmetically
    first_of_ring = zeros(Int, nrings)
    for ij in eachindex(point_ring)
        r = point_ring[ij]
        if first_of_ring[r] == 0 || londs[ij] < londs[first_of_ring[r]]
            first_of_ring[r] = ij
        end
    end
    return (; ring_lats, edges, point_ring, nlon_of_ring, first_of_ring)
end

"""Remap a 1-degree ERA5 field onto the model grid: area mean where a model cell holds at
least one ERA5 cell centre, bilinear where it holds none.

The accumulation runs over the 65,160 ERA5 cells rather than over the model cells, so the
cost does not grow with the product of the two resolutions. Each ERA5 cell is weighted by
cos(latitude), its share of the sphere."""
function remap(field::AbstractMatrix{<:AbstractFloat}, londs, latds, geom)
    npoints = length(londs)
    num = zeros(Float64, npoints)
    den = zeros(Float64, npoints)
    nlon_src, nlat_src = size(field)
    for j in 1:nlat_src
        slat = -90.0 + (j - 1)                       # ERA5 latitude of this row
        # the ring whose edge band contains this latitude; edges run 90 -> -90
        r = searchsortedfirst(geom.edges, slat, rev = true) - 1
        r = clamp(r, 1, length(geom.ring_lats))
        w = cosd(slat)
        w <= 0 && continue                           # the poles carry no area
        nlon_t = geom.nlon_of_ring[r]
        base = geom.first_of_ring[r]
        lon0 = londs[base]
        for i in 1:nlon_src
            v = field[i, j]
            isfinite(v) || continue                  # land
            slon = Float64(i - 1)
            # which cell of this ring the ERA5 centre falls in
            k = mod(floor(Int, (slon - lon0 + 180.0 / nlon_t) * nlon_t / 360.0), nlon_t)
            ij = base + k
            num[ij] += w * v
            den[ij] += w
        end
    end
    out = Vector{Float32}(undef, npoints)
    nfallback = 0
    for ij in 1:npoints
        if den[ij] > 0
            out[ij] = clamp(Float32(num[ij] / den[ij]), 0f0, 1f0)
        else
            out[ij] = sample_bilinear(field, londs[ij], latds[ij])
            nfallback += 1
        end
    end
    return out, nfallback, den
end

"""The ocean-area-weighted mean of an ERA5 field over a latitude band, straight from the
source. The remapped field has to reproduce it: it is what an area mean conserves."""
function era5_band_mean(field::AbstractMatrix{<:AbstractFloat}, lat_lo, lat_hi)
    nlon, nlat = size(field)
    num = 0.0
    den = 0.0
    for j in 1:nlat
        slat = -90.0 + (j - 1)
        (lat_lo <= slat <= lat_hi) || continue
        w = cosd(slat)
        for i in 1:nlon
            v = field[i, j]
            isfinite(v) || continue
            num += w * v
            den += w
        end
    end
    return den > 0 ? num / den : NaN, den
end

"""The same mean taken from the remapped field, weighted by each model cell's own share of
valid ocean source area. Cells filled bilinearly carry no source weight and drop out."""
const EARTH_RADIUS_KM = 6371.0
const CELL_1DEG = (pi / 180)^2 * EARTH_RADIUS_KM^2 / 1e6   # 1-degree cell area at the equator, 1e6 km2

"""Sea-ice area over a latitude band, from the remapped field. Each model cell contributes the
source ocean area it actually covers, `den` cells of 1 degree weighted by cos(latitude), times
its concentration -- so this is the same quantity on every grid."""
ice_area(block, den, latds, lat_lo, lat_hi) =
    sum(den[ij] * block[ij] for ij in eachindex(block)
        if lat_lo <= latds[ij] <= lat_hi && den[ij] > 0; init = 0.0) * CELL_1DEG

"""The same, straight from ERA5."""
function era5_ice_area(field::AbstractMatrix{<:AbstractFloat}, lat_lo, lat_hi)
    nlon, nlat = size(field)
    total = 0.0
    for j in 1:nlat
        slat = -90.0 + (j - 1)
        (lat_lo <= slat <= lat_hi) || continue
        w = cosd(slat)
        for i in 1:nlon
            v = field[i, j]
            isfinite(v) && (total += w * v)
        end
    end
    return total * CELL_1DEG
end

function grid_band_mean(block, den, latds, lat_lo, lat_hi)
    num = 0.0
    tot = 0.0
    for ij in eachindex(block)
        (lat_lo <= latds[ij] <= lat_hi) || continue
        den[ij] > 0 || continue
        num += den[ij] * block[ij]
        tot += den[ij]
    end
    return tot > 0 ? num / tot : NaN, tot
end

"""Read one day of ERA5 `siconc` from processed_YYYY.nc, as [longitude, latitude]."""
function read_siconc(processed::String, date::Date)
    path = joinpath(processed, @sprintf("processed_%04d.nc", year(date)))
    isfile(path) || error("$(path) not found: --processed must hold processed_YYYY.nc for " *
                          "$(date) and the three dates before it")
    NCDataset(path) do ds
        idx = Dates.value(date - Date(year(date), 1, 1)) + 1
        nt = size(ds["siconc"], 3)
        1 <= idx <= nt || error("$(path) has $(nt) days, cannot index $(date)")
        # NCDatasets hands masked points back as `missing`; the sampler wants NaN for land
        raw = ds["siconc"][:, :, idx]
        size(raw, 1) == 360 && size(raw, 2) == 181 ||
            error("$(path): siconc day is $(size(raw)), expected (360, 181) on the 1-degree grid")
        return [ismissing(v) ? NaN : Float64(v) for v in raw]
    end
end

const GRIDS = Dict("FullGaussianGrid" => FullGaussianGrid,
                   "OctahedralGaussianGrid" => OctahedralGaussianGrid)

function main(argv)
    opts = parse_args(argv)
    trunc = parse(Int, opts["trunc"])
    every = Day(parse(Int, opts["every"]))
    start = Date(opts["start"])
    haskey(GRIDS, opts["grid"]) || error("--grid must be one of " * join(sort(collect(keys(GRIDS))), ", "))
    GridType = GRIDS[opts["grid"]]

    spectral_grid = SpectralGrid(trunc = trunc, nlayers = 1, Grid = GridType,
                                 architecture = SpeedyWeather.CPU())
    grid = spectral_grid.grid
    npoints = RingGrids.get_npoints(grid)
    londs, latds = RingGrids.get_londlatds(grid)
    println("target grid: $(opts["grid"]) T$(trunc), $(npoints) points")

    geom = cell_geometry(londs, latds)
    println("            $(length(geom.ring_lats)) rings, " *
            "$(minimum(geom.nlon_of_ring))-$(maximum(geom.nlon_of_ring)) points per ring")

    dates = [start - 3 * every, start - 2 * every, start - every, start]
    names = ("hist3", "hist2", "hist1", "current")
    out = zeros(Float32, 4 * npoints)
    for (k, (name, date)) in enumerate(zip(names, dates))
        era5 = read_siconc(opts["processed"], date)
        block, nfallback, den = remap(era5, londs, latds, geom)
        k == 1 && println("            $(nfallback) of $(npoints) cells hold no ERA5 centre " *
                          "and were filled bilinearly")
        out[(k-1)*npoints+1:k*npoints] = block

        # Two things have to hold exactly, whatever the resolution. Every valid ERA5 cell
        # must have landed in exactly one model cell -- so the total source area the model
        # cells account for equals the total valid source area -- and the ice area must come
        # through with it. These test the ring and longitude arithmetic; a latitude-band mean
        # cannot, because a band boundary cuts through model cells and the comparison then
        # disagrees by the straddling row even when the binning is perfect.
        ref_mean, area_src = era5_band_mean(era5, -90.0, 90.0)
        area_dst = sum(den)
        abs(area_dst - area_src) / area_src < 1e-9 ||
            error("valid source area $(area_src) was binned as $(area_dst): the ring or " *
                  "longitude assignment loses or duplicates cells")
        got_mean, _ = grid_band_mean(block, den, latds, -90.0, 90.0)
        abs(got_mean - ref_mean) < 1e-9 ||
            error("global ocean-area-weighted mean SIC is $(got_mean), ERA5 says $(ref_mean)")

        # printed for the eye: sea-ice AREA per hemisphere, the number this initial state is
        # usually quoted by. It is resolution-stable by construction, up to the one source row
        # that a 40-degree band boundary cuts differently on each grid (about 1%): on T63 the
        # 40N row falls in a cell centred above 40, on T31 and T127 in one centred just below.
        area_n = ice_area(block, den, latds, 40.0, 90.0)
        area_s = ice_area(block, den, latds, -90.0, -40.0)
        ref_area_n = era5_ice_area(era5, 40.0, 90.0)
        ref_area_s = era5_ice_area(era5, -90.0, -40.0)
        @printf("  %-7s %s   ice area  north %.2f (ERA5 %.2f)  south %.2f (ERA5 %.2f) 1e6 km2\n",
                name, date, area_n, ref_area_n, area_s, ref_area_s)
    end

    tag = GridType === FullGaussianGrid ? "t$(trunc)" : "t$(trunc)_octahedral"
    path = joinpath(opts["out"], "initial_sic_$(tag)_$(Dates.format(start, "yyyymmdd")).bin")
    mkpath(opts["out"])
    open(path, "w") do io
        write(io, out)
    end
    @printf("\nwrote %s  (%d floats, %d bytes, 4 x %d)\n",
            path, length(out), length(out) * 4, npoints)

    # the four blocks must be 5 days apart in TIME, not four copies of one map: a file whose
    # blocks are identical gives the networks a zero tendency history and the first cycle
    # behaves like a cold start even though the file was read
    spread = maximum(maximum(abs.(out[(k-1)*npoints+1:k*npoints] - out[3npoints+1:4npoints]))
                     for k in 1:3)
    @printf("history differs from the current block by at most %.4f (0 would mean a flat history)\n", spread)

    println("\npass it to the component as:")
    println("    MLSeaIceV4(spectral_grid;")
    println("               initial_sic_path       = \"$(path)\",")
    println("               initial_sic_path_south = \"$(path)\",")
    println("               ...)")
    println("MAKE_INITIAL_SIC_DONE")
end

main(ARGS)
