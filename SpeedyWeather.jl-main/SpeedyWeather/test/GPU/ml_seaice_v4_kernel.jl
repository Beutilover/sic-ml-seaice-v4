# The sea-ice component on the GPU. Runs on whichever backend is present: included from
# test/GPU/runtests.jl, which has already chosen one and bound `gpu_backend`, or standalone, in
# which case it picks one itself.
#
# Three things are checked, and they are not the same claim:
#
#  1. the component driven on the device while the MODEL stays on the host. That split is worth
#     having on a backend where the model is slower on the GPU than on the CPU, which is the case
#     for Metal here (the spectral transform is 45-260x slower) but not for CUDA.
#  2. the component GPU-SAFE: both entry points driven against device-resident fields with scalar
#     indexing switched OFF, so any element-at-a-time access is an immediate error rather than a
#     silent crawl. This is the failure that stops the component running next to a parallel model,
#     and `allowscalar(false)` is what turns it from an opinion into a test.
#  3. END TO END, the model itself on the GPU, which is what "the component follows a parallel
#     run" actually means. This one needs a backend that can run SpeedyWeather's dynamical core:
#     on Metal it cannot (Base `sind(::Float32)` goes through a Float64 intermediate, which Metal
#     has no instruction for), so it SKIPS with the reason printed rather than failing. On CUDA it
#     runs, and it is the only one of the three that proves the whole path.
#
# This test replaced a standalone version that carried its own copies of the kernels. Once the
# kernels moved into the package those copies became a second implementation to keep in step with
# the first, which is the exact failure mode that has cost this project twice.
using SpeedyWeather, Test, Dates, Printf, Statistics
using SpeedyWeather: on_architecture, get_prognostic_step

# Standalone: pick a backend. Wrapped in a function, as test/GPU/runtests.jl does, because a
# `try` block is its own scope and an assignment inside one at top level does not reliably rebind
# an outer binding -- which silently left this `nothing` with the backend already loaded.
function _load_gpu_backend()
    backend = nothing
    try
        @eval using AMDGPU
        backend = :AMDGPU
    catch
    end
    if backend === nothing
        try
            @eval using CUDA
            backend = :CUDA
        catch
        end
    end
    if backend === nothing
        try
            @eval using Metal
            backend = :Metal
        catch
        end
    end
    backend === nothing && error("no GPU backend available (CUDA, AMDGPU or Metal)")
    return backend
end

if !@isdefined(gpu_backend)
    gpu_backend = _load_gpu_backend()
end

# CUDA, AMDGPU and Metal all re-export GPUArraysCore's `allowscalar`, so go through the backend
# module that is actually loaded rather than taking a direct dependency on GPUArraysCore (it is
# only an indirect one here, and `import` of it fails from this project).
allowscalar!(flag::Bool) = getfield(Main, gpu_backend).allowscalar(flag)

const SEAICE_WEIGHTS = joinpath(@__DIR__, "..", "..", "src", "parameterizations")
const SEAICE_SET = "ml_seaice_weights_v4_lapmtas"

seaice_weights_available() =
    isdir(joinpath(SEAICE_WEIGHTS, "$(SEAICE_SET)_unified_mid")) &&
    isfile(joinpath(SEAICE_WEIGHTS, "ml_seaice_weights", "initial_sic_t31.bin"))

function build_seaice_sim(; arch = SpeedyWeather.CPU(), kwargs...)
    spectral_grid = SpectralGrid(trunc = 31, nlayers = 15, Grid = FullGaussianGrid,
                                 architecture = arch)
    sea_ice = MLSeaIceV4(spectral_grid;
        initial_sic_path = joinpath(SEAICE_WEIGHTS, "ml_seaice_weights", "initial_sic_t31.bin"),
        initial_sic_path_south = joinpath(SEAICE_WEIGHTS, "ml_seaice_weights", "initial_sic_t31.bin"),
        weights_dir_mid = joinpath(SEAICE_WEIGHTS, "$(SEAICE_SET)_unified_mid"),
        weights_dir_highsic = joinpath(SEAICE_WEIGHTS, "$(SEAICE_SET)_north_high"),
        weights_dir_lowsic = joinpath(SEAICE_WEIGHTS, "$(SEAICE_SET)_unified_low"),
        curvature_tanh_scale = 0.5, curvature_laplacian = true,
        tos_ceiling_from_training = 1.0, kwargs...)
    model = PrimitiveWetModel(spectral_grid; sea_ice = sea_ice)
    simulation = initialize!(model)
    start = DateTime(1989, 1, 1)
    simulation.variables.prognostic.clock.time = start
    simulation.variables.prognostic.clock.start = start
    sea_ice.last_update[] = start
    return simulation, sea_ice
end

if !seaice_weights_available()
    @info "MLSeaIceV4 GPU sweep: weight package $(SEAICE_SET) not present, skipping"
else
    @testset "MLSeaIceV4 kernel_sweep on the GPU" begin
        both = (flat_forward = true, cross_from_snapshot = true)
        configs = (
            serial = (;),
            cpu_kernel = (kernel_sweep = true,),
            gpu_kernel = (kernel_sweep = true, kernel_architecture = SpeedyWeather.GPU()),
        )
        cfg_names = keys(configs)
        measured = map(cfg_names) do name
            extra = configs[name]
            SpeedyWeather.ML_TIME .= 0
            simulation, sea_ice = build_seaice_sim(; both..., extra...)
            run!(simulation, period = Day(6))     # one 5-day update
            ocean = simulation.variables.prognostic.ocean
            update_seconds = SpeedyWeather.ML_TIME[2]
            # FIRST update only, so this is dominated by first-launch compilation and is NOT the
            # cost of the path: measured in the production script at T63, marginally over six
            # updates, the steady state is 28 ms serial, 28 ms CPU kernel, 38 ms Metal. On Metal with
            # the model on the host the kernel path is break-even at best -- 9 vectors in, 7 out, 6
            # launches and 4 synchronisations cost more than the compute they save. Its value is
            # portability to a device where the model's own fields are already resident.
            @printf("  %-11s first update %7.2f ms (incl. compile)   sic_sum %.9f\n", name,
                    1000 * update_seconds, sum(Float64.(Array(ocean.sea_ice_concentration))))
            return (sic = copy(Array(ocean.sea_ice_concentration)),
                    sst = copy(Array(ocean.sea_surface_temperature)),
                    cost = update_seconds)
        end
        results = NamedTuple{cfg_names}(measured)

        # the CPU kernel path is the same arithmetic, so it is held to equality
        @test isequal(results.serial.sic, results.cpu_kernel.sic)
        @test isequal(results.serial.sst, results.cpu_kernel.sst)

        # the GPU path is the same computation in a different transcendental library
        d = abs.(Float64.(results.serial.sic) .- Float64.(results.gpu_kernel.sic))
        @printf("  GPU vs serial: max|d| = %.3e   mean|d| = %.3e   identical %d/%d\n",
                maximum(d), mean(d), count(iszero, d), length(d))
        @test maximum(d) < 1.0e-5
        @test all(0 .<= results.gpu_kernel.sic .<= 1)       # still a concentration
        @test results.gpu_kernel.cost > 0                   # the update really ran
    end

    # ---------------------------------------------------------------------------------------------
    # The configuration this component was made GPU-capable FOR: the model itself parallel, its
    # fields already on the device, nothing crossing the bus. It cannot be run end to end on this
    # backend that cannot run the dynamical core, so the model is built on the host and its fields
    # are handed to the component as DEVICE arrays, which is exactly what a parallel model hands it.
    # On a backend that CAN run the model, the end-to-end testset below supersedes this one.
    #
    # Both entry points are driven: the per-timestep accumulation (the one that matters, it runs
    # every step) and the 5-day sweep.
    @testset "MLSeaIceV4 next to a resident (parallel) model" begin
        simulation, sea_ice = build_seaice_sim(; flat_forward = true, cross_from_snapshot = true,
                                               kernel_sweep = true,
                                               kernel_architecture = SpeedyWeather.GPU())
        run!(simulation, period = Day(6))        # one update, to fill the carried state
        vars = simulation.variables
        model = simulation.model
        npoints = length(vars.prognostic.ocean.sea_ice_concentration)

        gpu = SpeedyWeather.GPU()
        dev(x) = SpeedyWeather.on_architecture(gpu, Array(x))
        # a buffer set that believes the fields are resident, which is what a GPU model gives
        B = SpeedyWeather.ml_seaice_kernel_buffers(sea_ice, gpu, gpu, npoints)
        @test B.resident

        sic = dev(vars.prognostic.ocean.sea_ice_concentration)
        sst = dev(vars.prognostic.ocean.sea_surface_temperature)
        land = dev(model.land_sea_mask.land_fraction)
        rsds = dev(vars.parameterizations.surface_shortwave_down)
        rld = dev(vars.parameterizations.surface_longwave_down)
        wind = dev(vars.parameterizations.surface_wind_speed)
        tas = dev(vars.parameterizations.surface_air_temperature)
        pres = dev(vars.parameterizations.surface_pressure)
        humid = dev(SpeedyWeather.get_prognostic_step(vars.grid.humidity, model.time_stepping, sea_ice))
        nlayers = model.geometry.nlayers

        allowscalar!(false)
        try
            # every timestep, in a real run
            for _ in 1:3
                SpeedyWeather.ml_seaice_accumulate_kernel!(
                    sea_ice, B, sst, rsds, rld, wind, tas, land, humid, pres, nlayers, true)
            end
            @test all(isfinite, Array(B.acc_rsds))
            @test maximum(Array(B.acc_rsds)) > 0            # it really accumulated

            # and the 5-day update, reading those accumulators where they already are
            before = copy(Array(sic))
            SpeedyWeather.ml_seaice_sweep_kernels!(sea_ice, B, sic, sst, land, 3.0, true, 432000.0)
            after = Array(sic)
            @test all(0 .<= after .<= 1)
            @test after != before                            # the sweep moved the ice
            @test all(isfinite, Array(sst))
        finally
            allowscalar!(true)
        end
    end

    # ---------------------------------------------------------------------------------------------
    # 3. END TO END: the model ITSELF on the GPU, which is what "the component follows a parallel
    # run" means. Nothing about kernels is passed here -- the point is that a user running a
    # parallel GCM does not have to know this component exists.
    #
    # [2026-11-11] This used to claim that a successful run proved GPU-safety by itself. It did
    # not: the testset above restores allowscalar(true) when it finishes, so a scalar index inside
    # the component would have been served slowly instead of throwing -- and one in initialize!
    # (land_fraction read point by point to build the hemisphere masks) duly went unnoticed here
    # and only surfaced in a real T63 run. The component's own initialize! is therefore now run
    # under allowscalar(false) below. `run!` is not: a full timestep goes through the whole library
    # and a throw there would be SpeedyWeather's, not this component's.
    #
    # Skipped, with the reason printed, on a backend that cannot run SpeedyWeather's dynamical core
    # (Metal: Base `sind(::Float32)` goes through a Float64 intermediate). That is a property of the
    # backend, not of this component, and the testset above covers what can be covered there.
    gpu_model_ok = try
        sg = SpectralGrid(trunc = 31, nlayers = 15, Grid = FullGaussianGrid,
                          architecture = SpeedyWeather.GPU())
        initialize!(PrimitiveWetModel(sg))
        true
    catch e
        @info "MLSeaIceV4 end-to-end: a PrimitiveWetModel does not initialise on $(gpu_backend), " *
              "so the model-on-GPU testset is skipped. This is the backend, not the component: " *
              first(replace(sprint(showerror, e), r"\s+" => " "), 200)
        false
    end

    if gpu_model_ok
        @testset "MLSeaIceV4 follows a model that is itself on the GPU" begin
            gpu = SpeedyWeather.GPU()
            # NOTHING about the kernel path is passed: it has to follow on its own
            allowscalar!(false)
            simulation, sea_ice = try
                build_seaice_sim(; arch = gpu)                # <- the component's initialize!
            finally
                allowscalar!(true)
            end
            @test sea_ice.kernel_sweep                       # it followed, unasked
            @test sea_ice.flat_forward                       # with what it requires
            @test sea_ice.cross_from_snapshot
            @test sea_ice.kernel_buffers.resident            # and nothing crosses the bus

            SpeedyWeather.ML_TIME .= 0
            run!(simulation, period = Day(6))                # one 5-day update
            ocean = simulation.variables.prognostic.ocean
            ice = Array(ocean.sea_ice_concentration.data)    # .data: Array(::Field) is the scalar path
            sst = Array(ocean.sea_surface_temperature.data)

            @test all(isfinite, ice)
            @test all(0 .<= ice .<= 1)                       # still a concentration
            @test any(>(0.15f0), ice)                        # and there is ice to speak of
            @test all(x -> isnan(x) || isfinite(x), sst)
            @test SpeedyWeather.ML_TIME[3] > 0               # the per-timestep accumulation ran
            @test SpeedyWeather.ML_TIME[2] > 0               # and the 5-day update ran
            @test any(isfinite, sea_ice.tos_prev_mean)      # carried state made it back to the host
            @test count(sea_ice.initialized) > 0

            @printf("  model on %s: ML cost accum %.2f ms, update %.2f ms over %d steps\n",
                    gpu_backend, 1000 * SpeedyWeather.ML_TIME[1],
                    1000 * SpeedyWeather.ML_TIME[2], Int(SpeedyWeather.ML_TIME[3]))
        end
    end
end
