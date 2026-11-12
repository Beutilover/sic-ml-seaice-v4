# The kernel sweep must be the SAME computation as the serial one, not merely a close one: it runs
# in production in place of it. Checked here on the CPU backend (so no GPU is needed) through a real
# coupled run, because that is the only way the carried state -- the history, the previous tos and
# the initialised flags -- gets exercised across more than one update.
using SpeedyWeather, Test, Dates

const SEAICE_WEIGHTS = joinpath(@__DIR__, "..", "..", "src", "parameterizations")
const SEAICE_SET = "ml_seaice_weights_v4_lapmtas"

seaice_weights_available() =
    isdir(joinpath(SEAICE_WEIGHTS, "$(SEAICE_SET)_unified_mid")) &&
    isfile(joinpath(SEAICE_WEIGHTS, "ml_seaice_weights", "initial_sic_t31.bin"))

function build_seaice_sim(; kwargs...)
    spectral_grid = SpectralGrid(trunc = 31, nlayers = 15, Grid = FullGaussianGrid,
                                 architecture = SpeedyWeather.CPU())
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
    @info "MLSeaIceV4 kernel sweep: weight package $(SEAICE_SET) not present, skipping"
else
    @testset "MLSeaIceV4 kernel_sweep preconditions" begin
        # all three are refused rather than silently switched on, because each one is a statement
        # about what makes the parallel form legal or meaningful
        @test_throws ErrorException build_seaice_sim(kernel_sweep = true, cross_from_snapshot = true)
        @test_throws ErrorException build_seaice_sim(kernel_sweep = true, flat_forward = true)
        @test_throws ErrorException build_seaice_sim(kernel_sweep = true, flat_forward = true,
                                                    cross_from_snapshot = true,
                                                    cross_snapshot_report = 2)
    end

    # [2026-11-11] The two paths are BIT-IDENTICAL in an ordinary build -- verified on a real
    # coupled T31 run over one, two and three 5-day updates, on the prognostic fields and on the
    # carried state. They are not bit-identical under `--check-bounds=yes`, and that is a property
    # of the FLAG rather than of this component: measured at one update, the serial path compared
    # against ITSELF across the two build modes gives a global sea-ice sum of 666.8807758484124
    # with the flag against 666.8807756109252 without it. Forcing bounds checks suppresses
    # vectorisation, which changes a reduction order, and the serial loop and the kernel are two
    # inlining contexts of the same arithmetic that the compiler then treats differently (kernel
    # under the flag: 666.8807761166333). Every one of those numbers is reproducible run to run, so
    # the test demands exactness where exactness is meaningful and a tight bound otherwise.
    const EXACT = Base.JLOptions().check_bounds != 1

    same(a, b; atol) = EXACT ? isequal(a, b) : all(isapprox.(a, b; atol = atol, nans = true))

    @testset "MLSeaIceV4 kernel_sweep matches the serial sweep" begin
        both = (flat_forward = true, cross_from_snapshot = true)
        results = map((false, true)) do kernel
            simulation, sea_ice = build_seaice_sim(; both..., kernel_sweep = kernel)
            run!(simulation, period = Day(6))      # one 5-day update, at day 5
            ocean = simulation.variables.prognostic.ocean
            return (sic = copy(Array(ocean.sea_ice_concentration)),
                    sst = copy(Array(ocean.sea_surface_temperature)),
                    hist1 = copy(sea_ice.sic_hist1), hist2 = copy(sea_ice.sic_hist2),
                    hist3 = copy(sea_ice.sic_hist3), tos = copy(sea_ice.tos_prev_mean),
                    initialized = copy(sea_ice.initialized))
        end
        serial, kernel = results

        # the update has to have done something, or the comparison below is vacuous
        @test count(serial.initialized) > 0
        @test any(isfinite, serial.tos)
        @test any(>(0.15f0), serial.sic)

        @test same(serial.sic, kernel.sic; atol = 1.0e-5)
        @test same(serial.sst, kernel.sst; atol = 1.0e-2)

        # the carried state, which the kernel path holds on the device and copies back. The run
        # script checkpoints these HOST arrays and restores them on restart, so a stale copy would
        # silently reset the sea-ice history -- a wrong answer with no error.
        @test serial.initialized == kernel.initialized
        for (name, atol) in ((:hist1, 1.0e-5), (:hist2, 1.0e-5), (:hist3, 1.0e-5), (:tos, 1.0e-2))
            a = getfield(serial, name); b = getfield(kernel, name)
            @test isequal(isfinite.(a), isfinite.(b))   # the same points are live on both paths
            @test same(a, b; atol = atol)
        end
    end
end
