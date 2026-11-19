# Regression test for the curvature-column class of defect, of which there have been three:
# 2026-09-29 (feature order), 2026-10-15 (the Antarctic branch built q4 unconditionally) and
# 2026-10-27 (`lapmtas` missing from a run script's curvature_tanh_scale list, which silently fed
# the network the 4th-difference column instead of the tanh physical Laplacian its weights were
# trained on, invalidated three 3-year coupled runs, and made a separate experiment a no-op).
#
# Every one of these was possible because the column's DEFINITION -- curvature_tanh_scale together
# with curvature_laplacian -- lived in a caller's flag list, separately from the weights fitted to
# it, and the variants are all three-column forms of the same shape, so a mismatch is silent.
# Each shipped weight package now carries a curvature.txt declaring the column it was trained on,
# and MLSeaIceV4.initialize! refuses to run on a mismatch.

@testset "MLSeaIceV4 curvature declaration" begin
    spectral_grid = SpectralGrid(trunc=31, nlayers=8)

    @testset "shipped defaults agree with their own weights" begin
        sea_ice = MLSeaIceV4(spectral_grid)
        @test SpeedyWeather.check_curvature_declarations(sea_ice) === nothing
    end

    @testset "a mismatch is an error, not a silent wrong column" begin
        # the shipped package is a tanh physical-Laplacian one, so either half of the pair being
        # wrong must be caught -- these are exactly the two 2026-10-27 failure modes.
        sea_ice = MLSeaIceV4(spectral_grid)
        sea_ice.curvature_laplacian = !sea_ice.curvature_laplacian
        @test_throws ErrorException SpeedyWeather.check_curvature_declarations(sea_ice)

        sea_ice = MLSeaIceV4(spectral_grid)
        sea_ice.curvature_tanh_scale = 0.0      # the 4th-difference column
        @test_throws ErrorException SpeedyWeather.check_curvature_declarations(sea_ice)
    end

    @testset "declarations are readable and self-consistent" begin
        dir = dirname(pathof(SpeedyWeather))
        packages = filter(readdir(joinpath(dir, "parameterizations"))) do name
            startswith(name, "ml_seaice_weights_v4_") &&
                isdir(joinpath(dir, "parameterizations", name))
        end
        @test !isempty(packages)
        for name in packages
            path = joinpath(dir, "parameterizations", name)
            decl = SpeedyWeather.read_curvature_declaration(path)
            @test decl !== nothing                       # every shipped package declares one
            @test decl.cross_columns >= 2
            if decl.cross_columns >= 3
                # the only two columns ever exported: tanh(q/0.5) and the raw 4th difference
                @test decl.curvature_tanh_scale in (0.0, 0.5)
                # the Laplacian form only exists for the tanh column
                @test !(decl.curvature_laplacian && decl.curvature_tanh_scale == 0.0)
            else
                @test !haskey(decl, :curvature_tanh_scale)
            end
        end
    end

    @testset "an undeclared package warns rather than failing to load" begin
        mktempdir() do tmp
            @test SpeedyWeather.read_curvature_declaration(tmp) === nothing
            sea_ice = MLSeaIceV4(spectral_grid; weights_dir_mid=tmp)
            @test_logs (:warn,) match_mode=:any SpeedyWeather.check_curvature_declarations(sea_ice)
        end
    end
end
