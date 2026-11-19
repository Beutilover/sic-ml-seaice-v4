# The ocean-temperature column has two forms. The affine one standardises tos with the package's own
# norm_mean/norm_std. The indicator one replaces it by tanh((tos - tos_centre)/tos_scale), which is
# NOT standardised, and appends one extra input carrying the above-fill bit.
#
# Why the second form exists: under ice the reanalysis reports a fill value, the seawater freezing
# point, for most of a dense-ice band's rows, and the real ocean temperatures begin 3.05e-05 K above
# it -- 3.5e-05 of the affine column's span. The indicator form spends the span on the distinction
# that carries the information instead.
#
# The resulting width matches NEITHER layout `_infer_has_tos` can infer, so the form has to be
# declared in `tos_column.txt` beside the weights. A package without that file is the affine form.

@testset "MLSeaIceV4 tos column form" begin
    M = SpeedyWeather.MLSeaIceModelV4

    @testset "declaration" begin
        mktempdir() do dir
            # no file at all: the affine column, which is what every package before 2026-11-18 is
            d = M.read_tos_declaration(dir)
            @test d.tos_form == Int32(0)

            write(joinpath(dir, "tos_column.txt"),
                  "# a comment\ntos_form=indicator\ntos_centre=271.46\ntos_scale=3.0\ntos_tol=0.001\n")
            d = M.read_tos_declaration(dir)
            @test d.tos_form == Int32(1)
            @test d.tos_centre ≈ 271.46f0
            @test d.tos_scale ≈ 3f0
            @test d.tos_tol ≈ 0.001f0

            write(joinpath(dir, "tos_column.txt"), "tos_form=something_else\n")
            @test_throws ErrorException M.read_tos_declaration(dir)
        end
    end

    @testset "layout inference" begin
        # the indicator form is one wider than the plain with-tos layout, and is only accepted when
        # declared: undeclared, that width is a packaging error and must raise
        for ncross in (2, 3), has_tdps in (false, true)
            ntdps = has_tdps ? 1 : 0
            n_ind = 7 + ntdps + ncross
            @test M._infer_has_tos(zeros(Float32, n_ind, 4), ncross, has_tdps, Int32(1))
            @test_throws ErrorException M._infer_has_tos(zeros(Float32, n_ind, 4), ncross, has_tdps)
        end
        # the shipped indicator high band: 6 base + 3 cross + 1 flag
        @test M._infer_has_tos(zeros(Float32, 10, 64), 3, false, Int32(1))
    end

    # identity normalisation, so the returned vector IS the feature ordering and the column's value
    function weights(tos_form; ncross = 3, has_tdps = false, centre = 271.46f0, scale = 3f0, tol = 1f-3)
        nin = 6 + (has_tdps ? 1 : 0) + ncross + (tos_form == Int32(1) ? 1 : 0)
        return M.MLWeightsV4(
            [zeros(Float32, nin, 4), zeros(Float32, 4, 4), zeros(Float32, 4, 4), zeros(Float32, 4, 4)],
            [zeros(Float32, 4) for _ in 1:4],
            zeros(Float32, 4, 3), zeros(Float32, 4),
            zeros(Float32, 3, 4), zeros(Float32, 3),
            zeros(Float32, 1, 4), zeros(Float32, 1),
            zeros(Float32, 1, 4), zeros(Float32, 1),
            0f0, 0f0, 1f0, 5f0,
            zeros(Float32, 6), ones(Float32, 6),
            zeros(Float32, ncross), ones(Float32, ncross),
            0f0, 1f0,
            has_tdps, 0f0, 1f0,
            true, tos_form, centre, scale, tol,
        )
    end

    @testset "the column and the appended bit" begin
        cross = (0.1f0, 0.2f0, 0.3f0)
        # a real ocean temperature, well above the fill value
        x = M.ml_normalize_features_v4(weights(Int32(1)), 0.9f0, 274.0f0, 100f0, 200f0, 5f0, 260f0, 0f0, cross)
        @test length(x) == 10
        @test x[2] ≈ tanh((274.0f0 - 271.46f0) / 3f0)
        @test x[10] == 1f0                       # above the fill value
        @test x[[1, 3, 4, 5, 6]] ≈ Float32[0.9, 100, 200, 5, 260]
        @test x[7:9] ≈ Float32[0.1, 0.2, 0.3]    # the bit goes AFTER the cross columns

        # exactly at the fill value: the bit is off and the column is 0
        x0 = M.ml_normalize_features_v4(weights(Int32(1)), 0.9f0, 271.46f0, 100f0, 200f0, 5f0, 260f0, 0f0, cross)
        @test x0[2] ≈ 0f0
        @test x0[10] == 0f0

        # the affine form is untouched: no extra input, and the column is standardised
        xa = M.ml_normalize_features_v4(weights(Int32(0)), 0.9f0, 274.0f0, 100f0, 200f0, 5f0, 260f0, 0f0, cross)
        @test length(xa) == 9
        @test xa[2] ≈ 274.0f0
    end

    @testset "the flat path agrees with the allocating one" begin
        cross = (0.1f0, 0.2f0, 0.3f0)
        for tos in (271.46f0, 271.4601f0, 274.0f0, 280.0f0), form in (Int32(0), Int32(1))
            w = weights(form)
            fw = M.flatten_weights_v4(w)
            xb = zeros(Float32, fw.nin, 1)
            M.ml_normalize_flat_v4!(xb, 1, fw, 0.9f0, tos, 100f0, 200f0, 5f0, 260f0, 0f0,
                                    cross[1], cross[2], cross[3])
            @test vec(xb) ≈ M.ml_normalize_features_v4(w, 0.9f0, tos, 100f0, 200f0, 5f0, 260f0, 0f0, cross)
        end
    end

    # the shipped package, read the way a run reads it: the form travels inside the directory, so
    # loading it is the only check that the declaration and the weights agree
    @testset "the shipped wttos package declares the form it was trained on" begin
        root = joinpath(pkgdir(SpeedyWeather), "src", "parameterizations")
        high = joinpath(root, "ml_seaice_weights_v4_wttos_north_high")
        if isdir(high)
            w = M.load_ml_weights_v4(high)
            @test w.has_tos
            @test w.tos_form == Int32(1)
            @test w.tos_centre ≈ 271.46f0
            @test w.tos_scale ≈ 3f0
            @test w.tos_tol ≈ 0.001f0
            @test size(w.W[1], 1) == 10          # 6 base + 3 cross + the appended bit
            @test !w.has_tdps
            # the other two bands of the same package keep the affine column
            for band in ("unified_low", "unified_mid")
                d = joinpath(root, "ml_seaice_weights_v4_wttos_" * band)
                isdir(d) || continue
                ww = band == "unified_low" ? M.load_ml_weights_changegate_v4(d) : M.load_ml_weights_v4(d)
                @test ww.tos_form == Int32(0)
                @test size(ww.W[1], 1) == 10     # 6 base + tdps + 3 cross, no bit
            end
        end
    end
end
