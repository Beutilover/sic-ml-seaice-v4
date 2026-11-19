# A band may be trained WITHOUT the ocean temperature column. In the high band tos is the seawater
# freezing point the reanalysis reports under ice for 98% of the training rows, so it is not
# identifiable there and the column is dropped rather than normalised.
#
# Which layout a weight package carries is inferred from the first layer's input width, so no extra
# file is written beside the weights:
#     with tos      in_dim = 6 + has_tdps + n_cross
#     without tos   in_dim = 5 + has_tdps + n_cross
# A width matching neither is a packaging error and must raise rather than silently feed the network
# a shifted feature vector -- the failure mode of the 2026-09-29 feature-order bug.

@testset "MLSeaIceV4 without the tos column" begin
    M = SpeedyWeather.MLSeaIceModelV4

    @testset "layout inference" begin
        for ncross in (2, 3), has_tdps in (false, true)
            ntdps = has_tdps ? 1 : 0
            # _read_bin yields (in_dim, hidden), the orientation the forward pass needs
            @test M._infer_has_tos(zeros(Float32, 6 + ntdps + ncross, 4), ncross, has_tdps)
            @test !M._infer_has_tos(zeros(Float32, 5 + ntdps + ncross, 4), ncross, has_tdps)
            # off by more than one: neither layout, so it must raise
            @test_throws ErrorException M._infer_has_tos(
                zeros(Float32, 8 + ntdps + ncross, 4), ncross, has_tdps)
        end
        # the shipped widths are unambiguous: high band 9 with tos and 8 without, low/mid 10 and 9
        @test M._infer_has_tos(zeros(Float32, 9, 64), 3, false)
        @test !M._infer_has_tos(zeros(Float32, 8, 64), 3, false)
        @test M._infer_has_tos(zeros(Float32, 10, 64), 3, true)
        @test !M._infer_has_tos(zeros(Float32, 9, 64), 3, true)
    end

    # an identity normalisation, so the returned vector IS the feature ordering
    function weights(; ncross, has_tdps, has_tos)
        nin = (has_tos ? 6 : 5) + (has_tdps ? 1 : 0) + ncross
        return M.MLWeightsV4(
            [zeros(Float32, nin, 4), zeros(Float32, 4, 4), zeros(Float32, 4, 4), zeros(Float32, 4, 4)],
            [zeros(Float32, 4) for _ in 1:4],
            zeros(Float32, 4, 3), zeros(Float32, 4),
            zeros(Float32, 3, 4), zeros(Float32, 3),
            zeros(Float32, 1, 4), zeros(Float32, 1),
            zeros(Float32, 1, 4), zeros(Float32, 1),
            0f0, 0f0, 1f0, 5f0,
            zeros(Float32, 6), ones(Float32, 6),            # norm_mean / norm_std -> identity
            zeros(Float32, ncross), ones(Float32, ncross),  # cross_mean / cross_std
            0f0, 1f0,                                       # rld_mean / rld_std
            has_tdps, 0f0, 1f0,                             # has_tdps, tdps_mean / tdps_std
            has_tos,
        )
    end

    @testset "feature vector drops tos and keeps the order of the rest" begin
        # sentinels: sic=1 tos=2 rsds=3 rld=4 wind=5 tas=6 tdps=7 cross=(8,9)
        w = weights(ncross = 2, has_tdps = false, has_tos = false)
        x = SpeedyWeather.ml_normalize_features_v4(w, 1f0, 2f0, 3f0, 4f0, 5f0, 6f0, 7f0, (8f0, 9f0))
        @test x == Float32[1, 3, 4, 5, 6, 8, 9]        # tos (2) absent, nothing else moved
        @test length(x) == size(w.W[1], 1)

        w = weights(ncross = 2, has_tdps = true, has_tos = false)
        x = SpeedyWeather.ml_normalize_features_v4(w, 1f0, 2f0, 3f0, 4f0, 5f0, 6f0, 7f0, (8f0, 9f0))
        @test x == Float32[1, 3, 4, 5, 6, 7, 8, 9]     # tdps still precedes the cross columns
        @test length(x) == size(w.W[1], 1)

        # the with-tos path is untouched
        w = weights(ncross = 2, has_tdps = true, has_tos = true)
        x = SpeedyWeather.ml_normalize_features_v4(w, 1f0, 2f0, 3f0, 4f0, 5f0, 6f0, 7f0, (8f0, 9f0))
        @test x == Float32[1, 2, 3, 4, 5, 6, 7, 8, 9]
    end

    @testset "tos is genuinely ignored, not merely reordered" begin
        w = weights(ncross = 3, has_tdps = false, has_tos = false)
        a = SpeedyWeather.ml_normalize_features_v4(w, 0.9f0, 271.46f0, 100f0, 200f0, 5f0, 250f0, 0f0,
                                                   (0.1f0, 0.2f0, 0.3f0))
        b = SpeedyWeather.ml_normalize_features_v4(w, 0.9f0, 350.00f0, 100f0, 200f0, 5f0, 250f0, 0f0,
                                                   (0.1f0, 0.2f0, 0.3f0))
        @test a == b
    end

    @testset "the real exported packages load with the right flag" begin
        root = joinpath(@__DIR__, "..", "..", "src", "parameterizations")
        for (pkg, want_tos) in (("ml_seaice_weights_v4_lapmtas_north_high", true),
                                ("ml_seaice_weights_v4_lapmtasnotos_north_high", false))
            dir = joinpath(root, pkg)
            isdir(dir) || continue
            w = M.load_ml_weights_v4(dir; tau_floor = 1.0f0, dt = 5.0f0)
            @test w.has_tos == want_tos
            @test size(w.W[1], 1) == (want_tos ? 6 : 5) + length(w.cross_mean)
        end
    end

    @testset "the change-gate weights carry the flag too" begin
        ncross = 2
        nin = 5 + ncross
        w = M.MLWeightsChangeGateV4(
            [zeros(Float32, nin, 4), zeros(Float32, 4, 4), zeros(Float32, 4, 4), zeros(Float32, 4, 4)],
            [zeros(Float32, 4) for _ in 1:4],
            zeros(Float32, 4, 3), zeros(Float32, 4),
            zeros(Float32, 2, 4), zeros(Float32, 2),
            zeros(Float32, 1, 4), zeros(Float32, 1),
            0f0, 1f0, 5f0,
            zeros(Float32, 6), ones(Float32, 6),
            zeros(Float32, ncross), ones(Float32, ncross),
            0f0, 1f0,
            false, 0f0, 1f0,
            false,
        )
        x = SpeedyWeather.ml_normalize_features_v4(w, 1f0, 2f0, 3f0, 4f0, 5f0, 6f0, 7f0, (8f0, 9f0))
        @test x == Float32[1, 3, 4, 5, 6, 8, 9]
        @test length(x) == nin
    end
end
