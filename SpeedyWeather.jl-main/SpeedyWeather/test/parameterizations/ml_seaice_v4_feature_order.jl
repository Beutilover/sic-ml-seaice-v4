# Regression test for the 2026-09-29 feature-order bug in ml_normalize_features_v4.
#
# The weights are trained in Python on the layout emitted by
# sic_pipeline_lib.build_feature_matrix_from_raw followed by the cross columns:
#     [sic, tos, rsds, rld, sfcwind, tas, tdps, cross...]
# This file used to append tdps AFTER the cross columns, cycling the last three inputs for every
# band that carries tdps. The high band has none, so it stayed correct and the fault went unnoticed
# through every online run until a Torch/Julia parity check was written.
#
# The test does not need the real weights: it builds a tiny MLWeightsV4 whose normalization is the
# identity, so the returned vector IS the feature order, and asserts tdps sits at position 7.

@testset "MLSeaIceV4 feature order: tdps precedes the cross columns" begin
    ncross = 2
    nin = 6 + 1 + ncross
    # a weights object whose normalization is the identity, so the returned vector IS the ordering.
    # Field order follows the struct definition: ..., norm_mean, norm_std, cross_mean, cross_std,
    # rld_mean, rld_std, has_tdps, tdps_mean, tdps_std, has_tos.
    w = SpeedyWeather.MLSeaIceModelV4.MLWeightsV4(
        [zeros(Float32, 4, nin), zeros(Float32, 4, 4), zeros(Float32, 4, 4), zeros(Float32, 4, 4)],
        [zeros(Float32, 4) for _ in 1:4],
        zeros(Float32, 4, 3), zeros(Float32, 4),
        zeros(Float32, 3, 4), zeros(Float32, 3),
        zeros(Float32, 1, 4), zeros(Float32, 1),
        zeros(Float32, 1, 4), zeros(Float32, 1),
        0f0, 0f0, 1f0, 5f0,
        zeros(Float32, 6), ones(Float32, 6),            # norm_mean / norm_std -> identity
        zeros(Float32, ncross), ones(Float32, ncross),  # cross_mean / cross_std
        0f0, 1f0,                                       # rld_mean / rld_std
        true, 0f0, 1f0,                                 # has_tdps, tdps_mean / tdps_std
        true,                                           # has_tos
    )
    # distinct sentinel values so any permutation is visible
    x = SpeedyWeather.ml_normalize_features_v4(w, 1f0, 2f0, 3f0, 4f0, 5f0, 6f0, 7f0, (8f0, 9f0))
    @test length(x) == nin
    @test x == Float32[1, 2, 3, 4, 5, 6, 7, 8, 9]      # tdps (7) BEFORE cross (8, 9)
    @test x[7] == 7f0                                   # the slot the bug filled with cross_ns
end
