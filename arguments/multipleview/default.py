ModelParams = dict(
    # Synthetic multi-view datasets (e.g. Blender) assume a white background.
    # Keep this enabled by default so renders do not collapse to black.
    white_background=True,
)

ModelHiddenParams = dict(
    kplanes_config={
        "grid_dimensions": 2,
        "input_coordinate_dim": 4,
        "output_coordinate_dim": 24,
        "resolution": [96, 96, 96, 200],
    },
    multires=[1, 2, 4],
    defor_depth=1,
    net_width=192,
    plane_tv_weight=0.00028,
    time_smoothness_weight=0.0007,
    l1_time_planes=0.0001,
    no_do=False,
    no_dshs=False,
    no_ds=False,
    empty_voxel=False,
    render_process=True,
    static_mlp=False,
)

OptimizationParams = dict(
    dataloader=True,
    iterations=18_000,
    batch_size=3,
    coarse_iterations=3_500,
    densify_until_iter=17_200,
    opacity_reset_interval=0,
    opacity_threshold_coarse=0.0025,
    opacity_threshold_fine_init=0.0025,
    opacity_threshold_fine_after=0.0025,
    random_background=True,
    random_background_coarse_only=True,
)

test_iterations = [5_000, 11_000, 15_000, 18_000]
save_iterations = [11_000, 15_000, 18_000]
checkpoint_iterations = [11_000, 15_000]

post_render_train = True
