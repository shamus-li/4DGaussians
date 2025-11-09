from typing import Optional


def _snap_iteration(value: int, base: int = 1000) -> int:
    """Round iteration counts to the nearest multiple of base (default 1k)."""
    if value <= 0:
        return base
    rounded = int(round(value / base)) * base
    return max(base, rounded)


def generate_multiview_config(
    camera_count: int,
    frame_count: int,
    *,
    dataset_name: Optional[str] = None,
) -> str:
    """Return a tuned config.py template for multi-view datasets."""
    cameras = max(1, int(camera_count or 0))
    frames = max(1, int(frame_count or 0))

    batch_size = max(1, min(4, cameras))

    dataset_hint = (dataset_name or "").lower()
    is_special = any(token in dataset_hint for token in ("iphone", "stereo"))

    iterations = 18_000
    coarse_iterations = 3_500
    base_densify_margin = 2_000
    densify_until = max(6_000, iterations - base_densify_margin)

    opacity_threshold = 0.0025
    opacity_reset_interval = 3_000
    max_screen_size = 18
    percent_dense = 0.02 if cameras <= 4 else 0.028
    pruning_interval = 100
    prune_min_points = 60000 if cameras > 8 else 30000
    # Default: allow densification and pruning from iteration 500
    densify_from_iter = 500
    pruning_from_iter = 500

    if is_special:
        if cameras <= 3:
            iterations = 24_000
            coarse_iterations = 4_000
            densify_until = max(densify_until, iterations - 500)
            percent_dense = 0.04
            pruning_interval = 100
            max_screen_size = 25
            # CRITICAL: Very conservative thresholds for few-camera scenarios to prevent collapse
            prune_min_points = 100_000  # Effectively disable explicit pruning
            opacity_threshold = 0.002   # More conservative than 0.0035
            opacity_reset_interval = 2000  # Enable periodic recovery
            # Disable densification initially - let initialization stabilize first
            densify_from_iter = 2000  # Delay densification significantly
            pruning_from_iter = iterations + 1  # Disable explicit pruning entirely
        elif cameras <= 6:
            iterations = max(iterations, 20_000)
            coarse_iterations = max(coarse_iterations, 3_800)
            densify_until = max(densify_until, iterations - 1_200)
            percent_dense = max(percent_dense, 0.038)
            pruning_interval = 100
            max_screen_size = 22
            prune_min_points = 18_000
            opacity_threshold = max(opacity_threshold, 0.0032)
            opacity_reset_interval = max(opacity_reset_interval, 6_000)
        elif cameras <= 9:
            iterations = max(iterations, 19_000)
            densify_until = max(densify_until, iterations - 1_500)
            percent_dense = max(percent_dense, 0.036)
            pruning_interval = 100
            max_screen_size = 20
            prune_min_points = 20_000
            opacity_threshold = max(opacity_threshold, 0.003)
        elif cameras <= 15:
            iterations = max(iterations, 18_500)
            densify_until = max(densify_until, iterations - 2_000)
            percent_dense = max(percent_dense, 0.033)
            pruning_interval = 100
            max_screen_size = 20
            prune_min_points = 25_000
            opacity_threshold = max(opacity_threshold, 0.003)
        else:
            opacity_threshold = max(opacity_threshold, 0.003)
            percent_dense = max(percent_dense, 0.03)
            pruning_interval = min(pruning_interval, 100)
            max_screen_size = min(max_screen_size, 20)
            prune_min_points = min(prune_min_points, 30_000)
    else:
        if cameras <= 3:
            iterations = max(iterations, 21_000)
            coarse_iterations = max(coarse_iterations, 3_800)
            densify_until = max(densify_until, iterations - 800)
            percent_dense = max(percent_dense, 0.036)
            pruning_interval = 100
            max_screen_size = 22
            prune_min_points = 15_000
            opacity_threshold = max(opacity_threshold, 0.0028)
            opacity_reset_interval = 0
        elif cameras <= 6:
            iterations = max(iterations, 19_000)
            densify_until = max(densify_until, iterations - 1_200)
            percent_dense = max(percent_dense, 0.033)
            pruning_interval = 100
            max_screen_size = 20
            prune_min_points = 20_000
            opacity_reset_interval = max(opacity_reset_interval, 6_000)
        elif cameras <= 9:
            percent_dense = max(percent_dense, 0.031)
            pruning_interval = 100
            max_screen_size = 20
            prune_min_points = min(prune_min_points, 28_000)
            densify_until = max(densify_until, iterations - 1_500)
        else:
            percent_dense = max(percent_dense, 0.028)
            pruning_interval = min(pruning_interval, 100)
            max_screen_size = min(max_screen_size, 20)
            prune_min_points = min(prune_min_points, 36_000)
            densify_until = max(densify_until, iterations - 1_800)

    densify_until = max(densify_until, coarse_iterations + 1_000)
    densify_until = min(densify_until, iterations - 200)

    temporal_resolution = max(150, min(300, frames))

    first_test = _snap_iteration(max(coarse_iterations + 1_000, 4_000))
    mid_test = _snap_iteration(max(int(iterations * 0.6), first_test + 1_000))
    densify_test = _snap_iteration(densify_until)
    final_test = iterations
    test_iterations = sorted({first_test, mid_test, densify_test, final_test})

    save_candidates = sorted({mid_test, _snap_iteration(densify_until), iterations})
    save_iterations = [value for value in save_candidates if value <= iterations]

    checkpoint_iterations = save_iterations[:-1] if len(save_iterations) > 1 else []

    header_comment = (
        f"# Auto-generated config for {dataset_name}\n"
        if dataset_name
        else "# Auto-generated config for multi-view dataset\n"
    )

    header_line = header_comment.rstrip("\n")

    lines = [
        header_line,
        f"# cameras: {cameras}, frames per camera: {frames}",
        "",
        "ModelParams = dict(",
        "    white_background=True,",
        ")",
        "",
        "ModelHiddenParams = dict(",
        "    kplanes_config={",
        "        'grid_dimensions': 2,",
        "        'input_coordinate_dim': 4,",
        "        'output_coordinate_dim': 24,",
        f"        'resolution': [96, 96, 96, {temporal_resolution}]",
        "    },",
        "    multires=[1, 2, 4],",
        "    defor_depth=1,",
        "    net_width=192,",
        "    plane_tv_weight=0.00028,",
        "    time_smoothness_weight=0.0007,",
        "    l1_time_planes=0.0001,",
        "    no_do=False,",
        "    no_dshs=False,",
        "    no_ds=False,",
        "    empty_voxel=False,",
        "    render_process=True,",
        "    static_mlp=False,",
        ")",
        "",
        "OptimizationParams = dict(",
        "    dataloader=True,",
        f"    iterations={iterations},",
        f"    batch_size={batch_size},",
        f"    coarse_iterations={coarse_iterations},",
        f"    densify_until_iter={densify_until},",
        f"    densify_from_iter={densify_from_iter},",
        f"    pruning_from_iter={pruning_from_iter},",
        f"    opacity_reset_interval={opacity_reset_interval},",
        f"    opacity_threshold_coarse={opacity_threshold},",
        f"    opacity_threshold_fine_init={opacity_threshold},",
        f"    opacity_threshold_fine_after={opacity_threshold},",
        f"    percent_dense={percent_dense},",
        f"    pruning_interval={pruning_interval},",
        f"    max_screen_size={max_screen_size},",
        f"    prune_min_points={prune_min_points},",
        "    random_background=True,",
        "    random_background_coarse_only=False,",
    ")",
        "",
        f"test_iterations = {test_iterations}",
        f"save_iterations = {save_iterations}",
        f"checkpoint_iterations = {checkpoint_iterations}",
        "",
    ]

    return "\n".join(lines)
