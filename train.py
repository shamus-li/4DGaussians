#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import copy
import os
import random
import sys
from argparse import ArgumentParser, Namespace
from random import randint

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from arguments import ModelHiddenParams, ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import render
from scene import GaussianModel, Scene
from utils.general_utils import safe_state
from utils.image_utils import psnr
from utils.loader_utils import FineSampler, get_stamp_list
from utils.loss_utils import l1_loss, ssim
from utils.params_utils import load_config, merge_hparams
from utils.scene_utils import render_training_image
from utils.timer import Timer

to8b = lambda x: (255 * np.clip(x.cpu().numpy(), 0, 1)).astype(np.uint8)

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def scene_reconstruction(
    dataset,
    opt,
    hyper,
    pipe,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
    debug_from,
    gaussians,
    scene,
    stage,
    tb_writer,
    train_iter,
    timer,
):
    first_iter = 0

    gaussians.training_setup(opt)
    print(f"[{stage}] Initial gaussian count: {gaussians.get_xyz.shape[0]}")
    if checkpoint:
        # breakpoint()
        if stage == "coarse" and stage not in checkpoint:
            print("start from fine stage, skip coarse stage.")
            # process is in the coarse stage, but start from fine stage
            return
        if stage in checkpoint:
            (model_params, first_iter) = torch.load(
                checkpoint, map_location="cuda", weights_only=False
            )
            gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_psnr_for_log = 0.0

    final_iter = train_iter

    guard_min_points = getattr(opt, "min_gaussians", 0) if stage == "coarse" else 0
    guard_resume_iter = getattr(opt, "min_gaussians_warmup", 0)
    guard_cooldown = max(getattr(opt, "min_gaussians_cooldown", 200), 50)

    progress_bar = tqdm(range(first_iter, final_iter), desc="Training progress")
    first_iter += 1
    # lpips_model = lpips.LPIPS(net="alex").cuda()
    video_cams = scene.getVideoCameras()
    test_cams = scene.getTestCameras()
    train_cams = scene.getTrainCameras()

    if not viewpoint_stack and not opt.dataloader:
        # dnerf's branch
        viewpoint_stack = [i for i in train_cams]
        temp_list = copy.deepcopy(viewpoint_stack)
    #
    batch_size = opt.batch_size
    print("data loading done")

    max_screen_size = getattr(opt, "max_screen_size", 0.0)
    prune_min_points = getattr(opt, "prune_min_points", 200000)

    if opt.dataloader:
        viewpoint_stack = scene.getTrainCameras()
        if opt.custom_sampler is not None:
            sampler = FineSampler(viewpoint_stack)
            viewpoint_stack_loader = DataLoader(
                viewpoint_stack,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=0,
                collate_fn=list,
            )
            random_loader = False
        else:
            viewpoint_stack_loader = DataLoader(
                viewpoint_stack,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                collate_fn=list,
            )
            random_loader = True
        loader = iter(viewpoint_stack_loader)

    # dynerf, zerostamp_init
    # breakpoint()
    if stage == "coarse" and opt.zerostamp_init:
        load_in_memory = True
        # batch_size = 4
        temp_list = get_stamp_list(viewpoint_stack, 0)
        viewpoint_stack = temp_list.copy()
    else:
        load_in_memory = False
        #
    count = 0
    for iteration in range(first_iter, final_iter + 1):
        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Safeguard: Check for Gaussian collapse and reinitialize if needed
        current_count = gaussians.get_xyz.shape[0]
        critical_threshold = max(guard_min_points, 100) if guard_min_points else 100

        if iteration > guard_resume_iter and current_count < critical_threshold:
            if getattr(scene, "base_point_cloud", None) is not None and current_count > 0:
                print(
                    f"[SAFEGUARD] Reinitializing Gaussians from source point cloud at iteration {iteration} "
                    f"(count={current_count}, threshold={critical_threshold})"
                )
                gaussians.create_from_pcd(
                    scene.base_point_cloud, scene.cameras_extent, scene.maxtime
                )
                gaussians.training_setup(opt)
                guard_resume_iter = iteration + guard_cooldown
                continue
            elif current_count == 0:
                # CRITICAL: Cannot continue with 0 Gaussians
                error_msg = (
                    f"[CRITICAL ERROR] Gaussian collapse detected at iteration {iteration}!\n"
                    f"All {critical_threshold if guard_min_points else 'tracked'} Gaussians have been pruned.\n"
                )
                if not getattr(scene, "base_point_cloud", None):
                    error_msg += "No base point cloud available for recovery.\n"
                error_msg += "Training cannot continue. Please check your hyperparameters."
                print(error_msg)
                raise RuntimeError(error_msg)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera

        # dynerf's branch
        if opt.dataloader and not load_in_memory:
            try:
                viewpoint_cams = next(loader)
            except StopIteration:
                if not random_loader:
                    viewpoint_stack_loader = DataLoader(
                        viewpoint_stack,
                        batch_size=opt.batch_size,
                        shuffle=True,
                        num_workers=0,
                        collate_fn=list,
                    )
                    random_loader = True
                loader = iter(viewpoint_stack_loader)

        else:
            idx = 0
            viewpoint_cams = []

            while idx < batch_size:
                viewpoint_cam = viewpoint_stack.pop(
                    randint(0, len(viewpoint_stack) - 1)
                )
                if not viewpoint_stack:
                    viewpoint_stack = temp_list.copy()
                viewpoint_cams.append(viewpoint_cam)
                idx += 1
            if len(viewpoint_cams) == 0:
                continue
        # print(len(viewpoint_cams))
        # breakpoint()
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        images = []
        gt_images = []
        radii_list = []
        visibility_filter_list = []
        viewspace_point_tensor_list = []
        alpha_masks = []
        all_cams_have_alpha = True
        use_random_background = getattr(opt, "random_background", False)
        coarse_only = getattr(opt, "random_background_coarse_only", False)
        if use_random_background and (not coarse_only or stage == "coarse"):
            iteration_background = torch.rand_like(background)
        else:
            iteration_background = background
        for viewpoint_cam in viewpoint_cams:
            render_pkg = render(
                viewpoint_cam,
                gaussians,
                pipe,
                iteration_background,
                stage=stage,
                cam_type=scene.dataset_type,
            )
            image, viewspace_point_tensor, visibility_filter, radii = (
                render_pkg["render"],
                render_pkg["viewspace_points"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
            )
            if viewspace_point_tensor.shape[0] == 0 or viewspace_point_tensor.shape[1] == 0:
                print(
                    f"[WARN] Empty viewspace tensor at iteration {iteration} (shape={viewspace_point_tensor.shape})"
                )
            expected_vp_shape = viewspace_point_tensor.shape
            viewspace_point_tensor.retain_grad()
            viewspace_point_tensor.register_hook(
                lambda grad: grad
                if grad.shape == expected_vp_shape
                else grad.new_zeros(expected_vp_shape)
            )
            images.append(image.unsqueeze(0))
            if scene.dataset_type != "PanopticSports":
                gt_image = viewpoint_cam.original_image.cuda()
                alpha_mask = getattr(viewpoint_cam, "alpha_mask", None)
                if alpha_mask is not None:
                    alpha = alpha_mask.to(gt_image.device)
                    alpha_masks.append(alpha.unsqueeze(0))
                else:
                    alpha = torch.ones((1, gt_image.shape[1], gt_image.shape[2]), device=gt_image.device)
                    all_cams_have_alpha = False
                gt_image = gt_image * alpha + iteration_background.view(3, 1, 1) * (1.0 - alpha)
            else:
                gt_image = viewpoint_cam["image"].cuda()
                all_cams_have_alpha = False

            gt_images.append(gt_image.unsqueeze(0))
            radii_list.append(radii.unsqueeze(0))
            visibility_filter_list.append(visibility_filter.unsqueeze(0))
            viewspace_point_tensor_list.append(viewspace_point_tensor)

        radii = torch.cat(radii_list, 0).max(dim=0).values
        visibility_filter = torch.cat(visibility_filter_list).any(dim=0)
        image_tensor = torch.cat(images, 0)
        gt_image_tensor = torch.cat(gt_images, 0)
        # Loss
        # breakpoint()
        if (
            scene.dataset_type != "PanopticSports"
            and all_cams_have_alpha
            and len(alpha_masks) == len(viewpoint_cams)
            and alpha_masks
        ):
            alpha_tensor = torch.cat(alpha_masks, dim=0).to(image_tensor.device)
            alpha_expanded = alpha_tensor.expand(-1, image_tensor.shape[1], -1, -1)
            masked_prediction = image_tensor * alpha_expanded
            masked_target = gt_image_tensor[:, :3, :, :] * alpha_expanded
            mask_mass = alpha_tensor.sum(dim=(1, 2, 3)).clamp_min(1e-6)
            Ll1 = (
                torch.abs(masked_prediction - masked_target).sum(dim=(1, 2, 3))
                / (mask_mass * image_tensor.shape[1])
            ).mean()
        else:
            Ll1 = l1_loss(image_tensor, gt_image_tensor[:, :3, :, :])

        psnr_ = psnr(image_tensor, gt_image_tensor).mean().double()
        # norm

        loss = Ll1
        if stage == "fine" and hyper.time_smoothness_weight != 0:
            # tv_loss = 0
            tv_loss = gaussians.compute_regulation(
                hyper.time_smoothness_weight,
                hyper.l1_time_planes,
                hyper.plane_tv_weight,
            )
            loss += tv_loss
        if opt.lambda_dssim != 0:
            ssim_loss = ssim(image_tensor, gt_image_tensor)
            loss += opt.lambda_dssim * (1.0 - ssim_loss)
        # if opt.lambda_lpips !=0:
        #     lpipsloss = lpips_loss(image_tensor,gt_image_tensor,lpips_model)
        #     loss += opt.lambda_lpips * lpipsloss

        loss.backward()
        if torch.isnan(loss).any():
            print("loss is nan,end training, reexecv program now.")
            os.execv(sys.executable, [sys.executable] + sys.argv)
        viewspace_point_tensor_grad = torch.zeros_like(viewspace_point_tensor)
        for idx in range(0, len(viewspace_point_tensor_list)):
            viewspace_point_tensor_grad = (
                viewspace_point_tensor_grad + viewspace_point_tensor_list[idx].grad
            )
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_psnr_for_log = 0.4 * psnr_ + 0.6 * ema_psnr_for_log
            total_point = gaussians._xyz.shape[0]
            if iteration % 10 == 0:
                progress_bar.set_postfix(
                    {
                        "Loss": f"{ema_loss_for_log:.{7}f}",
                        "psnr": f"{psnr_:.{2}f}",
                        "point": f"{total_point}",
                    }
                )
                progress_bar.update(10)
            if iteration % 500 == 0 or iteration == first_iter:
                gaussian_count = gaussians.get_xyz.shape[0]
                if gaussian_count > 0:
                    opacity_stats = gaussians.get_opacity
                    print(
                        f"[{stage}] Iteration {iteration}: {gaussian_count} gaussians, "
                        f"opacity mean={opacity_stats.mean().item():.4f}, "
                        f"min={opacity_stats.min().item():.4f}, max={opacity_stats.max().item():.4f}"
                    )
                else:
                    print(f"[{stage}] Iteration {iteration}: WARNING - 0 gaussians remaining!")
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            timer.pause()
            training_report(
                tb_writer,
                iteration,
                Ll1,
                loss,
                l1_loss,
                iter_start.elapsed_time(iter_end),
                testing_iterations,
                scene,
                render,
                [pipe, background],
                stage,
                scene.dataset_type,
            )
            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration, stage)
            if dataset.render_process:
                if (
                    (iteration < 1000 and iteration % 10 == 9)
                    or (iteration < 3000 and iteration % 50 == 49)
                    or (iteration < 60000 and iteration % 100 == 99)
                ):
                    # breakpoint()
                    render_training_image(
                        scene,
                        gaussians,
                        [test_cams[iteration % len(test_cams)]],
                        render,
                        pipe,
                        background,
                        stage + "test",
                        iteration,
                        timer.get_elapsed_time(),
                        scene.dataset_type,
                    )
                    render_training_image(
                        scene,
                        gaussians,
                        [train_cams[iteration % len(train_cams)]],
                        render,
                        pipe,
                        background,
                        stage + "train",
                        iteration,
                        timer.get_elapsed_time(),
                        scene.dataset_type,
                    )
                    # render_training_image(scene, gaussians, train_cams, render, pipe, background, stage+"train", iteration,timer.get_elapsed_time(),scene.dataset_type)

                # total_images.append(to8b(temp_image).transpose(1,2,0))
            timer.start()
            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
                )
                gaussians.add_densification_stats(
                    viewspace_point_tensor_grad, visibility_filter
                )

                if stage == "coarse":
                    opacity_threshold = opt.opacity_threshold_coarse
                    densify_threshold = opt.densify_grad_threshold_coarse
                else:
                    opacity_threshold = opt.opacity_threshold_fine_init - iteration * (
                        opt.opacity_threshold_fine_init
                        - opt.opacity_threshold_fine_after
                    ) / (opt.densify_until_iter)
                    densify_threshold = (
                        opt.densify_grad_threshold_fine_init
                        - iteration
                        * (
                            opt.densify_grad_threshold_fine_init
                            - opt.densify_grad_threshold_after
                        )
                        / (opt.densify_until_iter)
                    )

                # Adaptive opacity threshold: More conservative when Gaussian count is low
                current_count = gaussians.get_xyz.shape[0]
                if current_count < 10000:
                    # Scale down threshold to be more conservative
                    # When count is low, we need to be more careful about pruning
                    scale_factor = max(0.3, current_count / 10000)  # 0.3x to 1.0x scaling
                    original_threshold = opacity_threshold
                    opacity_threshold = opacity_threshold * scale_factor
                    if iteration % 500 == 0 and scale_factor < 1.0:  # Log when adapting
                        print(f"[ADAPTIVE THRESHOLD] count={current_count}, "
                              f"opacity_threshold={original_threshold:.4f} -> {opacity_threshold:.4f} "
                              f"(scale={scale_factor:.2f})")
                if (
                    iteration > opt.densify_from_iter
                    and iteration % opt.densification_interval == 0
                    and gaussians.get_xyz.shape[0] < 360000
                ):
                    pre_densify_count = gaussians.get_xyz.shape[0]
                    if iteration % 500 == 0:  # Log every 500 iterations
                        print(f"[DENSIFY] Iteration {iteration}: Running densification (pre_count={pre_densify_count}, threshold={densify_threshold})")

                    if max_screen_size and max_screen_size > 0:
                        size_threshold = max_screen_size
                    else:
                        size_threshold = (
                            20 if iteration > opt.opacity_reset_interval else None
                        )

                    gaussians.densify(
                        densify_threshold,
                        opacity_threshold,
                        scene.cameras_extent,
                        size_threshold,
                        5,
                        5,
                        scene.model_path,
                        iteration,
                        stage,
                    )

                    post_densify_count = gaussians.get_xyz.shape[0]
                    if iteration % 500 == 0:  # Log every 500 iterations
                        delta = post_densify_count - pre_densify_count
                        print(f"[DENSIFY] Iteration {iteration}: Complete (post_count={post_densify_count}, delta={delta:+d})")
                elif iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    if iteration % 500 == 0:  # Log skipped densification
                        print(f"[DENSIFY] Iteration {iteration}: Skipped (count={gaussians.get_xyz.shape[0]} >= 360000 or disabled)")
                # Per-stage pruning control: Skip pruning in coarse stage
                # Coarse stage is for initialization - pruning can destabilize before convergence
                skip_pruning_coarse = (stage == "coarse")

                if (
                    iteration > opt.pruning_from_iter
                    and iteration % opt.pruning_interval == 0
                    and gaussians.get_xyz.shape[0] > prune_min_points
                    and not skip_pruning_coarse
                ):
                    pre_prune_count = gaussians.get_xyz.shape[0]
                    if iteration % 500 == 0:  # Log every 500 iterations
                        print(f"[PRUNE] Iteration {iteration}: Running explicit pruning (pre_count={pre_prune_count}, opacity_threshold={opacity_threshold:.4f})")

                    if max_screen_size and max_screen_size > 0:
                        size_threshold = max_screen_size
                    else:
                        size_threshold = (
                            20 if iteration > opt.opacity_reset_interval else None
                        )

                    gaussians.prune(
                        densify_threshold,
                        opacity_threshold,
                        scene.cameras_extent,
                        size_threshold,
                    )

                    post_prune_count = gaussians.get_xyz.shape[0]
                    if iteration % 500 == 0:  # Log every 500 iterations
                        delta = post_prune_count - pre_prune_count
                        print(f"[PRUNE] Iteration {iteration}: Complete (post_count={post_prune_count}, delta={delta:+d})")
                elif iteration > opt.pruning_from_iter and iteration % opt.pruning_interval == 0:
                    if iteration % 500 == 0:  # Log skipped pruning
                        skip_reason = []
                        if gaussians.get_xyz.shape[0] <= prune_min_points:
                            skip_reason.append(f"count={gaussians.get_xyz.shape[0]}<={prune_min_points}")
                        if skip_pruning_coarse:
                            skip_reason.append("coarse_stage")
                        if not skip_reason:
                            skip_reason.append("disabled by config")
                        print(f"[PRUNE] Iteration {iteration}: Skipped ({', '.join(skip_reason)})")

                # if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0 :
                if (
                    iteration % opt.densification_interval == 0
                    and gaussians.get_xyz.shape[0] < 360000
                    and opt.add_point
                ):
                    gaussians.grow(5, 5, scene.model_path, iteration, stage)
                    # torch.cuda.empty_cache()
                if (
                    opt.opacity_reset_interval > 0
                    and iteration % opt.opacity_reset_interval == 0
                    and iteration < opt.iterations
                    and iteration < getattr(opt, 'densify_until_iter', opt.iterations)
                ):
                    print("reset opacity")
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if iteration in checkpoint_iterations:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save(
                    (gaussians.capture(), iteration),
                    scene.model_path
                    + "/chkpnt"
                    + f"_{stage}_"
                    + str(iteration)
                    + ".pth",
                )


def training(
    dataset,
    hyper,
    opt,
    pipe,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
    debug_from,
    expname,
):
    # first_iter = 0
    tb_writer = prepare_output_and_logger(expname)
    gaussians = GaussianModel(dataset.sh_degree, hyper)
    dataset.model_path = args.model_path
    timer = Timer()
    scene = Scene(dataset, gaussians, load_coarse=None)
    timer.start()
    scene_reconstruction(
        dataset,
        opt,
        hyper,
        pipe,
        testing_iterations,
        saving_iterations,
        checkpoint_iterations,
        checkpoint,
        debug_from,
        gaussians,
        scene,
        "coarse",
        tb_writer,
        opt.coarse_iterations,
        timer,
    )
    scene_reconstruction(
        dataset,
        opt,
        hyper,
        pipe,
        testing_iterations,
        saving_iterations,
        checkpoint_iterations,
        checkpoint,
        debug_from,
        gaussians,
        scene,
        "fine",
        tb_writer,
        opt.iterations,
        timer,
    )


def prepare_output_and_logger(expname):
    if not args.model_path:
        # if os.getenv('OAR_JOB_ID'):
        #     unique_str=os.getenv('OAR_JOB_ID')
        # else:
        #     unique_str = str(uuid.uuid4())
        unique_str = expname

        args.model_path = os.path.join("./output/", unique_str)
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(
    tb_writer,
    iteration,
    Ll1,
    loss,
    l1_loss,
    elapsed,
    testing_iterations,
    scene: Scene,
    renderFunc,
    renderArgs,
    stage,
    dataset_type,
):
    if tb_writer:
        tb_writer.add_scalar(
            f"{stage}/train_loss_patches/l1_loss", Ll1.item(), iteration
        )
        tb_writer.add_scalar(
            f"{stage}/train_loss_patchestotal_loss", loss.item(), iteration
        )
        tb_writer.add_scalar(f"{stage}/iter_time", elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        #
        validation_configs = (
            {
                "name": "test",
                "cameras": [
                    scene.getTestCameras()[idx % len(scene.getTestCameras())]
                    for idx in range(10, 5000, 299)
                ],
            },
            {
                "name": "train",
                "cameras": [
                    scene.getTrainCameras()[idx % len(scene.getTrainCameras())]
                    for idx in range(10, 5000, 299)
                ],
            },
        )

        for config in validation_configs:
            if config["cameras"] and len(config["cameras"]) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config["cameras"]):
                    image = torch.clamp(
                        renderFunc(
                            viewpoint,
                            scene.gaussians,
                            stage=stage,
                            cam_type=dataset_type,
                            *renderArgs,
                        )["render"],
                        0.0,
                        1.0,
                    )
                    if dataset_type == "PanopticSports":
                        gt_image = torch.clamp(viewpoint["image"].to("cuda"), 0.0, 1.0)
                    else:
                        gt_image = torch.clamp(
                            viewpoint.original_image.to("cuda"), 0.0, 1.0
                        )
                    try:
                        if tb_writer and (idx < 5):
                            tb_writer.add_images(
                                stage
                                + "/"
                                + config["name"]
                                + "_view_{}/render".format(viewpoint.image_name),
                                image[None],
                                global_step=iteration,
                            )
                            if iteration == testing_iterations[0]:
                                tb_writer.add_images(
                                    stage
                                    + "/"
                                    + config["name"]
                                    + "_view_{}/ground_truth".format(
                                        viewpoint.image_name
                                    ),
                                    gt_image[None],
                                    global_step=iteration,
                                )
                    except:
                        pass
                    l1_test += l1_loss(image, gt_image).mean().double()
                    # mask=viewpoint.mask

                    psnr_test += psnr(image, gt_image, mask=None).mean().double()
                psnr_test /= len(config["cameras"])
                l1_test /= len(config["cameras"])
                print(
                    "\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(
                        iteration, config["name"], l1_test, psnr_test
                    )
                )
                # print("sh feature",scene.gaussians.get_features.shape)
                if tb_writer:
                    tb_writer.add_scalar(
                        stage + "/" + config["name"] + "/loss_viewpoint - l1_loss",
                        l1_test,
                        iteration,
                    )
                    tb_writer.add_scalar(
                        stage + "/" + config["name"] + "/loss_viewpoint - psnr",
                        psnr_test,
                        iteration,
                    )

        if tb_writer:
            tb_writer.add_histogram(
                f"{stage}/scene/opacity_histogram",
                scene.gaussians.get_opacity,
                iteration,
            )

            tb_writer.add_scalar(
                f"{stage}/total_points", scene.gaussians.get_xyz.shape[0], iteration
            )
            tb_writer.add_scalar(
                f"{stage}/deformation_rate",
                scene.gaussians._deformation_table.sum()
                / scene.gaussians.get_xyz.shape[0],
                iteration,
            )
            tb_writer.add_histogram(
                f"{stage}/scene/motion_histogram",
                scene.gaussians._deformation_accum.mean(dim=-1) / 100,
                iteration,
                max_bins=500,
            )

        torch.cuda.empty_cache()


def post_training_evaluation(
    args,
    dataset_params,
    hyper_params,
    pipeline_params,
    final_iteration,
):
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

    try:
        from render import render_sets
        from pathlib import Path

        # Prefer the requested final iteration, but fall back to the latest saved one if missing
        model_dir = Path(args.model_path)
        final_ply = model_dir / "point_cloud" / f"iteration_{final_iteration}" / "point_cloud.ply"
        chosen_iteration = final_iteration
        if not final_ply.exists():
            # Find latest saved iteration
            iters = []
            pc_dir = model_dir / "point_cloud"
            if pc_dir.exists():
                for d in pc_dir.iterdir():
                    name = d.name
                    if name.startswith("iteration_") and (d / "point_cloud.ply").exists():
                        try:
                            iters.append(int(name.split("_")[-1]))
                        except Exception:
                            pass
            if iters:
                chosen_iteration = max(iters)
                print(
                    f"[WARN] Requested final iteration {final_iteration} not found; "
                    f"falling back to latest saved iteration {chosen_iteration}"
                )

        print(
            f"\nPost-training rendering for iteration {chosen_iteration} at {args.model_path}"
        )
        render_sets(
            dataset_params,
            hyper_params,
            chosen_iteration,
            pipeline_params,
            skip_train=not args.post_render_train,
            skip_test=args.post_render_skip_test,
            skip_video=args.post_render_skip_video,
        )
    except Exception as exc:  # pragma: no cover - diagnostics
        print(f"[WARN] Rendering after training failed: {exc}")
        return

    if args.post_render_skip_test:
        print(
            "[INFO] Skipping metrics evaluation because test views were not rendered."
        )
        return

    try:
        from metrics import evaluate as metrics_evaluate

        print("\nPost-training metrics evaluation")
        metrics_evaluate([args.model_path])
    except Exception as exc:  # pragma: no cover - diagnostics
        print(f"[WARN] Metric evaluation after training failed: {exc}")


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def apply_filtered_training_overrides(args):
    """Adjust optimization knobs when --filter_training is enabled to keep Gaussians from collapsing."""
    if not getattr(args, "filter_training", False):
        return args

    # Keep substantially more points alive before allowing any pruning/densification to remove them.
    args.min_gaussians = max(getattr(args, "min_gaussians", 0), 4000)
    args.min_gaussians_warmup = max(getattr(args, "min_gaussians_warmup", 0), 800)
    args.min_gaussians_cooldown = max(getattr(args, "min_gaussians_cooldown", 0), 200)

    # Disable explicit pruning entirely for filtered training; the coarse stage is prone to over-pruning otherwise.
    args.pruning_from_iter = max(
        getattr(args, "pruning_from_iter", 0), args.iterations + 1
    )
    args.prune_min_points = max(getattr(args, "prune_min_points", 0), 1_000_000)

    # CRITICAL FIX: Also disable densification to prevent implicit pruning via densify_and_split.
    # Densification causes Gaussian collapse in few-camera scenarios because densify_and_split
    # prunes the original Gaussians after splitting them, which can remove all Gaussians when
    # the selection criteria select all points (common with poor few-camera initialization).
    args.densify_from_iter = max(
        getattr(args, "densify_from_iter", 0), args.iterations + 1
    )

    # Slow down remaining densification operations (though densify_from_iter should disable them)
    args.percent_dense = min(getattr(args, "percent_dense", 0.04), 0.025)
    args.random_background = False
    args.random_background_coarse_only = True

    # Give the coarse stage more room to converge and extend total iterations for the tougher setting.
    args.coarse_iterations = max(getattr(args, "coarse_iterations", 0), 6000)
    args.iterations = max(getattr(args, "iterations", 0), 28_000)

    print(f"[FILTERED TRAINING] Applied stability overrides:")
    print(f"  - min_gaussians: {args.min_gaussians}")
    print(f"  - pruning_from_iter: {args.pruning_from_iter}")
    print(f"  - densify_from_iter: {args.densify_from_iter}")
    print(f"  - coarse_iterations: {args.coarse_iterations}")
    print(f"  - iterations: {args.iterations}")

    return args

if __name__ == "__main__":
    # Set up command line argument parser
    # torch.set_default_tensor_type('torch.FloatTensor')
    torch.cuda.empty_cache()
    parser = ArgumentParser(description="Training script parameters")
    setup_seed(6666)
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6009)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument(
        "--test_iterations", nargs="+", type=int, default=[3000, 7000, 14000]
    )
    parser.add_argument(
        "--save_iterations",
        nargs="+",
        type=int,
        default=[14000, 20000, 30_000, 45000, 60000],
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--expname", type=str, default="")
    parser.add_argument("--configs", type=str, default="")
    parser.add_argument("--skip_post_eval", action="store_true")
    parser.add_argument(
        "--post_render_train",
        action="store_true",
        help="Render training views after training (default: skip)",
    )
    parser.add_argument(
        "--post_render_skip_test",
        action="store_true",
        help="Skip rendering test views after training",
    )
    parser.add_argument(
        "--post_render_skip_video",
        action="store_true",
        help="Skip rendering video trajectory after training",
    )
    # Note: --match_string is automatically added by ModelParams class

    args = parser.parse_args(sys.argv[1:])
    # Auto-detect config.py in source directory if --configs not provided
    if not args.configs:
        from pathlib import Path
        auto_config = Path(args.source_path) / "config.py"
        if auto_config.exists():
            args.configs = str(auto_config)
            print(f"Auto-detected config at: {args.configs}")

    if args.configs:
        config = load_config(args.configs)
        args = merge_hparams(args, config)
    args = apply_filtered_training_overrides(args)
    # Ensure the final iteration checkpoint is saved after configs and overrides
    try:
        if getattr(args, "save_iterations", None) is None:
            args.save_iterations = []
        if args.iterations not in args.save_iterations:
            args.save_iterations.append(args.iterations)
    except Exception:
        pass
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    model_params = lp.extract(args)
    hyper_params = hp.extract(args)
    opt_params = op.extract(args)
    pipe_params = pp.extract(args)
    training(
        model_params,
        hyper_params,
        opt_params,
        pipe_params,
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.debug_from,
        args.expname,
    )

    if not args.skip_post_eval:
        post_training_evaluation(
            args,
            model_params,
            hyper_params,
            pipe_params,
            opt_params.iterations,
        )

    # All done
    print("\nTraining complete.")
