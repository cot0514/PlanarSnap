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

import os
import gc
import time
import numpy as np
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,garbage_collection_threshold:0.8"

import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, build_scaling_rotation
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def total_variation_loss(img):
    bs_img, c_img, h_img, w_img = img.size()
    tv_h = torch.pow(img[:, :, 1:, :] - img[:, :, :-1, :], 2).sum()
    tv_w = torch.pow(img[:, :, :, 1:] - img[:, :, :, :-1], 2).sum()
    return (tv_h + tv_w) / (bs_img * c_img * h_img * w_img)

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, add_sky_box=opt.add_sky_box, max_read_points=opt.max_read_points, sphere_point=opt.sphere_point)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint, weights_only=False)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_dist_for_log = 0.0
    ema_normal_for_log = 0.0
    ema_texture_for_log = 0.0

    initial_texture_alpha = gaussians.get_texture_alpha[0:1].detach().clone()

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    _wall_prev = time.perf_counter()
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):

        iter_start.record()

        xyz_lr = gaussians.update_learning_rate(iteration)
        # After densification stops, freeze geometry (xyz/scaling/rotation) to prevent drift.
        # MCMC converges geometry by iter densify_until_iter; further updates only degrade quality.
        # Texture (alpha/color) and SH (f_dc/f_rest) continue training for appearance refinement.
        if iteration >= opt.densify_until_iter:
            xyz_lr = 0.0
            for param_group in gaussians.optimizer.param_groups:
                if param_group["name"] in ("xyz", "scaling", "rotation"):
                    param_group['lr'] = 0.0

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        
        _profile = (iteration <= 5 or iteration % 500 == 0 or (1100 <= iteration <= 1115) or (1200 <= iteration <= 1320))
        _raster_debug = False

        with torch.no_grad():
            # Only replace NaN/Inf — do NOT clamp xyz, that restricts valid scene coordinates.
            gaussians._xyz.data = torch.nan_to_num(gaussians._xyz.data, nan=0.0, posinf=1e4, neginf=-1e4)
            if hasattr(gaussians, '_scaling'):
                gaussians._scaling.data = torch.nan_to_num(torch.clamp(gaussians._scaling.data, min=-15.0, max=8.0), nan=-5.0, posinf=8.0, neginf=-15.0)
            if hasattr(gaussians, '_rotation'):
                rot_bad = ~torch.isfinite(gaussians._rotation.data).all(dim=-1)
                if rot_bad.any():
                    gaussians._rotation.data[rot_bad] = torch.tensor([1.0, 0.0, 0.0, 0.0], device='cuda')

        if _profile: torch.cuda.synchronize(); _t0 = time.perf_counter()
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, raster_debug=_raster_debug)
        if _profile: torch.cuda.synchronize(); _t1 = time.perf_counter()
        image = render_pkg["render"]
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]
        impact = render_pkg["impact"]

        gt_image = viewpoint_cam.original_image.cuda()
        gt_image = gt_image[:, :viewpoint_cam.image_height, :viewpoint_cam.image_width]

        Ll1 = l1_loss(image, gt_image)
        ssim_map = ssim(image, gt_image, size_average=False)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_map.mean())

        # regularization
        lambda_normal = opt.lambda_normal if iteration > 7000 else 0.0
        lambda_dist = opt.lambda_dist if iteration > 3000 else 0.0

        rend_dist = render_pkg["rend_dist"]
        rend_normal = render_pkg['rend_normal']
        surf_normal = render_pkg['surf_normal']
        normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
        normal_loss = lambda_normal * (normal_error).mean()
        dist_loss = lambda_dist * (rend_dist).mean()

        if _profile: torch.cuda.synchronize(); _t2 = time.perf_counter()
        weights = opt.max_impact_threshold - torch.clamp(impact[visibility_filter], 0, opt.max_impact_threshold)
        textures_reg = (gaussians.get_texture_color[visibility_filter].mean(dim=[1, 2, 3]) * weights).mean() * opt.lambda_texture_value
        textures_reg += torch.abs((gaussians.get_texture_alpha[visibility_filter] - initial_texture_alpha).mean(dim=[1, 2]) * weights).mean() * opt.lambda_alpha_value

        # loss
        total_loss = loss + dist_loss + normal_loss + textures_reg
        # For MCMC sampler: opacity_reg only active during densification phase.
        # After densify_until_iter, no MCMC relocation occurs — keeping this penalty
        # active would keep pushing alpha toward 0, degrading quality past the peak.
        current_opacity_reg = opt.opacity_reg if iteration < opt.densify_until_iter else 0.0
        total_loss += current_opacity_reg * gaussians.get_texture_alpha.mean()
        if _profile: torch.cuda.synchronize(); _t3 = time.perf_counter()
        total_loss.backward()

        iter_end.record()
        if _profile:
            torch.cuda.synchronize(); _t4 = time.perf_counter()
            print(f"[PROFILE iter={iteration}] render={(_t1-_t0)*1000:.1f}ms  loss={(_t2-_t1)*1000:.1f}ms  texreg={(_t3-_t2)*1000:.1f}ms  backward={(_t4-_t3)*1000:.1f}ms  N={gaussians._xyz.shape[0]}")

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_dist_for_log = 0.4 * dist_loss.item() + 0.6 * ema_dist_for_log
            ema_normal_for_log = 0.4 * normal_loss.item() + 0.6 * ema_normal_for_log
            ema_texture_for_log = 0.4 * textures_reg.item() + 0.6 * ema_texture_for_log


            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "distort": f"{ema_dist_for_log:.{5}f}",
                    "normal": f"{ema_normal_for_log:.{5}f}",
                    "texture": f"{ema_texture_for_log:.{5}f}",
                    "Points": f"{len(gaussians.get_xyz)}"
                }
                progress_bar.set_postfix(loss_dict)

                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            if tb_writer is not None:
                tb_writer.add_scalar('train_loss_patches/dist_loss', ema_dist_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/normal_loss', ema_normal_for_log, iteration)

            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
            if iteration in testing_iterations or iteration in saving_iterations:
                gc.collect()
                torch.cuda.empty_cache()

            # Texture LR schedule:
            # [texture_from_iter, densify_until_iter): full LR (MCMC active phase)
            # [densify_until_iter, texture_to_iter):  exponential LR decay to 1% of init
            #   — continued refinement at decaying LR avoids the constant-LR oscillation
            #     that caused PSNR collapse in v12 (constant LR after MCMC stop).
            # [texture_to_iter, ...):                 LR = 0 (fully deactivated)
            if opt.texture_from_iter <= iteration < opt.densify_until_iter:
                gaussians.activate_texture_training()
            elif opt.densify_until_iter <= iteration < opt.texture_to_iter:
                decay_t = (iteration - opt.densify_until_iter) / max(1, opt.texture_to_iter - opt.densify_until_iter)
                alpha_lr = opt.texture_opacity_lr * (0.01 ** decay_t)
                color_lr = opt.texture_color_lr * (0.01 ** decay_t)
                for param_group in gaussians.optimizer.param_groups:
                    if param_group["name"] == "texture_alpha":
                        param_group['lr'] = alpha_lr
                    elif param_group["name"] == "texture_color":
                        param_group['lr'] = color_lr
            elif iteration >= opt.texture_to_iter:
                gaussians.deactivate_texture_training()

            if iteration > opt.position_lr_max_steps:
                gaussians.deactivate_gaussians_training()

            # Densification
            if iteration < opt.densify_until_iter and iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                _t_dens = time.perf_counter()
                size = gaussians._texture_alpha.shape[0]
                dead_mask = (gaussians.get_texture_alpha.view(size, -1).mean(1) <= opt.dead_opacity).squeeze(-1)
                _t_mask = time.perf_counter()
                gaussians.relocate_gs(dead_mask=dead_mask)
                _t_rel = time.perf_counter()
                gaussians.add_new_gs(cap_max=opt.cap_max)
                _t_add = time.perf_counter()
                new_size = gaussians._xyz.shape[0]
                print(f"[DENS iter={iteration}] mask={(_t_mask-_t_dens)*1000:.0f}ms  relocate={(_t_rel-_t_mask)*1000:.0f}ms  add_gs={(_t_add-_t_rel)*1000:.0f}ms  total={(_t_add-_t_dens)*1000:.0f}ms  N={new_size}")
                gc.collect()
                torch.cuda.empty_cache()

            # Optimizer step
            if iteration < opt.iterations:
                if _profile: torch.cuda.synchronize(); _t5 = time.perf_counter()
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

                # Clamp + sanitize immediately after optimizer step so covariance
                # computation below never sees extreme or NaN/Inf values.
                gaussians._xyz.data = torch.nan_to_num(gaussians._xyz.data, nan=0.0, posinf=1e4, neginf=-1e4)
                if hasattr(gaussians, '_scaling'):
                    gaussians._scaling.data = torch.nan_to_num(torch.clamp(gaussians._scaling.data, min=-15.0, max=8.0), nan=-5.0, posinf=8.0, neginf=-15.0)
                if hasattr(gaussians, '_rotation'):
                    rot_bad = ~torch.isfinite(gaussians._rotation.data).all(dim=-1)
                    if rot_bad.any():
                        gaussians._rotation.data[rot_bad] = torch.tensor([1.0, 0.0, 0.0, 0.0], device='cuda')
                # Clamp SH features to prevent unbounded growth (causes rendered
                # color >> 1 and rising L1 loss when no clamping before loss).
                if hasattr(gaussians, '_features_dc'):
                    gaussians._features_dc.data.clamp_(-5.0, 5.0)
                if hasattr(gaussians, '_features_rest'):
                    gaussians._features_rest.data.clamp_(-2.0, 2.0)

                L = build_scaling_rotation(gaussians.get_scaling, gaussians.get_rotation)
                actual_covariance = L @ L.transpose(1, 2)

                def op_sigmoid(x, k=100, x0=0.995):
                    return 1 / (1 + torch.exp(-k * (x - x0)))

                #size = len(gaussians.get_texture_alpha)
                #opacity = gaussians.get_texture_alpha.view(size, -1).mean(1, keepdim=True) * 10 # Rescale to get maximum = 1
                opacity = torch.ones([gaussians._xyz.shape[0], 1], dtype=torch.float32, device="cuda") # Fix opacity to 1 (results in the paper obtained this way)
                noise = torch.randn_like(gaussians._xyz) * (op_sigmoid(1 - opacity)) * opt.noise_lr * xyz_lr
                noise = torch.bmm(actual_covariance, noise.unsqueeze(-1)).squeeze(-1)
                gaussians._xyz.add_(noise)
                gaussians._xyz.data = torch.nan_to_num(gaussians._xyz.data, nan=0.0, posinf=1e4, neginf=-1e4)
                if _profile: torch.cuda.synchronize(); _t6 = time.perf_counter(); print(f"[PROFILE iter={iteration}]  optimizer+noise={(_t6-_t5)*1000:.1f}ms")

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

        if iteration % 20 == 0:
            gc.collect()

        _wall_now = time.perf_counter()
        if iteration % 100 == 0 or iteration <= 5:
            print(f"[WALL iter={iteration}] total_iter={(_wall_now-_wall_prev)*1000:.0f}ms  N={gaussians._xyz.shape[0]}")
        _wall_prev = _wall_now

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

@torch.no_grad()
def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/reg_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)

                    if image.shape != gt_image.shape:
                        min_h = min(image.shape[1], gt_image.shape[1])
                        min_w = min(image.shape[2], gt_image.shape[2])
                        image = image[:, :min_h, :min_w]
                        gt_image = gt_image[:, :min_h, :min_w]

                    if tb_writer and (idx < 5):
                        from utils.general_utils import colormap
                        depth = render_pkg["surf_depth"]
                        norm = depth.max()
                        depth = depth / norm
                        depth = colormap(depth.cpu().numpy()[0], cmap='turbo')
                        tb_writer.add_images(config['name'] + "_view_{}/depth".format(viewpoint.image_name), depth[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)

                        try:
                            rend_alpha = render_pkg['rend_alpha']
                            rend_normal = render_pkg["rend_normal"] * 0.5 + 0.5
                            surf_normal = render_pkg["surf_normal"] * 0.5 + 0.5
                            tb_writer.add_images(config['name'] + "_view_{}/rend_normal".format(viewpoint.image_name), rend_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/surf_normal".format(viewpoint.image_name), surf_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/rend_alpha".format(viewpoint.image_name), rend_alpha[None], global_step=iteration)

                            rend_dist = render_pkg["rend_dist"]
                            rend_dist = colormap(rend_dist.cpu().numpy()[0])
                            tb_writer.add_images(config['name'] + "_view_{}/rend_dist".format(viewpoint.image_name), rend_dist[None], global_step=iteration)
                        except:
                            pass

                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1_000, 7_000, 10_000, 15_000, 20_000, 25_000, 30_000, 32_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1_000, 7_000, 10_000, 15_000, 20_000, 30_000, 32_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint)

    # All done
    print("\nTraining complete.")
