
import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, render_env_map, network_gui2
import sys
from scene import Scene, GaussianModel
from scene.cameras import Camera
from utils.general_utils import safe_state
import uuid
import cv2, time
import numpy as np
from tqdm import tqdm
from torchvision.transforms import GaussianBlur
from utils.image_utils import psnr
import torchvision
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.general_utils import gamma_tonemap
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

'''
Add decay opacity for refl gaussians (banned)
Add reset refl ratio
Add refl smooth loss (banned)
SH0, and no densify (banned)
INIT_ITER 5000, cbmp_lr = 0.01 $

densify -> 30k
densify_intv in prop -> 100
'''

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, server=None):
    network_gui2.try_connect()

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)

    INIT_UNITIL_ITER = opt.init_until_iter #3000
    FR_OPTIM_FROM_ITER = opt.feature_rest_from_iter
    NORMAL_PROP_UNTIL_ITER = opt.normal_prop_until_iter + opt.longer_prop_iter #24_000
    OPAC_LR0_INTERVAL = opt.opac_lr0_interval # 200
    DENSIFIDATION_INTERVAL_WHEN_PROP = opt.densification_interval_when_prop #500

    TOT_ITER = opt.iterations + opt.longer_prop_iter + 1
    DENSIFY_UNTIL_ITER = opt.densify_until_iter + opt.longer_prop_iter

    # for real scenes
    USE_ENV_SCOPE = opt.use_env_scope # False
    if USE_ENV_SCOPE:
        center = [float(c) for c in opt.env_scope_center]
        ENV_CENTER = torch.tensor(center, device='cuda')
        ENV_RADIUS = opt.env_scope_radius
        REFL_MSK_LOSS_W = 0.4

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians) # init all parameters(pos,scale,rot...) from pcds
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, TOT_ITER), desc="Training progress")
    first_iter += 1
    iteration = first_iter

    print('propagation until: {}'.format(NORMAL_PROP_UNTIL_ITER))
    print('densify until: {}'.format(DENSIFY_UNTIL_ITER))
    print('total iter: {}'.format(TOT_ITER))

    initial_stage = True

    # Toycar
    #ENV_CENTER = torch.tensor([0.6810, 0.8080, 4.4550], device='cuda') # None
    #ENV_RANGE = 2.707

    # Garden
    #ENV_CENTER = torch.tensor([-0.2270,  1.9700,  1.7740], device='cuda') # None
    #ENV_RANGE = 0.974

    # Sedan
    #ENV_CENTER = torch.tensor([-0.032,0.808,0.751], device='cuda') # None
    #ENV_RANGE = 2.138

    while iteration < TOT_ITER:

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration > FR_OPTIM_FROM_ITER and iteration % 1000 == 0:
            gaussians.oneupSHdegree()
        if iteration > INIT_UNITIL_ITER:
            initial_stage = False

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, initial_stage=initial_stage)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # GT
        gt_image = viewpoint_cam.original_image.cuda()
        # Loss
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))


        def get_outside_msk():
            return None if not USE_ENV_SCOPE else \
                torch.sum((gaussians.get_xyz - ENV_CENTER[None])**2, dim=-1) > ENV_RADIUS**2

        if USE_ENV_SCOPE and 'refl_strength_map' in render_pkg:
            refls = gaussians.get_refl
            refl_msk_loss = refls[get_outside_msk()].mean()
            loss += REFL_MSK_LOSS_W * refl_msk_loss

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f} #points: {len(gaussians.get_xyz)}"})
                progress_bar.update(10)
            if iteration == TOT_ITER:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations or iteration == TOT_ITER-1):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < DENSIFY_UNTIL_ITER:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration <= INIT_UNITIL_ITER:
                    opacity_reset_intval = 3000
                    densification_interval = 100
                elif iteration <= NORMAL_PROP_UNTIL_ITER:
                    opacity_reset_intval = 3000 # 2:1 (reset 1: reset 0)
                    densification_interval = DENSIFIDATION_INTERVAL_WHEN_PROP
                else:
                    opacity_reset_intval = 3000
                    densification_interval = 100

                if iteration > opt.densify_from_iter and iteration % densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold,
                        opt.prune_opacity_threshold,
                        scene.cameras_extent, size_threshold,
                    )

                HAS_RESET0 = False
                if iteration % opacity_reset_intval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    HAS_RESET0 = True
                    outside_msk = get_outside_msk()
                    gaussians.reset_opacity0()
                    gaussians.reset_refl(exclusive_msk=outside_msk) ###
                if  OPAC_LR0_INTERVAL > 0 and (INIT_UNITIL_ITER < iteration <= NORMAL_PROP_UNTIL_ITER) and iteration % OPAC_LR0_INTERVAL == 0: ## 200->50
                    gaussians.set_opacity_lr(opt.opacity_lr)
                if  (INIT_UNITIL_ITER < iteration <= NORMAL_PROP_UNTIL_ITER) and iteration % 1000 == 0:
                    if not HAS_RESET0:
                        outside_msk = get_outside_msk()
                        gaussians.reset_opacity1(exclusive_msk=outside_msk)
                        gaussians.dist_color(exclusive_msk=outside_msk) #
                        gaussians.reset_scale(exclusive_msk=outside_msk)
                        if OPAC_LR0_INTERVAL > 0 and iteration != NORMAL_PROP_UNTIL_ITER:
                            gaussians.set_opacity_lr(0.0)

            # Optimizer step
            if iteration < TOT_ITER:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
        iteration += 1

        if server is None:
            continue

        render_type = network_gui2.on_gui_change()
        with torch.no_grad():
            client = server["client"]
            RT_w2v = viser.transforms.SE3(wxyz_xyz=np.concatenate([client.camera.wxyz, client.camera.position], axis=-1)).inverse()
            R = torch.tensor(RT_w2v.rotation().as_matrix().astype(np.float32)).numpy()
            T = torch.tensor(RT_w2v.translation().astype(np.float32)).numpy()
            FoVx = viewpoint_cam.FoVx
            FoVy = viewpoint_cam.FoVy

            camera = Camera(
                colmap_id=None,
                R=R,
                T=T,
                FoVx=FoVx,
                FoVy=FoVy,
                image=gt_image,
                gt_alpha_mask = None,
                image_name="",
                uid=None,
                HWK=viewpoint_cam.HWK,
            )

            # render and extract outputs
            render_pkg = render(camera, gaussians, pipe, background, initial_stage=initial_stage)
            image = render_pkg["render"]
            envmaps = render_env_map(scene.gaussians)
            if not initial_stage:
                normal_map  = render_pkg['normal_map']
                base_color_map = render_pkg['base_color_map']
                refl_strength_map = render_pkg['refl_strength_map']

            output = None
            if initial_stage or render_type == "rendered color":
                image = torch.clamp(image, 0.0, 1.0)
                rendered_image = image.detach().cpu().permute(1, 2, 0)
                rendered_image = rendered_image * 255
                rendered_image = rendered_image.byte().numpy()
                output = rendered_image
            elif not initial_stage and render_type == "base color":
                image = torch.clamp(base_color_map, 0.0, 1.0)
                rendered_image = image.detach().cpu().permute(1, 2, 0)
                rendered_image = rendered_image * 255
                rendered_image = rendered_image.byte().numpy()
                output = rendered_image
            elif not initial_stage and render_type == "refl strength":
                image = torch.clamp(refl_strength_map, 0.0, 1.0).repeat(3,1,1)
                rendered_image = image.detach().cpu().permute(1, 2, 0)
                rendered_image = rendered_image * 255
                rendered_image = rendered_image.byte().numpy()
                output = rendered_image
            elif not initial_stage and render_type == "normal":
                rendered_image = (normal_map.detach().cpu().permute(1, 2, 0) + 1)/2
                rendered_image = rendered_image * 255
                rendered_image = rendered_image.byte().numpy()
                output = rendered_image
            elif not initial_stage and render_type == "envmap cood1":
                image = envmaps['env_cood1']
                rendered_image = image.detach().cpu().permute(1, 2, 0)
                rendered_image = gamma_tonemap(rendered_image) * 255
                rendered_image = rendered_image.byte().numpy()
                output = rendered_image
            elif not initial_stage and render_type == "envmap cood2":
                image = envmaps['env_cood2']
                rendered_image = image.detach().cpu().permute(1, 2, 0)
                rendered_image = gamma_tonemap(rendered_image) * 255
                rendered_image = rendered_image.byte().numpy()
                output = rendered_image
            else:
                print(f"Unsupported render type: {render_type}")

            client.scene.set_background_image(
                output,
                format="jpeg"
            )


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        #args.model_path = os.path.join("./output/", unique_str[0:10])
        args.model_path = os.path.join("./output/", os.path.basename(args.source_path))

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

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

    # Report test and samples of training set
    if iteration % 10_000 == 0:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()},
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        env_res = render_env_map(scene.gaussians)
        for env_name in env_res.keys():
            if tb_writer:
                tb_writer.add_image("#envmap/{}".format(env_name), env_res[env_name], global_step=iteration)

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    res = renderFunc(viewpoint, scene.gaussians, more_debug_infos = True, *renderArgs)
                    image = torch.clamp(res["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        for maps_name in res.keys():
                            if 'map' in maps_name:
                                if 'normal' in maps_name:
                                     res[maps_name] = res[maps_name]*0.5+0.5
                                tb_writer.add_image(config['name'] + "_view_{}/{}".format(viewpoint.image_name, maps_name), res[maps_name], global_step=iteration)
                        tb_writer.add_image(config['name'] + "_view_{}/2_render".format(viewpoint.image_name), image, global_step=iteration)
                        if iteration == 10_000:
                            tb_writer.add_image(config['name'] + "_view_{}/1_ground_truth".format(viewpoint.image_name), gt_image, global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            #tb_writer.add_scalar("refl_gauss_ratio", scene.gaussians.get_refl_strength_to_total.item(), iteration)

        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 15_000, 30_000, 60_000, 100_000, 150_000])
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Start GUI server, configure and run training
    if args.gui:
        import viser
        server = network_gui2.init()
    else:
        server = None

    # Initialize system state (RNG)
    # safe_state(args.quiet)

    # Start GUI server, configure and run training
    # torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args),
             args.test_iterations, args.save_iterations, args.checkpoint_iterations,
             args.start_checkpoint, args.debug_from, server)

    # All done
    print("\nTraining complete.")
