
import torch
from scene import Scene
import os, time
import numpy as np
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render, render_env_map
import torchvision
from utils.general_utils import safe_state, make_cubemap_faces
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.image_utils import psnr
from utils.loss_utils import ssim
from lpipsPyTorch import get_lpips_model
import imageio

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2

def gamma_tonemap(img, gamma=2.2):
    if isinstance(img, torch.Tensor):
        return torch.clamp(img ** (1.0 / gamma), 0, 1)
    elif isinstance(img, np.ndarray):
        return np.clip(img ** (1.0 / gamma), 0, 1)
    else:
        raise RuntimeWarning(f"gamma_tonemap is not defined for type {type(img)}")


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, args):
    if args.save_images:
        render_path = os.path.join(model_path, name, "ours_{}".format(iteration))
        gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
        color_path = os.path.join(render_path, 'render')
        normal_path = os.path.join(render_path, 'normal')
        refl_path = os.path.join(render_path, 'refl')
        base_color_path = os.path.join(render_path, 'base_color')
        makedirs(color_path, exist_ok=True)
        makedirs(normal_path, exist_ok=True)
        makedirs(refl_path, exist_ok=True)
        makedirs(base_color_path, exist_ok=True)
        makedirs(gts_path, exist_ok=True)

    LPIPS = get_lpips_model(net_type='vgg').cuda()
    ssims = []
    psnrs = []
    lpipss = []
    render_times = []

    if args.save_images: # save env light
        ltres = render_env_map(gaussians)
        torchvision.utils.save_image(gamma_tonemap(ltres['env_cood1']), os.path.join(model_path, name, "ours_{}".format(iteration), 'light1.png'))
        torchvision.utils.save_image(gamma_tonemap(ltres['env_cood2']), os.path.join(model_path, name, "ours_{}".format(iteration), 'light2.png'))

    # evaluate and save views
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        view.refl_mask = None # when evaluating, refl mask is banned
        t1 = time.time()
        rendering = render(view, gaussians, pipeline, background)
        render_time = time.time() - t1

        render_color = rendering["render"][None]
        refl_strength = rendering["refl_strength_map"][None]
        base_color = rendering["base_color_map"][None]
        gt = view.original_image[None, 0:3, :, :]
        mask = view.gt_alpha_mask.bool()

        # rescale colors to match with GT
        if args.relight_gt_path and args.relight_envmap_path and args.rescale_relighted:
            if True: # multiscale rescaling
                gt_mean = gt[:, :3, mask.squeeze()].mean(axis=2)
                render_mean = render_color[:, :3, mask.squeeze()].mean(axis=2)
            else:
                gt_mean = gt[:3, mask.squeeze()].mean(None, None, True).squeeze(-1)
                render_mean = render_color[:, :3, mask.squeeze()].mean(None, None, True).squeeze(-1)

            factor = gt_mean / render_mean
            render_color = factor[:, :, None, None] * render_color * mask + (1 - mask.byte())

        ssims.append(ssim(render_color, gt).item())
        psnrs.append(psnr(render_color, gt).item())
        lpipss.append(LPIPS(render_color, gt).item())
        render_times.append(render_time)

        if args.save_images:
            normal_map = rendering['normal_map'] * 0.5 + 0.5
            torchvision.utils.save_image(render_color, os.path.join(color_path, '{0:05d}.png'.format(idx)))
            torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}.png'.format(idx)))
            torchvision.utils.save_image(refl_strength, os.path.join(refl_path, 'refl_strength_{0:05d}.png'.format(idx)))
            torchvision.utils.save_image(base_color, os.path.join(base_color_path, 'base_color_{0:05d}.png'.format(idx)))
            torchvision.utils.save_image(normal_map, os.path.join(normal_path, '{0:05d}.png'.format(idx)))

    ssim_v = np.array(ssims).mean()
    psnr_v = np.array(psnrs).mean()
    lpip_v = np.array(lpipss).mean()
    fps = 1.0/np.array(render_times).mean()
    print('psnr:{},ssim:{},lpips:{},fps:{}'.format(psnr_v, ssim_v, lpip_v, fps))
    dump_path = os.path.join(model_path, name, "ours_{}".format(iteration), 'metric.txt')
    with open(dump_path, 'w') as f:
        f.write('psnr:{}, ssim:{}, lpips:{}, fps:{}'.format(psnr_v, ssim_v, lpip_v, fps))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, args):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        relight = args.relight_gt_path and args.relight_envmap_path

        if relight:
            dataset.source_path = args.relight_gt_path

        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, relight=relight)

        # load relighted envmap
        if relight:
            envmap = cv2.cvtColor(cv2.imread(args.relight_envmap_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
            envmap = cv2.resize(envmap, (1024, 512), interpolation=cv2.INTER_LINEAR)
            envmap = gamma_tonemap(envmap)  # apply tonemap to envmap
            orig_envmap = envmap.copy()
            orig_envmap = np.roll(orig_envmap, shift=envmap.shape[1] // 4, axis=1)
            envmap = envmap * 10 - 5  # expected range of envmap is [-5, 5]
            envmap = np.roll(envmap, shift=envmap.shape[1] // 4, axis=1)
            faces = make_cubemap_faces(envmap, face_size=gaussians.env_map.resolution)
            faces = torch.from_numpy(faces).permute(0,3,1,2).float().cuda()
            fail_value = torch.zeros(gaussians.env_map.output_dim).float().cuda()
            gaussians.env_map.set_faces(faces, fail_value)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        test_dir = "test" if not relight else os.path.join("relight", args.relight_envmap_path.split('/')[-1])

        render_set(dataset.model_path, test_dir, scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, args)
        if relight:
            torchvision.utils.save_image(torch.from_numpy(orig_envmap.transpose(2, 0, 1)), os.path.join(dataset.model_path, test_dir, "ours_{}".format(scene.loaded_iter), 'relight_envmap.png'))


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--save_images", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--relight_envmap_path", default="", help="The envmap to use to relight the scene")
    parser.add_argument("--relight_gt_path", default="", help="The relighted dataset to compare against")
    parser.add_argument("--rescale_relighted", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    # safe_state(args.quiet)

    # render sets
    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args)