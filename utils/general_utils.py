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

import torch
import sys
from datetime import datetime
import numpy as np
import random
import open3d as o3d
import math

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)

def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper

def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3]+1e-6)

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

def safe_state(silent):
    old_f = sys.stdout
    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(x.replace("\n", " [{}]\n".format(str(datetime.now().strftime("%d/%m %H:%M:%S")))))
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = F(silent)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))

# in: X,K
# out: X,K+K*2*order
def positional_encoding(pts, order):
    if order == 0: return pts
    exps = torch.exp2(torch.tensor(range(order)))
    out_tensor = [pts]
    for e in exps:
        out_tensor.append(torch.sin(e*pts))
        out_tensor.append(torch.cos(e*pts))
    out_tensor = torch.cat(out_tensor, dim=-1)
    return out_tensor

def get_pencoding_len(dim, order):
    return dim*(1+2*order)

def write2ply_norgb(pts, save_path):
    rgbs = np.ones_like(pts)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(rgbs)
    print('write ply file...')
    o3d.io.write_point_cloud(save_path, pcd, write_ascii=True)
    print('point cloud generate complete')

# env_map: 16,7
env_rayd1 = None
def init_envrayd1(H,W):
    i, j = np.meshgrid(
        np.linspace(-np.pi, np.pi, W, dtype=np.float32),
        np.linspace(0, np.pi, H, dtype=np.float32),
        indexing='xy'
    )
    xy1 = np.stack([i, j], axis=2)
    z = np.cos(xy1[..., 1])
    x = np.sin(xy1[..., 1])*np.cos(xy1[...,0])
    y = np.sin(xy1[..., 1])*np.sin(xy1[...,0])
    global env_rayd1
    env_rayd1 = torch.tensor(np.stack([x,y,z], axis=-1)).cuda()

def get_env_rayd1(H,W):
    if env_rayd1 is None:
        init_envrayd1(H,W)
    return env_rayd1

env_rayd2 = None
def init_envrayd2(H,W):
    gy, gx = torch.meshgrid(torch.linspace( 0.0 + 1.0 / H, 1.0 - 1.0 / H, H, device='cuda'),
                            torch.linspace(-1.0 + 1.0 / W, 1.0 - 1.0 / W, W, device='cuda'),
                            # indexing='ij')
                            )

    sintheta, costheta = torch.sin(gy*np.pi), torch.cos(gy*np.pi)
    sinphi, cosphi     = torch.sin(gx*np.pi), torch.cos(gx*np.pi)

    reflvec = torch.stack((
        sintheta*sinphi,
        costheta,
        -sintheta*cosphi
        ), dim=-1)
    global env_rayd2
    env_rayd2 = reflvec

def get_env_rayd2(H, W):
    global env_rayd2
    # re-init if shape mismatches or first call
    if env_rayd2 is None or env_rayd2.shape[0] != H or env_rayd2.shape[1] != W:
        init_envrayd2(H, W)
    return env_rayd2

pixel_camera = None
def sample_camera_rays(HWK, R, T):
    H,W,K = HWK
    R = R.T # NOTE!!! the R rot matrix is transposed save in 3DGS

    global pixel_camera
    if pixel_camera is None or pixel_camera.shape[0] != H:
        K = K.astype(np.float32)
        i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                        np.arange(H, dtype=np.float32),
                        indexing='xy')
        xy1 = np.stack([i, j, np.ones_like(i)], axis=2)
        pixel_camera = np.dot(xy1, np.linalg.inv(K).T)
        pixel_camera = torch.tensor(pixel_camera).cuda()

    rays_o = (-R.T @ T.unsqueeze(-1)).flatten()
    pixel_world = (pixel_camera - T[None, None]).reshape(-1, 3) @ R
    rays_d = pixel_world - rays_o[None]
    rays_d = rays_d / torch.norm(rays_d, dim=1, keepdim=True)
    rays_d = rays_d.reshape(H,W,3)
    return rays_d


def gamma_tonemap(color, gamma=2.2):
    """Apply gamma tonemapping to an HDR color tensor.

    Args:
        color (torch.Tensor): Input HDR color tensor in the range [0,1].
        gamma (float): Gamma correction value (default 2.2 for sRGB).

    Returns:
        torch.Tensor: Tonemapped color tensor.
    """
    if isinstance(color, torch.Tensor):
        return torch.clamp(color ** (1.0 / gamma), 0, 1)
    elif isinstance(color, np.ndarray):
        return np.clip(color ** (1.0 / gamma), 0, 1)
    else:
        raise RuntimeWarning(f"gamma_tonemap is not defined for type {type(color)}")

import numpy as np, math

def make_cubemap_faces(equi_map, face_size):
    H, W, _ = equi_map.shape

    # shoot rays at the same centers as get_env_rayd2
    vs = np.linspace(-1 + 1/face_size, 1 - 1/face_size, face_size)
    yy, xx = np.meshgrid(vs, vs, indexing='ij')

    faces = []
    face_specs = [
        # name,    u_axis,           v_axis,       face_center
        ('+X', np.array([ 0,  0, -1]), np.array([0, -1,  0]), np.array([ 1,  0,  0])),
        ('-X', np.array([ 0,  0,  1]), np.array([0, -1,  0]), np.array([-1,  0,  0])),
        ('+Y', np.array([ 1,  0,  0]), np.array([0,  0, 1]), np.array([0,1,0])),
        ('-Y', np.array([ 1,  0,  0]), np.array([0,  0, 1]), np.array([0,-1,0])),
        # ('+Y', np.array([ 1,  0,  0]), np.array([0,  0,  -1]), np.array([ 0,  -1,  0])),
        # ('-Y', np.array([ 1,  0,  0]), np.array([0,  0, -1]), np.array([ 0, 1,  0])),
        ('+Z', np.array([ 1,  0,  0]), np.array([0, -1,  0]), np.array([ 0,  0,  1])),
        ('-Z', np.array([-1,  0,  0]), np.array([0, -1,  0]), np.array([ 0,  0, -1])),
    ]

    for name, u_axis, v_axis, center in face_specs:
        # 1) build ray directions
        dirs = center[None,None,:] \
             + xx[:,:,None]*u_axis[None,None,:] \
             + yy[:,:,None]*v_axis[None,None,:]
        dirs = dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)
        dx, dy, dz = dirs[...,0], dirs[...,1], dirs[...,2]

        # 2) spherical coords
        theta = np.arccos(np.clip(dy, -1, 1))   # [0, π]
        # *** FIXED φ ***
        phi   = np.arctan2(dx, -dz)             # invert your get_env_rayd2 convention!

        # 3) map into pixel coords with a -0.5 texel shift
        u = (phi   / (2*math.pi) + 0.5) * W - 0.5
        v = (theta / math.pi)          * H - 0.5

        # 4) bilinear sample
        i0 = np.floor(u).astype(int); j0 = np.floor(v).astype(int)
        i1 = np.minimum(i0+1, W-1);      j1 = np.minimum(j0+1, H-1)
        wu = u - i0;                      wv = v - j0

        c00 = equi_map[j0, i0]; c10 = equi_map[j0, i1]
        c01 = equi_map[j1, i0]; c11 = equi_map[j1, i1]

        # w00 = ((1-wu)*(1-wv))[...,None]
        # w10 = ((   wu)*(1-wv))[...,None]
        # w01 = ((1-wu)*(   wv))[...,None]
        # w11 = ((   wu)*(   wv))[...,None]
        # faces.append(c00*w00 + c10*w10 + c01*w01 + c11*w11)

        face = (
             c00 * ((1-wu)*(1-wv))[..., None] +
             c10 * ((  wu)*(1-wv))[..., None] +
             c01 * ((1-wu)*(  wv))[..., None] +
             c11 * ((  wu)*(  wv))[..., None]
        )
        # the encoder assumes “LEFT_TOP_AS_ORIGIN” so flip each face vertically:
        face = face[::-1, ...]     # flip along the image’s row axis
        faces.append(face)

    return np.stack(faces, axis=0)  # (6, face_size, face_size, 3)

