import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def disp_to_depth(disp, min_depth, max_depth):
    """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth


def transformation_from_parameters(axisangle, translation, invert=False):
    """Convert the network's (axisangle, translation) output into a 4x4 matrix
    """
    R = rot_from_axisangle(axisangle)
    t = translation.clone()

    if invert:
        R = R.transpose(1, 2)
        t *= -1

    T = get_translation_matrix(t)

    if invert:
        M = torch.matmul(R, T)
    else:
        M = torch.matmul(T, R)

    return M


def get_translation_matrix(translation_vector):
    """Convert a translation vector into a 4x4 transformation matrix
    """
    T = torch.zeros(translation_vector.shape[0], 4, 4).to(device=translation_vector.device)

    t = translation_vector.contiguous().view(-1, 3, 1)

    T[:, 0, 0] = 1
    T[:, 1, 1] = 1
    T[:, 2, 2] = 1
    T[:, 3, 3] = 1
    T[:, :3, 3, None] = t

    return T


def rot_from_axisangle(vec):
    """Convert an axisangle rotation into a 4x4 transformation matrix
    (adapted from https://github.com/Wallacoloo/printipi)
    Input 'vec' has to be Bx1x3
    """
    angle = torch.norm(vec, 2, 2, True)
    axis = vec / (angle + 1e-7)

    ca = torch.cos(angle)
    sa = torch.sin(angle)
    C = 1 - ca

    x = axis[..., 0].unsqueeze(1)
    y = axis[..., 1].unsqueeze(1)
    z = axis[..., 2].unsqueeze(1)

    xs = x * sa
    ys = y * sa
    zs = z * sa
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    rot = torch.zeros((vec.shape[0], 4, 4)).to(device=vec.device)

    rot[:, 0, 0] = torch.squeeze(x * xC + ca)
    rot[:, 0, 1] = torch.squeeze(xyC - zs)
    rot[:, 0, 2] = torch.squeeze(zxC + ys)
    rot[:, 1, 0] = torch.squeeze(xyC + zs)
    rot[:, 1, 1] = torch.squeeze(y * yC + ca)
    rot[:, 1, 2] = torch.squeeze(yzC - xs)
    rot[:, 2, 0] = torch.squeeze(zxC - ys)
    rot[:, 2, 1] = torch.squeeze(yzC + xs)
    rot[:, 2, 2] = torch.squeeze(z * zC + ca)
    rot[:, 3, 3] = 1

    return rot


class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU
    """

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out


class Conv3x3(nn.Module):
    """Layer to pad and convolve input
    """

    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out


class BackprojectDepth(nn.Module):
    """Layer to transform a depth image into a point cloud
    """

    def __init__(self, batch_size, height, width):
        super(BackprojectDepth, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords),
                                      requires_grad=False)

        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
                                 requires_grad=False)

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1),
                                       requires_grad=False)

    def forward(self, depth, inv_K):
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)
        cam_points = depth.view(self.batch_size, 1, -1) * cam_points
        cam_points = torch.cat([cam_points, self.ones], 1)

        return cam_points


class Project3D(nn.Module):
    """Layer which projects 3D points into a camera with intrinsics K and at position T
    """

    def __init__(self, batch_size, height, width, eps=1e-7):
        super(Project3D, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps

    def forward(self, points, K, T , resflow=None):
        P = torch.matmul(K, T)[:, :3, :]

        cam_points = torch.matmul(P, points)

        pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + self.eps)
        pix_coords = pix_coords.view(self.batch_size, 2, self.height, self.width)
        if resflow is not None:
            pix_coords = pix_coords.permute(0, 2, 3, 1) + resflow
        else:
            pix_coords = pix_coords.permute(0, 2, 3, 1)
        pix_coords[..., 0] /= self.width - 1
        pix_coords[..., 1] /= self.height - 1
        pix_coords = (pix_coords - 0.5) * 2
        return pix_coords


def upsample(x):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=2, mode="nearest")


def get_smooth_loss(disp, img):
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    """
    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp( -1.0 * grad_img_x)
    grad_disp_y *= torch.exp( -1.0 * grad_img_y)

    return grad_disp_x.mean() + grad_disp_y.mean()

def gradient(D):
    D_dy = D[:, :, 1:] - D[:, :, :-1]
    D_dx = D[:, :, :, 1:] - D[:, :, :, :-1]
    return D_dx, D_dy

def flow_smooth_loss(flow):
    dx, dy = gradient(flow)
    return dx.abs().mean() + dy.abs().mean()

class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """

    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool = nn.AvgPool2d(3, 1)
        self.mu_y_pool = nn.AvgPool2d(3, 1)
        self.sig_x_pool = nn.AvgPool2d(3, 1)
        self.sig_y_pool = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


def compute_depth_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = torch.max((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).float().mean()
    a2 = (thresh < 1.25 ** 2).float().mean()
    a3 = (thresh < 1.25 ** 3).float().mean()

    rmse = (gt - pred) ** 2
    rmse = torch.sqrt(rmse.mean())

    rmse_log = (torch.log(gt) - torch.log(pred)) ** 2
    rmse_log = torch.sqrt(rmse_log.mean())

    abs_rel = torch.mean(torch.abs(gt - pred) / gt)

    sq_rel = torch.mean((gt - pred) ** 2 / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def meshgrid(height, width, is_homogeneous=True):
    x = torch.ones(height).float().view(height, 1)
    # shape : (h,w)
    x = torch.matmul(x, torch.linspace(0, 1, width).view(1, width))

    y = torch.linspace(0, 1, height).view(height, 1)
    # shape : (h,w)
    y = torch.matmul(y, torch.ones(width).float().view(1, width))

    x = x * (width - 1)
    y = y * (height - 1)

    if is_homogeneous:
        ones = torch.ones(height, width).float()
        coords = torch.stack((x, y, ones), dim=2)  # shape: h,w,3
    else:
        coords = torch.stack((x, y), dim=2)  # shape: h,w,2

    # shape:  (h, w, 2 or 3)
    return coords


def compute_rigid_flow(pose, depth, intrinsics, reverse_pose):
    '''Compute the rigid flow from src view to tgt view 
        input shapes:
            pose: #batch,4,4
            depth: #batch,h,w
            intrinsics: #batch,3,3
    '''
    batch_size, h, w = depth.shape
    # shape: (#batch,4,4)
    pose_mat = pose
    if reverse_pose:
        pose_mat = torch.inverse(pose_mat)

    # shape: (#batch,1,3,3)
    intrinsics_inv = torch.inverse(intrinsics).unsqueeze_(1)

    # shape: (#h*w,3,1)
    src_coords = meshgrid(h, w, True).contiguous().view(h * w, 3, 1).cuda()

    # shape: matmul( (#batch,1,3,3) ,(h*w,3,1)) = (#batch,h*w,3,1)
    tgt_coords = torch.matmul(intrinsics_inv, src_coords)

    # shape: (#batch, h*w,3,1)
    _depth = depth.view(batch_size, h * w).repeat(3, 1,
                                                  1).permute(1, 2, 0).unsqueeze_(3)

    # point-wise multiplication : shape: (# batch, h*w,3,1)
    tgt_coords = _depth * tgt_coords

    ones = torch.ones(batch_size, h * w, 1, 1).float().cuda()
    # shape: (#batch, h*w,4,1)
    tgt_coords = torch.cat((tgt_coords, ones), dim=2)

    # shape: (#batch,h*w,4, 4)
    pose_mat = pose_mat.repeat(1, h * w, 1, 1)

    # shape: matmul((#batch,h*w,4,4),(#batch,h*w,4,1)) = (#batch,h*w,4,1) -> (#batch,h*w,3,1)
    tgt_coords = torch.matmul(pose_mat, tgt_coords)[:, :, :3, :]

    # shape: (#batch,h*w,3, 3)
    intrinsics = intrinsics.repeat(h * w, 1, 1, 1).transpose(1, 0)

    # shape: matmul((#batch,h*w,3,3),(#batch,h*w,3,1)) = (#batch,h*w,3,1)
    tgt_coords = torch.matmul(intrinsics, tgt_coords)

    # shape: (#batch,h*w,2)
    src_coords = src_coords.repeat(batch_size, 1, 1, 1).squeeze_(-1)[:, :, :2]
    # shape: (#batch,h*w,3,1)
    # rigid_flow = tgt_coords-src_coords
    # shape: (#batch,h*w,2)
    tgt_depth = tgt_coords[:, :, 2, :].clone().repeat(1, 1, 2)  # require grad but also require modify (repeat) here
    # shape: (#batch,h*w,2)
    tgt_coords = tgt_coords[:, :, :2, :].squeeze_(-1) / tgt_depth

    # shape: (#batch,h*w,2)
    rigid_flow = tgt_coords - src_coords
    # shape: (#batch,2,h,w)
    rigid_flow = rigid_flow.contiguous().view(batch_size, h, w, 2).permute(0, 3, 1, 2)
    return rigid_flow


def flow_to_tgt_coords(src2tgt_flow):
    # shape: (#batch,2,h,w)
    batch_size, _, h, w = src2tgt_flow.shape

    # shape: (#batch,h,w,2)
    src2tgt_flow = src2tgt_flow.clone().permute(0, 2, 3, 1)

    # shape: (#batch,h,w,2)
    src_coords = meshgrid(h, w, False).repeat(batch_size, 1, 1, 1).cuda()

    tgt_coords = src_coords + src2tgt_flow

    normalizer = torch.tensor([(2. / w), (2. / h)]).repeat(batch_size, h, w, 1).float().cuda()
    # shape: (#batch,h,w,2)
    tgt_coords = tgt_coords * normalizer - 1

    # shape: (#batch,h,w,2)
    return tgt_coords


def flow_warp(src_img, src2tgt_flow):
    tgt_coords = flow_to_tgt_coords(src2tgt_flow)
    tgt_img = F.grid_sample(src_img, tgt_coords, padding_mode="border", align_corners=True)
    return tgt_img

def image_similarity(x, y, alpha=0.85):
    DSSIM = SSIM()
    return alpha * DSSIM(x,y) + (1-alpha) * torch.abs(x-y)

def L2_norm(x, dim, keep_dims=True):
    curr_offset = 1e-10
    l2_norm = torch.norm(torch.abs(x) + curr_offset,
                         dim=dim, keepdim=keep_dims)
    return l2_norm

def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img


def coords_grid(batch, ht, wd, device):
    coords = torch.meshgrid(torch.arange(ht, device=device), torch.arange(wd, device=device))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)