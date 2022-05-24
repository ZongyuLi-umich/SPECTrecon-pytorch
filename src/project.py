# project.py
import torch
from utils import *

def project_angle(image, mumap, psf, dy, viewangle):
    """
    :param image: nx * ny * nz
    :param mumap: nx * ny * nz
    :param psf: px * pz * ny
    :param dy: floating point
    :param viewangle: floating point
    """

    rot_image = imrotate(image.permute(2, 0, 1).unsqueeze(1), viewangle.item()).squeeze().permute(1, 2, 0)  # nx*ny*nz -> nz*nx*ny -> nx*ny*nz
    rot_mumap = imrotate(mumap.permute(2, 0, 1).unsqueeze(1), viewangle.item()).squeeze().permute(1, 2, 0)
    view_list = torch.zeros_like(image)
    for y in range(image.shape[1]):
        mumapr = 0.5 * rot_mumap[:, y, :]  # initialize
        sum_mumapr = mumapr + torch.sum(rot_mumap[:, 0:y, :], dim=1)  # add 0,1,...,y-1
        exp_mumapr = torch.exp(sum_mumapr * -dy)
        mu_rot_image = rot_image[:, y, :] * exp_mumapr  # apply depth-dependent attenuation
        view_list[:, y, :] = fft_conv(mu_rot_image, psf[:, :, y])
    return torch.sum(view_list, dim=1) # nx * nz


def project(image, mumap, psfs, dy):
    """
    :param image: nx * ny * nz
    :param mumap: nx * ny * nz
    :param psfs: px * pz * ny * nview
    :param dy: floating point
    """
    nx, ny, nz = image.shape
    nview = psfs.shape[-1]
    views = torch.zeros(nx, nz, nview).to(image.device).to(image.dtype)
    anglelist = torch.linspace(0, 360, nview + 1)[0:-1]
    for (viewidx, viewangle) in enumerate(anglelist):
        views[:, :, viewidx] = project_angle(image, mumap, psfs[:, :, :, viewidx], dy, viewangle)
    return views  # nx * nz * nview


def backproject_angle(view, mumap, psf, dy, viewangle):
    """
    :param view: nx * nz
    :param mumap: nx * ny * nz
    :param psf: px * pz * ny
    :param dy: floating point
    :param viewangle: floating point
    """
    rot_mumap = imrotate(mumap.permute(2, 0, 1).unsqueeze(1), viewangle.item()).squeeze().permute(1, 2, 0)
    image = torch.zeros_like(mumap)
    for y in range(image.shape[1]):
        mumapr = 0.5 * rot_mumap[:,y,:] # initialize
        sum_mumapr = mumapr + torch.sum(rot_mumap[:,0:y,:], dim=1) # add 0,1,...,y-1
        exp_mumapr = torch.exp(sum_mumapr * -dy)
        image[:, y, :] = fft_conv_adj(view, psf[:,:,y]) * exp_mumapr # apply depth-dependent attenuation
    rot_image = imrotate(image.permute(2, 0, 1).unsqueeze(1), -viewangle.item()).squeeze().permute(1, 2, 0)
    return rot_image


def backproject(view, mumap, psfs, dy):
    """
    :param view: nx * nz * nview
    :param mumap: nx * ny * nz
    :param psfs: px * pz * ny * nview
    :param dy: floating point
    """
    nx, nz = view.shape[0], view.shape[1]
    ny = mumap.shape[1]
    nview = psfs.shape[-1]
    image_list = torch.zeros(nx, ny, nz, nview).to(view.device).to(view.dtype)
    anglelist = torch.linspace(0, 360, nview + 1)[0:-1]
    for (viewidx, viewangle) in enumerate(anglelist):
        image_list[:,:,:,viewidx] = backproject_angle(view[:,:,viewidx], mumap, psfs[:,:,:,viewidx], dy, viewangle)
    return torch.sum(image_list, dim=-1) # nx * ny * nz
