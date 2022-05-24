# utils.py
import torch
import torchvision
import numpy as np
import torch.nn as nn

def imrotate(img, angle):
    """
    :param img: N * C * H * W tensor
    :param angle: in degree
    :return: rotated img
    """
    return torchvision.transforms.functional.rotate(img,
            angle, interpolation=torchvision.transforms.InterpolationMode.BILINEAR, fill=0)


def fft2(img):
    """
    :param img: H * W tensor
    :return: 2D FFT of the img
    """
    return torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(img)))


def ifft2(img):
    """
    :param img: H * W tensor
    :return: 2D iFFT of the img
    """
    return torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(img)))


def power2(x):
    """
    :param x: floating point
    :return:
    """
    return (2 ** (np.ceil(np.log2(x)))).astype(int)


def _padup(nx, px):
    """
    :param nx: floating point
    :param px: floating point
    :return:
    """
    return np.ceil((power2(nx + px - 1) - nx) / 2).astype(int)


def _paddown(nx, px):
    """
    :param nx: floating point
    :param px: floating point
    :return:
    """
    return np.floor((power2(nx + px - 1) - nx) / 2).astype(int)


def _padleft(nz, pz):
    """
    :param nz: floating point
    :param pz: floating point
    :return:
    """
    return np.ceil((power2(nz + pz - 1) - nz) / 2).astype(int)


def _padright(nz, pz):
    """
    :param nz: floating point
    :param pz: floating point
    :return:
    """
    return np.floor((power2(nz + pz - 1) - nz) / 2).astype(int)


def pad2sizezero(img, padx, padz):
    """
    :param img: H * W tensor
    :param padx: floating point
    :param padz: floating point
    :return:
    """
    px, pz = img.shape
    pad_img = torch.zeros(padx, padz).to(img.device).to(img.dtype)
    padx_dims = np.ceil((padx - px) / 2).astype(int)
    padz_dims = np.ceil((padz - pz) / 2).astype(int)
    pad_img[padx_dims:padx_dims + px, padz_dims:padz_dims + pz] = img
    return pad_img


def fft_conv(img, ker):
    """
    :param img: nx * nz
    :param ker: px * pz
    :return: nx * nz
    """
    nx, nz = img.shape[0], img.shape[1]
    px, pz = ker.shape[0], ker.shape[1]
    padup = _padup(nx, px)
    paddown = _paddown(nx, px)
    padleft = _padleft(nz, pz)
    padright = _padright(nz, pz)
    m = nn.ReplicationPad2d((padleft, padright, padup, paddown))
    pad_img = m(img.unsqueeze(0).unsqueeze(0)).squeeze()

    padx, padz = pad_img.shape[0], pad_img.shape[1]

    pad_ker = pad2sizezero(ker, padx, padz)
    pad_img_fft = fft2(pad_img)
    pad_ker_fft = fft2(pad_ker)
    freq = torch.mul(pad_img_fft, pad_ker_fft)
    xout = torch.real(ifft2(freq))
    return xout[padup:padup + nx, padleft:padleft + nz]


def fft_conv_adj(img, ker):
    """
      :param img: nx * nz
      :param ker: px * pz
      :return: nx * nz
    """
    nx, nz = img.shape[0], img.shape[1]
    px, pz = ker.shape[0], ker.shape[1]
    padup = _padup(nx, px)
    paddown = _paddown(nx, px)
    padleft = _padleft(nz, pz)
    padright = _padright(nz, pz)
    m = nn.ZeroPad2d((padleft, padright, padup, paddown))
    pad_img = m(img.unsqueeze(0).unsqueeze(0)).squeeze()

    padx, padz = pad_img.shape[0], pad_img.shape[1]

    pad_ker = pad2sizezero(ker, padx, padz)
    pad_img_fft = fft2(pad_img)
    pad_ker_fft = fft2(pad_ker)
    freq = torch.mul(pad_img_fft, pad_ker_fft)
    xout = torch.real(ifft2(freq))
    xout[padup, :] += torch.sum(xout[0:padup, :], dim=0)
    xout[nx + padup - 1, :] += torch.sum(xout[nx + padup:, :], dim=0)
    xout[:, padleft] += torch.sum(xout[:, 0:padleft], dim=1)
    xout[:, nz + padleft - 1] += torch.sum(xout[:, nz + padleft:], dim=1)
    return xout[padup:padup + nx, padleft:padleft + nz]
