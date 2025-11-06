import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.utils import make_grid, save_image
from torch.optim import Adam
from tqdm import tqdm
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import glob
from pathlib import Path
import random
import pandas as pd
import torch as th


def save_images2pil(generated, real, condition, save_path, flag, label=None):
    """
    Save generated and real images
    """

    # Denormalize
    def denormalize(x):
        samples = ((x * 0.5 + 0.5) * 255).clamp(0, 255).to(th.uint8)
        samples = samples.permute(0, 2, 3, 1)
        samples = samples.contiguous()
        return samples

    if isinstance(flag, list):
        rpath = save_path
    else:
        rpath = save_path + '/' + flag
    generated = denormalize(generated)
    real = denormalize(real)
    condition = denormalize(condition)

    if len(condition.shape) == 4:
        condition = condition.squeeze(-1)
    generated = generated.cpu().numpy()
    condition = condition.cpu().numpy()
    real = real.cpu().numpy()
    for i in range(len(generated)):
        if isinstance(flag, list):
            Image.fromarray(generated[i, :, :, :]).save(rpath + '/' + flag[i] + '_' + str(i) + '_gen.png')
            Image.fromarray(real[i, :, :, :]).save(rpath + '/' + flag[i] + '_' + str(i) + '_opt.png')
            Image.fromarray(condition[i, :, :]).save(rpath + '/' + flag[i] + '_' + str(i) + '_sar.png')
            if label is not None:
                Image.fromarray(label[i, :, :]).save(rpath + '/' + flag[i] + '_' + str(i) + '_label.png')
        else:
            Image.fromarray(generated[i, :, :, :]).save(rpath + '_' + str(i) + '_gen.png')
            Image.fromarray(real[i, :, :, :]).save(rpath + '_' + str(i) + '_opt.png')
            Image.fromarray(condition[i, :, :]).save(rpath + '_' + str(i) + '_sar.png')
            if label is not None:
                Image.fromarray(label[i, :, :]).save(rpath + '_' + str(i) + '_label.png')


def summary_metrics(gens, reals, save_path, flag, files=None):
    psnrs = []
    ssims = []

    # Calculate metrics for each image pair
    for i in range(gens.shape[0]):
        gen = gens[i]  # [ch, width, height]
        real = reals[i]  # [ch, width, height]
        mse = F.mse_loss(gen, real).item()  # .item() already converts to Python scalar
        psnr = -10 * math.log10(mse)
        ssim = calculate_ssim(gen, real)

        # Ensure we're storing Python scalars, not tensors
        if isinstance(psnr, torch.Tensor):
            psnr = psnr.cpu().item()
        if isinstance(ssim, torch.Tensor):
            ssim = ssim.cpu().item()

        psnrs.append(psnr)
        ssims.append(ssim)

    # Calculate mean values (these should now be Python scalars)
    mean_psnr = sum(psnrs) / len(psnrs) if psnrs else 0
    mean_ssim = sum(ssims) / len(ssims) if ssims else 0

    # Add mean values to the lists
    psnrs.append(mean_psnr)
    ssims.append(mean_ssim)

    # Create IDs for the DataFrame
    ids = [i for i in range(1, len(psnrs))] + [-1]

    # Create DataFrame with Python scalars (not tensors)
    if files is None:
        result = pd.DataFrame({'id': ids, 'ssim': ssims, 'psnr': psnrs}, index=ids)
    else:
        files = files + ['patchmean']
        result = pd.DataFrame({'id': ids, 'file': files, 'ssim': ssims, 'psnr': psnrs}, index=ids)

    # Ensure directory exists
    os.makedirs(save_path, exist_ok=True)
    result.to_csv(os.path.join(save_path, f'{flag}_gensummary.csv'), index=False)

    return result


def calculate_ssim(img1, img2, data_range=1.0, window_size=11):
    """
    Calculate SSIM between two images

    Args:
        img1, img2: Input images (B, C, H, W)
        data_range: Range of input images (1.0 for [0,1], 255.0 for [0,255])
        window_size: Size of sliding window
    """

    # Ensure correct data type
    img1 = img1.float()
    img2 = img2.float()

    # Auto-detect data range if not specified
    if data_range == 1.0 and (img1.max() > 1.0 or img2.max() > 1.0):
        data_range = 255.0
        print(f"Auto-detected data range: {data_range}")

    # Constants scaled by data range
    k1, k2 = 0.01, 0.03
    C1 = (k1 * data_range) ** 2
    C2 = (k2 * data_range) ** 2

    print(f"Using C1={C1:.6f}, C2={C2:.6f}")

    # Calculate means
    mu1 = F.avg_pool2d(img1, kernel_size=window_size, stride=1,
                       padding=window_size // 2)
    mu2 = F.avg_pool2d(img2, kernel_size=window_size, stride=1,
                       padding=window_size // 2)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    # Calculate variances and covariance
    sigma1_sq = F.avg_pool2d(img1 * img1, kernel_size=window_size, stride=1,
                             padding=window_size // 2) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2 * img2, kernel_size=window_size, stride=1,
                             padding=window_size // 2) - mu2_sq
    sigma12 = F.avg_pool2d(img1 * img2, kernel_size=window_size, stride=1,
                           padding=window_size // 2) - mu1_mu2

    # Clamp variances to prevent negative values
    eps = 1e-8
    sigma1_sq = torch.clamp(sigma1_sq, min=eps)
    sigma2_sq = torch.clamp(sigma2_sq, min=eps)

    # Calculate SSIM
    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

    ssim_map = numerator / denominator

    # Debug prints
    print(f"SSIM map range: [{ssim_map.min():.3f}, {ssim_map.max():.3f}]")

    return ssim_map.mean()
