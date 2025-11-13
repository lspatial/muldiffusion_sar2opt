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
import matplotlib.pyplot as plt
from PIL import Image
import glob
from pathlib import Path
import random
from torch.cuda.amp import autocast, GradScaler
import pandas as pd
import torchvision.transforms as transforms

class SAROptPairlDataset(Dataset):
    """
    Dataset for paired SAR and optical images
    """

    def __init__(self, data_dir,channels,summaryfl,subset_size=None,random_seed=3):
        self.data_dir = Path(data_dir)
        self.channels = channels
        self.random_seed = random_seed
        self.dataDf = pd.read_csv(summaryfl)
        # If subset size is specified, randomly select patches

        if  subset_size is not None and subset_size < len(self.dataDf):
            self.dataDf = self.dataDf.sample(n=subset_size, random_state=self.random_seed)

        # Reset index after sampling (optional)
        self.dataDf = self.dataDf.reset_index(drop=True)

        # Normalization for SAR and optical images
        self.normalize_sar = transforms.Normalize([0.5], [0.5])
        self.normalize_optical = transforms.Normalize([0.5] * channels, [0.5] * channels)

    def __len__(self):
        return len(self.dataDf)

    def __getitem__(self, idx):
        # Get filenames
        failoptfl = os.path.basename(self.dataDf.iloc[idx]['png_filename'])
        winoptfl = failoptfl.replace('gen.png', 'opt.png')
        sarfile = failoptfl.replace('gen.png', 'sar.png')

        # Create paths
        fail_path = self.data_dir / failoptfl
        win_path = self.data_dir / winoptfl
        sar_path = self.data_dir / sarfile

        # Load images as numpy arrays
        fail_img = np.array(Image.open(fail_path))
        win_img = np.array(Image.open(win_path))
        sar_img = np.array(Image.open(sar_path))

        # Convert to tensors
        fail_tensor = torch.from_numpy(fail_img).float()
        win_tensor = torch.from_numpy(win_img).float()
        sar_tensor = torch.from_numpy(sar_img).float()

        # Normalize to [0, 1] if uint8
        if fail_img.dtype == np.uint8:
            fail_tensor = fail_tensor / 255.0
            win_tensor = win_tensor / 255.0
            sar_tensor = sar_tensor / 255.0
            win_tensor = torch.clamp(win_tensor, -1, 1)
            fail_tensor = torch.clamp(fail_tensor, -1, 1)
            sar_tensor = torch.clamp(sar_tensor, -1, 1)

        # If SAR is multi-channel, handle appropriately
        if sar_tensor.ndim == 2:
            sar_tensor = sar_tensor.unsqueeze(0)
        elif sar_tensor.ndim == 3 and sar_tensor.shape[0] > 1:
            sar_tensor = sar_tensor[0].unsqueeze(0)
        # Rearrange dimensions from HWC to CHW (Height, Width, Channels -> Channels, Height, Width)
        if len(fail_tensor.shape) == 3:
            fail_tensor = fail_tensor.permute(2, 0, 1)
            win_tensor = win_tensor.permute(2, 0, 1)
        win_tensor = self.normalize_optical(win_tensor)
        fail_tensor = self.normalize_optical(fail_tensor)
        sar_tensor = self.normalize_sar(sar_tensor)
        filename = failoptfl.replace('gen.png', '') 
        return {
            'sar': sar_tensor,
            'win': win_tensor,
            'fail':fail_tensor,
            'filename':filename,
        }


def create_predictloader(data_dir, summaryfl,channels=3,batch_size=8, num_workers=4,
                random_seed=1,subset_size=None):
    """Create training and validation dataloaders from patch dataset"""

    #data_dir,channels,summaryfl,subset_size=None,random_seed=3
    dataset = SAROptPairlDataset(
        data_dir=data_dir,
        channels=channels,
        summaryfl=summaryfl,
        subset_size=subset_size,
        random_seed=random_seed
    )

    # Create dataloaders
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return data_loader

def create_pairdataloaders(data_dir, summaryfl,channels=3,batch_size=8, num_workers=4, apply_augmentation=True,
                       val_split=0.2, subset_size=None, random_seed=1):
    """Create training and validation dataloaders from patch dataset"""

    # Define transforms
    if apply_augmentation:
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip()
        ])
    else:
        transform = None

    #data_dir,channels,summaryfl,subset_size=None,random_seed=3
    dataset = SAROptPairlDataset(
        data_dir=data_dir,
        channels=channels,
        summaryfl=summaryfl,
        subset_size=subset_size,
        random_seed=random_seed
    )

    # Split into train and validation
    val_size = int(val_split * len(dataset))
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"Created train dataloader with {len(train_dataset)} samples and "
          f"validation dataloader with {len(val_dataset)} samples")

    return train_loader, val_loader
