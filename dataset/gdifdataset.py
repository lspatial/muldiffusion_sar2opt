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
from PIL import Image
from dataset.pixels2text import TextInfoRetriever
import pandas as pd 
import torchvision.transforms as transforms

class SAROpticalDataset(Dataset):
    """
    Dataset for paired SAR and optical images
    """

    def __init__(self, data_dir,channels,resize_to=224, mode='train',transform=None,
                 subset_size=None, crop_size=None, addlabel=False,random_seed=42):
        self.channels = channels
        self.transform = transform
        self.resize_to = resize_to
        self.random_seed = random_seed
        self.crop_size = crop_size
        self.addlabel = addlabel
        self.mode = mode
        self.patch_files = []
        if Path(data_dir).is_dir(): 
            # Find all patch directories
            self.patches_dir = Path(data_dir)
            self.pair_dirs = [d for d in self.patches_dir.iterdir() if d.is_dir() and d.name.startswith('pair_')]
            # Find all patch files
            for pair_dir in self.pair_dirs:
                patch_files = sorted(list(pair_dir.glob('patch_*.npy')))
                self.patch_files.extend([(pair_dir, patch_file) for patch_file in patch_files])
        elif Path(data_dir).is_file():
            dataflDf = pd.read_csv(data_dir)
            patch_files  = dataflDf['fpath'].tolist() 
            for patch_file in patch_files:
                self.patch_files.append((os.path.dirname(patch_file), patch_file))

        print(f"Found {len(self.patch_files)} total patch samples!!!")

        # If subset size is specified, randomly select patches
        if subset_size is not None and subset_size < len(self.patch_files):
            random.seed(self.random_seed)
            self.patch_files = random.sample(self.patch_files, subset_size)
            print(f"Randomly selected {subset_size} patch pairs")

        # Normalization for SAR and optical images
        self.normalize_sar = transforms.Normalize([0.5], [0.5])
        self.normalize_optical = transforms.Normalize([0.5] * channels, [0.5] * channels)

    def __len__(self):
        return len(self.patch_files)

    def __getitem__(self, idx):
        # Load SAR image (assuming single channel)
        pair_dir, patch_file = self.patch_files[idx]

        # Load the numpy file containing the patch pair
        patch_data = np.load(patch_file, allow_pickle=True).item()
        opt_img = patch_data['optical']  # Shape: [4, 512, 512]
        sar_img = patch_data['sar']  # Shape: [1, 512, 512]
        label_img = patch_data['label']  # Shape: [1, 512, 512]
        label_txtcode = patch_data['textcode']
        position = patch_data['position']  # Tuple: (h_start, w_start)

        # Convert to torch tensors
        opt_tensor = torch.from_numpy(opt_img).float()
        sar_tensor = torch.from_numpy(sar_img).float()
        label_tensor = torch.from_numpy(label_img).float()
        textcode_tensor = torch.from_numpy(label_txtcode).float()

        # Resize patches if specified
        if self.resize_to is not None:
            # Resize optical patch
            opt_tensor = F.interpolate(
                opt_tensor.unsqueeze(0),  # Add batch dimension for interpolate
                size=(self.resize_to, self.resize_to),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)  # Remove batch dimension

            # Resize SAR patch
            sar_tensor = F.interpolate(
                sar_tensor.unsqueeze(0),  # Add batch dimension
                size=(self.resize_to, self.resize_to),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)  # Remove batch dimension

            # Resize SAR patch
            label_tensor = F.interpolate(
                label_tensor.unsqueeze(0),  # Add batch dimension
                size=(self.resize_to, self.resize_to),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
            # Normalize optical data (assuming reflectance values)
        opt_tensor = opt_tensor / 255.0  # Typical scale for surface reflectance
        opt_tensor = torch.clamp(opt_tensor, -1, 1)

        sar_tensor = sar_tensor / 255.0
        sar_tensor = torch.clamp(sar_tensor, -1, 1)

        # If SAR is multi-channel, handle appropriately
        if sar_tensor.shape[0] > 1:
            # Use first channel or compute intensity
            sar_tensor = sar_tensor[0].unsqueeze(0)

          # Ensure correct number of channels
        if opt_tensor.shape[0] != self.channels:
            if opt_tensor.shape[0] > self.channels:
                # Take first n channels
                opt_tensor = opt_tensor[:self.channels]
            else:
                # Duplicate channels if needed
                while opt_tensor.shape[0] < self.channels:
                    opt_tensor = torch.cat([opt_tensor, opt_tensor[:1]], dim=0)

        opt_tensor = self.normalize_optical(opt_tensor)
        sar_tensor = self.normalize_sar(sar_tensor)
        if self.crop_size is not None:
            # Center crop
            left = (256 - 224) // 2
            top = (256 - 224) // 2
            opt_tensor = opt_tensor[:, top:top + 224, left:left + 224]
            sar_tensor = sar_tensor[:, top:top + 224, left:left + 224]
            label_tensor = label_tensor[:, top:top + 224, left:left + 224]
        return {
            'sar': sar_tensor,
            'optical': opt_tensor,
            'label': label_tensor,
            'labeltext': textcode_tensor,
            'position': position,
            'filename': os.path.basename(os.path.dirname(patch_file)) + "_" +os.path.splitext(os.path.basename(patch_file))[0]
              #os.path.basename(patch_file)
        }


def create_dataloaders(patches_dir, channels=3,batch_size=8, num_workers=4, apply_augmentation=True,
                       val_split=0.2, subset_size=None, resize_to=256,crop_size=None,label2text=False):
    """Create training and validation dataloaders from patch dataset"""


    # Define transforms
    if apply_augmentation:
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip()
        ])
    else:
        transform = None

    # Create dataset
    dataset = SAROpticalDataset(
        data_dir=patches_dir,
        channels=channels,
        resize_to = resize_to,
        mode='train',
        transform=transform,
        crop_size=crop_size,
        subset_size=subset_size,
    )

    if val_split > 0:
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
    else:
        train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        val_loader = None
        
        print(f"Created prediction dataloader with {len(dataset)} samples ")

    return train_loader, val_loader


def test():
    data_path = 'D:/wdataspace/WHU-OPT-SAR/jointpatch224fulltext/'
    train_loader, val_loader = create_dataloaders(data_path, channels=3, batch_size=8, num_workers=4,
                                                  apply_augmentation=False, val_split=0.2, subset_size=50,
                                                  resize_to=None)
    rpath = 'D:/wdataspace/WHU-OPT-SAR/textcode/'
    for bid, batch in enumerate(train_loader):
         print(bid,'::',batch['labeltext'].shape)
         files = batch['filename']
         for afile in files:
             tpath = os.path.join(rpath, afile)
