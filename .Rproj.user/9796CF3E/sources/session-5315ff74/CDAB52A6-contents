import os
import numpy as np
import rasterio
from pathlib import Path
import multiprocessing
from functools import partial
import argparse
from tqdm import tqdm
import time
import json

import torch, open_clip
from PIL import Image
from pixels2text import TextInfoRetriever

def extract_non_overlapping_patches(image, patch_size=512):
    """Extract non-overlapping patches from an image in a grid pattern"""
    if len(image.shape) == 2:
        # Single band image, add channel dimension
        image = np.expand_dims(image, axis=0)

    channels, height, width = image.shape

    # Calculate number of complete patches that fit in the image
    num_patches_h = height // patch_size
    num_patches_w = width // patch_size

    # Extract all complete patches in a grid
    patches = []
    positions = []  # To keep track of where each patch came from

    for i in range(num_patches_h):
        for j in range(num_patches_w):
            h_start = i * patch_size
            w_start = j * patch_size

            # Extract patch
            patch = image[:, h_start:h_start + patch_size, w_start:w_start + patch_size]

            # Only add if patch is the right size (should always be true for grid-based approach)
            if patch.shape[1] == patch_size and patch.shape[2] == patch_size:
                patches.append(patch)
                positions.append((h_start, w_start))

    return patches, positions


def process_image_pair(opt_file, sar_file, label_file,output_dir,patch_size=512, pair_id=None):
    """Process a pair of optical and SAR images to extract and save non-overlapping paired patches"""
    tokenizer, remoteclip, img2text = pretrainedModel()
    try:
        # Read optical image
        with rasterio.open(opt_file) as src:
            optical_img = src.read()
            optical_profile = src.profile

        # Read SAR image
        with rasterio.open(sar_file) as src:
            sar_img = src.read()
            sar_profile = src.profile

        # Read label
        with rasterio.open(label_file) as src:
            label_img = src.read()
            label_profile = src.profile

        # Extract patches from both images (non-overlapping)
        optical_patches, opt_positions = extract_non_overlapping_patches(optical_img, patch_size)
        sar_patches, sar_positions = extract_non_overlapping_patches(sar_img, patch_size)
        label_patches, label_positions = extract_non_overlapping_patches(label_img, patch_size)
        print('sar max:', np.max(sar_img),'; min:', np.min(sar_img))
        print('opt max:', np.max(optical_img),'; min:', np.min(optical_img))
        print('opt max:', np.max(label_img), '; min:', np.min(label_img))
        # Verify we have the same number of patches
        if len(optical_patches) != len(sar_patches):
            print(f"Warning: Different number of patches extracted from {opt_file} and {sar_file}!")
            print(f"Optical patches: {len(optical_patches)}, SAR patches: {len(sar_patches)}")
            # Use the minimum number of patches
            min_patches = min(len(optical_patches), len(sar_patches),len(label_patches))
            optical_patches = optical_patches[:min_patches]
            sar_patches = sar_patches[:min_patches]
            label_patches = label_patches[:min_patches]
            opt_positions = opt_positions[:min_patches]

        # Create pair ID if not provided
        if pair_id is None:
            pair_id = Path(opt_file).stem

        # Create subdirectory for this pair
        pair_dir = os.path.join(output_dir, f"pair_{pair_id}")
        os.makedirs(pair_dir, exist_ok=True)

        # Save metadata about the pair
        metadata = {
            "optical_file": os.path.basename(opt_file),
            "sar_file": os.path.basename(sar_file),
            "optical_shape": optical_img.shape,
            "sar_shape": sar_img.shape,
            "label_shape": label_img.shape,
            "patch_size": patch_size,
            "overlap": 0,  # Explicitly state no overlap
            "num_patches": len(optical_patches),
            "patch_positions": [{"h_start": pos[0], "w_start": pos[1]} for pos in opt_positions]
        }

        with open(os.path.join(pair_dir, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)

        prompts_img = img2text.retrieveTextInfoBatchOptimized(np.concat(label_patches))
        textfeatures0 = retrieveTextCode(tokenizer,remoteclip,prompts_img)
        textfeatures0 = list(textfeatures0)
        textfeatures = np.float32(textfeatures0)
        # Save each patch pair
        for i, (optical_patch, sar_patch, label_patch,txt_patch) in enumerate(zip(optical_patches, sar_patches,label_patches,textfeatures)):
            # Create a dictionary with both patches
            patch_pair = {
                "optical": optical_patch,
                "sar": sar_patch,
                "label": label_patch,
                'textcode':txt_patch,
                "position": opt_positions[i]
            }

            # Save as numpy file
            np.save(os.path.join(pair_dir, f"patch_{i:04d}.npy"), patch_pair)
        return {
            "pair_id": pair_id,
            "num_patches": len(optical_patches),
            "success": True
        }

    except Exception as e:
        print(f"Error processing {opt_file} and {sar_file}: {str(e)}")
        return {
            "pair_id": pair_id if pair_id else "unknown",
            "error": str(e),
            "success": False
        }

def pretrainedModel(model_name = 'ViT-B-32' ): # 'RN50' or 'ViT-B-32' or 'ViT-L-14'
    model, _, preprocess = open_clip.create_model_and_transforms(model_name)
    tokenizer = open_clip.get_tokenizer(model_name)
    tpath='/deva/wkspace/guidedDiffusionSimServer/checkpoints/models--chendelong--RemoteCLIP/snapshots/bf1d8a3ccf2ddbf7c875705e46373bfe542bce38/RemoteCLIP-ViT-B-32.pt'
    ckpt = torch.load(tpath, map_location="cpu")
    message = model.load_state_dict(ckpt)
    remoteclip = model#.cuda().eval()
    tokenizer = open_clip.get_tokenizer(model_name)
    img2text = TextInfoRetriever()
    return tokenizer,remoteclip,img2text

def retrieveTextCode(tokenizer,remoteclip,prompts_img):
    text = tokenizer(prompts_img)
    with torch.no_grad():
        x = remoteclip.token_embedding(text)  # [1, 77, 512]
        x = x + remoteclip.positional_embedding
        x = x.permute(1, 0, 2)  # [77, 1, 512]
        x = remoteclip.transformer(x)  # [77, 1, 512]
        x = x.permute(1, 0, 2)  # [1, 77, 512]
    return x.squeeze()

def match_files(optical_dir, sar_dir, label_dir, matching_method='name'):
    """Match optical and SAR files to create pairs"""
    optical_files = sorted([os.path.join(optical_dir, f) for f in os.listdir(optical_dir)
                            if f.endswith(('.tif', '.tiff'))])
    sar_files = sorted([os.path.join(sar_dir, f) for f in os.listdir(sar_dir)
                        if f.endswith(('.tif', '.tiff'))])
    label_files = sorted([os.path.join(label_dir, f) for f in os.listdir(label_dir)
                        if f.endswith(('.tif', '.tiff'))])
    pairs = []

    if matching_method == 'name':
        # Match by filename
        opt_stems = [Path(f).stem for f in optical_files]
        sar_stems = [Path(f).stem for f in sar_files]
        label_stems = [Path(f).stem for f in label_files]

        for i, opt_stem in enumerate(opt_stems):
            # Try exact match
            if opt_stem in sar_stems:
                sar_idx = sar_stems.index(opt_stem)
                label_idx = label_stems.index(opt_stem)
                pairs.append((optical_files[i], sar_files[sar_idx],label_files[label_idx], opt_stem))
            # Try substring match
            else:
                matches = [j for j, sar_stem in enumerate(sar_stems)
                           if sar_stem in opt_stem or opt_stem in sar_stem]
                if matches:
                    pairs.append((optical_files[i], sar_files[matches[0]], opt_stem))
    elif matching_method == 'index':
        # Match by index (assumes files are sorted)
        n = min(len(optical_files), len(sar_files))
        pairs = [(optical_files[i], sar_files[i], label_files[i],f"pair_{i:04d}") for i in range(n)]

    return pairs

def process_pair_wrapper(args):
    """Wrapper function for multiprocessing"""
    opt_file, sar_file, label_file, pair_id, output_dir, patch_size = args
    return process_image_pair(opt_file, sar_file,label_file, output_dir, patch_size, pair_id)
