import torch
from dataset.gdifdataset import create_dataloaders
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import multiprocessing
from mappingknow.preswinunet import preSwinUnet
torch.serialization.add_safe_globals([preSwinUnet])
import numpy as np

def modelinf():
    data_path = '/devb/WHU-OPT-SAR/jointpatch256'
    train_loader, test_loader = create_dataloaders(data_path, channels=3, batch_size=8, num_workers=0,
                                                   apply_augmentation=False, val_split=0.2, subset_size=1000,
                                                   resize_to=None, crop_size=224)
    pretrained_file='/devb/sar2opt_diff/mapping/model_statedict_best.tor'
    device = torch.device('cuda:' + str(0))
    model = torch.load(pretrained_file, map_location=device,weights_only=False)
    torch.cuda.empty_cache()
    masks_pred_all, true_masks_all = [], []
    for i, batch in enumerate(tqdm(test_loader)):
        sar = batch['sar'].to(device)
        optical = batch['optical'].to(device)
        sar.to(device)
        optical.to(device)
        masks_pred = model(sar)
        masks_pred = masks_pred.cpu().detach()
        true_masks = optical.cpu().detach()
        masks_pred_all.append(masks_pred)
        true_masks_all.append(true_masks)
        del optical, sar, true_masks, masks_pred
        torch.cuda.empty_cache()
    
    true_masks_all = np.concatenate(true_masks_all)
    masks_pred_all = np.concatenate(masks_pred_all)
    def denormalize(x):
        samples = ((x * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
        samples = np.transpose(samples, (0, 2, 3, 1))
        return samples
    
    masks_pred_all = denormalize(masks_pred_all)
    true_masks_all = denormalize(true_masks_all)
    output_dir='/devb/sar2opt_diff/mapping/figs'
    multiprocessing.freeze_support()
    for i, prediction in enumerate(masks_pred_all):
        ground_truth = true_masks_all[i]
        if len(prediction.shape) == 2:
            # It's already 2D, use as is
            pass
        elif len(prediction.shape) == 3 and prediction.shape[2] == 1:
            # Squeeze 3D with single channel to 2D
            prediction = prediction.squeeze(-1)
            ground_truth = ground_truth.squeeze(-1)
    
        # Create figure with two subplots side by side
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
        # Plot prediction on the left
        im1 = axes[0].imshow(prediction, cmap='viridis')  # Adjust cmap as needed
        axes[0].set_title('Prediction')
        axes[0].axis('off')  # Hide axes
    
        # Plot ground truth on the right
        im2 = axes[1].imshow(ground_truth, cmap='viridis')  # Adjust cmap as needed
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')  # Hide axes
    
        # Adjust layout
        plt.tight_layout()
    
        # Save the figure
        plt.savefig(f'{output_dir}/comparison_{i}.png', dpi=300, bbox_inches='tight')
    
        # Close the figure to free memory
        plt.close(fig)

