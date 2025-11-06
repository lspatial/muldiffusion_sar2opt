import torch
import warnings
import os
from piqa import SSIM
from torch import nn 
from kornia.losses import ssim_loss
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt 

def generateImg(true_masks_all,masks_pred_all):
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

class SSIMLoss(nn.Module):
    def forward(self, x, y):
        return 1. - kl.ssim(x, y, window_size=11, reduction='mean')

class SSIMMAELoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(SSIMMAELoss, self).__init__()
        self.l1 = nn.L1Loss()
        self.ssim = SSIMLoss()
        self.alpha = alpha
        
    def forward(self, x, y):
        l1_loss = self.l1(x, y)
        ssim_loss = self.ssim(x, y)
        return self.alpha * l1_loss + (1 - self.alpha) * ssim_loss
      
class RobustSSIMMAELoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(RobustSSIMMAELoss, self).__init__() 
        self.l1 = nn.L1Loss()
        self.alpha = alpha
        
    def forward(self, x, y):
        l1_loss = self.l1(x, y)
        ssim_value = ssim_loss(x, y, window_size=11, reduction='mean')
        return self.alpha * l1_loss + (1 - self.alpha) * ssim_value    
      
      
class RobustSSIMMSELoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(RobustSSIMMSELoss, self).__init__()
        self.rse = nn.MSELoss()
        self.alpha = alpha
        
    def forward(self, x, y):
        lrse_loss = self.rse(x, y)
        ssim_value = ssim_loss(x, y, window_size=11, reduction='mean')
        return self.alpha * lrse_loss + (1 - self.alpha) * ssim_value          
      
def silence_all_warnings():
    # Environment variable (affects subprocesses too)
    os.environ["PYTHONWARNINGS"] = "ignore"

    # Python warning filters
    warnings.filterwarnings("ignore")

    # Common specific warning types
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=ImportWarning)
    warnings.filterwarnings("ignore", category=ResourceWarning)

    # Disable warning printing to stderr
    warnings.simplefilter("ignore")

    # Force warnings.warn to do nothing
    original_warn = warnings.warn
    warnings.warn = lambda *args, **kwargs: None


def r2_score(y_true, y_pred):
    """
    Calculate R² coefficient of determination

    Args:
        y_true: Tensor of ground truth values
        y_pred: Tensor of predicted values

    Returns:
        r2: R² score
    """
    # Ensure tensors are on the same device and have the same shape
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)

    # Calculate mean of true values
    y_mean = torch.mean(y_true)

    # Calculate total sum of squares
    ss_tot = torch.sum((y_true - y_mean) ** 2)

    # Calculate sum of squared residuals
    ss_res = torch.sum((y_true - y_pred) ** 2)

    # Calculate R²
    r2 = 1 - (ss_res / ss_tot)

    return r2


def rmsec(y_true, y_pred):
    """
    Calculate Root Mean Square Error

    Args:
        y_true: Tensor of ground truth values
        y_pred: Tensor of predicted values

    Returns:
        rmse: Root Mean Square Error
    """
    # Flatten the tensors
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)

    # Calculate MSE
    mse = torch.mean((y_true - y_pred) ** 2)

    # Take square root to get RMSE
    return torch.sqrt(mse)


def pearson_correlation(x, y):
    """
    Calculate Pearson correlation coefficient between two variables

    Args:
        x: Tensor of first variable
        y: Tensor of second variable

    Returns:
        correlation: Pearson correlation coefficient
    """
    # Ensure tensors are flattened
    x = x.reshape(-1)
    y = y.reshape(-1)

    # Calculate means
    x_mean = torch.mean(x)
    y_mean = torch.mean(y)

    # Calculate numerator (covariance)
    numerator = torch.sum((x - x_mean) * (y - y_mean))

    # Calculate denominator (product of standard deviations)
    x_std = torch.sqrt(torch.sum((x - x_mean) ** 2))
    y_std = torch.sqrt(torch.sum((y - y_mean) ** 2))
    denominator = x_std * y_std

    # Calculate correlation
    correlation = numerator / denominator

    return correlation
