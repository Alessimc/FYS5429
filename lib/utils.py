import logging
import sys
import torch
import numpy as np
import torch.nn.functional as F
from dataloader import split_data
import os

def init_logging():
    """Log to console"""
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    log_info = logging.StreamHandler(sys.stdout)
    log_info.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(log_info)
    return logger

logger = init_logging()

def get_split_datapaths():
    all_paths = []
    for year in range(2003, 2021):
        for month in range(1, 13):
            folder_path = f"/lustre/storeB/project/metkl/DigitalSeaIce/are-phd/SuperResolutionSeaIce/Dataset/AMSRSSMI/{year}/{month:02d}"
            if os.path.exists(folder_path):
                for file_name in os.listdir(folder_path):
                    if file_name.endswith(".nc"):
                        all_paths.append(os.path.join(folder_path, file_name))
            else:
                logger.info(f"Warning: Directory {folder_path} does not exist. Skipping...")

    train_paths, val_paths, test_paths = split_data(all_paths)
    return train_paths, val_paths, test_paths

pol_max_val = {
    "V": 294.8545227050781,
    "H": 302.40582275390625
}

def psnr(predicted, target, pol):
    """Calculate the PSNR between predicted and target images using the given polarizations max value."""
    # Calculate MSE (Mean Squared Error)
    mse = F.mse_loss(predicted, target)
    
    # Get the max pixel value of the given polarization
    max_value = pol_max_val[pol] 
    
    # Calculate PSNR
    psnr_value = 10 * torch.log10((max_value ** 2) / mse)
    return psnr_value
