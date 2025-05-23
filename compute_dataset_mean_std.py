import numpy as np
import xarray as xr
import os
from lib.utils import init_logging, get_split_datapaths

logger = init_logging()

def compute_mean_std(file_paths):
    """Compute per-channel mean, std, and max over the entire dataset efficiently."""
    sum_values = 0  # Running sum (all channels)
    sum_squared = 0  # Running sum of squares (all channels)
    
    sum_v = 0  # Running sum for V channels
    sum_v_squared = 0  # Running sum of squares for V channels
    
    sum_h = 0  # Running sum for H channels
    sum_h_squared = 0  # Running sum of squares for H channels

    max_v = float('-inf')  # Max value for V channels
    max_h = float('-inf')  # Max value for H channels

    total_pixels = 0  # Total number of pixels
    total_pixels_v = 0  # Total pixels for V channels
    total_pixels_h = 0  # Total pixels for H channels

    # 103460 values per channel excluding nan values (173056 - 69596)
    values_per_file = 4 * 103460  # Assuming each file has 4 channels of 416x416
    values_per_file_vh = 2 * 103460  # Only V or H channels per file

    file_nr = 0

    for file_path in file_paths:
        ds = xr.open_dataset(file_path)

        # Extract values 
        ssmi_v = ds['ssmi_tb37v'].values
        ssmi_h = ds['ssmi_tb37h'].values
        amsr_v = ds['amsr_tb37v'].values
        amsr_h = ds['amsr_tb37h'].values

        # Compute sum and sum of squares for all channels
        sum_values += np.nansum(ssmi_v) + np.nansum(ssmi_h) + np.nansum(amsr_v) + np.nansum(amsr_h)
        sum_squared += np.nansum(ssmi_v**2) + np.nansum(ssmi_h**2) + np.nansum(amsr_v**2) + np.nansum(amsr_h**2)

        # Compute sum and sum of squares for V and H separately
        sum_v += np.nansum(ssmi_v) + np.nansum(amsr_v)
        sum_v_squared += np.nansum(ssmi_v**2) + np.nansum(amsr_v**2)
        
        sum_h += np.nansum(ssmi_h) + np.nansum(amsr_h)
        sum_h_squared += np.nansum(ssmi_h**2) + np.nansum(amsr_h**2)

        # Track max values
        max_v = max(max_v, np.nanmax(ssmi_v), np.nanmax(amsr_v))
        max_h = max(max_h, np.nanmax(ssmi_h), np.nanmax(amsr_h))

        # Update total pixel count
        total_pixels += values_per_file
        total_pixels_v += values_per_file_vh
        total_pixels_h += values_per_file_vh

        file_nr += 1
        logger.info(f"-----File nr {file_nr}: {file_path}")

    # Compute overall mean and std
    mean = sum_values / total_pixels
    std = np.sqrt(sum_squared / total_pixels - mean**2)

    # Compute per-channel mean and std
    mean_v = sum_v / total_pixels_v
    std_v = np.sqrt(sum_v_squared / total_pixels_v - mean_v**2)

    mean_h = sum_h / total_pixels_h
    std_h = np.sqrt(sum_h_squared / total_pixels_h - mean_h**2)

    return mean, std, mean_v, std_v, mean_h, std_h, max_v, max_h


train_paths, val_paths, test_paths = get_split_datapaths()

mean, std, mean_v, std_v, mean_h, std_h, max_v, max_h = compute_mean_std(train_paths)

logger.info(f"IGNORING nan values and not handling them as 0")
logger.info(f"Computed dataset mean: {mean}, std: {std}")
logger.info(f"Computed V mean: {mean_v}, V std: {std_v}, Max V: {max_v}")
logger.info(f"Computed H mean: {mean_h}, H std: {std_h}, Max H: {max_h}")