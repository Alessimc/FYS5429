import logging
import sys
import torch
import numpy as np
import torch.nn.functional as F
try:
    from dataloader import split_data
except ImportError:
    from lib.dataloader import split_data
import os
import torch.nn as nn
import csv
from skimage.metrics import structural_similarity as ssim


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


def assess_model(model_pth, model, test_loader, test_dataset, mask_land=False, denormalize = True, outfile_path = "assess_models.csv", model_name="No name given!"):

    model.load_state_dict(torch.load(model_pth, map_location=torch.device("cpu")))
    model.eval()  # Set the model to evaluation mode

    # Move model to device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    num_samples = 0

    # Initialize total PSNR accumulators
    total_psnr_v = 0
    total_psnr_h = 0
    total_psnr_baseline_v = 0
    total_psnr_baseline_h = 0

    # Initialize total SSIM accumulators
    total_ssim_v = 0
    total_ssim_h = 0
    total_ssim_baseline_v = 0
    total_ssim_baseline_h = 0

    with torch.no_grad():
        for low_res, high_res in test_loader:
            low_res, high_res = low_res.to(device), high_res.to(device)
            outputs = model(low_res)

            # Denormalize if needed
            if denormalize:
                outputs = test_dataset.denormalize(outputs)
                high_res = test_dataset.denormalize(high_res)
                low_res = test_dataset.denormalize(low_res)

            # Apply land mask if enabled
            if mask_land:
                outputs[high_res == 0] = 0

            # Compute PSNR and SSIM
            batch_psnr_v = batch_psnr_h = 0
            batch_ssim_v = batch_ssim_h = 0
            batch_psnr_baseline_v = batch_psnr_baseline_h = 0
            batch_ssim_baseline_v = batch_ssim_baseline_h = 0

            for i in range(len(outputs)):
                # PSNR Computation
                psnr_value_v = psnr(outputs[i][0], high_res[i][0], pol="V")
                psnr_value_h = psnr(outputs[i][1], high_res[i][1], pol="H")
                psnr_baseline_value_v = psnr(low_res[i][0], high_res[i][0], pol="V")
                psnr_baseline_value_h = psnr(low_res[i][1], high_res[i][1], pol="H")

                # SSIM Computation
                ssim_output_v = ssim(high_res[i][0].cpu().numpy(), outputs[i][0].cpu().numpy(), 
                                    data_range=outputs[i][0].max().item() - outputs[i][0].min().item())
                ssim_output_h = ssim(high_res[i][1].cpu().numpy(), outputs[i][1].cpu().numpy(), 
                                    data_range=outputs[i][1].max().item() - outputs[i][1].min().item())
                ssim_low_res_v = ssim(high_res[i][0].cpu().numpy(), low_res[i][0].cpu().numpy(), 
                                    data_range=low_res[i][0].max().item() - low_res[i][0].min().item())
                ssim_low_res_h = ssim(high_res[i][1].cpu().numpy(), low_res[i][1].cpu().numpy(), 
                                    data_range=low_res[i][1].max().item() - low_res[i][1].min().item())

                # Accumulate batch values
                batch_psnr_v += psnr_value_v.item()
                batch_psnr_h += psnr_value_h.item()
                batch_psnr_baseline_v += psnr_baseline_value_v.item()
                batch_psnr_baseline_h += psnr_baseline_value_h.item()

                batch_ssim_v += ssim_output_v
                batch_ssim_h += ssim_output_h
                batch_ssim_baseline_v += ssim_low_res_v
                batch_ssim_baseline_h += ssim_low_res_h

            # Accumulate total metrics
            total_psnr_v += batch_psnr_v
            total_psnr_h += batch_psnr_h
            total_psnr_baseline_v += batch_psnr_baseline_v
            total_psnr_baseline_h += batch_psnr_baseline_h

            total_ssim_v += batch_ssim_v
            total_ssim_h += batch_ssim_h
            total_ssim_baseline_v += batch_ssim_baseline_v
            total_ssim_baseline_h += batch_ssim_baseline_h

            num_samples += len(outputs)

    # Compute final averages
    average_psnr_v = total_psnr_v / num_samples if num_samples > 0 else 0
    average_psnr_h = total_psnr_h / num_samples if num_samples > 0 else 0
    average_psnr_baseline_v = total_psnr_baseline_v / num_samples if num_samples > 0 else 0
    average_psnr_baseline_h = total_psnr_baseline_h / num_samples if num_samples > 0 else 0

    average_ssim_v = total_ssim_v / num_samples if num_samples > 0 else 0
    average_ssim_h = total_ssim_h / num_samples if num_samples > 0 else 0
    average_ssim_baseline_v = total_ssim_baseline_v / num_samples if num_samples > 0 else 0
    average_ssim_baseline_h = total_ssim_baseline_h / num_samples if num_samples > 0 else 0

    # Log results
    logger.info(f"V-Pol = Test PSNR: {average_psnr_v:.6f}, Test SSIM: {average_ssim_v:.6f}, Baseline PSNR: {average_psnr_baseline_v:.6f}, Baseline SSIM: {average_ssim_baseline_v:.6f}")
    logger.info(f"H-Pol = Test PSNR: {average_psnr_h:.6f}, Test SSIM: {average_ssim_h:.6f}, Baseline PSNR: {average_psnr_baseline_h:.6f}, Baseline SSIM: {average_ssim_baseline_h:.6f}")

    # Append results to CSV
    new_row = [model_name, average_psnr_v, average_psnr_h, average_ssim_v, average_ssim_h]

    # Check if file exists
    file_exists = os.path.isfile(outfile_path)

    # Read existing data
    rows = []
    model_found = False

    if file_exists:
        with open(outfile_path, mode="r", newline="") as file:
            reader = csv.reader(file)
            for row in reader:
                if row and row[0] == model_name:
                    rows.append(new_row)  # Replace existing model row
                    model_found = True
                else:
                    rows.append(row)

    # If model name wasn't found, add new row
    if not model_found:
        rows.append(new_row)

    # Write updated data back to file
    with open(outfile_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        
        if not file_exists:
            writer.writerow(["Model_name", "v_pol_psnr", "h_pol_psnr", "v_pol_ssim", "h_pol_ssim"])
            writer.writerow(["Baseline", average_psnr_baseline_v, average_psnr_baseline_h])

        writer.writerows(rows)