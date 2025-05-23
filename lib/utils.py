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
from datetime import datetime


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
    print(f"assessing {model_pth}")
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

            part = model_pth.split("/")[0]
            if "VAE" in part or "GAN" in part:
                outputs = model(low_res)[0] # handel VAE and GAN output slightly differently
            else:
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
                # Mask to exclude land (where high_res is 0)
                mask_v = high_res[i][0] > 0
                mask_h = high_res[i][1] > 0

                # Apply mask to filter only valid pixels
                filtered_output_v = outputs[i][0][mask_v]
                filtered_high_res_v = high_res[i][0][mask_v]
                filtered_low_res_v = low_res[i][0][mask_v]

                filtered_output_h = outputs[i][1][mask_h]
                filtered_high_res_h = high_res[i][1][mask_h]
                filtered_low_res_h = low_res[i][1][mask_h]

                # PSNR Computation
                psnr_value_v = psnr(filtered_output_v, filtered_high_res_v, pol="V")
                psnr_value_h = psnr(filtered_output_h, filtered_high_res_h, pol="H")
                psnr_baseline_value_v = psnr(filtered_low_res_v, filtered_high_res_v, pol="V")
                psnr_baseline_value_h = psnr(filtered_low_res_h, filtered_high_res_h, pol="H")

                # Convert to NumPy for SSIM computation
                filtered_output_v_np = filtered_output_v.cpu().numpy()
                filtered_output_h_np = filtered_output_h.cpu().numpy()
                filtered_high_res_v_np = filtered_high_res_v.cpu().numpy()
                filtered_high_res_h_np = filtered_high_res_h.cpu().numpy()
                filtered_low_res_v_np = filtered_low_res_v.cpu().numpy()
                filtered_low_res_h_np = filtered_low_res_h.cpu().numpy()

                # SSIM Computation
                ssim_output_v = ssim(filtered_high_res_v_np, filtered_output_v_np, 
                                    data_range=filtered_high_res_v_np.max() - filtered_high_res_v_np.min())
                ssim_output_h = ssim(filtered_high_res_h_np, filtered_output_h_np, 
                                    data_range=filtered_high_res_h_np.max() - filtered_high_res_h_np.min())
                ssim_low_res_v = ssim(filtered_high_res_v_np, filtered_low_res_v_np, 
                                    data_range=filtered_high_res_v_np.max() - filtered_high_res_v_np.min())
                ssim_low_res_h = ssim(filtered_high_res_h_np, filtered_low_res_h_np, 
                                    data_range=filtered_high_res_h_np.max() - filtered_high_res_h_np.min())



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
            writer.writerow(["Baseline", average_psnr_baseline_v, average_psnr_baseline_h, average_ssim_baseline_v, average_ssim_baseline_h])

        writer.writerows(rows)


def create_metric_timeseries(model_pth, model, test_paths, test_dataset, device="cuda" if torch.cuda.is_available() else "cpu", denormalize = True):
    """
    Compute PSNR and SSIM for each test sample and write to a csv file.
    """

    for sample_idx in range(len(test_paths)):
        time = test_paths[sample_idx].split("_")[-1].replace(".nc","")
        dt = datetime.strptime(str(time), "%Y%m%d%H")
        dt.strftime("%Y-%m-%d %H:00")
        date = str(dt).split(' ')[0]
        print(date)

        model.load_state_dict(torch.load(model_pth, map_location=torch.device("cpu")))
        model.eval()  # Set the model to evaluation mode

        # Get a single test sample
        low_res, high_res = test_dataset[sample_idx]

        # Convert to batch format and move to device
        low_res_tensor = low_res.clone().detach().unsqueeze(0).to(device)
        high_res_tensor = high_res.clone().detach().unsqueeze(0).to(device)

        # Run model inference
        with torch.no_grad():
            part = model_pth.split("/")[0]
            if "VAE" in part or "GAN" in part:
                output = model(low_res_tensor)[0] # handel VAE and GAN output slightly differently
            else:
                output = model(low_res_tensor)

        # Move data back to CPU and convert to NumPy
        if denormalize:
            low_res = test_dataset.denormalize(low_res_tensor.squeeze(0)).cpu().numpy()
            high_res = test_dataset.denormalize(high_res_tensor.squeeze(0)).cpu().numpy()
            output = test_dataset.denormalize(output.squeeze(0)).cpu().numpy()
        else:
            low_res = low_res_tensor.squeeze(0).cpu().numpy()
            high_res = high_res_tensor.squeeze(0).cpu().numpy()
            output = output.squeeze(0).cpu().numpy()

        filtered_low_res_v = low_res[0][low_res[0] > 0]
        filtered_high_res_v = high_res[0][high_res[0] > 0]
        combined_filtered_v = np.concatenate([filtered_low_res_v, filtered_high_res_v])
        v_min = np.min(combined_filtered_v)
        v_max = np.max(combined_filtered_v)

        filtered_low_res_h = low_res[1][low_res[1] > 0]
        filtered_high_res_h = high_res[1][high_res[1] > 0]
        combined_filtered_h = np.concatenate([filtered_low_res_h, filtered_high_res_h])
        h_min = np.min(combined_filtered_h)
        h_max = np.max(combined_filtered_h)

        # Mask to exclude regions where high_res is 0
        mask_v = high_res[0] > 0
        mask_h = high_res[1] > 0

        # Filtered tensors
        filtered_output_v = output[0][mask_v]
        filtered_high_res_v = high_res[0][mask_v]
        filtered_low_res_v = low_res[0][mask_v]

        filtered_output_h = output[1][mask_h]
        filtered_high_res_h = high_res[1][mask_h]
        filtered_low_res_h = low_res[1][mask_h]

        # PSNR computation
        psnr_value_v = psnr(torch.tensor(filtered_high_res_v), torch.tensor(filtered_output_v), pol="V").item()
        psnr_baseline_v = psnr(torch.tensor(filtered_high_res_v), torch.tensor(filtered_low_res_v), pol="V").item()

        psnr_value_h = psnr(torch.tensor(filtered_high_res_h), torch.tensor(filtered_output_h), pol="H").item()
        psnr_baseline_h = psnr(torch.tensor(filtered_high_res_h), torch.tensor(filtered_low_res_h), pol="H").item()

        # SSIM computation
        ssim_low_res_v = ssim(filtered_high_res_v, filtered_low_res_v, data_range=filtered_high_res_v.max() - filtered_high_res_v.min())
        ssim_low_res_h = ssim(filtered_high_res_h, filtered_low_res_h, data_range=filtered_high_res_h.max() - filtered_high_res_h.min())

        ssim_output_v = ssim(filtered_high_res_v, filtered_output_v, data_range=filtered_high_res_v.max() - filtered_high_res_v.min())
        ssim_output_h = ssim(filtered_high_res_h, filtered_output_h, data_range=filtered_high_res_h.max() - filtered_high_res_h.min())

        # Append results to CSV
        new_row = [date, psnr_baseline_v, psnr_baseline_h, psnr_value_v, psnr_value_h, ssim_low_res_v, ssim_low_res_h, ssim_output_v, ssim_output_h]

        # Check if file exists
        outfile_path = "test_metric_timeseries/" + model.__class__.__name__ + "_test_metric_timeseries.csv"

        file_exists = os.path.isfile(outfile_path)

        # Read existing data
        rows = []
        date_found = False

        if file_exists:
            with open(outfile_path, mode="r", newline="") as file:
                reader = csv.reader(file)
                for row in reader:
                    if row and row[0] == date:
                        rows.append(new_row)  # Replace existing date row
                        date_found = True
                    else:
                        rows.append(row)

        # If date wasn't found, add new row
        if not date_found:
            rows.append(new_row)

        # Write updated data back to file
        with open(outfile_path, mode="w", newline="") as file:
            writer = csv.writer(file)
            
            if not file_exists:
                writer.writerow(["date", "psnr_baseline_v", "psnr_baseline_h", "psnr_value_v", "psnr_value_h", "ssim_low_res_v", "ssim_low_res_h", "ssim_output_v", "ssim_output_h"])
                # writer.writerow(new_row)

            writer.writerows(rows)