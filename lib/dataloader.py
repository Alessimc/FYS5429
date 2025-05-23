import torch
from torch.utils.data import Dataset, DataLoader
import xarray as xr
import numpy as np
import os
import cv2
from torchvision.transforms import ToTensor

import xarray as xr
import numpy as np
import torch
import cv2
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset

class PassiveMicrowaveDataset(Dataset):
    def __init__(self, data_paths, transform=ToTensor(), normalize=True, use_bicubic=False):
        """
        Optimized dataset: loads data lazily instead of storing everything in memory.
        """
        self.data_paths = data_paths
        self.transform = transform
        self.use_bicubic = use_bicubic
        self.normalize = normalize
        self.mean = 126.07822402554218  # dataset mean
        self.std = 106.56974184133237  # dataset std

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        file_path = self.data_paths[idx]
        ds = xr.open_dataset(file_path)

        # Load only the necessary variables, convert to float32, and replace NaNs with 0
        ssmi_v = np.nan_to_num(ds['ssmi_tb37v'].astype(np.float32).values, nan=0)
        ssmi_h = np.nan_to_num(ds['ssmi_tb37h'].astype(np.float32).values, nan=0)
        amsr_v = np.nan_to_num(ds['amsr_tb37v'].astype(np.float32).values, nan=0)
        amsr_h = np.nan_to_num(ds['amsr_tb37h'].astype(np.float32).values, nan=0)

        # Stack along channel dimension
        low_res = np.stack((ssmi_v, ssmi_h), axis=-1)  # (H, W, C)
        high_res = np.stack((amsr_v, amsr_h), axis=-1)  # (H, W, C)

        # Apply bicubic interpolation if needed
        if self.use_bicubic:
            low_res = self.bicubic_interpolation(low_res, high_res.shape[0], high_res.shape[1])

        # Convert to tensor and normalize
        if self.transform:
            low_res = self.transform(low_res)
            high_res = self.transform(high_res)
        if self.normalize:
            low_res = (low_res - self.mean) / self.std
            high_res = (high_res - self.mean) / self.std

        return low_res, high_res

    def bicubic_interpolation(self, low_res, target_height, target_width):
        """Resize each channel separately using bicubic interpolation."""
        upscaled = np.stack([
            cv2.resize(low_res[..., i], (target_width, target_height), interpolation=cv2.INTER_CUBIC)
            for i in range(low_res.shape[-1])
        ], axis=-1)
        return upscaled

    def denormalize(self, tensor):
        """Reverse normalization to original scale."""
        return tensor * self.std + self.mean if self.normalize else tensor


# class PassiveMicrowaveDataset(Dataset):
#     def __init__(self, data_paths, transform=ToTensor(), normalize=True, use_bicubic=False):
#         """
#         Note: Data is normalized by default, normalize=False for raw data.
#         """
#         self.data_paths = data_paths
#         self.transform = transform
#         self.use_bicubic = use_bicubic
#         self.data = self.load_data()
#         self.normalize = normalize
#         self.mean = 126.07822402554218 # hardcoded dataset mean
#         self.std = 106.56974184133237 # hardcoded dataset std

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         low_res_data, high_res_data = self.data[idx]

#         # Apply transforms (convert to tensor, normalize)
#         if self.transform:
#             low_res_data = self.transform(low_res_data)
#             high_res_data = self.transform(high_res_data)
#         if self.normalize:
#             low_res_data = (low_res_data - self.mean) / self.std
#             high_res_data = (high_res_data - self.mean) / self.std

#         return low_res_data, high_res_data

#     def load_data(self):
#         data = []
#         for file_path in self.data_paths:
#             ds = xr.open_dataset(file_path)
#             ssmi_v = np.nan_to_num(ds['ssmi_tb37v'].values, nan=0)
#             ssmi_h = np.nan_to_num(ds['ssmi_tb37h'].values, nan=0)
#             amsr_v = np.nan_to_num(ds['amsr_tb37v'].values, nan=0)
#             amsr_h = np.nan_to_num(ds['amsr_tb37h'].values, nan=0)

#             # Stack along channel dimension
#             low_res = np.stack((ssmi_v, ssmi_h), axis=-1)  # (H, W, C)
#             high_res = np.stack((amsr_v, amsr_h), axis=-1)  # (H, W, C)

#             # Apply bicubic interpolation if needed
#             if self.use_bicubic:
#                 low_res = self.bicubic_interpolation(low_res, high_res.shape[1], high_res.shape[2])

#             data.append((low_res, high_res))

#         return data
    
#     def denormalize(self, tensor):
#         """Reverse normalization to original scale."""
#         return tensor * self.std + self.mean if self.normalize else tensor

#     def bicubic_interpolation(self, low_res, target_height, target_width):
#         # Resize each channel separately
#         upscaled = np.stack([
#             cv2.resize(low_res[i], (target_width, target_height), interpolation=cv2.INTER_CUBIC)
#             for i in range(low_res.shape[0])
#         ], axis=0)
#         return upscaled


def split_data(all_paths):
    train_files, val_files, test_files = [], [], []

    for path in all_paths:
        year = int(path.split('/')[-3])
        if 2003 <= year <= 2018:
            train_files.append(path)
        elif year == 2019:
            val_files.append(path)
        elif year == 2020:
            test_files.append(path)

    return train_files, val_files, test_files

