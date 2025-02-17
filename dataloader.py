import torch
from torch.utils.data import Dataset, DataLoader
import xarray as xr
import numpy as np
import os
import cv2
from torchvision.transforms import ToTensor


class PassiveMicrowaveDataset(Dataset):
    def __init__(self, data_paths, transform=None, use_bicubic=False):
        self.data_paths = data_paths
        self.transform = transform
        self.use_bicubic = use_bicubic
        self.data = self.load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        low_res_data, high_res_data = self.data[idx]

        # Apply transforms (convert to tensor, normalize)
        if self.transform:
            low_res_data = self.transform(low_res_data)
            high_res_data = self.transform(high_res_data)

        return low_res_data, high_res_data

    def load_data(self):
        data = []
        for file_path in self.data_paths:
            ds = xr.open_dataset(file_path)
            ssmi_v = np.nan_to_num(ds['ssmi_tb37v'].values, nan=0)
            ssmi_h = np.nan_to_num(ds['ssmi_tb37h'].values, nan=0)
            amsr_v = np.nan_to_num(ds['amsr_tb37v'].values, nan=0)
            amsr_h = np.nan_to_num(ds['amsr_tb37h'].values, nan=0)

            # Stack along channel dimension
            low_res = np.stack((ssmi_v, ssmi_h), axis=-1)  # (H, W, C)
            high_res = np.stack((amsr_v, amsr_h), axis=-1)  # (H, W, C)

            # Apply bicubic interpolation if needed
            if self.use_bicubic:
                low_res = self.bicubic_interpolation(low_res, high_res.shape[1], high_res.shape[2])

            data.append((low_res, high_res))

        return data

    def bicubic_interpolation(self, low_res, target_height, target_width):
        # Resize each channel separately
        upscaled = np.stack([
            cv2.resize(low_res[i], (target_width, target_height), interpolation=cv2.INTER_CUBIC)
            for i in range(low_res.shape[0])
        ], axis=0)
        return upscaled


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


# all_paths = []
# for year in range(2003, 2021):
#     for month in range(1, 13):
#         folder_path = f"/lustre/storeB/project/metkl/DigitalSeaIce/are-phd/SuperResolutionSeaIce/Dataset/AMSRSSMI/{year}/{month:02d}"
#         if os.path.exists(folder_path):
#             for file_name in os.listdir(folder_path):
#                 if file_name.endswith(".nc"):
#                     all_paths.append(os.path.join(folder_path, file_name))
#         else:
#             print(f"Warning: Directory {folder_path} does not exist. Skipping...")

# # Split data into train, validation, and test sets
# train_files, val_files, test_files = PassiveMicrowaveDataset.split_data(None, all_paths)

# # Create datasets for each split with bicubic interpolation enabled for SRCNN
# train_dataset = PassiveMicrowaveDataset(train_files, split='train', use_bicubic=True)
# val_dataset = PassiveMicrowaveDataset(val_files, split='val', use_bicubic=True)  
# test_dataset = PassiveMicrowaveDataset(test_files, split='test', use_bicubic=True)

# # Create DataLoaders for each split
# train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
# val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)
# test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# # Accessing a batch from the train DataLoader
# for low_res, high_res in train_dataloader:
#     print(low_res.shape, high_res.shape)
#     break
