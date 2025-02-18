import torch.optim as optim
import torch.nn as nn
from srcnn_model import SRCNN
from dataloader import PassiveMicrowaveDataset, split_data
import torch
from torch.utils.data import Dataset, DataLoader
import xarray as xr
import numpy as np
import os
import cv2
from torchvision.transforms import ToTensor
import os
import logging
import sys

output_dir = os.path.dirname(os.path.abspath(__file__))

def init_logging():
    # Log to console
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    log_info = logging.StreamHandler(sys.stdout)
    log_info.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(log_info)
    return logger

logger = init_logging()

# Initialize model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SRCNN().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
# optimizer = optim.SGD([
#     {'params': model.conv1.parameters(), 'lr': 1e-4},  # First layer
#     {'params': model.conv2.parameters(), 'lr': 1e-4},  # Second layer
#     {'params': model.conv3.parameters(), 'lr': 1e-5}   # Last layer
# ], lr=1e-4)#, momentum=0.9)

# Load dataset
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

# set how much of the dataset to use
nr_samples = "all"
if nr_samples == "all":
    train_dataset = PassiveMicrowaveDataset(train_paths, transform=ToTensor(), use_bicubic=True)
    val_dataset = PassiveMicrowaveDataset(val_paths, transform=ToTensor(), use_bicubic=True)
    logger.info("Using all samples")
else:    
    val_samples = int(nr_samples/4)
    train_dataset = PassiveMicrowaveDataset(train_paths[:nr_samples], transform=ToTensor(), use_bicubic=True)
    val_dataset = PassiveMicrowaveDataset(val_paths[:val_samples], transform=ToTensor(), use_bicubic=True)
    logger.info(f"Using {nr_samples} samples")

batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Training loop
num_epochs = 30

for epoch in range(num_epochs):
    model.train()
    train_loss = 0

    for low_res, high_res in train_loader:
        low_res, high_res = low_res.to(device), high_res.to(device)
        optimizer.zero_grad()
        outputs = model(low_res)
        loss = criterion(outputs, high_res)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for low_res, high_res in val_loader:
            low_res, high_res = low_res.to(device), high_res.to(device)
            outputs = model(low_res)
            loss = criterion(outputs, high_res)
            val_loss += loss.item()

    logger.info(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss/len(train_loader):.6f}, Val Loss: {val_loss/len(val_loader):.6f}")
logger.info(f"Training completed. Saving model to {output_dir}...")
# Save the model
torch.save(model.state_dict(), f"{output_dir}/srcnn_model_epochs{num_epochs}_batchsize{batch_size}_samples{nr_samples}.pth")