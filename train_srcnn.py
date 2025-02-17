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

# Initialize model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SRCNN().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

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
            print(f"Warning: Directory {folder_path} does not exist. Skipping...")

train_paths, val_paths, test_paths = split_data(all_paths)

train_dataset = PassiveMicrowaveDataset(train_paths[:60], transform=ToTensor(), use_bicubic=True)
val_dataset = PassiveMicrowaveDataset(val_paths[:30], transform=ToTensor(), use_bicubic=True)

train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False)

# Training loop
num_epochs = 10

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

    print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss/len(train_loader):.6f}, Val Loss: {val_loss/len(val_loader):.6f}")

# Save the model
torch.save(model.state_dict(), "srcnn_model.pth")
