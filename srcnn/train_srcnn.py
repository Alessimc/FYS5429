import torch.optim as optim
import torch.nn as nn
from srcnn_model import SRCNN, SRCNN2, SRCNN3
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "lib")))
from utils import init_logging, get_split_datapaths
from dataloader import PassiveMicrowaveDataset
import argparse
import csv

logger = init_logging()

# Dictionary to map model numbers to their classes
MODEL_MAP = {
    1: SRCNN,
    2: SRCNN2,
    3: SRCNN3
}

def main(args):
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trained_models")
    os.makedirs(output_dir, exist_ok=True)
    metrics_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_train_val_metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    batch_size = args.batch_size
    num_epochs = args.epochs
    nr_samples = args.nr_samples if args.nr_samples != "all" else "all"

    file_name = f"srcnn_model{args.model}_epochs{num_epochs}_batchsize{batch_size}_samples{nr_samples}_normalized.pth"
    metrics_file = f"{metrics_dir}/{file_name.replace('.pth', '.csv')}"

    # Initialize model, loss function, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Select the correct model based on argument
    model_class = MODEL_MAP.get(args.model)
    if model_class is None:
        raise ValueError(f"Invalid model number {args.model}. Choose from 1, 2, or 3.")
    
    model = model_class().to(device)

    criterion = nn.MSELoss()
    # Define optimizer with different learning rates
    if args.model in [1, 2]:  
        optimizer = optim.Adam([
            {'params': model.conv1.parameters(), 'lr': 1e-4},  # First layer
            {'params': model.conv2.parameters(), 'lr': 1e-4},  # Second layer
            {'params': model.conv3.parameters(), 'lr': 1e-5}   # Last layer
        ])
    elif args.model == 3:  
        optimizer = optim.Adam([
            {'params': model.conv1.parameters(), 'lr': 1e-4},  # First layer
            {'params': model.conv2.parameters(), 'lr': 1e-4},  # Second layer
            {'params': model.conv3.parameters(), 'lr': 1e-4},  # Third layer
            {'params': model.conv4.parameters(), 'lr': 1e-5}   # Last layer
        ])

    # Load dataset paths
    train_paths, val_paths, test_paths = get_split_datapaths()

    # Set how much of the dataset to use
    normalize = True

    if nr_samples == "all":
        train_dataset = PassiveMicrowaveDataset(train_paths, transform=ToTensor(), normalize=normalize, use_bicubic=True)
        val_dataset = PassiveMicrowaveDataset(val_paths, transform=ToTensor(), normalize=normalize, use_bicubic=True)
        logger.info("Using all samples")
    else:
        nr_samples = int(nr_samples)
        val_samples = nr_samples // 4
        train_dataset = PassiveMicrowaveDataset(train_paths[:nr_samples], transform=ToTensor(), normalize=normalize, use_bicubic=True)
        val_dataset = PassiveMicrowaveDataset(val_paths[:val_samples], transform=ToTensor(), normalize=normalize, use_bicubic=True)
        logger.info(f"Using {nr_samples} training samples and {val_samples} validation samples")

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Training info logging
    logger.info(f"Training SRCNN Model {args.model}")
    logger.info(f"Optimizer: Adam (lr per layer: 1e-4. Last layer: 1e-5)")
    logger.info(f"Nr of training samples: {nr_samples} (Normalized: {normalize})")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Number of epochs: {num_epochs}")

    with open(metrics_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss"]) 

        # Training loop
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

                train_loss_avg = train_loss / len(train_loader)
                val_loss_avg = val_loss / len(val_loader)

                # Log results
                logger.info(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss_avg:.6f}, Val Loss: {val_loss_avg:.6f}")

                # Save to CSV
                writer.writerow([epoch + 1, train_loss_avg, val_loss_avg])

    logger.info(f"Training metrics saved to {metrics_file}")
    # Save the trained model    
    logger.info(f"Training completed. Saving model to {output_dir}/{file_name} ...")
    torch.save(model.state_dict(), os.path.join(output_dir, file_name))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to train the SRCNN.")

    parser.add_argument("--model", type=int, required=True, choices=[1, 2, 3], help="Choose model: 1, 2, or 3.")
    parser.add_argument("--batch_size", type=int, required=True, help="Batch size.")
    parser.add_argument("--epochs", type=int, required=True, help="Number of epochs.")
    parser.add_argument("--nr_samples", type=str, default="all", help='Number of samples ("all" or an integer).')

    args = parser.parse_args()
    main(args)
