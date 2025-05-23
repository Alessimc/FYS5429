import torch.optim as optim
import torch.nn as nn
from srcnn_model import SRCNN, SRCNN_L4, SRCNN_W128, SRCNN_W256, SRCNN_K3, SRCNN_K5
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
import time

logger = init_logging()

# Dictionary to map model keys to their classes
MODEL_MAP = {
    "default": SRCNN,
    "K3": SRCNN_K3,
    "K5": SRCNN_K5,
    "L4": SRCNN_L4,
    "W128": SRCNN_W128,
    "W256": SRCNN_W256
}

def main(args):
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trained_models")
    os.makedirs(output_dir, exist_ok=True)
    metrics_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_train_val_metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    batch_size = args.batch_size
    nr_samples = args.nr_samples if args.nr_samples != "all" else "all"

    file_name = f"srcnn_model_{args.model}_24h_batchsize{batch_size}_samples{nr_samples}_normalized.pth"
    model_path = os.path.join(output_dir, file_name)
    metrics_file = f"{metrics_dir}/{file_name.replace('.pth', '.csv')}"

    # Initialize model, loss function, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    logger.info(f"cuda available: {torch.cuda.is_available()}")
    logger.info(f"torch version {torch.__version__}")

    model_class = MODEL_MAP.get(args.model)
    if model_class is None:
        raise ValueError(f"Invalid model {args.model}.")

    model = model_class().to(device)

    criterion = nn.MSELoss()

    if args.model in ["L4"]:  
        optimizer = optim.Adam([
            {'params': model.conv1.parameters(), 'lr': 1e-4},  
            {'params': model.conv2.parameters(), 'lr': 1e-4},  
            {'params': model.conv3.parameters(), 'lr': 1e-4},  
            {'params': model.conv4.parameters(), 'lr': 1e-5}   
        ])
    elif args.model == 5:
        optimizer = optim.Adam([
            {'params': model.conv1.parameters(), 'lr': 1e-4},  
            {'params': model.conv2.parameters(), 'lr': 1e-4},  
            {'params': model.conv3.parameters(), 'lr': 1e-4},
            {'params': model.conv4.parameters(), 'lr': 1e-4},  
            {'params': model.conv5.parameters(), 'lr': 1e-5}   
        ])
    else:
        optimizer = optim.Adam([
            {'params': model.conv1.parameters(), 'lr': 1e-4},  
            {'params': model.conv2.parameters(), 'lr': 1e-4},  
            {'params': model.conv3.parameters(), 'lr': 1e-5}   
        ])

    scaler = torch.amp.GradScaler('cuda')  # Automatic mixed precision scaler

    # Load dataset
    train_paths, val_paths, test_paths = get_split_datapaths()
    normalize = True
    if nr_samples == "all":
        train_dataset = PassiveMicrowaveDataset(train_paths, transform=ToTensor(), normalize=normalize, use_bicubic=True)
        val_dataset = PassiveMicrowaveDataset(val_paths, transform=ToTensor(), normalize=normalize, use_bicubic=True)
    else:
        nr_samples = int(nr_samples)
        val_samples = nr_samples // 4
        train_dataset = PassiveMicrowaveDataset(train_paths[:nr_samples], transform=ToTensor(), normalize=normalize, use_bicubic=True)
        val_dataset = PassiveMicrowaveDataset(val_paths[:val_samples], transform=ToTensor(), normalize=normalize, use_bicubic=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Time Tracking
    start_time = time.time()
    max_duration = 24 * 3600  # 24 hours in seconds
    patience = 20  # Stop if no improvement for 10 epochs
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    logger.info(f"Training SRCNN Model {args.model} for a max of 24 hours with early stopping")
    logger.info(f"Nr of training samples: {nr_samples} (Normalized: {normalize})")
    logger.info(f"Batch size: {batch_size}")

    with open(metrics_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "max_memory_GiB", "elapsed_time_sec", "train_loss", "val_loss"]) 

        epoch = 0
        while True:
            epoch += 1
            model.train()
            train_loss = 0

            for low_res, high_res in train_loader:
                low_res, high_res = low_res.to(device), high_res.to(device)
                optimizer.zero_grad()
                with torch.amp.autocast('cuda'):  # Enable AMP
                    outputs = model(low_res)
                    loss = criterion(outputs, high_res)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                train_loss += loss.item()

            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for low_res, high_res in val_loader:
                    low_res, high_res = low_res.to(device), high_res.to(device)
                    
                    with torch.amp.autocast('cuda'):
                        outputs = model(low_res)
                        loss = criterion(outputs, high_res)
                    
                    val_loss += loss.item()

            train_loss_avg = train_loss / len(train_loader)
            val_loss_avg = val_loss / len(val_loader)

            # max memory usage so far
            max_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)  # Convert bytes to GB

            #time since start of training
            elapsed_time = time.time() - start_time

            # Log results
            logger.info(f"Epoch {epoch} - Train Loss: {train_loss_avg:.6f}, Val Loss: {val_loss_avg:.6f}")
            writer.writerow([epoch, max_memory, elapsed_time, train_loss_avg, val_loss_avg])

            # Check early stopping condition
            if val_loss_avg < best_val_loss:
                best_val_loss = val_loss_avg
                epochs_without_improvement = 0
                torch.save(model.state_dict(), model_path)  # Save best model
                logger.info(f"New best model saved at epoch {epoch} with val_loss {val_loss_avg:.6f}")
            else:
                epochs_without_improvement += 1
                logger.info(f"No improvement for {epochs_without_improvement} epochs")

            # Stop if no improvement for `patience` epochs
            if epochs_without_improvement >= patience:
                logger.info(f"Early stopping triggered after {epoch} epochs")
                break

            # Stop if 24 hours have passed
            elapsed_time = time.time() - start_time
            if elapsed_time > max_duration:
                logger.info(f"Time limit reached ({elapsed_time / 3600:.2f} hours). Stopping training.")
                break

    logger.info(f"Training completed after {epoch} epochs. Best model saved at {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to train the SRCNN.")

    parser.add_argument("--model", type=str, required=True, choices=["default", "L4", "W128", "W256","K3", "K5"], help="Choose model: default, K3, K5.")
    parser.add_argument("--batch_size", type=int, required=True, help="Batch size.")
    parser.add_argument("--nr_samples", type=str, default="all", help='Number of samples ("all" or an integer).')

    args = parser.parse_args()
    main(args)
