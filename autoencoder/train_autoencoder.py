import torch.optim as optim
import torch.nn as nn
from autoencoder_model import REDNet10, REDNet10_W256, REDNet20, REDNet30
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import os
import sys
import time
import csv
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "lib")))
from utils import init_logging, get_split_datapaths
from dataloader import PassiveMicrowaveDataset
import argparse

logger = init_logging()

MODEL_MAP = {
    10: REDNet10,
    256: REDNet10_W256,
    20: REDNet20,
    30: REDNet30
}

def main(args):
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trained_models")
    os.makedirs(output_dir, exist_ok=True)
    metrics_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_train_val_metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    batch_size = args.batch_size
    nr_samples = args.nr_samples if args.nr_samples != "all" else "all"
    masked_loss = args.masked_loss

    add_masked_loss = ""
    if masked_loss:
        add_masked_loss = "_masked_loss"

    file_name = f"REDNet{args.model}_24h_batchsize{batch_size}_samples{nr_samples}_normalized{add_masked_loss}.pth"
    model_path = os.path.join(output_dir, file_name)
    metrics_file = os.path.join(metrics_dir, file_name.replace('.pth', '.csv'))

    # Initialize model, loss function, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    logger.info(f"cuda available: {torch.cuda.is_available()}")
    logger.info(f"torch version {torch.__version__}")

    model_class = MODEL_MAP.get(args.model)
    if model_class is None:
        raise ValueError(f"Invalid model number {args.model}. Choose from 10, 20, or 30.")
    
    model = model_class().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    scaler = torch.amp.GradScaler('cuda')  # Automatic mixed precision scaler

    # Load dataset paths
    train_paths, val_paths, _ = get_split_datapaths()
    normalize = True

    if nr_samples == "all":
        train_dataset = PassiveMicrowaveDataset(train_paths, transform=ToTensor(), normalize=normalize, use_bicubic=False)
        val_dataset = PassiveMicrowaveDataset(val_paths, transform=ToTensor(), normalize=normalize, use_bicubic=False)
    else:
        nr_samples = int(nr_samples)
        val_samples = nr_samples // 4
        train_dataset = PassiveMicrowaveDataset(train_paths[:nr_samples], transform=ToTensor(), normalize=normalize, use_bicubic=False)
        val_dataset = PassiveMicrowaveDataset(val_paths[:val_samples], transform=ToTensor(), normalize=normalize, use_bicubic=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Time Tracking & Early Stopping
    start_time = time.time()
    max_duration = 24 * 3600  # 24 hours in seconds
    patience = 20  # Stop if no improvement for 10 epochs
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    logger.info(f"Training REDNet{args.model} for a max of 24 hours with early stopping")
    logger.info(f"Nr of training samples: {nr_samples} (Normalized: {normalize})")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Masked loss: {masked_loss}")


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
                    if masked_loss:
                        mask = (high_res > -1)
                        loss = criterion(outputs[mask], high_res[mask])
                    else:
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
                        if masked_loss:
                            mask = (high_res > -1)
                            loss = criterion(outputs[mask], high_res[mask])
                        else:
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
    parser = argparse.ArgumentParser(description="Script to train REDNet Autoencoder.")
    parser.add_argument("--model", type=int, required=True, choices=[10, 256, 20, 30], help="Choose model: 10, 256, 20, or 30.")
    parser.add_argument("--batch_size", type=int, required=True, help="Batch size.")
    parser.add_argument("--nr_samples", type=str, default="all", help='Number of samples ("all" or an integer).')
    parser.add_argument("--masked_loss", type=bool, default=False, help='Mask out nan regions during loss calculation.')

    args = parser.parse_args()
    main(args)
