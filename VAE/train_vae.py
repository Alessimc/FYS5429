import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import os
import sys
import time
import csv
import argparse

# Custom imports
from vae_model import VAE, REDNetVAE
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "lib")))
from utils import init_logging, get_split_datapaths
from dataloader import PassiveMicrowaveDataset

logger = init_logging()

def main(args):
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trained_models")
    os.makedirs(output_dir, exist_ok=True)
    metrics_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_train_val_metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    file_name = f"REDNetVAE_24h_batch{args.batch_size}_samples{args.nr_samples}.pth"
    model_path = os.path.join(output_dir, file_name)
    metrics_file = os.path.join(metrics_dir, file_name.replace('.pth', '.csv'))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    model = REDNetVAE(in_channels=2, latent_dim=128).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler(enabled=(device.type == 'cuda'))

    # === Dataset ===
    train_paths, val_paths, _ = get_split_datapaths()
    normalize = True

    if args.nr_samples == "all":
        train_dataset = PassiveMicrowaveDataset(train_paths, transform=ToTensor(), normalize=normalize, use_bicubic=False)
        val_dataset = PassiveMicrowaveDataset(val_paths, transform=ToTensor(), normalize=normalize, use_bicubic=False)
    else:
        n = int(args.nr_samples)
        val_samples = n // 4
        train_dataset = PassiveMicrowaveDataset(train_paths[:n], transform=ToTensor(), normalize=normalize, use_bicubic=False)
        val_dataset = PassiveMicrowaveDataset(val_paths[:val_samples], transform=ToTensor(), normalize=normalize, use_bicubic=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    start_time = time.time()
    max_duration = 24 * 3600
    patience = 20
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    logger.info("Starting REDNetVAE training...")

    with open(metrics_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "max_memory_GiB", "elapsed_time_sec", "train_loss", "val_loss", "val_KLD", "val_recon"])

        epoch = 0
        while True:
            epoch += 1
            model.train()
            train_loss = 0

            for low_res, high_res in train_loader:
                low_res, high_res = low_res.to(device), high_res.to(device)
                optimizer.zero_grad()
                with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
                    output = model(low_res)
                    loss_dict = model.loss_function(*output, target=high_res, M_N=args.batch_size / len(train_dataset))
                    loss = loss_dict['loss']

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                train_loss += loss.item()

            model.eval()
            val_loss = 0
            val_kl = 0
            val_recon = 0
            with torch.no_grad():
                for low_res, high_res in val_loader:
                    low_res = low_res.to(device)
                    high_res = high_res.to(device)
                    with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
                        output = model(low_res)

                        kl_anneal_epochs = 10  # Gradually increase KL weight over first 10 epochs
                        kl_weight = min(1.0, epoch / kl_anneal_epochs)
                        loss_dict = model.loss_function(*output, target=high_res, M_N=args.batch_size / len(val_dataset), kl_weight=kl_weight)
                        loss = loss_dict['loss']
                        kl_div = loss_dict['KLD']
                        recon = loss_dict['Reconstruction_Loss'] 

                    val_loss += loss.item()
                    val_kl += kl_div.item()
                    val_recon += recon.item()

            train_avg = train_loss / len(train_loader)
            val_avg = val_loss / len(val_loader)
            val_kl_avg = val_kl / len(val_loader)
            val_recon_avg = val_recon / len(val_loader)
            elapsed = time.time() - start_time
            max_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)

            logger.info(f"Epoch {epoch} | Train Loss: {train_avg:.6f} | Val Loss: {val_avg:.6f}")
            writer.writerow([epoch, max_mem, elapsed, train_avg, val_avg, val_kl_avg, val_recon_avg])
            f.flush()
            
            if val_avg < best_val_loss:
                best_val_loss = val_avg
                epochs_without_improvement = 0
                torch.save(model.state_dict(), model_path)
                logger.info(f"Saved best model at epoch {epoch}")
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                logger.info("Early stopping: no improvement")
                break
            if elapsed > max_duration:
                logger.info("Training stopped: 24h limit reached")
                break

    logger.info(f"Training complete. Best model saved at {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train REDNetVAE")
    parser.add_argument("--batch_size", type=int, required=True, help="Batch size")
    parser.add_argument("--nr_samples", type=str, default="all", help='Number of samples ("all" or an integer)')
    args = parser.parse_args()
    main(args)
