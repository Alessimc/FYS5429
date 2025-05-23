import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import os
import sys
import time
import csv
import argparse

# Custom imports
from vae_gan_model import REDNetVAEGenerator, SRGANDiscriminator
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "lib")))
from utils import init_logging, get_split_datapaths
from dataloader import PassiveMicrowaveDataset

logger = init_logging()

def train_gan(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # === Paths ===
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trained_models")
    os.makedirs(output_dir, exist_ok=True)
    metrics_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_train_val_metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    file_name = f"REDNetVAEGAN_batch{args.batch_size}_samples{args.nr_samples}.pth"
    gen_model_path = os.path.join(output_dir, "gen_" + file_name)
    disc_model_path = os.path.join(output_dir, "disc_" + file_name)
    metrics_file = os.path.join(metrics_dir, file_name.replace('.pth', '.csv'))

    # === Models ===
    generator = REDNetVAEGenerator(in_channels=2, out_channels=2).to(device)
    discriminator = SRGANDiscriminator(in_channels=2).to(device)

    optimizer_G = optim.Adam(generator.parameters(), lr=1e-4)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=1e-5)
    criterion_adv = nn.BCEWithLogitsLoss()
    criterion_recon = nn.MSELoss()

    # === Dataset ===
    train_paths, val_paths, _ = get_split_datapaths()
    normalize = True

    if args.nr_samples == "all":
        train_dataset = PassiveMicrowaveDataset(train_paths, transform=ToTensor(), normalize=normalize)
        val_dataset = PassiveMicrowaveDataset(val_paths, transform=ToTensor(), normalize=normalize)
    else:
        n = int(args.nr_samples)
        val_samples = n // 4
        train_dataset = PassiveMicrowaveDataset(train_paths[:n], transform=ToTensor(), normalize=normalize)
        val_dataset = PassiveMicrowaveDataset(val_paths[:val_samples], transform=ToTensor(), normalize=normalize)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    start_time = time.time()
    max_duration = 24 * 3600
    patience = 20
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    logger.info("Starting REDNetVAEGAN training...")

    with open(metrics_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "max_memory_GiB", "elapsed_time_sec", "train_loss", "val_loss"])

        epoch = 0
        while True:
            epoch += 1
            generator.train()
            discriminator.train()
            train_loss = 0

            for low_res, high_res in train_loader:
                low_res, high_res = low_res.to(device), high_res.to(device)

                # === Train Discriminator ===
                optimizer_D.zero_grad()

                recon, _, mu, log_var = generator(low_res)
                pred_real = discriminator(high_res)
                pred_fake = discriminator(recon.detach())
                valid = torch.ones_like(pred_real) * 0.9
                fake_lbl = torch.zeros_like(pred_fake)

                loss_D_real = criterion_adv(pred_real, valid)
                loss_D_fake = criterion_adv(pred_fake, fake_lbl)
                loss_D = (loss_D_real + loss_D_fake) / 2

                loss_D.backward()
                
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)

                optimizer_D.step()

                # === Train Generator (VAE-GAN) ===
                optimizer_G.zero_grad()

                recon, _, mu, log_var = generator(low_res)
                pred_fake = discriminator(recon)

                vae_losses = generator.loss_function(recon, low_res, mu, log_var, target=high_res)
                loss_G_vae = vae_losses['loss']
                loss_G_adv = criterion_adv(pred_fake, valid)
                adv_weight = min(0.1, epoch * 0.01)
                loss_G = loss_G_vae + adv_weight * loss_G_adv 

                loss_G.backward()

                torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)

                optimizer_G.step()

                train_loss += loss_G.item()


            # === Validation ===
            generator.eval()
            val_loss = 0
            with torch.no_grad():
                for low_res, high_res in val_loader:
                    low_res, high_res = low_res.to(device), high_res.to(device)
                    recon, _, _, _ = generator(low_res)
                    loss = criterion_recon(recon, high_res)
                    val_loss += loss.item()

            train_avg = train_loss / len(train_loader)
            val_avg = val_loss / len(val_loader)
            elapsed = time.time() - start_time
            max_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)

            logger.info(f"Epoch {epoch} | Train Loss: {train_avg:.6f} | Val Loss: {val_avg:.6f}")
            writer.writerow([epoch, max_mem, elapsed, train_avg, val_avg])
            f.flush()

            if val_avg < best_val_loss:
                best_val_loss = val_avg
                epochs_without_improvement = 0
                torch.save(generator.state_dict(), gen_model_path)
                torch.save(discriminator.state_dict(), disc_model_path)
                logger.info(f"Saved best models at epoch {epoch}")
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                logger.info("Early stopping: no improvement")
                break
            if elapsed > max_duration:
                logger.info("Training stopped: 24h limit reached")
                break

    logger.info(f"Training complete. Generator saved to {gen_model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train REDNetVAEGAN")
    parser.add_argument("--batch_size", type=int, required=True, help="Batch size")
    parser.add_argument("--nr_samples", type=str, default="all", help='Number of samples ("all" or an integer)')
    args = parser.parse_args()
    train_gan(args)
