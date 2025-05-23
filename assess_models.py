from lib.utils import assess_model, init_logging, get_split_datapaths
from lib.dataloader import PassiveMicrowaveDataset
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from autoencoder.autoencoder_model import REDNet10, REDNet20, REDNet30, REDNet10_W256
from srcnn.srcnn_model import SRCNN, SRCNN_W256 # note need new names here!!!
from transformer.transformer_model import ELAN
from transformer.swinir_model import SwinIR
from VAE.vae_model import REDNetVAE
from GAN.gan_model import REDNetGenerator
from GAN.vae_gan_model import REDNetVAEGenerator


train_paths, val_paths, test_paths = get_split_datapaths()

# Initialize test dataset with normalization
test_dataset = PassiveMicrowaveDataset(test_paths, transform=ToTensor(), normalize=True, use_bicubic=False)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)


# assess_model(model_pth = "srcnn/trained_models/srcnn_model_W256_24h_batchsize8_samplesall_normalized.pth",
#              model = SRCNN_W256(),
#              test_loader = test_loader,
#              test_dataset = test_dataset,
#              mask_land = True,
#              denormalize = True,
#              outfile_path = "assess_models.csv",
#              model_name = "SRCNN_W256")

### MASKED LOSS
# assess_model(model_pth = "autoencoder/trained_models/REDNet10_24h_batchsize8_samplesall_normalized_masked_loss.pth",
#              model = REDNet10(),
#              test_loader = test_loader,
#              test_dataset = test_dataset,
#              mask_land = True,
#              denormalize = True,
#              outfile_path = "assess_models.csv",
#              model_name = "REDNet10_masked_loss")

# assess_model(model_pth = "autoencoder/trained_models/REDNet20_24h_batchsize8_samplesall_normalized_masked_loss.pth",
#              model = REDNet20(),
#              test_loader = test_loader,
#              test_dataset = test_dataset,
#              mask_land = True,
#              denormalize = True,
#              outfile_path = "assess_models.csv",
#              model_name = "REDNet20_masked_loss")


# assess_model(model_pth = "autoencoder/trained_models/REDNet30_24h_batchsize8_samplesall_normalized_masked_loss.pth",
#              model = REDNet30(),
#              test_loader = test_loader,
#              test_dataset = test_dataset,
#              mask_land = True,
#              denormalize = True,
#              outfile_path = "assess_models.csv",
#              model_name = "REDNet30_masked_loss")

# assess_model(model_pth = "autoencoder/trained_models/REDNet256_24h_batchsize8_samplesall_normalized_masked_loss.pth",
#              model = REDNet10_W256(),
#              test_loader = test_loader,
#              test_dataset = test_dataset,
#              mask_land = True,
#              denormalize = True,
#              outfile_path = "assess_models.csv",
#              model_name = "REDNet256_masked_loss")

### NOT MASKED LOSS
# assess_model(model_pth = "autoencoder/trained_models/REDNet10_24h_batchsize8_samplesall_normalized.pth",
#              model = REDNet10(),
#              test_loader = test_loader,
#              test_dataset = test_dataset,
#              mask_land = True,
#              denormalize = True,
#              outfile_path = "assess_models.csv",
#              model_name = "REDNet10")


# assess_model(model_pth = "autoencoder/trained_models/REDNet20_24h_batchsize8_samplesall_normalized.pth",
#              model = REDNet20(),
#              test_loader = test_loader,
#              test_dataset = test_dataset,
#              mask_land = True,
#              denormalize = True,
#              outfile_path = "assess_models.csv",
#              model_name = "REDNet20")

# assess_model(model_pth = "autoencoder/trained_models/REDNet30_24h_batchsize8_samplesall_normalized.pth",
#              model = REDNet30(),
#              test_loader = test_loader,
#              test_dataset = test_dataset,
#              mask_land = True,
#              denormalize = True,
#              outfile_path = "assess_models.csv",
#              model_name = "REDNet30")

# assess_model(model_pth = "autoencoder/trained_models/REDNet256_24h_batchsize8_samplesall_normalized.pth",
#              model = REDNet10_W256(),
#              test_loader = test_loader,
#              test_dataset = test_dataset,
#              mask_land = True,
#              denormalize = True,
#              outfile_path = "assess_models.csv",
#              model_name = "REDNet256")

# assess_model(model_pth = "transformer/trained_models/ELAN_light1_24h_batchsize8_samplesall_normalized.pth",
#              model = ELAN(),
#              test_loader = test_loader,
#              test_dataset = test_dataset,
#              mask_land = True,
#              denormalize = True,
#              outfile_path = "assess_models.csv",
#              model_name = "ELAN")

# assess_model(model_pth = "GAN/trained_models/gen_REDNetGAN_batch8_samplesall.pth",
#              model = REDNetGenerator(),
#              test_loader = test_loader,
#              test_dataset = test_dataset,
#              mask_land = False,
#              denormalize = True,
#              outfile_path = "assess_models.csv",
#              model_name = "REDNetGAN")

# assess_model(model_pth = "GAN/trained_models/gen_REDNetVAEGAN_batch8_samplesall.pth",
#              model = REDNetVAEGenerator(),
#              test_loader = test_loader,
#              test_dataset = test_dataset,
#              mask_land = False,
#              denormalize = True,
#              outfile_path = "assess_models.csv",
#              model_name = "REDNetVAE-GAN")


# assess_model(model_pth = "VAE/trained_models/REDNetVAE_24h_batch8_samplesall.pth",
#              model = REDNetVAE(),
#              test_loader = test_loader,
#              test_dataset = test_dataset,
#              mask_land = True,
#              denormalize = True,
#              outfile_path = "assess_models.csv",
#              model_name = "REDNetVAE")

# assess_model(model_pth = "transformer/trained_models/SwinIR_24h_batchsize8_samplesall_normalized.pth",
#              model = SwinIR( img_size=416,
#                         in_chans=2,
#                         upscale=1,
#                         window_size=4, 
#                         upsampler=None,  
#                         img_range=1.0,
#                         embed_dim=24,
#                         num_heads=[4, 4, 4, 4]
#                         ),
#              test_loader = test_loader,
#              test_dataset = test_dataset,
#              mask_land = True,
#              denormalize = True,
#              outfile_path = "assess_models.csv",
#              model_name = "SwinIR")