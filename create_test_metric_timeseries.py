from lib.utils import create_metric_timeseries, init_logging, get_split_datapaths
from lib.dataloader import PassiveMicrowaveDataset
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from autoencoder.autoencoder_model import REDNet10, REDNet20, REDNet30, REDNet10_W256
from VAE.vae_model import REDNetVAE
from transformer.swinir_model import SwinIR
from srcnn.srcnn_model import SRCNN, SRCNN_W256
from transformer.transformer_model import ELAN


train_paths, val_paths, test_paths = get_split_datapaths()

# Initialize test dataset with normalization
test_dataset = PassiveMicrowaveDataset(test_paths, transform=ToTensor(), normalize=True, use_bicubic=False)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# create_metric_timeseries(model_pth = "autoencoder/trained_models/REDNet256_24h_batchsize8_samplesall_normalized.pth",
#                         model = REDNet10_W256(),
#                         test_paths = test_paths,
#                         test_dataset = test_dataset)

# create_metric_timeseries(model_pth = "srcnn/trained_models/srcnn_model_W256_24h_batchsize8_samplesall_normalized.pth",
#                         model = SRCNN_W256(),
#                         test_paths = test_paths,
#                         test_dataset = test_dataset)

# create_metric_timeseries(model_pth = "transformer/trained_models/ELAN_light1_24h_batchsize8_samplesall_normalized copy.pth",
#                         model = ELAN(),
#                         test_paths = test_paths,
#                         test_dataset = test_dataset)

create_metric_timeseries(model_pth = "VAE/trained_models/REDNetVAE_24h_batch8_samplesall.pth",
                        model = REDNetVAE(),
                        test_paths = test_paths,
                        test_dataset = test_dataset)

create_metric_timeseries(model_pth = "transformer/trained_models/SwinIR_24h_batchsize8_samplesall_normalized.pth",
                        model = SwinIR( img_size=416,
                        in_chans=2,
                        upscale=1,
                        window_size=4, 
                        upsampler=None,  
                        img_range=1.0,
                        embed_dim=24,
                        num_heads=[4, 4, 4, 4]
                        ),
                        test_paths = test_paths,
                        test_dataset = test_dataset)