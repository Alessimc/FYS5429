from lib.utils import assess_model, init_logging, get_split_datapaths
from lib.dataloader import PassiveMicrowaveDataset
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from autoencoder.autoencoder_model import REDNet10, REDNet20, REDNet30
from srcnn.srcnn_model import SRCNN, SRCNN2, SRCNN3, SRCNN6


train_paths, val_paths, test_paths = get_split_datapaths()

# Initialize test dataset with normalization
test_dataset = PassiveMicrowaveDataset(test_paths, transform=ToTensor(), normalize=True, use_bicubic=False)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)


# assess_model(model_pth = "srcnn/trained_models/srcnn_model2_24h_batchsize8_samplesall_normalized.pth",
#              model = SRCNN2(),
#              test_loader = test_loader,
#              test_dataset = test_dataset,
#              mask_land = False,
#              denormalize = True,
#              outfile_path = "assess_models.csv",
#              model_name = "SRCNN2_unmasked")

assess_model(model_pth = "srcnn/trained_models/srcnn_model6_24h_batchsize8_samplesall_normalized.pth",
             model = SRCNN6(),
             test_loader = test_loader,
             test_dataset = test_dataset,
             mask_land = False,
             denormalize = True,
             outfile_path = "assess_models.csv",
             model_name = "SRCNN6_unmasked")


# assess_model(model_pth = "srcnn/trained_models/srcnn_model2_24h_batchsize8_samplesall_normalized.pth",
#              model = SRCNN2(),
#              test_loader = test_loader,
#              test_dataset = test_dataset,
#              mask_land = True,
#              denormalize = True,
#              outfile_path = "assess_models.csv",
#              model_name = "SRCNN2")

# assess_model(model_pth = "autoencoder/trained_models/REDNet10_24h_batchsize8_samplesall_normalized_masked_loss.pth",
#              model = REDNet10(),
#              test_loader = test_loader,
#              test_dataset = test_dataset,
#              mask_land = True,
#              denormalize = True,
#              outfile_path = "assess_models.csv",
#              model_name = "REDNet10_masked_loss")

# assess_model(model_pth = "autoencoder/trained_models/REDNet10_24h_batchsize8_samplesall_normalized.pth",
#              model = REDNet10(),
#              test_loader = test_loader,
#              test_dataset = test_dataset,
#              mask_land = True,
#              denormalize = True,
#              outfile_path = "assess_models.csv",
#              model_name = "REDNet10")

# assess_model(model_pth = "autoencoder/trained_models/REDNet20_24h_batchsize8_samplesall_normalized_masked_loss.pth",
#              model = REDNet20(),
#              test_loader = test_loader,
#              test_dataset = test_dataset,
#              mask_land = True,
#              denormalize = True,
#              outfile_path = "assess_models.csv",
#              model_name = "REDNet20_masked_loss")

# assess_model(model_pth = "autoencoder/trained_models/REDNet20_24h_batchsize8_samplesall_normalized.pth",
#              model = REDNet20(),
#              test_loader = test_loader,
#              test_dataset = test_dataset,
#              mask_land = True,
#              denormalize = True,
#              outfile_path = "assess_models.csv",
#              model_name = "REDNet20")

# assess_model(model_pth = "autoencoder/trained_models/REDNet30_24h_batchsize8_samplesall_normalized_masked_loss.pth",
#              model = REDNet30(),
#              test_loader = test_loader,
#              test_dataset = test_dataset,
#              mask_land = True,
#              denormalize = True,
#              outfile_path = "assess_models.csv",
#              model_name = "REDNet30_masked_loss")

# assess_model(model_pth = "autoencoder/trained_models/REDNet30_24h_batchsize8_samplesall_normalized.pth",
#              model = REDNet30(),
#              test_loader = test_loader,
#              test_dataset = test_dataset,
#              mask_land = True,
#              denormalize = True,
#              outfile_path = "assess_models.csv",
#              model_name = "REDNet30")

