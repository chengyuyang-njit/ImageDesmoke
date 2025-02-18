import argparse
import pandas as pd
import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
from torchvision import transforms

from PIL import Image
import matplotlib.pyplot as plt

from torchvision import models
from torchsummary import summary

import torch.optim as optim

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        
        # Encoder (Downsampling)
        for feature in features:
            self.encoder.append(self._conv_block(in_channels, feature))
            in_channels = feature
        
        # Bottleneck
        self.bottleneck = self._conv_block(features[-1], features[-1] * 2)
        
        # Decoder (Upsampling)
        for feature in reversed(features):
            self.decoder.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.decoder.append(self._conv_block(feature * 2, feature))
        
        # Final Convolution
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
    
    def forward(self, x):
        skip_connections = []
        
        # Encoding
        for layer in self.encoder:
            x = layer(x)
            skip_connections.append(x)
            x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoding
        skip_connections = skip_connections[::-1]
        for idx in range(0, len(self.decoder), 2):
            x = self.decoder[idx](x)
            skip_connection = skip_connections[idx // 2]
            
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode='bilinear', align_corners=True)
            
            x = torch.cat((skip_connection, x), dim=1)
            x = self.decoder[idx + 1](x)
        
        return self.final_conv(x)
    
    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

class PairedSmokeImageDataset(Dataset):

    def __init__(self, csv_file, root_dir, transform = None):
        # data loading
        self.paired_images = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __getitem__(self,idx):
        # dataset[0]
        if torch.is_tensor(idx):
            idx = idx.tolist()

        clear_image_name = os.path.join(self.root_dir,
                                    self.paired_images.iloc[idx, 1])
        smoked_image_name = os.path.join(self.root_dir,
                                    self.paired_images.iloc[idx, 0])
        
        clear_image = Image.open(clear_image_name).convert("RGB")
        smoked_image = Image.open(smoked_image_name).convert("RGB")

        if self.transform:
            clear_image = self.transform(clear_image)
            smoked_image = self.transform(smoked_image)

        sample = {'clear_image': clear_image, 'smoked_image': smoked_image}

        return sample

    def __len__(self):
        # len(dataset)
        return len(self.paired_images)

def load_data():
    # load the dataset
    print("[INFO] loading the paired desmoke image dataset...")

    dataset = PairedSmokeImageDataset(
        csv_file = '..\\..\\datasets\\DesmokeData-paired\\DesmokeData-main\\images\\paired_images.csv',
        root_dir = '..\\..\\datasets\\DesmokeData-paired\\DesmokeData-main\\images\\dataset',
        transform =  transforms.Compose([transforms.ToTensor()]))

    num_train_samples = int(len(dataset) * args.TRAIN_SPLIT) + 1
    num_val_samples = int(len(dataset) * (1 - args.TRAIN_SPLIT - args.TEST_SPLIT))
    num_test_samples = int(len(dataset) * args.TEST_SPLIT)

    (train_data, val_data, test_data) = random_split(dataset, [num_train_samples, num_val_samples, num_test_samples],
                                                    generator=torch.Generator().manual_seed(42))

    print("[INFO] paired desmoke image dataset loaded...")
    return {"train":train_data, "val":val_data, "test": test_data}

def arg_parse():

    ap = argparse.ArgumentParser()
    ap.add_argument("-trs", "--TRAIN_SPLIT", type = float, required = True,
                    help = "percentage of training samples")
    ap.add_argument("-ts", "--TEST_SPLIT", type = float, required = True,
                    help = "percentage of testing samples")
    ap.add_argument("-lr", "--INIT_LR", type = float, required = True,
                    help = "initial learning rate for training the model")
    ap.add_argument("-bs", "--BATCH_SIZE", type = int, required = True,
                    help = "batch size for the training")
    ap.add_argument("-epcs", "--EPOCHS", type = int, required = True,
                    help = "number of epochs to train the model")

    # ap.add_argument("-m", "--model", type = str, required = True,
    #                 help = "path to output trained model")
    # ap.add_argument("-p", "--plot", type = str, required = True,
    #                 help = "path to output loss/accuracy plot")
    return ap.parse_args()

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("cuda available:", torch.cuda.is_available())

    model = UNet(in_channels=3, out_channels=3).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr = 1e-3)

    num_epochs = args.EPOCHS

    data = load_data()
    train_loader = DataLoader(data["train"], batch_size = args.BATCH_SIZE, shuffle = True)
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        for samples in train_loader:
            input = samples["smoked_image"].to(device)
            target = samples["clear_image"].to(device)

            optimizer.zero_grad()
            output = model(input)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            # print(loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}], loss: {epoch_loss / len(train_loader):.4f}")

    return model


args = arg_parse()
# Load model later with:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(in_channels=3, out_channels=3).to(device)
model.load_state_dict(torch.load("unet_model.pth"))
model.eval()

data = load_data()
eval_loader = DataLoader(data["val"], batch_size = args.BATCH_SIZE, shuffle = True)
# with torch.no_grad():
#     for samples in eval_loader:
#         input = samples["smoked_image"].to(device)
#         target = samples["clear_image"].to(device)
#         output = model(input)

def visualize_sample(inputs, targets, outputs):
    inputs = inputs.cpu().numpy().transpose(1, 2, 0)
    targets = targets.cpu().numpy().transpose(1, 2, 0)
    outputs = outputs.cpu().numpy().transpose(1, 2, 0)

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(inputs)
    ax[0].set_title("Input")
    ax[1].imshow(targets)
    ax[1].set_title("Target")
    ax[2].imshow(outputs)
    ax[2].set_title("Output")
    plt.show()

# Get a sample batch
samples = next(iter(eval_loader))
input = samples["smoked_image"].to(device)
target = samples["clear_image"].to(device)
with torch.no_grad():
    output = model(input)

# Visualize
visualize_sample(input[0], target[0], output[0])