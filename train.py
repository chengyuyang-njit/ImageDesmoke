# main script to start training
# token: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

import argparse
import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import transforms

import matplotlib.pyplot as plt
import torch.optim as optim
from utils.util import _save_checkpoint

from base.base_model import UNet
from dataloader import dataloaders

import json

def load_data():
    # load the dataset
    print("[INFO] loading the paired desmoke image dataset...")

    dataset = dataloaders.PairedSmokeImageDataset(
        csv_file = config['dataloader']['args']['csv_dir'],
        root_dir = config['dataloader']['args']['data_dir'],
        transform =  transforms.Compose([transforms.ToTensor()]))

    num_train_samples = int(len(dataset) * 
                            config['dataloader']['args']['train_split']) + 1
    num_val_samples = int(len(dataset) * 
                          config['dataloader']['args']['validation_split'])
    num_test_samples = int(len(dataset) * 
                           (1 - config['dataloader']['args']['train_split'] - 
                            config['dataloader']['args']['validation_split']))

    (train_data, val_data, test_data) = random_split(
        dataset, [num_train_samples, num_val_samples, num_test_samples],
                                                    generator=torch.Generator().manual_seed(42))

    print("[INFO] paired desmoke image dataset loaded...")
    return {"train":train_data, "val":val_data, "test": test_data}


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("cuda available:", torch.cuda.is_available())

    model = UNet(in_channels=3, out_channels=3).to(device)
    if config['loss'] == "MSELoss":
        criterion = torch.nn.MSELoss()
    if config['optimizer']['type'] == "Adam":
        optimizer = optim.Adam(
            model.parameters(), lr = config['optimizer']['args']['lr'], 
                           amsgrad = config['optimizer']['args']['amsgrad'])
    if config['lr_scheduler']['used']:
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size = config['lr_scheduler']['args']['step_size'],
                                              gamma = config['lr_scheduler']['args']['gamma'])
    num_epochs = config['epochs']

    data = load_data()
    train_loader = DataLoader(
        data["train"], batch_size = config['dataloader']['args']['batch_size'], 
                              shuffle = config['dataloader']['args']['shuffle'])
    total_loss = 0
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

            if config['lr_scheduler']['used']:
                scheduler.step()

            epoch_loss += loss.item()
            # print(loss)
        total_loss += epoch_loss
        print(f"Epoch [{epoch + 1}/{num_epochs}], loss: {epoch_loss / len(train_loader):.4f}")

    return model, optimizer, total_loss


# Example usage:
if __name__ == "__main__":
    args = argparse.ArgumentParser(description="ImageDesmoke")
    args.add_argument('-c', '--CONFIG', default = None, type = str, required = True,
                      help = 'config file path (default : None)')
    args.add_argument('-r', '--RESUME', default = None, type = str,
                      help = 'path to latest checkpoint (default : None)')
    args.add_argument('-d', '--DEVICE', default = None, type = str,
                      help = "indices of GPUs to enable (default : all)")

    args = args.parse_args()


    # x = torch.randn((1, 3, 700, 350))  # Example input tensor
    # preds = model(x)
    # print(preds.shape)  # Should output torch.Size([1, 3, 256, 256])
    # model = model.cuda()
    # summary(model, (3,700,350))

    checkpoint_path = "./saved/models"
    with open(args.CONFIG, 'r') as f:
        config = json.load(f)
    # print(config['dataloader']['args']['batch_size'])
    trained_model, optimizer_used, total_loss = train()
    _save_checkpoint(trained_model, optimizer_used, total_loss, config, checkpoint_path)
    


