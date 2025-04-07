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
from torch.autograd import Variable

from base.base_model import UNet
from model.model import UNetWithWiener

from dataloader import dataloaders

import numpy as np

from model.loss import SSIMLoss
from model.loss import PerceptualLoss

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

    if config['arch']['type'] == "UNetWithWiener":
        model = UNetWithWiener(in_channels=3, out_channels=3).to(device)
    else:
        model = UNet(in_channels=3, out_channels=3).to(device)


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
    
    loss_mse = torch.nn.MSELoss()
    loss_ssim = SSIMLoss()
    loss_perceptual = PerceptualLoss()



    total_loss = 0
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        epoch_mse_loss = 0
        epoch_ssim_loss = 0
        epoch_perc_loss = 0

        for samples in train_loader:
            input = samples["smoked_image"].to(device)
            target = samples["clear_image"].to(device)

            optimizer.zero_grad()
            output = model(input)
            # print(f"output shape:{np.shape(output)}")
            # print(f"target shape:{np.shape(target)}")
            # print(f"data range:{torch.max(target)}")

            # target = Variable(target, requires_grad = False)
            # output = Variable(output, requires_grad = True)
            data_range = target.max().unsqueeze(0)
            if config['loss'] == "MSELoss":
                mse_loss = loss_mse(target, output)
                loss = mse_loss
            elif config['loss'] == "MSELoss+SSIMLoss":
                mse_loss = loss_mse(target, output)
                ssim_loss = loss_ssim(target, output, data_range)
                loss = 0.5 * mse_loss + 0.5 * ssim_loss
            elif config['loss'] == "MSELoss+SSIMLoss+PerceptualLoss":
                mse_loss = loss_mse(target, output)
                ssim_loss = loss_ssim(target, output, data_range)
                perc_loss = loss_perceptual(output, target)
                loss = mse_loss / 3.0 + ssim_loss / 3.0 + perc_loss / 3.0

            loss.backward()
            optimizer.step()

            if config['lr_scheduler']['used']:
                scheduler.step()

            epoch_loss += loss.item()
            if config['loss'] == "MSELoss":
                epoch_mse_loss += mse_loss.item()
            elif config['loss'] == "MSELoss+SSIMLoss":
                epoch_mse_loss += mse_loss.item()
                epoch_ssim_loss += ssim_loss.item()
            elif config['loss'] == "MSELoss+SSIMLoss+PerceptualLoss":
                epoch_mse_loss += mse_loss.item()
                epoch_ssim_loss += ssim_loss.item()
                epoch_perc_loss += perc_loss.item()

        total_loss += epoch_loss
        print(f"Epoch [{epoch + 1}/{num_epochs}], loss: {epoch_loss/len(train_loader):.4f}, mse_loss:{epoch_mse_loss/len(train_loader):.4f}, ssim_loss:{epoch_ssim_loss/len(train_loader):.4f}, perc_loss:{epoch_perc_loss/len(train_loader):.4f}")

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
    name = config['arch']['type'] + '-' + config['loss']
    _save_checkpoint(trained_model, optimizer_used, total_loss, config, checkpoint_path, name)
    


