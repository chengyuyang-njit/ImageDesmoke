import argparse
import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import transforms

import matplotlib.pyplot as plt
import torch.optim as optim

from base.base_model import UNet
from dataloader import dataloaders

import json

def load_data():
    # load the dataset
    print("[INFO] loading the paired desmoke image dataset...")

    dataset = dataloaders.PairedSmokeImageDataset(
        csv_file = '..\\..\\datasets\\DesmokeData-main\\images\\paired_images.csv',
        root_dir = '..\\..\\datasets\\DesmokeData-main\\images\\dataset',
        transform =  transforms.Compose([transforms.ToTensor()]))

    num_train_samples = int(len(dataset) * config["dataloader"]["args"]["train_split"]) + 1
    num_val_samples = int(len(dataset) * config["dataloader"]["args"]["validation_split"])
    num_test_samples = int(len(dataset) * (1 - config["dataloader"]["args"]["train_split"] - 
                                          config["dataloader"]["args"]["validation_split"]))

    (train_data, val_data, test_data) = random_split(dataset, [num_train_samples, num_val_samples, num_test_samples],
                                                    generator=torch.Generator().manual_seed(42))

    print("[INFO] paired desmoke image dataset loaded...")
    return {"train":train_data, "val":val_data, "test": test_data}



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

if __name__ == "__main__":
    args = argparse.ArgumentParser(description="ImageDesmoke")
    args.add_argument('-ckp', '--CHECKPOINT', default = None, type = str, required = True,
                      help = "path to the checkpoint file of the model that you want to evaluate")

    args = args.parse_args()
    ckp = torch.load(args.CHECKPOINT)


    print(json.dumps(ckp["config"], indent = 4))
    config = ckp["config"]
    

    # Load model later with:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=3, out_channels=3).to(device)
    model.load_state_dict(torch.load(args.CHECKPOINT))
    model.eval()

    data = load_data()
    eval_loader = DataLoader(data["val"], batch_size = args.BATCH_SIZE, shuffle = True)
# with torch.no_grad():
#     for samples in eval_loader:
#         input = samples["smoked_image"].to(device)
#         target = samples["clear_image"].to(device)
#         output = model(input)



# Get a sample batch
samples = next(iter(eval_loader))
input = samples["smoked_image"].to(device)
target = samples["clear_image"].to(device)
with torch.no_grad():
    output = model(input)

# Visualize
visualize_sample(input[0], target[0], output[0])



