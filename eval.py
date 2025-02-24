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
    ap.add_argument("-mp", "--MODEL_PATH", type = str, required = True,
                    help = "the path to the model saved")
    ap.add_argument("-bs", "--BATCH_SIZE", type = int, required = True,
                    help = "batch size for the training")

    return ap.parse_args()

if __name__ == "__main__":
    args = argparse.ArgumentParser(description="ImageDesmoke")
    args.add_argument('-ckp', '--CHECKPOINT', default = None, type = str, required = True,
                      help = "path to the checkpoint file of the model that you want to evaluate")

    args = args.parse_args()
    ckp = torch.load(args.CHECKPOINT)

    print(json.dumps(ckp["config"], indent = 4))
    checkpoint_path = "C:\\Users\\ycy99\\Documents\\NJIT\\research\\projects\\ImageDesmoke\\saved\\models"
    with open(args.CONFIG, 'r') as f:
        config = json.load(f)
    # print(config['dataloader']['args']['batch_size'])
    trained_model, optimizer_used, total_loss = train()
    _save_checkpoint(trained_model, optimizer_used, total_loss, config, checkpoint_path)
    

args = arg_parse()
# Load model later with:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(in_channels=3, out_channels=3).to(device)
model.load_state_dict(torch.load(args.MODEL_PATH))
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



