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
import numpy as np
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
from skimage.color import deltaE_ciede2000
from skimage import color
import scipy.signal


def wiener_filter(image, kernel_size=5):
    """
    Apply Wiener filter to each channel of the image.

    Parameters:
    - image: Input image (numpy array).
    - kernel_size: Size of the Wiener filter kernel (default is 5).

    Returns:
    - filtered_image: Image after applying Wiener filter.
    """
    filtered_image = np.zeros_like(image)
    for i in range(image.shape[2]):  # Iterate over R, G, B channels
        filtered_image[..., i] = scipy.signal.wiener(image[..., i], mysize=kernel_size)
    return filtered_image



def load_data():
    # load the dataset
    print("[INFO] loading the paired desmoke image dataset...")

    dataset = dataloaders.PairedSmokeImageDataset(
        csv_file = '..\\..\\datasets\\DesmokeData-paired\\DesmokeData-main\\images\\paired_images.csv',
        root_dir = '..\\..\\datasets\\DesmokeData-paired\\DesmokeData-main\\images\\dataset',
        transform = transforms.Compose([transforms.ToTensor()]))

    num_train_samples = int(len(dataset) * config["dataloader"]["args"]["train_split"]) + 1
    num_val_samples = int(len(dataset) * config["dataloader"]["args"]["validation_split"])
    num_test_samples = int(len(dataset) * (1 - config["dataloader"]["args"]["train_split"] - 
                                          config["dataloader"]["args"]["validation_split"]))

    (train_data, val_data, test_data) = random_split(dataset, [num_train_samples, num_val_samples, num_test_samples],
                                                    generator=torch.Generator().manual_seed(42))

    print("[INFO] paired desmoke image dataset loaded...")
    return {"train":train_data, "val":val_data, "test": test_data}



def visualize_sample(inputs, targets, outputs, filtered_image):
    inputs = inputs.cpu().numpy().transpose(1, 2, 0)
    targets = targets.cpu().numpy().transpose(1, 2, 0)
    outputs = outputs.cpu().numpy().transpose(1, 2, 0)

    fig, ax = plt.subplots(1, 4, figsize=(16, 4))
    ax[0].imshow(inputs)
    ax[0].set_title("Input")
    ax[1].imshow(targets)
    ax[1].set_title("Target")
    ax[2].imshow(outputs)
    ax[2].set_title("Output")
    ax[3].imshow(filtered_image)
    ax[3].set_title("Wiener Filter")
    plt.show()

def save_sample(inputs, targets, outputs, filtered_image, filename):


    fig, ax = plt.subplots(1, 4, figsize=(16, 4))
    ax[0].imshow(inputs)
    ax[0].set_title("Input")
    ax[1].imshow(targets)
    ax[1].set_title("Target")
    ax[2].imshow(outputs)
    ax[2].set_title("Output")
    ax[3].imshow(filtered_image)
    ax[3].set_title("Wiener Filter")
    plt.savefig(filename)
    plt.close()

if __name__ == "__main__":
    args = argparse.ArgumentParser(description="ImageDesmoke")
    args.add_argument('-ckp', '--CHECKPOINT', default = None, type = str, required = True,
                      help = "path to the checkpoint file of the model that you want to evaluate")
    args.add_argument('-sp', '--SAVE_PATH', default = None, type = str, required = True,
                      help = "Path to save the evaluation results")
    args = args.parse_args()
    ckp = torch.load(args.CHECKPOINT)


    print(json.dumps(ckp["config"], indent = 4))
    config = ckp["config"]
    

    # Load model later with:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=3, out_channels=3).to(device)
    model.load_state_dict(ckp["state_dict"])
    model.eval()

    data = load_data()
    eval_loader = DataLoader(data["val"], batch_size = config["dataloader"]["args"]["batch_size"], shuffle = False)
    number = 1
    with torch.no_grad():
        for samples in eval_loader:
            input = samples["smoked_image"].to(device)
            target = samples["clear_image"].to(device)
            output = model(input)
            # print(input.shape)
            filename = args.SAVE_PATH + "/" + str(number) + ".png"



            print(f"Evaluation Metrics for sample {number}:")

            input = input[0].cpu().numpy().transpose(1, 2, 0).astype(np.float32)
            target = target[0].cpu().numpy().transpose(1, 2, 0).astype(np.float32)
            output = output[0].cpu().numpy().transpose(1, 2, 0).astype(np.float32)
            # wiener filter
            filtered_image = wiener_filter(input)
            save_sample(input, target, output, filtered_image, filename)

            # print(target)
            ssim_value = ssim(target, output, channel_axis = -1 ,data_range = 1)
            ssim_value_ = ssim(target, filtered_image, channel_axis = -1 ,data_range = 1)
            print(f"SSIM: {ssim_value},{ssim_value_}")

            psnr_value = psnr(target, output)
            psnr_value_ = psnr(target, filtered_image)
            print(f"PSNR: {psnr_value} dB, {psnr_value_} dB")

            lab_image1 = color.rgb2lab(target)
            lab_image2 = color.rgb2lab(output)
            color_diff = deltaE_ciede2000(target, output)
            print(f"CIEDE2000: {color_diff}")

            # Compute statistics
            mean_diff = np.mean(color_diff)
            std_diff = np.std(color_diff)

            print(f"Mean CIEDE2000 Color Difference: {mean_diff}")
            print(f"Standard Deviation: {std_diff}")


            number += 1

            


# # Get a sample batch
# samples = next(iter(eval_loader))
# input = samples["smoked_image"].to(device)
# target = samples["clear_image"].to(device)
# with torch.no_grad():
#     output = model(input)

# Visualize
# visualize_sample(input[0], target[0], output[0])



