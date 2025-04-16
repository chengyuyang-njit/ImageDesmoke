import argparse
import torch
import cv2
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import transforms

import matplotlib.pyplot as plt
import torch.optim as optim

from base.base_model import UNet
from model.model import UNetWithWiener
from dataloader import dataloaders

import json
import numpy as np
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
from skimage import color
import scipy.signal
from scipy.io import savemat
from pyciede2000 import ciede2000

def wiener_filter(image, kernel_size=5, noise_var = 0.1):
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
        filtered_image[..., i] = scipy.signal.wiener(image[..., i], mysize=kernel_size, noise = noise_var)
    return filtered_image



def load_data():
    # load the dataset
    print("[INFO] loading the paired desmoke image dataset...")

    dataset = dataloaders.PairedSmokeImageDataset(
        csv_file = '/mmfs1/project/cliu/cy322/datasets/DesmokeData-main/images/paired_images.csv',
        root_dir = '/mmfs1/project/cliu/cy322/datasets/DesmokeData-main/images/dataset',
        transform = transforms.Compose([transforms.Resize((350,700)),transforms.ToTensor()]))

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

def save_sample(inputs, targets, outputs, filename):


    fig, ax = plt.subplots(1, 3, figsize=(16, 3))
    ax[0].imshow(inputs)
    ax[0].set_title("Input")
    ax[1].imshow(targets)
    ax[1].set_title("Target")
    ax[2].imshow(outputs)
    ax[2].set_title("Output")
    plt.savefig(filename)
    plt.close()




if __name__ == "__main__":
    args = argparse.ArgumentParser(description="ImageDesmoke")
    args.add_argument('-ckp', '--CHECKPOINT', default = None, type = str, required = True,
                      help = "path to the checkpoint file of the model that you want to evaluate")
    args.add_argument('-sp', '--SAVE_PATH', default = None, type = str, required = True,
                      help = "Path to save the evaluation results")
    args = args.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckp = torch.load(args.CHECKPOINT, map_location = device)


    print(json.dumps(ckp["config"], indent = 4))
    config = ckp["config"]
    

    # Load model later with:
    if config["arch"]["type"] == "UNetWithWiener":
        model = UNetWithWiener(in_channels=3, out_channels=3).to(device)
    else:
        model = UNet(in_channels=3, out_channels=3).to(device)
    model.load_state_dict(ckp["state_dict"])
    model.eval()

    data = load_data()
    eval_loader = DataLoader(data["val"], batch_size = 1, shuffle = False)

    with torch.no_grad():
        number, ssim_total, psnr_total, mse_total, ciede_total = 0, 0, 0, 0, 0
        for samples in eval_loader:

            input = samples["smoked_image"].to(device)
            target = samples["clear_image"].to(device)
            output = model(input)
            # print(input.shape)
            filename = args.SAVE_PATH + "/outputs/" + str(number) + ".png"



            # print(f"Evaluation Metrics for sample {number}:")

            input = input[0].cpu().numpy().transpose(1, 2, 0).astype(np.float32)
            target = target[0].cpu().numpy().transpose(1, 2, 0).astype(np.float32)
            output = output[0].cpu().numpy().transpose(1, 2, 0).astype(np.float32)
            # print(f"input:{input}, min:{np.min(input)}, max:{np.max(input)}")
            # print(f"target:{target}, min:{np.min(input)}, max:{np.max(input)}")
            # print(f"output:{output}, min:{np.min(output)}, max:{np.max(output)}")
            save_sample(input, target, output, f"{args.SAVE_PATH}/compare/compare{number}.png")
            cv2.imwrite(filename, cv2.cvtColor((output*255).astype(np.uint8),cv2.COLOR_RGB2BGR))
            cv2.imwrite(args.SAVE_PATH + "/inputs/" + str(number) + ".png", cv2.cvtColor((input*255).astype(np.uint8),cv2.COLOR_RGB2BGR))
            cv2.imwrite(args.SAVE_PATH + "/targets/" + str(number) + ".png", cv2.cvtColor((target*255).astype(np.uint8),cv2.COLOR_RGB2BGR))
            savemat(f"{args.SAVE_PATH}/targets_mat/target{number}.mat",{"image":target})
            savemat(f"{args.SAVE_PATH}/outputs_mat/output{number}.mat",{"image":output})
            number += 1
            # print(target)
            # break
            # savemat("target.mat",{"image":cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)})
            # savemat("output.mat",{"image":cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)})
            ssim_total += ssim(cv2.cvtColor(output, cv2.COLOR_BGR2GRAY), 
                               cv2.cvtColor(target, cv2.COLOR_BGR2GRAY),
                                            data_range = 1, win_size = 3, gaussian_weights=False, use_sample_covariance=False)

            psnr_total += psnr(target, output)

            mse_total += np.mean((output-target) ** 2)

        
        print(f"ssim:{ssim_total/number}, psnr:{psnr_total/number}, mse:{mse_total/number}")







