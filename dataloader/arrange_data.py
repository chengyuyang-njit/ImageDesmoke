import dataloaders
import torch
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import numpy as np
from torchvision import transforms
import os
import cv2

def load_data():
    # load the dataset
    print("[INFO] loading the paired desmoke image dataset...")

    dataset = dataloaders.PairedSmokeImageDataset(
        csv_file = '/mmfs1/project/cliu/cy322/datasets/DesmokeData-main/images/paired_images.csv',
        root_dir = '/mmfs1/project/cliu/cy322/datasets/DesmokeData-main/images/dataset',
        transform = transforms.Compose([
            transforms.Resize((350,700)),
            transforms.ToTensor()]))

    num_train_samples = int(len(dataset) * 0.8) + 1
    num_val_samples = int(len(dataset) * 0.1)
    num_test_samples = int(len(dataset) * 0.1)

    (train_data, val_data, test_data) = random_split(dataset, [num_train_samples, num_val_samples, num_test_samples],
                                                    generator=torch.Generator().manual_seed(42))
    print("[INFO] paired desmoke image dataset loaded...")
    return {"train":train_data, "val":val_data, "test": test_data}

data = load_data()
train_loader = DataLoader(
    data["train"], batch_size = 1, 
                            shuffle = False)
val_loader = DataLoader(
    data["val"], batch_size = 1, 
                            shuffle = False)
test_loader = DataLoader(
    data["test"], batch_size = 1, 
                            shuffle = False)

number = 0
for samples in test_loader:
    number += 1
    input = (np.transpose(torch.squeeze(samples["smoked_image"]).numpy(), (1,2,0)) * 255).astype(np.uint8)
    target = (np.transpose(torch.squeeze(samples["clear_image"]).numpy(), (1,2,0)) * 255).astype(np.uint8)
    im_AB = np.concatenate([input,target], 1)
    # print(f"input size:{np.shape(input)}, taget size:{np.shape(target)},im_AB size:{np.shape(im_AB)}")
    filename = os.path.join("/mmfs1/project/cliu/cy322/datasets/Desmoke-pixel2pixel/test", str(number)+".jpg")
    cv2.imwrite(filename, cv2.cvtColor(im_AB,cv2.COLOR_RGB2BGR))


number = 0
for samples in val_loader:
    number += 1
    input = (np.transpose(torch.squeeze(samples["smoked_image"]).numpy(), (1,2,0)) * 255).astype(np.uint8)
    target = (np.transpose(torch.squeeze(samples["clear_image"]).numpy(), (1,2,0)) * 255).astype(np.uint8)
    im_AB = np.concatenate([input,target], 1)
    # print(f"input size:{np.shape(input)}, taget size:{np.shape(target)},im_AB size:{np.shape(im_AB)}")
    filename = os.path.join("/mmfs1/project/cliu/cy322/datasets/Desmoke-pixel2pixel/val", str(number)+".jpg")
    cv2.imwrite(filename, cv2.cvtColor(im_AB,cv2.COLOR_RGB2BGR))

number = 0
for samples in train_loader:
    number += 1
    input = (np.transpose(torch.squeeze(samples["smoked_image"]).numpy(), (1,2,0)) * 255).astype(np.uint8)
    target = (np.transpose(torch.squeeze(samples["clear_image"]).numpy(), (1,2,0)) * 255).astype(np.uint8)
    im_AB = np.concatenate([input,target], 1)
    # print(f"input size:{np.shape(input)}, taget size:{np.shape(target)},im_AB size:{np.shape(im_AB)}")
    filename = os.path.join("/mmfs1/project/cliu/cy322/datasets/Desmoke-pixel2pixel/train", str(number)+".jpg")
    cv2.imwrite(filename, cv2.cvtColor(im_AB,cv2.COLOR_RGB2BGR))