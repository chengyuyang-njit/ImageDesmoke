# main script to start training

import argparse
import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import transforms

import matplotlib.pyplot as plt
import torch.optim as optim


from base.base_model import UNet
from dataloader import dataloaders

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


# Example usage:
if __name__ == "__main__":
    args = arg_parse()
    

    # x = torch.randn((1, 3, 700, 350))  # Example input tensor
    # preds = model(x)
    # print(preds.shape)  # Should output torch.Size([1, 3, 256, 256])
    # model = model.cuda()
    # summary(model, (3,700,350))

    trained_model = train()
    torch.save(trained_model.state_dict(), f"unet_model_{args.EPOCHS}_epochs.pth")


