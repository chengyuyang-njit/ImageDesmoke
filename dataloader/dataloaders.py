import torch
from torch.utils.data import Dataset
import pandas as pd
import pandas as pd
import os
from PIL import Image


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


# paired_surgery_dataset = PairedSmokeImageDataset(
#     csv_file = '/mmfs1/project/cliu/cy322/datasets/DesmokeData-main/images/paired_images.csv',
#     root_dir = '/mmfs1/project/cliu/cy322/datasets/DesmokeData-main/images/dataset')
# fig = plt.figure()

# for i, sample in enumerate(paired_surgery_dataset):
#     print(i, sample['clear_image'].shape, sample['smoked_image'].shape)

#     ax = plt.subplot(1, 4, i + 1)
#     plt.tight_layout()
#     ax.set_title("Sample # {}".format(i))
#     ax.axis('off')

#     if i == 3:
#         plt.show()
#         break