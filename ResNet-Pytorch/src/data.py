from torch.utils.data import Dataset
import torch
from pathlib import Path
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
import torchvision as tv

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]


class ChallengeDataset(Dataset):
    # TODO implement the Dataset class according to the description
    
    def __init__(self, data: Dataset, mode: str):
        self.data=data
        self.mode=mode

    def _transform(self, transforms):
        # return tv.transforms.v2.Compose(transforms)
        return tv.transforms.Compose(transforms)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        img_name = self.data.iloc[item,0]
        img = imread(img_name)
        img=gray2rgb(img)
        to_pil_image = tv.transforms.ToPILImage()
        to_tensor = tv.transforms.ToTensor()
        # if self.mode == 'train':
        normalize = tv.transforms.Normalize(mean=train_mean, std=train_std)
        # else:
            # normalize = tv.transforms.Normalize()
        transforms = self._transform([to_pil_image, to_tensor, normalize])
        img = transforms(img)

        labels=self.data.iloc[item, 1:]
        labels=torch.tensor(labels)

        return img, labels

    
    
