import numpy as np
import torch
from cv2 import imread
from torch.utils.data import Dataset
from sys import platform
import pickle
import os
from torchvision import transforms
from PIL import Image
# TODO: Change the global directory to where you normally hold the datasets.
# I use both Windows PC and Linux Server for this project so I have two dirs.

im_trans = transforms.Compose([
    Image.fromarray,
    transforms.Resize([256, 256]),
    transforms.ToTensor(),
    transforms.Normalize([.5, .5, .5], [.5, .5, .5]),
])

import glob

class ImageDataset(Dataset):
    def __init__(self, dir):
        image_names = glob.glob(os.path.join(dir, '*.jpg'))
        image_names.sort()
        self.image_names = image_names

    def __len__(self):
        return self.image_names.__len__()

    def __getitem__(self, item):
        name = self.image_names[item]
        image = im_trans(imread(name))
        return image, name

