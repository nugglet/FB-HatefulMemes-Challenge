# Utility functions for preprocessing
import os
import json
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.io import read_image

import cv2

################################## Data Loading and Pre-processing ###################################################

def resize_img(img_path, target_length=512, display=False):
    """
    img_path: relative path of the img
    Resizes and pads given img to target size (target_length x target_length)
    """ 
    img = cv2.imread(img_path)[:,:,:3]  # take only RGB img data
    img_width = img.shape[1]
    img_height = img.shape[0]

    padded_img = np.zeros((target_length, target_length, 3), np.uint8)

    if img_width >= img_height:
        # landscape or square, fix width at target_length, scale height accordingly
        new_height = int((float(target_length)/img_width) * img_height )
        img_scaled = cv2.resize(img, (target_length, new_height), interpolation = cv2.INTER_AREA)

        idx = round((target_length - new_height)/2)
        padded_img[idx : idx + new_height, :] = img_scaled
        
    else:
        # portrait, fix height at target_length, scale width accordingly
        new_width = int( (float(target_length)/img_height) * img_width )
        img_scaled = cv2.resize(img, (new_width, target_length), interpolation = cv2.INTER_AREA)

        idx = round((target_length - new_width)/2)
        padded_img[:, idx : idx + new_width] = img_scaled 

    padded_img = cv2.cvtColor(padded_img, cv2.COLOR_BGR2RGB) # change to rgb format
    PIL_img = Image.fromarray(padded_img).convert('RGB')
    PIL_img = Image.fromarray(padded_img.astype('uint8'), 'RGB')
        
    if display:
        plt.imshow(PIL_img, cmap='gray'); plt.title('Path: '+str(img_path)); plt.axis('off'); plt.show()        

    return PIL_img


# returns a dictionary of pandas dataframes of the json files and dictionary of torch DataLoaders
def load_data(**kwargs):
    loader_params = kwargs.pop('loader_params', None)
    transform = kwargs.pop('transform', None)
    label_transform = kwargs.pop("label_transform", None)
    img_size = kwargs.pop("img_size", 512)

    if not transform:
        transform = transforms.Compose([transforms.Resize(img_size + 1), transforms.CenterCrop(img_size - 1), transforms.ToTensor()])
    if not loader_params:
        loader_params = {'batch_size': 64, 'shuffle': True}

    files = ["dev_seen", "dev_unseen", "test_seen", "test_unseen", "train"]
    datasets ={}
    loaders = {}
    for name in files:
        json_path = "../hateful_memes/" + name + ".jsonl"

        # create dataloader
        dataset = MemeDataset(json_path, transform=transform, label_transform=label_transform)
        dataloader = DataLoader(dataset, **loader_params)
        datasets[name] = dataset
        loaders[name] = dataloader

    return loaders, datasets

# Custom Dataset
class MemeDataset(Dataset):
    def __init__(self, json_path, img_dir="../hateful_memes/", transform=None, label_transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.label_transform = label_transform

        with open(json_path, 'r') as json_file:
            self.json_list = list(json_file)
        
    def __len__(self):
        return len(self.json_list)

    def __getitem__(self, index):
        sample = json.loads(self.json_list[index])
        img_path = os.path.join(self.img_dir, sample["img"])

        # image = Image.open(img_path).convert('RGB')
        image = resize_img(img_path)
        label = sample["label"]
        text = sample["text"]

        if self.transform:
            image = self.transform(image)
        if self.label_transform:
            label = self.label_transform(label)
        return image, text, label

###################################### Data Visualisation ########################################

def img_grid(dataset: Dataset, num_imgs=10, rows=5):
    imgs = torch.stack([dataset[i][0] for i in range(num_imgs)])
    grid = torchvision.utils.make_grid(imgs, nrow=rows)
    plt.figure(figsize=(16, 16), dpi=75)
    plt.imshow(grid.permute(1, 2, 0))