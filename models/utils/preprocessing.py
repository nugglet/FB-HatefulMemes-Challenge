# Utility functions for preprocessing
from cProfile import label
import os
import json
from sys import maxsize
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.io import read_image

################################## Data Loading and Pre-processing ###################################################

# returns a dictionary of pandas dataframes of the json files and dictionary of torch DataLoaders
def load_data(**kwargs):
    loader_params = kwargs.pop('loader_params', None)
    transform = kwargs.pop('transform', None)
    label_transform = kwargs.pop("label_transform", None)
    img_size = kwargs.pop("img_size", 512)

    # TODO: Transforms should pad all images to same size instead of cropping to prevent data loss
    if not transform:
        transform = transforms.Compose([transforms.Resize(img_size+1), transforms.CenterCrop(img_size), transforms.ToTensor()])
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
        image = Image.open(img_path).convert('RGB')
        label = sample["label"]
        text = sample["text"]
        if self.transform:
            image = self.transform(image)
        if self.label_transform:
            label = self.label_transform(label)
        return image, text, label

###################################### Data Visualisation ########################################

def img_grid(dataset: Dataset, rows=5):
    imgs = torch.stack([dataset[i][0] for i in range(10)])
    grid = torchvision.utils.make_grid(imgs, nrow=rows)
    plt.figure(figsize=(16, 16), dpi=75)
    plt.imshow(grid.permute(1, 2, 0))