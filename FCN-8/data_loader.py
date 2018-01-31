import torch
import os
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import h5py
import numpy as np

class FlowersDataset(Dataset):
    def __init__(self, image_path, transform, mode):
        self.transform = transform
        self.mode = mode
        self.data = h5py.File(image_path, 'r')
        self.num_data = self.data["train_images"].shape[0]
        self.attr2idx = {}
        self.idx2attr = {}

        print ('Start preprocessing dataset..!')
        self.preprocess()
        print ('Finished preprocessing dataset..!')

        if self.mode == 'train':
            self.num_data = self.data["train_images"].shape[0]
        elif self.mode == 'test':
            self.num_data = self.data["test_images"].shape[0]

    def preprocess(self):
        main_colors = ["blue","orange","pink","purple","red","white","yellow"]
        for i, attr in enumerate(main_colors):
            self.attr2idx[attr] = i
            self.idx2attr[i] = attr


    def __getitem__(self, index):
        image = Image.fromarray(np.uint8(self.data[self.mode+"_images"][index]))
        feature = np.float32(self.data[self.mode+"_feature"][index])
        identity = int(self.data[self.mode+"_class"][index])


        return self.transform(image), torch.FloatTensor(feature), identity

    def __len__(self):
        return self.num_data

def get_loader(image_path, crop_size, image_size, batch_size, dataset='Flowers', mode='train'):
    """Build and return data loader."""

    if mode == 'train':
        transform = transforms.Compose([
            transforms.RandomCrop(crop_size),
            transforms.Scale(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    else:
        transform = transforms.Compose([
            transforms.CenterCrop(crop_size),
            transforms.Scale(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    if dataset == 'Flowers':
        dataset = FlowersDataset(image_path, transform, mode)

    shuffle = False
    if mode == 'train':
        shuffle = True

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle)
    return data_loader
