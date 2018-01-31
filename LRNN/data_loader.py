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
import collections


class RandomCropGenerator(object):
    def __call__(self, img):
        self.x1 = random.uniform(0, 1)
        self.y1 = random.uniform(0, 1)
        return img

class RandomCrop(object):
    def __init__(self, size, padding=0, gen=None):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self._gen = gen

    def __call__(self, img):
        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)
        w, h = img.size
        th, tw = self.size
        if w == tw and h == th:
            return img

        if self._gen is not None:
            x1 = math.floor(self._gen.x1 * (w - tw))
            y1 = math.floor(self._gen.y1 * (h - th))
        else:
            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)

        return img.crop((x1, y1, x1 + tw, y1 + th))

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


class PascalVOC2012(Dataset):

    class_names = np.array([
        'background',
        'aeroplane',
        'bicycle',
        'bird',
        'boat',
        'bottle',
        'bus',
        'car',
        'cat',
        'chair',
        'cow',
        'diningtable',
        'dog',
        'horse',
        'motorbike',
        'person',
        'potted plant',
        'sheep',
        'sofa',
        'train',
        'tv/monitor',
    ])
    mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])

    def __init__(self, root, transform=None, mode='train'):
        self.root = root

        if mode == 'train':
            self.split = 'train'
        else:
            self.split = 'val'


        self.transform = transform

        # VOC2011 and others are subset of VOC2012
        dataset_dir = os.path.join(self.root, 'VOC/VOCdevkit/VOC2012')
        self.files = collections.defaultdict(list)
        for split in ['train', 'val']:
            imgsets_file = os.path.join(
                dataset_dir, 'ImageSets/Segmentation/%s.txt' % split)
            for did in open(imgsets_file):
                did = did.strip()
                img_file = os.path.join(dataset_dir, 'JPEGImages/%s.jpg' % did)
                lbl_file = os.path.join(
                    dataset_dir, 'SegmentationClass/%s.png' % did)
                self.files[split].append({
                    'img': img_file,
                    'lbl': lbl_file,
                })
    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        data_file = self.files[self.split][index]
        # load image
        img_file = data_file['img']
        img = Image.open(img_file)
        # img = np.array(img, dtype=np.uint8)
        # load label
        lbl_file = data_file['lbl']
        lbl = Image.open(lbl_file)
        lbl = np.array(lbl, dtype=np.int32)
        lbl[lbl == 255] = -1

        img, lbl = self.preprocess(img,lbl)

        img = self.transform(img)
        im_size = img.size()

        rand = torch.zeros(1, im_size[1], im_size[2])
        rand.random_(0,2)
        rand = rand.expand(3, im_size[1], im_size[2])
        img_corrupted = img * rand
        lbl = torch.from_numpy(lbl).long()
        
        return img_corrupted, img

    def preprocess(self,img, lbl):
        # img = img[:, :, ::-1]  # RGB -> BGR
        # img = img.astype(np.float64)
        # img -= self.mean_bgr
        # img = img.transpose(2, 0, 1)

        return img, lbl       

    def postprocess(self, img, lbl):
        # img = img.numpy()
        # img = img.transpose(1, 2, 0)
        # img += self.mean_bgr
        # img = img.astype(np.uint8)
        # img = img[:, :, ::-1]
        # lbl = lbl.numpy()
        return img, lbl



def get_loader(image_path, crop_size, image_size, batch_size, dataset='PascalVOC2012', mode='train'):
    """Build and return data loader."""

    if mode == 'train':
        transform = transforms.Compose([
            # transforms.RandomCrop(crop_size),
            transforms.Scale([image_size, image_size]),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor()])
    else:
        transform = transforms.Compose([
            transforms.CenterCrop(crop_size),
            transforms.Scale(image_size),
            transforms.ToTensor()])

    if dataset == 'PascalVOC2012':
        dataset = PascalVOC2012(image_path, transform, mode)

    shuffle = False
    if mode == 'train':
        shuffle = True

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle)
    return data_loader
