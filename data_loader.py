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
import numbers
import math

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
    '''
    color map
    0=background, 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle # 6=bus, 7=car, 8=cat, 9=chair, 10=cow, 11=diningtable,
    12=dog, 13=horse, 14=motorbike, 15=person # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
    '''
    palette = [0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128,
               128, 128, 128, 64, 0, 0, 192, 0, 0, 64, 128, 0, 192, 128, 0, 64, 0, 128, 192, 0, 128,
               64, 128, 128, 192, 128, 128, 0, 64, 0, 128, 64, 0, 0, 192, 0, 128, 192, 0, 0, 64, 128]


    def __init__(self, root, transform, crop_size, image_size, mode='train'):
        self.root = root

        if mode == 'train':
            self.split = 'train'
        else:
            self.split = 'val'

        self.crop_size = crop_size
        self.image_size=image_size

        self._transform = transform
        zero_pad = 256 * 3 - len(self.palette)
        for i in range(zero_pad):
            self.palette.append(0)
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

    def colorize_mask(self,mask):
        # mask: numpy array of the mask

        new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
        new_mask.putpalette(self.palette)

        return new_mask.convert('RGB')
    def colorize_mask_batch(self,masks):
        color_masks = np.zeros((masks.shape[0],3,masks.shape[1],masks.shape[2]))
        toTensor = transforms.ToTensor()
        for i in range(masks.shape[0]):
            color_masks[i] = np.array(self.colorize_mask(masks[i])).transpose(2,0,1)

        return torch.from_numpy(color_masks).float()

    def __getitem__(self, index):
        data_file = self.files[self.split][index]
        # load image
        img_file = data_file['img']
        img_pil = Image.open(img_file)
        gen = RandomCropGenerator()
        onlyBgPatch = True
        while onlyBgPatch:

            transform_img = transforms.Compose([
                        gen,
                        RandomCrop(self.crop_size, gen=gen),
                        transforms.Resize([self.image_size, self.image_size])])

            img = np.array(transform_img(img_pil),dtype=np.uint8)

            transform_mask = transforms.Compose([
                RandomCrop(self.crop_size, gen=gen),
                transforms.Resize([self.image_size, self.image_size],interpolation=Image.NEAREST)])

            # load label
            lbl_file = data_file['lbl']
            lbl_pil = Image.open(lbl_file)

            lbl_cropped = transform_mask(lbl_pil)
            lbl = np.array(transform_mask(lbl_pil), dtype=np.int32)
            lbl[lbl == 255] = -1
            unique_vals = np.unique(lbl)
            
            if len(unique_vals) >= 2:
                onlyBgPatch = False
                # for i in unique_vals:
                #     percentage_covered = np.sum(lbl==i) / (self.image_size*self.image_size)
                    
                #     if percentage_covered >= 0.98:
                #         onlyBgPatch = True
                #         break
                    
        if self._transform:
            return self.transform(img, lbl)
        else:
            return img, lbl

    def transform(self, img, lbl):
        img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float64)
        img -= self.mean_bgr
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()
        return img, lbl

    def untransform(self, img):
        img = img.transpose(1, 2, 0)
        img += self.mean_bgr
        img = img.astype(np.uint8)
        img = img[:, :, ::-1] / 255
        img = img.transpose(2, 0, 1)
        return img
    def untransform_batch(self, img):
        img = img.numpy()
        for i in range(img.shape[0]):
            img[i] = self.untransform(img[i])

        return img


def get_loader(image_path, crop_size, image_size, batch_size, transform=False, dataset='PascalVOC2012', mode='train'):
    """Build and return data loader."""

    if dataset == 'PascalVOC2012':
        dataset = PascalVOC2012(image_path, transform, crop_size, image_size, mode)

    shuffle = False
    if mode == 'train':
        shuffle = True

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle)
    return data_loader
