# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import PIL
import pandas as pd

import torch
from torchvision import datasets, transforms
from torchvision.datasets.folder import default_loader


from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

class ImageListFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None, target_transform=None,
                 ann_file=None, loader=default_loader):
        self.root = root
        self.transform = transform
        self.loader = loader
        self.target_transform = target_transform
        self.nb_classes = 1000

        assert ann_file is not None
        print('load info from', ann_file)

        self.samples = []
        ann = open(ann_file)
        for elem in ann.readlines():
            cut = elem.split(',')
            path_current = os.path.join(root, cut[0])
            target_current = int(cut[1])
            self.samples.append((path_current, target_current))
        ann.close()

        print('load finish')
        
        # self.imgs = self.samples
        
    # def __getitem__(self, index):
    #     # this is what ImageFolder normally returns 
    #     original_tuple = super(ImageListFolder, self).__getitem__(index)
    #     # the image file path
    #     path = self.imgs[index][0]
    #     # make a new tuple that includes original and the path
    #     tuple_with_path = (original_tuple + (path,))
    #     return tuple_with_path

class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path
    
    
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root, csv_path, transform = None):
        
        self.df = pd.read_csv(csv_path)
        self.images_folder = root
        self.transform = transform
        self.class2index = {"negative":0, "positive":1}

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        
        filename = self.df.at[index, "filename"]
        label = self.df.at[index, "label"]
        image_path = os.path.join(self.images_folder, filename)
        image = PIL.Image.open(image_path)
        image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        
        final_tuple = ((image, label) + (image_path,))
        return final_tuple



def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    # root = os.path.join(args.data_path, 'train' if is_train else 'val')
    root = args.data_path
    fold = args.fold
    if not is_train:
        # dataset = ImageFolderWithPaths(root, transform=transform)
        dataset = CustomDataset(root, csv_path=f'/home/aarjav/scratch/Her2Neu_slices/data_folds/folds_normal/{fold}/test.csv',  transform=transform)
    else:
        # dataset = datasets.ImageFolder(root, transform=transform)
        dataset = CustomDataset(root, csv_path=f'/home/aarjav/scratch/Her2Neu_slices/data_folds/folds_normal/{fold}/train.csv', transform=transform)

    # samples = dataset.samples

    # print(dataset)

    # if(not is_train):
    #     return (dataset, samples)
    return dataset


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
