import os
from pathlib import Path

import numpy as np
import random

import torch
import torchvision.transforms as T
import PIL.Image as Image

from config import cfg
from datasets import uniform

random.seed(cfg.RANDOM_SEED)
root = cfg.DATASET.AGRISEG_DIR
num_classes = 1
ignore_label = None

def AgriSeg_DataLoader(args, augment=False):
    dataset = AgriSeg(args, augment=False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=24)
    return dataloader


class AgriSeg(torch.utils.data.Dataset):
    """Image (semantic) segmentation dataset."""

    def __init__(self, args, root_dir=root, augment=False, transform=None, target_transform=None, target_aux_transform=None, test=False):
        self.root_dir = Path(root_dir)  
        self.args = args
        self.augment = augment
        self.images, self.masks = [], []
        self.target_transform = target_transform
        self.transform = transform
        self.target_aux_transform = target_aux_transform
        self.test = test
        self.perc = (1.0 - self.args.val_perc) if not test else 1.0
        
        self.get_file_lists_multi()
        self.get_transforms()
        
    def get_file_lists_multi(self):
        if self.test:
            self.get_file_lists(self.root_dir.joinpath(self.args.target))
            return
        
        for d in self.args.source:
            if d == self.args.target:
                continue
            self.get_file_lists(self.root_dir.joinpath(d))

    def get_file_lists(self, subd=None):
        if subd is None:
            subd = self.root_dir
        
        for subdir in subd.iterdir():
            if subdir.is_file() or subdir.name.startswith('.'): continue
            print(subdir)
            image_file_names = [list(f.glob('**/*'))[0].absolute() 
                                for f in subdir.joinpath('images').iterdir()]
            mask_file_names = [list(f.glob('**/*'))[0].absolute() 
                               for f in subdir.joinpath('masks').iterdir()]
            
            print(len(image_file_names))

            if self.perc < 1.0:
                random.seed(cfg.RANDOM_SEED)
                self.images += random.sample(sorted(image_file_names), int(len(image_file_names) * self.perc))
                random.seed(cfg.RANDOM_SEED)
                self.masks += random.sample(sorted(mask_file_names), int(len(mask_file_names) * self.perc))
            else:
                self.images += sorted(image_file_names)
                self.masks += sorted(mask_file_names)
        
    
    def get_transforms(self):
        if self.augment:
            self.image_transforms = T.Compose([
                T.RandomResizedCrop(size=(self.args.crop_size, self.args.crop_size), 
                                   scale=(0.5, 1.0),
                                   ratio=(1,1),
                                   interpolation=T.InterpolationMode.BILINEAR),
                T.RandomHorizontalFlip(0.5),
                T.ColorJitter(brightness=0.4,
                              contrast=0.4),
                T.RandomGrayscale(0.1),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
            ])
            
            self.mask_transforms = T.Compose([
                T.RandomResizedCrop(size=(self.args.crop_size, self.args.crop_size), 
                                   scale=(0.5, 1.0),
                                   ratio=(1,1),
                                   interpolation=T.InterpolationMode.NEAREST),
                T.RandomHorizontalFlip(0.5),
                T.ToTensor(),
                T.Lambda(lambda mask: torch.where(mask > 0, 1.0, 0.0))  
            ])
            
        else:
            self.image_transforms = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
            ])
            
            self.mask_transforms = T.Compose([
                T.ToTensor(),
                T.Lambda(lambda mask: torch.where(mask > 0, 1.0, 0.0))
            ])
        
    def __len__(self):
        return len(self.images)
    
    def build_epoch(self, cut=False):
        """
        Perform Uniform Sampling per epoch to create a new list for training such that it
        uniformly samples all classes
        """
        if self.class_uniform_pct > 0:
            if cut:
                # after max_cu_epoch, we only fine images to fine tune
                self.imgs_uniform = uniform.build_epoch(self.imgs,
                                                        self.fine_centroids,
                                                        num_classes,
                                                        cfg.CLASS_UNIFORM_PCT)
            else:
                self.imgs_uniform = uniform.build_epoch(self.imgs + self.aug_imgs,
                                                        self.centroids,
                                                        num_classes,
                                                        cfg.CLASS_UNIFORM_PCT)
        else:
            self.imgs_uniform = self.imgs
            
    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        mask = Image.open(self.masks[idx]).convert('L')
        
        self.seed = np.random.randint(2147483647) # make a seed with numpy generator 

        image = self.preprocess_image(image)
        mask = self.preprocess_mask(mask)

        if self.transform is not None:
            #img = self.transform(img)
            pass

        if self.target_aux_transform is not None:
            mask_aux = self.target_aux_transform(mask)
        else:
            mask_aux = torch.tensor([0])
        if self.target_transform is not None:
            mask = self.target_transform(mask)

        return image, mask, str(self.images[idx]), mask_aux
    
    def preprocess_image(self, image):
        random.seed(self.seed) 
        torch.manual_seed(self.seed) 
        return self.image_transforms(image)
    
    def preprocess_mask(self, mask):
        random.seed(self.seed) 
        torch.manual_seed(self.seed) 
        return self.mask_transforms(mask)
    
    def build_epoch(self, cut=False):
        """
        Perform Uniform Sampling per epoch to create a new list for training such that it
        uniformly samples all classes
        """
        if self.args.class_uniform_pct > 0:
            if cut:
                # after max_cu_epoch, we only fine images to fine tune
                self.imgs_uniform = uniform.build_epoch(self.images,
                                                        self.fine_centroids,
                                                        num_classes,
                                                        cfg.CLASS_UNIFORM_PCT)
            else:
                self.imgs_uniform = uniform.build_epoch(self.images,
                                                        self.centroids,
                                                        num_classes,
                                                        cfg.CLASS_UNIFORM_PCT)
        else:
            self.imgs_uniform = self.imgs