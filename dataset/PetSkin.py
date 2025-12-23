
import torch.utils.data
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import os
import os.path
from os.path import join
import glob
import json
from sklearn.model_selection import train_test_split

from dataset.utils import noisify

class PetSkin(torch.utils.data.Dataset):
    '''
    Pet Skin Lesion Dataset (A1-A6)
    '''

    def __init__(self,
                 root,
                 train=0, # 0: train, 1: test, 2: val
                 transform=None,
                 target_transform=None,
                 noise_type='clean',
                 noise_rate=0.00,
                 device=1,
                 split_ratios=(0.7, 0.15, 0.15),
                 random_seed=0
                 ):
        
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.train = train 
        self.device = device
        self.noise_type = noise_type
        self.labelOrder = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6']
        self.class_map = {label: i for i, label in enumerate(self.labelOrder)}

        # Gather all data
        self.all_data = []
        self.all_labels = []
        
        # Check if root has Datasets folder or is the Datasets folder
        search_root = root
        if os.path.isdir(join(root, 'Datasets')):
            search_root = join(root, 'Datasets')

        print(f"Loading PetSkin data from {search_root}")

        for label_name in self.labelOrder:
            class_dir = join(search_root, label_name)
            if not os.path.isdir(class_dir):
                print(f"Warning: Directory {class_dir} not found.")
                continue
            
            # Find all jpg files
            # Pattern: Recursive search or just top level? User said "Datasets/A1~A6... folder inside image"
            # Assuming flat structure inside A1, or recursive. optimizing for recursive just in case.
            # Using specific logic: files are like IMG_D_A2_088247.jpg
            # glob is slightly safer
            # images = glob.glob(join(class_dir, "**", "*.jpg"), recursive=True) 
            # Simple listdir might be faster/sufficient if flat
            
            for root_dir, dirs, files in os.walk(class_dir):
                for file in files:
                    if file.lower().endswith('.jpg') or file.lower().endswith('.jpeg'):
                        self.all_data.append(join(root_dir, file))
                        self.all_labels.append(self.class_map[label_name])

        self.all_data = np.array(self.all_data)
        self.all_labels = np.array(self.all_labels)

        self.all_data = np.array(self.all_data)
        self.all_labels = np.array(self.all_labels)

        # Splitting with Stratification
        # To ensure consistent splits across train/val/test instances, we use a fixed random seed
        num_samples = len(self.all_data)
        indices = np.arange(num_samples)
        
        # 1. Split Train vs (Val + Test)
        # Stratify based on all_labels to preserve class distribution
        train_indices, temp_indices, _, temp_labels = train_test_split(
            indices, self.all_labels,
            train_size=split_ratios[0],
            stratify=self.all_labels,
            random_state=random_seed
        )

        # 2. Split Val vs Test from the remaining data
        # Calculate relative ratio: if val=0.15, test=0.15, then val is 50% of the remaining 0.30
        relative_val_size = split_ratios[1] / (split_ratios[1] + split_ratios[2])
        
        val_indices, test_indices = train_test_split(
            temp_indices,
            train_size=relative_val_size,
            stratify=temp_labels,
            random_state=random_seed
        )

        if self.train == 0:
            self.data = self.all_data[train_indices]
            self.labels = self.all_labels[train_indices]
        elif self.train == 1:
            self.data = self.all_data[test_indices]
            self.labels = self.all_labels[test_indices]
        elif self.train == 2:
            self.data = self.all_data[val_indices]
            self.labels = self.all_labels[val_indices]
        
        # Load images into memory if device=1 (RAM)
        # Note: 32k images might be too large for RAM depending on resolution.
        # ISIC loader does this. I'll stick to it but warn/add check.
        # Actually ISIC resizes first.
        
        if image_size := get_image_size_from_transform(transform):
             self.resize_dim = (image_size, image_size)
        else:
             self.resize_dim = (224, 224) 

        self.loaded_data = []
        if self.device == 1:
            print(f"Loading {len(self.data)} images into RAM...")
            for i, img_path in enumerate(self.data):
                self.loaded_data.append(self.img_loader(img_path))
                if i % 1000 == 0:
                    print(f"\rLoaded {i}/{len(self.data)}", end='')
            print(" Done.")
            self.loaded_data = np.array(self.loaded_data)
        
        # Noisy labels generation
        # "noise_type" argument dictates if we ADD synthetic noise.
        # Since usage of this algo implies combating noise, strict "clean" evaluation 
        # is only possible if we trust the provided labels as ground truth.
        self.labels = np.asarray(self.labels)
        
        if noise_type == 'clean':
            self.noise_or_not = np.ones([len(self.labels)], dtype=bool)
            self.noisy_labels = self.labels
        else:
            self.noisy_labels, self.actual_noise_rate = noisify(dataset="petskin",
                                                                  nb_classes=6,
                                                                  train_labels=np.expand_dims(self.labels, 1),
                                                                  noise_type=noise_type,
                                                                  noise_rate=noise_rate,
                                                                  random_state=random_seed)
            self.noisy_labels = self.noisy_labels.squeeze()
            self.noise_or_not = self.noisy_labels == self.labels

    def img_loader(self, img_path):
        # Resize on load to save RAM if caching
        img = Image.open(img_path).convert('RGB')
        return np.asarray(img.resize(self.resize_dim, Image.NEAREST)).astype(np.uint8)

    def __getitem__(self, index):
        if self.device == 1:
            img = self.loaded_data[index]
        else:
            img = self.img_loader(self.data[index])
            
        target = self.noisy_labels[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        return len(self.data)

def get_image_size_from_transform(transform):
    if not transform: return None
    if hasattr(transform, 'transforms'):
        for t in transform.transforms:
            if isinstance(t, transforms.Resize):
                return t.size[0] if isinstance(t.size, tuple) else t.size
    return None
