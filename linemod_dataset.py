 import os
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np

class LineMODDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = sorted([os.path.join(root_dir, 'images', f) for f in os.listdir(os.path.join(root_dir, 'images'))])
        self.mask_paths = sorted([os.path.join(root_dir, 'masks', f) for f in os.listdir(os.path.join(root_dir, 'masks'))])
        self.vector_fields = sorted([os.path.join(root_dir, 'vectors', f) for f in os.listdir(os.path.join(root_dir, 'vectors'))])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_paths[idx], 0)  # single-channel
        vector = np.load(self.vector_fields[idx])  # shape: H x W x (2*K)

        if self.transform:
            image = self.transform(image)

        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(mask).long()
        vector = torch.from_numpy(vector).permute(2, 0, 1).float()

        return image, vector, mask

