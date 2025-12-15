import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np

class SegmentationDataset(Dataset):
    def __init__(self, size=100):
        self.size = size
        # We generate data on the fly to simulate a dataset
        self.img_size = 256
        
        # --- THE TRAP ---
        # Defining separate transform objects implies separate random states.
        # When self.img_transform(img) is called, it picks a random crop (e.g., x=10, y=10).
        # When self.mask_transform(mask) is called, it picks a NEW random crop (e.g., x=50, y=90).
        # The mask no longer matches the image.
        
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomCrop(128),
            transforms.RandomHorizontalFlip(p=0.5)
        ])
        
        self.mask_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomCrop(128),
            transforms.RandomHorizontalFlip(p=0.5)
        ])

    def __len__(self):
        return self.size

    def _generate_synthetic_sample(self, idx):
        """
        Generates a black image with a white square.
        The mask is identical to the image (perfect correlation).
        """
        # Deterministic generation based on index to be reproducible
        np.random.seed(idx)
        
        # Create blank
        img_np = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
        
        # Draw a square at a random location
        w = 50
        x = np.random.randint(0, self.img_size - w)
        y = np.random.randint(0, self.img_size - w)
        img_np[y:y+w, x:x+w] = 255
        
        # In this synthetic task, Image == Mask exactly.
        # This makes checking alignment easy: Transformed Image should equal Transformed Mask.
        return Image.fromarray(img_np), Image.fromarray(img_np)

    def __getitem__(self, idx):
        img, mask = self._generate_synthetic_sample(idx)
        
        # Apply transforms independently (BUG)
        img = self.img_transform(img)
        mask = self.mask_transform(mask)
        
        return img, mask