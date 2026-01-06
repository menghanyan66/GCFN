"""
Dataset Handler for Dynamic Coarse-to-Fine Inpainting
WITH SAMPLE LIMITING FOR FAST TRAINING
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import random
import numpy as np

class InpaintingDataset(Dataset):
    """
    Image inpainting dataset with sample limiting
    Expected structure:
        data/
        â”œâ”€â”€ train/
        â”‚   â”œâ”€â”€ image1.jpg
        â”‚   â”œâ”€â”€ image2.jpg
        â”‚   â””â”€â”€ ...
        â””â”€â”€ val/
            â”œâ”€â”€ image1.jpg
            â””â”€â”€ ...
    """
    def __init__(self, config, data_dir, split='train'):
        self.config = config
        self.split = split
        self.data_dir = data_dir
        
        # Get all image paths
        self.image_paths = []
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            self.image_paths.extend(
                [os.path.join(data_dir, f) for f in os.listdir(data_dir) 
                 if f.endswith(ext)]
            )
        
        self.image_paths = sorted(self.image_paths)
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {data_dir}")
        
        # LIMIT SAMPLES FOR FAST TRAINING
        original_count = len(self.image_paths)
        if split == 'train' and hasattr(config, 'max_train_samples'):
            if len(self.image_paths) > config.max_train_samples:
                # Randomly sample images
                random.seed(42)  # For reproducibility
                self.image_paths = random.sample(self.image_paths, config.max_train_samples)
                print(f"âš¡ Limited training samples: {original_count} â†’ {len(self.image_paths)}")
        
        elif split == 'val' and hasattr(config, 'max_val_samples'):
            if len(self.image_paths) > config.max_val_samples:
                # Randomly sample images
                random.seed(42)  # For reproducibility
                self.image_paths = random.sample(self.image_paths, config.max_val_samples)
                print(f"âš¡ Limited validation samples: {original_count} â†’ {len(self.image_paths)}")
        
        # Image transforms
        if split == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((config.image_size, config.image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((config.image_size, config.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        
        print(f"{split.upper()} dataset: {len(self.image_paths)} images from {data_dir}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def generate_mask(self, image_size):
        """
        Generate irregular mask using random walk
        """
        h, w = image_size
        mask = torch.zeros(1, h, w)
        
        # Random mask parameters
        mask_ratio = random.uniform(*self.config.mask_ratio_range)
        target_area = int(mask_ratio * h * w)
        
        # Generate irregular mask with multiple strokes
        num_strokes = random.randint(3, 8)
        current_area = 0
        
        for _ in range(num_strokes):
            if current_area >= target_area:
                break
            
            # Random starting point
            start_x = random.randint(0, w - 1)
            start_y = random.randint(0, h - 1)
            
            # Random stroke parameters
            stroke_length = random.randint(20, min(h, w) // 2)
            stroke_width = random.randint(8, 20)
            
            # Create stroke using random walk
            x, y = start_x, start_y
            for step in range(stroke_length):
                # Create circular brush
                for dy in range(-stroke_width, stroke_width + 1):
                    for dx in range(-stroke_width, stroke_width + 1):
                        if dx*dx + dy*dy <= stroke_width*stroke_width:
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < h and 0 <= nx < w:
                                if mask[0, ny, nx] == 0:
                                    mask[0, ny, nx] = 1
                                    current_area += 1
                
                # Random walk
                angle = random.uniform(0, 2 * np.pi)
                x = int(np.clip(x + stroke_width * np.cos(angle), 0, w - 1))
                y = int(np.clip(y + stroke_width * np.sin(angle), 0, h - 1))
        
        return mask
    
    def __getitem__(self, idx):
        # Load image
        image_path = self.image_paths[idx]
        
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            return self.__getitem__((idx + 1) % len(self))
        
        # Transform
        image = self.transform(image)
        
        # Generate mask
        mask = self.generate_mask(image.shape[-2:])
        
        # Apply mask
        masked_image = image * (1 - mask)
        
        return {
            'original_image': image,
            'masked_image': masked_image,
            'mask': mask,
            'image_id': idx
        }


def create_dataloaders(config, data_root):
    """
    Create train and validation dataloaders
    
    Args:
        config: configuration object
        data_root: root directory containing train/ and val/ folders
    """
    train_dir = os.path.join(data_root, 'train')
    val_dir = os.path.join(data_root, 'val')
    
    # Verify directories exist
    if not os.path.exists(train_dir):
        raise ValueError(f"Training directory not found: {train_dir}")
    if not os.path.exists(val_dir):
        raise ValueError(f"Validation directory not found: {val_dir}")
    
    # Create datasets
    train_dataset = InpaintingDataset(config, train_dir, split='train')
    val_dataset = InpaintingDataset(config, val_dir, split='val')
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=False
    )
    
    print(f"\n{'='*60}")
    print(f"Dataloaders created:")
    print(f"  Train batches: {len(train_loader)} (batch_size={config.batch_size})")
    print(f"  Val batches: {len(val_loader)} (batch_size={config.batch_size})")
    print(f"  Total train images: {len(train_dataset)}")
    print(f"  Total val images: {len(val_dataset)}")
    print(f"{'='*60}\n")
    
    return train_loader, val_loader