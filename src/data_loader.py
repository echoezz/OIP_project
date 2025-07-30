import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
import os
import json
from typing import Tuple, List
import random
import numpy as np

class OriginalDataset(Dataset):
    """Original dataset loader that keeps images as PIL Images"""
    
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.images = []
        self.labels = []
        self.classes = []
        
        # Load class names
        for class_name in sorted(os.listdir(data_dir)):
            class_path = os.path.join(data_dir, class_name)
            if os.path.isdir(class_path):
                self.classes.append(class_name)
        
        # Load images and labels
        for class_idx, class_name in enumerate(self.classes):
            class_path = os.path.join(data_dir, class_name)
            for img_name in os.listdir(class_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_path, img_name)
                    self.images.append(img_path)
                    self.labels.append(class_idx)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Return PIL Image (not transformed)
        image = Image.open(img_path).convert('RGB')
        
        return image, label

class AugmentedDataset(Dataset):
    """Dataset wrapper that applies augmentation to training data"""
    
    def __init__(self, original_dataset, augmentation_factor=15, is_training=True):
        self.original_dataset = original_dataset
        self.augmentation_factor = augmentation_factor if is_training else 1
        self.is_training = is_training
        
        # Heavy augmentation for training
        self.heavy_augment = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224, padding=32, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(45),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), scale=(0.8, 1.2)),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            transforms.RandomGrayscale(p=0.1),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Light augmentation for training
        self.light_augment = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # No augmentation for validation
        self.val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.original_dataset) * self.augmentation_factor
    
    def __getitem__(self, idx):
        original_idx = idx // self.augmentation_factor
        augmentation_type = idx % self.augmentation_factor
        
        # Get PIL image and label from original dataset
        image, label = self.original_dataset[idx // self.augmentation_factor]
        
        # Apply appropriate transform based on training/validation and augmentation type
        if not self.is_training:
            # Validation - no augmentation
            transformed_image = self.val_transform(image)
        elif augmentation_type < 5:
            # Heavy augmentation for first 5 versions
            transformed_image = self.heavy_augment(image)
        else:
            # Light augmentation for remaining versions
            transformed_image = self.light_augment(image)
        
        return transformed_image, label

class CustomSubset(Dataset):
    """Custom subset that preserves PIL Images for augmentation"""
    
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

class PestDataLoader:
    def __init__(self, data_dir: str, batch_size: int = 16, img_size: int = 224, 
                 augmentation_factor: int = 15, validation_split: float = 0.2):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.augmentation_factor = augmentation_factor
        self.validation_split = validation_split
    
    def get_data_loaders(self) -> Tuple[DataLoader, DataLoader, List[str]]:
        """Get train and validation data loaders"""
        
        # Create original dataset (returns PIL Images)
        original_dataset = OriginalDataset(self.data_dir)
        classes = original_dataset.classes
        
        # Split into train and validation indices
        dataset_size = len(original_dataset)
        indices = list(range(dataset_size))
        np.random.shuffle(indices)
        
        val_size = int(self.validation_split * dataset_size)
        train_indices = indices[val_size:]
        val_indices = indices[:val_size]
        
        # Create custom subsets that preserve PIL Images
        train_subset = CustomSubset(original_dataset, train_indices)
        val_subset = CustomSubset(original_dataset, val_indices)
        
        # Create augmented datasets
        augmented_train_dataset = AugmentedDataset(
            train_subset, 
            augmentation_factor=self.augmentation_factor,
            is_training=True
        )
        
        augmented_val_dataset = AugmentedDataset(
            val_subset,
            augmentation_factor=1,  # No augmentation for validation
            is_training=False
        )
        
        # Create data loaders with reduced num_workers for Windows
        train_loader = DataLoader(
            augmented_train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,  # Set to 0 for Windows compatibility
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            augmented_val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,  # Set to 0 for Windows compatibility
            pin_memory=True
        )
        
        print(f"📊 Dataset Statistics:")
        print(f"   Original training images: {len(train_indices)}")
        print(f"   Augmented training samples: {len(augmented_train_dataset)}")
        print(f"   Validation images: {len(val_indices)}")
        print(f"   Classes: {len(classes)}")
        print(f"   Augmentation factor: {self.augmentation_factor}x")
        
        return train_loader, val_loader, classes
    
    def analyze_dataset_balance(self):
        """Analyze class distribution in dataset"""
        class_counts = {}
        
        for class_name in os.listdir(self.data_dir):
            class_path = os.path.join(self.data_dir, class_name)
            if os.path.isdir(class_path):
                count = len([f for f in os.listdir(class_path) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                class_counts[class_name] = count
        
        print("📊 Class Distribution:")
        total_images = sum(class_counts.values())
        for class_name, count in sorted(class_counts.items()):
            percentage = (count / total_images) * 100
            print(f"   {class_name}: {count} images ({percentage:.1f}%)")
        
        # Check for imbalanced classes
        counts = list(class_counts.values())
        if max(counts) / min(counts) > 3:
            print("⚠️  Warning: Significant class imbalance detected!")
            print("   Consider collecting more data for underrepresented classes")
        
        return class_counts

# Simple data loader function for basic usage
def get_simple_data_loaders(data_dir: str, batch_size: int = 16, validation_split: float = 0.2):
    """Simplified data loader without heavy augmentation"""
    
    # Simple transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets with transforms
    class SimpleDataset(Dataset):
        def __init__(self, data_dir, transform=None):
            self.data_dir = data_dir
            self.transform = transform
            self.images = []
            self.labels = []
            self.classes = []
            
            for class_name in sorted(os.listdir(data_dir)):
                class_path = os.path.join(data_dir, class_name)
                if os.path.isdir(class_path):
                    self.classes.append(class_name)
            
            for class_idx, class_name in enumerate(self.classes):
                class_path = os.path.join(data_dir, class_name)
                for img_name in os.listdir(class_path):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(class_path, img_name)
                        self.images.append(img_path)
                        self.labels.append(class_idx)
        
        def __len__(self):
            return len(self.images)
        
        def __getitem__(self, idx):
            img_path = self.images[idx]
            label = self.labels[idx]
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            return image, label
    
    # Create dataset
    full_dataset = SimpleDataset(data_dir)
    classes = full_dataset.classes
    
    # Split dataset
    dataset_size = len(full_dataset)
    val_size = int(validation_split * dataset_size)
    train_size = dataset_size - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # Apply transforms
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    
    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader, classes