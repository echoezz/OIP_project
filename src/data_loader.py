import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
import os
import json
from typing import Tuple, List, Dict
import random
import numpy as np
from collections import Counter
import math

class BalancedAugmentationDataset(Dataset):
    """Dataset that automatically balances classes through targeted augmentation"""
    
    def __init__(self, data_dir: str, target_samples_per_class: int = None, 
                 min_augmentation_factor: int = 3, max_augmentation_factor: int = 20):
        self.data_dir = data_dir
        self.min_augmentation_factor = min_augmentation_factor
        self.max_augmentation_factor = max_augmentation_factor
        
        # Load original data
        self.classes = []
        self.class_to_idx = {}
        self.original_samples = []  # [(image_path, class_idx)]
        self.class_counts = {}
        
        # Scan directories and count samples per class
        self._scan_dataset()
        
        # Calculate target samples per class
        if target_samples_per_class is None:
            # Use the count of the most populous class
            target_samples_per_class = max(self.class_counts.values())
        
        self.target_samples_per_class = target_samples_per_class
        
        # Calculate augmentation factors for each class
        self.class_augmentation_factors = self._calculate_augmentation_factors()
        
        # Generate balanced samples list
        self.balanced_samples = self._generate_balanced_samples()
        
        # Define augmentation transforms
        self._setup_transforms()
        
        self._print_balance_info()
    
    def _scan_dataset(self):
        """Scan dataset and build class information"""
        print("🔍 Scanning dataset for class balance analysis...")
        
        for class_name in sorted(os.listdir(self.data_dir)):
            class_path = os.path.join(self.data_dir, class_name)
            if os.path.isdir(class_path):
                image_files = [f for f in os.listdir(class_path) 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                
                if len(image_files) > 0:
                    class_idx = len(self.classes)
                    self.classes.append(class_name)
                    self.class_to_idx[class_name] = class_idx
                    self.class_counts[class_name] = len(image_files)
                    
                    # Add all images for this class
                    for img_file in image_files:
                        img_path = os.path.join(class_path, img_file)
                        self.original_samples.append((img_path, class_idx))
    
    def _calculate_augmentation_factors(self) -> Dict[str, int]:
        """Calculate how much to augment each class"""
        augmentation_factors = {}
        
        print(f"\n📊 Calculating augmentation factors (target: {self.target_samples_per_class} samples/class):")
        
        for class_name, count in self.class_counts.items():
            if count == 0:
                continue
                
            # Calculate needed augmentation factor
            needed_factor = math.ceil(self.target_samples_per_class / count)
            
            # Clamp to reasonable bounds
            augmentation_factor = max(self.min_augmentation_factor, 
                                    min(needed_factor, self.max_augmentation_factor))
            
            augmentation_factors[class_name] = augmentation_factor
            
            final_samples = count * augmentation_factor
            print(f"   📈 {class_name}: {count} → {final_samples} samples (×{augmentation_factor})")
        
        return augmentation_factors
    
    def _generate_balanced_samples(self) -> List[Tuple[str, int, int]]:
        """Generate balanced sample list: (image_path, class_idx, augmentation_type)"""
        balanced_samples = []
        
        for img_path, class_idx in self.original_samples:
            class_name = self.classes[class_idx]
            augmentation_factor = self.class_augmentation_factors[class_name]
            
            # Generate multiple augmented versions of each image
            for aug_type in range(augmentation_factor):
                balanced_samples.append((img_path, class_idx, aug_type))
        
        return balanced_samples
    
    def _setup_transforms(self):
        """Setup different levels of augmentation"""
        
        # HEAVY augmentation for severely underrepresented classes
        self.heavy_augment = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224, padding=32, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(p=0.7),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(60),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.3),
            transforms.RandomAffine(degrees=0, translate=(0.3, 0.3), scale=(0.7, 1.3), shear=15),
            transforms.RandomPerspective(distortion_scale=0.3, p=0.5),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 3.0)),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.15)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # MEDIUM augmentation for moderately underrepresented classes
        self.medium_augment = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224, padding=16, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), scale=(0.8, 1.2)),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # LIGHT augmentation for well-represented classes
        self.light_augment = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # NO augmentation (original image)
        self.no_augment = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _print_balance_info(self):
        """Print class balance information"""
        print(f"\n📊 BALANCED DATASET SUMMARY:")
        print(f"   Classes: {len(self.classes)}")
        print(f"   Original samples: {len(self.original_samples)}")
        print(f"   Balanced samples: {len(self.balanced_samples)}")
        print(f"   Target per class: {self.target_samples_per_class}")
        
        # Calculate final distribution
        final_distribution = Counter()
        for _, class_idx, _ in self.balanced_samples:
            final_distribution[self.classes[class_idx]] += 1
        
        print(f"\n📈 FINAL CLASS DISTRIBUTION:")
        for class_name in self.classes:
            original_count = self.class_counts[class_name]
            final_count = final_distribution[class_name]
            improvement = final_count / original_count if original_count > 0 else 0
            print(f"   {class_name}: {original_count} → {final_count} (+{improvement:.1f}×)")
    
    def __len__(self):
        return len(self.balanced_samples)
    
    def __getitem__(self, idx):
        img_path, class_idx, aug_type = self.balanced_samples[idx]
        
        # Load PIL image
        image = Image.open(img_path).convert('RGB')
        
        # Choose augmentation based on class needs and augmentation type
        class_name = self.classes[class_idx]
        original_count = self.class_counts[class_name]
        augmentation_factor = self.class_augmentation_factors[class_name]
        
        # Determine augmentation intensity based on how underrepresented the class is
        if augmentation_factor >= 15:  # Severely underrepresented
            if aug_type == 0:  # Keep one original
                image = self.no_augment(image)
            else:
                image = self.heavy_augment(image)
        elif augmentation_factor >= 8:  # Moderately underrepresented
            if aug_type == 0:  # Keep one original
                image = self.no_augment(image)
            elif aug_type < augmentation_factor // 2:
                image = self.heavy_augment(image)
            else:
                image = self.medium_augment(image)
        elif augmentation_factor >= 4:  # Slightly underrepresented
            if aug_type == 0:  # Keep one original
                image = self.no_augment(image)
            else:
                image = self.medium_augment(image)
        else:  # Well represented
            if aug_type == 0:  # Keep one original
                image = self.no_augment(image)
            else:
                image = self.light_augment(image)
        
        return image, class_idx

class ValidationDataset(Dataset):
    """Simple validation dataset without augmentation"""
    
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.samples = []
        self.classes = []
        
        # Load data
        for class_name in sorted(os.listdir(data_dir)):
            class_path = os.path.join(data_dir, class_name)
            if os.path.isdir(class_path):
                image_files = [f for f in os.listdir(class_path) 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                
                if len(image_files) > 0:
                    class_idx = len(self.classes)
                    self.classes.append(class_name)
                    
                    for img_file in image_files:
                        img_path = os.path.join(class_path, img_file)
                        self.samples.append((img_path, class_idx))
        
        # Validation transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, class_idx = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        return image, class_idx

class BalancedPestDataLoader:
    """Data loader with automatic class balancing"""
    
    def __init__(self, data_dir: str, batch_size: int = 16, validation_split: float = 0.2,
                 target_samples_per_class: int = None, min_augmentation_factor: int = 3, 
                 max_augmentation_factor: int = 20):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.target_samples_per_class = target_samples_per_class
        self.min_augmentation_factor = min_augmentation_factor
        self.max_augmentation_factor = max_augmentation_factor
    
    def get_balanced_data_loaders(self) -> Tuple[DataLoader, DataLoader, List[str]]:
        """Get balanced train and validation data loaders"""
        
        print("🎯 CREATING BALANCED DATA LOADERS")
        print("=" * 60)
        
        # Create full balanced dataset
        full_dataset = BalancedAugmentationDataset(
            self.data_dir,
            target_samples_per_class=self.target_samples_per_class,
            min_augmentation_factor=self.min_augmentation_factor,
            max_augmentation_factor=self.max_augmentation_factor
        )
        
        classes = full_dataset.classes
        
        if len(classes) == 0:
            raise ValueError("No valid classes found!")
        
        # Split into train and validation by original images (not augmented samples)
        original_indices = list(range(len(full_dataset.original_samples)))
        np.random.shuffle(original_indices)
        
        val_size = int(self.validation_split * len(original_indices))
        train_original_indices = set(original_indices[val_size:])
        val_original_indices = set(original_indices[:val_size])
        
        # Create train samples (balanced augmented)
        train_samples = []
        val_samples = []
        
        for i, (img_path, class_idx, aug_type) in enumerate(full_dataset.balanced_samples):
            # Find which original sample this augmented sample came from
            original_idx = None
            for j, (orig_path, orig_class_idx) in enumerate(full_dataset.original_samples):
                if orig_path == img_path and orig_class_idx == class_idx:
                    original_idx = j
                    break
            
            if original_idx in train_original_indices:
                train_samples.append(i)
            elif original_idx in val_original_indices:
                val_samples.append(i)
        
        # Create train subset (balanced)
        train_dataset = Subset(full_dataset, train_samples)
        
        # Create simple validation dataset (no augmentation)
        val_dataset = ValidationDataset(self.data_dir)
        val_indices = [i for i, (_, class_idx) in enumerate(val_dataset.samples) 
                      if any(val_dataset.samples[i][0] == full_dataset.original_samples[j][0] 
                            for j in val_original_indices)]
        val_dataset = Subset(val_dataset, val_indices)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        
        print(f"\n📊 FINAL LOADER STATISTICS:")
        print(f"   Training samples: {len(train_samples)}")
        print(f"   Validation samples: {len(val_dataset)}")
        print(f"   Classes: {len(classes)}")
        print(f"   Batch size: {self.batch_size}")
        
        return train_loader, val_loader, classes
    
    def analyze_class_balance(self):
        """Analyze original vs balanced class distribution"""
        print("🔍 ANALYZING CLASS BALANCE")
        print("=" * 60)
        
        # Original distribution
        original_counts = {}
        for class_name in os.listdir(self.data_dir):
            class_path = os.path.join(self.data_dir, class_name)
            if os.path.isdir(class_path):
                count = len([f for f in os.listdir(class_path) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                if count > 0:
                    original_counts[class_name] = count
        
        if not original_counts:
            print("❌ No images found!")
            return
        
        print("📊 ORIGINAL DISTRIBUTION:")
        total_original = sum(original_counts.values())
        min_count = min(original_counts.values())
        max_count = max(original_counts.values())
        
        for class_name, count in sorted(original_counts.items()):
            percentage = (count / total_original) * 100
            print(f"   {class_name}: {count} images ({percentage:.1f}%)")
        
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        print(f"\n⚠️  Imbalance ratio: {imbalance_ratio:.1f}:1 (max/min)")
        
        if imbalance_ratio > 5:
            print("🚨 SEVERE imbalance detected! Balanced augmentation will help significantly.")
        elif imbalance_ratio > 2:
            print("⚠️  MODERATE imbalance detected. Balanced augmentation recommended.")
        else:
            print("✅ Dataset is relatively balanced.")
        
        return original_counts

# Usage function
def get_balanced_data_loaders(data_dir: str, batch_size: int = 16, validation_split: float = 0.2,
                             target_samples_per_class: int = None, auto_balance: bool = True):
    """
    Get balanced data loaders that automatically augment underrepresented classes
    
    Args:
        data_dir: Path to dataset directory
        batch_size: Batch size for training
        validation_split: Fraction of data for validation
        target_samples_per_class: Target number of samples per class (None = use max class count)
        auto_balance: Whether to apply automatic class balancing
    """
    
    if auto_balance:
        print("🎯 Using BALANCED data loader with adaptive augmentation")
        loader = BalancedPestDataLoader(
            data_dir=data_dir,
            batch_size=batch_size,
            validation_split=validation_split,
            target_samples_per_class=target_samples_per_class
        )
        
        # Analyze balance first
        loader.analyze_class_balance()
        
        # Get balanced loaders
        return loader.get_balanced_data_loaders()
    else:
        print("📊 Using STANDARD data loader (no balancing)")
        # Fall back to your original simple loader
        return get_simple_data_loaders(data_dir, batch_size, validation_split)

# Example usage
if __name__ == "__main__":
    print("🎯 BALANCED PEST DATA LOADER TEST")
    print("=" * 60)
    
    # Test the balanced data loader
    try:
        train_loader, val_loader, classes = get_balanced_data_loaders(
            data_dir="datasets",
            batch_size=16,
            target_samples_per_class=200,  # Target 200 samples per class
            auto_balance=True
        )
        
        print(f"\n✅ Success! Created balanced data loaders:")
        print(f"   Train batches: {len(train_loader)}")
        print(f"   Val batches: {len(val_loader)}")
        print(f"   Classes: {classes}")
        
        # Test a batch
        for batch_idx, (images, labels) in enumerate(train_loader):
            print(f"\n📦 Sample batch {batch_idx + 1}:")
            print(f"   Images shape: {images.shape}")
            print(f"   Labels: {labels}")
            if batch_idx == 0:  # Only show first batch
                break
                
    except Exception as e:
        print(f"❌ Error: {e}")