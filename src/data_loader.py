import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
from PIL import Image
import shutil
from sklearn.model_selection import train_test_split

class JPGImageFolder(datasets.ImageFolder):
    """Custom ImageFolder that only loads JPG files"""
    def __init__(self, root, transform=None):
        super(JPGImageFolder, self).__init__(root, transform)
        # Filter to only include JPG files
        self.samples = [(path, class_idx) for path, class_idx in self.samples 
                       if path.lower().endswith(('.jpg', '.jpeg'))]
        self.imgs = self.samples

class PestDataLoader:
    def __init__(self, data_dir, batch_size=32, img_size=224):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.img_size = img_size
        
        # Standard transform for validation/testing
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Enhanced augmentation for JPG training images
        self.augment_transform = transforms.Compose([
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.RandomCrop((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=20),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def analyze_dataset(self):
        """Analyze your JPG dataset structure"""
        print("🔍 Analyzing JPG dataset...")
        
        # Get all directories (classes)
        all_items = os.listdir(self.data_dir)
        class_dirs = [d for d in all_items 
                     if os.path.isdir(os.path.join(self.data_dir, d)) 
                     and d not in ['train', 'val', 'test']]
        
        total_images = 0
        class_counts = {}
        
        for class_name in class_dirs:
            class_path = os.path.join(self.data_dir, class_name)
            jpg_files = [f for f in os.listdir(class_path) 
                        if f.lower().endswith(('.jpg', '.jpeg'))]
            
            class_counts[class_name] = len(jpg_files)
            total_images += len(jpg_files)
            
            print(f"  📁 {class_name}: {len(jpg_files)} JPG images")
        
        print(f"\n📊 Total: {total_images} JPG images across {len(class_dirs)} classes")
        return class_counts, total_images
    
    def create_train_val_split(self, train_ratio=0.8):
        """Create train/validation split from your JPG dataset"""
        print("\n🔄 Creating train/validation split...")
        
        # Check if split already exists
        if os.path.exists(os.path.join(self.data_dir, 'train')) and \
           os.path.exists(os.path.join(self.data_dir, 'val')):
            print("✅ Train/validation split already exists!")
            return
        
        # Get class directories
        class_dirs = [d for d in os.listdir(self.data_dir) 
                     if os.path.isdir(os.path.join(self.data_dir, d)) 
                     and d not in ['train', 'val', 'test']]
        
        train_dir = os.path.join(self.data_dir, 'train')
        val_dir = os.path.join(self.data_dir, 'val')
        
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        
        for class_name in class_dirs:
            class_path = os.path.join(self.data_dir, class_name)
            
            # Get all JPG files
            jpg_files = [f for f in os.listdir(class_path) 
                        if f.lower().endswith(('.jpg', '.jpeg'))]
            
            if len(jpg_files) < 2:
                print(f"⚠️  Warning: {class_name} has only {len(jpg_files)} image(s)")
                continue
            
            # Split files
            train_files, val_files = train_test_split(
                jpg_files, 
                train_size=train_ratio, 
                random_state=42,
                stratify=None
            )
            
            # Create class directories
            train_class_dir = os.path.join(train_dir, class_name)
            val_class_dir = os.path.join(val_dir, class_name)
            os.makedirs(train_class_dir, exist_ok=True)
            os.makedirs(val_class_dir, exist_ok=True)
            
            # Copy files
            for file in train_files:
                src = os.path.join(class_path, file)
                dst = os.path.join(train_class_dir, file)
                shutil.copy2(src, dst)
            
            for file in val_files:
                src = os.path.join(class_path, file)
                dst = os.path.join(val_class_dir, file)
                shutil.copy2(src, dst)
            
            print(f"  ✅ {class_name}: {len(train_files)} train, {len(val_files)} val")
    
    def get_data_loaders(self):
        """Get train and validation data loaders for JPG images"""
        
        # Analyze dataset first
        self.analyze_dataset()
        
        # Create train/val split if needed
        self.create_train_val_split()
        
        # Create datasets
        train_dataset = JPGImageFolder(
            os.path.join(self.data_dir, 'train'),
            transform=self.augment_transform
        )
        
        val_dataset = JPGImageFolder(
            os.path.join(self.data_dir, 'val'),
            transform=self.transform
        )
        
        print(f"\n📚 Dataset loaded:")
        print(f"  🎯 Training samples: {len(train_dataset)}")
        print(f"  🎯 Validation samples: {len(val_dataset)}")
        print(f"  🏷️  Classes: {len(train_dataset.classes)}")
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=2,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=2,
            pin_memory=True
        )
        
        return train_loader, val_loader, train_dataset.classes