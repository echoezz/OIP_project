import os
from src.data_loader import get_balanced_data_loaders

def setup_jpg_dataset():
    """Quick setup and analysis of your JPG dataset"""
    
    print("🌱 Setting up JPG Pest Dataset")
    print("=" * 50)
    
    dataset_path = "datasets"
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset directory '{dataset_path}' not found!")
        print("Please ensure your dataset is in the 'datasets' folder")
        return False
    
    try:
        # UPDATED: Direct call to get_balanced_data_loaders (it returns the loaders directly)
        train_loader, val_loader, classes = get_balanced_data_loaders(
            data_dir=dataset_path, 
            batch_size=16,
            auto_balance=True  # Enable class balancing
        )
        
        print(f"\n✅ Setup complete!")
        print(f"📋 Classes found: {len(classes)} classes")
        print(f"📂 Class names: {classes}")
        
        # Show training data stats
        print(f"\n📊 Data Statistics:")
        print(f"   Training batches: {len(train_loader)}")
        print(f"   Validation batches: {len(val_loader)}")
        
        # Test loading a batch
        print(f"\n🧪 Testing data loading...")
        sample_batch = next(iter(train_loader))
        images, labels = sample_batch
        print(f"✅ Batch test successful!")
        print(f"   Image batch shape: {images.shape}")
        print(f"   Label batch shape: {labels.shape}")
        print(f"   Image dtype: {images.dtype}")
        print(f"   Label dtype: {labels.dtype}")
        
        # Check for class balance in a few batches
        print(f"\n⚖️  Checking class balance in first 5 batches...")
        class_counts = [0] * len(classes)
        
        for i, (images, labels) in enumerate(train_loader):
            if i >= 5:  # Only check first 5 batches
                break
            for label in labels:
                class_counts[label.item()] += 1
        
        print(f"   Sample distribution: {class_counts}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during setup: {e}")
        print(f"💡 Make sure:")
        print(f"   • 'datasets' folder exists")
        print(f"   • Subfolders contain .jpg/.jpeg/.png images")
        print(f"   • data_loader.py is in src/ directory")
        return False

def analyze_dataset_structure():
    """Analyze the structure of your dataset"""
    
    print("\n🔍 ANALYZING DATASET STRUCTURE")
    print("=" * 50)
    
    dataset_path = "datasets"
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset path '{dataset_path}' not found!")
        return
    
    total_images = 0
    class_info = {}
    
    for class_name in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_name)
        
        if os.path.isdir(class_path):
            image_files = [f for f in os.listdir(class_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            if len(image_files) > 0:
                class_info[class_name] = len(image_files)
                total_images += len(image_files)
                print(f"   📁 {class_name}: {len(image_files)} images")
    
    print(f"\n📊 SUMMARY:")
    print(f"   Total classes: {len(class_info)}")
    print(f"   Total images: {total_images}")
    
    if class_info:
        min_images = min(class_info.values())
        max_images = max(class_info.values())
        avg_images = total_images / len(class_info)
        
        print(f"   Min per class: {min_images}")
        print(f"   Max per class: {max_images}")
        print(f"   Avg per class: {avg_images:.1f}")
        
        imbalance_ratio = max_images / min_images if min_images > 0 else float('inf')
        print(f"   Imbalance ratio: {imbalance_ratio:.1f}:1")
        
        if imbalance_ratio > 5:
            print(f"   ⚠️  SEVERE imbalance - balanced augmentation will help!")
        elif imbalance_ratio > 2:
            print(f"   📈 MODERATE imbalance - some balancing needed")
        else:
            print(f"   ✅ WELL BALANCED dataset")

if __name__ == "__main__":
    # Run dataset setup
    success = setup_jpg_dataset()
    
    if success:
        # Also analyze the raw dataset structure
        analyze_dataset_structure()