import os
from src.data_loader import PestDataLoader

def setup_jpg_dataset():
    """Quick setup and analysis of your JPG dataset"""
    
    print("🌱 Setting up JPG Pest Dataset")
    print("=" * 50)
    
    dataset_path = "datasets"
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset directory '{dataset_path}' not found!")
        print("Please ensure your dataset is in the 'dataset' folder")
        return False
    
    # Initialize data loader
    data_loader = PestDataLoader(dataset_path, batch_size=16)
    
    try:
        # This will analyze and create train/val split
        train_loader, val_loader, classes = data_loader.get_data_loaders()
        
        print(f"\n✅ Setup complete!")
        print(f"📋 Classes found: {classes}")
        
        # Test loading a batch
        sample_batch = next(iter(train_loader))
        images, labels = sample_batch
        print(f"🔧 Batch test successful: {images.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during setup: {e}")
        return False

if __name__ == "__main__":
    setup_jpg_dataset()