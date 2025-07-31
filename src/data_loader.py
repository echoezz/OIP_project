# data_loader.py - TensorFlow Version
import tensorflow as tf
import os
import numpy as np
from pathlib import Path

def create_data_generators(data_dir, batch_size=32, validation_split=0.2, image_size=(224, 224)):
    """Create TensorFlow data generators with augmentation"""
    
    print("📊 Creating TensorFlow data generators...")
    
    # Training data generator with augmentation
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        brightness_range=[0.8, 1.2],
        validation_split=validation_split
    )
    
    # Validation data generator (no augmentation)
    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        validation_split=validation_split
    )
    
    # Create training dataset
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True,
        seed=42
    )
    
    # Create validation dataset
    val_generator = val_datagen.flow_from_directory(
        data_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False,
        seed=42
    )
    
    # Get class information
    classes = list(train_generator.class_indices.keys())
    class_indices = train_generator.class_indices
    
    print(f"✅ Data generators created:")
    print(f"   📈 Training samples: {train_generator.samples}")
    print(f"   📊 Validation samples: {val_generator.samples}")
    print(f"   🏷️  Classes: {len(classes)}")
    print(f"   📋 Class names: {classes}")
    
    return train_generator, val_generator, classes

def get_balanced_data_loaders(data_dir, batch_size=32, validation_split=0.2, auto_balance=True):
    """Get balanced TensorFlow data loaders (compatible with your existing code)"""
    
    print("🔄 Loading balanced TensorFlow datasets...")
    
    # Print dataset statistics
    total_samples = 0
    class_counts = {}
    
    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)
        if os.path.isdir(class_path):
            count = len([f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            class_counts[class_name] = count
            total_samples += count
    
    print(f"📊 DATASET SUMMARY:")
    print(f"   Classes: {len(class_counts)}")
    print(f"   Total samples: {total_samples}")
    
    print(f"\n📈 CLASS DISTRIBUTION:")
    for class_name, count in class_counts.items():
        print(f"   {class_name}: {count} samples")
    
    # Create data generators
    train_gen, val_gen, classes = create_data_generators(
        data_dir, batch_size, validation_split
    )
    
    print(f"\n📊 FINAL LOADER STATISTICS:")
    print(f"   Training samples: {train_gen.samples}")
    print(f"   Validation samples: {val_gen.samples}")
    print(f"   Classes: {len(classes)}")
    print(f"   Batch size: {batch_size}")
    
    return train_gen, val_gen, classes