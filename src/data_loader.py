# data_loader.py - Heavy Augmentation Version
import tensorflow as tf
import os
import numpy as np

def create_data_generators(data_dir, batch_size=32, validation_split=0.2, image_size=(224, 224)):
    """Create heavily augmented data generators to prevent overfitting"""
    
    print("📊 Creating ANTI-OVERFITTING data generators...")
    
    # HEAVY augmentation for training
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,              # MORE rotation
        width_shift_range=0.2,          # MORE shifting
        height_shift_range=0.2,         # MORE shifting
        shear_range=0.1,                # ADD shear transformation
        zoom_range=0.2,                 # MORE zoom
        horizontal_flip=True,
        # vertical_flip=True,             # ADD vertical flip
        brightness_range=[0.8, 1.4],    # MORE brightness variation
        # channel_shift_range=30,         # ADD color shifting
        fill_mode='nearest',
        validation_split=validation_split
    )
    
    # Clean validation data (no augmentation)
    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        validation_split=validation_split
    )
    
    # Create generators
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True,
        seed=42
    )
    
    val_generator = val_datagen.flow_from_directory(
        data_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False,
        seed=42
    )
    
    classes = list(train_generator.class_indices.keys())
    
    print(f"✅ HEAVY AUGMENTATION applied:")
    print(f"   🔄 Rotation: ±40°")
    print(f"   📐 Shear: ±30%")
    print(f"   🔍 Zoom: ±30%")
    print(f"   💡 Brightness: 60-140%")
    print(f"   🎨 Color shift: ±30")
    print(f"   ↔️ Horizontal flip: Yes")
    print(f"   ↕️ Vertical flip: Yes")
    print(f"   📈 Training samples: {train_generator.samples}")
    print(f"   📊 Validation samples: {val_generator.samples}")
    
    return train_generator, val_generator, classes

def get_balanced_data_loaders(data_dir, batch_size=32, validation_split=0.2, auto_balance=True):
    """Get anti-overfitting data loaders"""
    
    print("🛡️ Loading ANTI-OVERFITTING datasets...")
    
    # Print dataset stats
    class_counts = {}
    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)
        if os.path.isdir(class_path):
            count = len([f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            class_counts[class_name] = count
    
    print(f"📊 DATASET SUMMARY:")
    for class_name, count in class_counts.items():
        print(f"   {class_name}: {count} samples")
    
    # Create generators with heavy augmentation
    train_gen, val_gen, classes = create_data_generators(
        data_dir, batch_size, validation_split
    )
    
    return train_gen, val_gen, classes