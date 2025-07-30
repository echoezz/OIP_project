import tensorflow as tf
import os
import json
import time
import numpy as np
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks

# ===== SMART GPU/CPU DETECTION AND SETUP =====
print("🔍 Smart GPU/CPU Detection and Setup")
print("=" * 60)

# Hide TensorFlow info messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# Initialize device configuration
DEVICE_CONFIG = {
    'device_type': 'CPU',
    'device_name': '/CPU:0',
    'batch_size': 8,
    'mixed_precision': False,
    'memory_optimized': True
}

print(f"📦 TensorFlow version: {tf.__version__}")

# Step 1: Detect GPUs
gpus = tf.config.list_physical_devices('GPU')
print(f"🎮 GPU devices found: {len(gpus)}")

if gpus:
    print("✅ GPU(s) detected:")
    for i, gpu in enumerate(gpus):
        print(f"   GPU {i}: {gpu}")
    
    try:
        # Step 2: Configure GPU settings
        print("🔧 Configuring GPU settings...")
        
        # Enable memory growth (prevents TensorFlow from allocating all GPU memory)
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        
        # Set visible devices
        tf.config.set_visible_devices(gpus, 'GPU')
        
        # Test GPU functionality
        print("🧪 Testing GPU functionality...")
        with tf.device('/GPU:0'):
            test_tensor = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            test_result = tf.matmul(test_tensor, test_tensor)
            
        print(f"✅ GPU test successful: {test_result.device}")
        
        # Update device configuration for GPU
        DEVICE_CONFIG.update({
            'device_type': 'GPU',
            'device_name': '/GPU:0',
            'batch_size': 16,
            'mixed_precision': True,
            'memory_optimized': False
        })
        
        print("🚀 GPU setup completed successfully!")
        print(f"   📊 Device: {DEVICE_CONFIG['device_type']}")
        print(f"   🎯 Batch size: {DEVICE_CONFIG['batch_size']}")
        print(f"   ⚡ Mixed precision: {DEVICE_CONFIG['mixed_precision']}")
        
    except Exception as e:
        print(f"❌ GPU setup failed: {e}")
        print("🔄 Falling back to CPU configuration...")
        DEVICE_CONFIG.update({
            'device_type': 'CPU',
            'device_name': '/CPU:0',
            'batch_size': 8,
            'mixed_precision': False,
            'memory_optimized': True
        })
        
else:
    print("❌ No GPU detected")
    print("💡 Training will use CPU with optimized settings")
    print("   • Reduced batch size for memory efficiency")
    print("   • Disabled mixed precision")
    print("   • Memory optimizations enabled")

print(f"\n🎯 Final Device Configuration:")
print(f"   Device Type: {DEVICE_CONFIG['device_type']}")
print(f"   Device Path: {DEVICE_CONFIG['device_name']}")
print(f"   Batch Size: {DEVICE_CONFIG['batch_size']}")
print(f"   Mixed Precision: {DEVICE_CONFIG['mixed_precision']}")
print("=" * 60)
print()

# ===== DATASET UTILITIES =====
def clean_dataset_directory(data_dir):
    """Remove empty directories and return valid classes"""
    valid_classes = []
    removed_dirs = []
    
    print("🧹 Checking dataset directories...")
    
    if not os.path.exists(data_dir):
        print(f"❌ Dataset directory not found: {data_dir}")
        return []
    
    for item in os.listdir(data_dir):
        item_path = os.path.join(data_dir, item)
        
        if os.path.isdir(item_path):
            # Count image files
            image_files = [f for f in os.listdir(item_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            if len(image_files) == 0:
                print(f"   ⚠️  Empty directory: {item} (will be skipped)")
                removed_dirs.append(item)
            else:
                valid_classes.append(item)
                print(f"   ✅ {item}: {len(image_files)} images")
    
    if removed_dirs:
        print(f"📊 Skipped {len(removed_dirs)} empty directories")
    
    print(f"✅ Valid classes: {len(valid_classes)}")
    return valid_classes

def create_smart_data_generators(data_dir, device_config, img_size=224, validation_split=0.2):
    """Create data generators optimized for detected device"""
    
    valid_classes = clean_dataset_directory(data_dir)
    
    if len(valid_classes) == 0:
        raise ValueError("No valid classes with images found in dataset directory!")
    
    # Adjust augmentation based on device type
    if device_config['device_type'] == 'GPU':
        print("🎮 Creating GPU-optimized data generators...")
        # More aggressive augmentation for GPU (can handle it)
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            rotation_range=30,
            width_shift_range=0.15,
            height_shift_range=0.15,
            shear_range=0.15,
            zoom_range=0.15,
            horizontal_flip=True,
            vertical_flip=False,
            brightness_range=[0.8, 1.2],
            fill_mode='reflect',
            validation_split=validation_split
        )
    else:
        print("💻 Creating CPU-optimized data generators...")
        # Lighter augmentation for CPU (faster processing)
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            brightness_range=[0.9, 1.1],
            validation_split=validation_split
        )
    
    # Validation generator (same for both)
    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        validation_split=validation_split
    )
    
    # Create generators with device-optimized batch size
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(img_size, img_size),
        batch_size=device_config['batch_size'],
        class_mode='categorical',
        subset='training',
        shuffle=True,
        seed=42
    )
    
    val_generator = val_datagen.flow_from_directory(
        data_dir,
        target_size=(img_size, img_size),
        batch_size=device_config['batch_size'],
        class_mode='categorical',
        subset='validation',
        shuffle=False,
        seed=42
    )
    
    return train_generator, val_generator, valid_classes

# ===== MODEL ARCHITECTURE =====
def create_device_optimized_model(num_classes, device_config, img_size=224):
    """Create model optimized for detected device"""
    
    print(f"🧠 Creating model optimized for {device_config['device_type']}...")
    
    # Build model within device context
    with tf.device(device_config['device_name']):
        inputs = layers.Input(shape=(img_size, img_size, 3))
        
        # Initial conv block
        x = layers.Conv2D(32, 3, padding='same', use_bias=False)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D(2)(x)  # 224->112
        
        # Residual Block 1: 32 -> 64 channels
        identity = layers.Conv2D(64, 1, padding='same', use_bias=False)(x)
        identity = layers.BatchNormalization()(identity)
        
        x = layers.Conv2D(64, 3, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(64, 3, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        
        x = layers.Add()([x, identity])  # Residual connection
        x = layers.ReLU()(x)
        x = layers.Dropout(0.1)(x)
        x = layers.MaxPooling2D(2)(x)  # 112->56
        
        # Residual Block 2: 64 -> 128 channels
        identity = layers.Conv2D(128, 1, padding='same', use_bias=False)(x)
        identity = layers.BatchNormalization()(identity)
        
        x = layers.Conv2D(128, 3, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(128, 3, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        
        x = layers.Add()([x, identity])  # Residual connection
        x = layers.ReLU()(x)
        x = layers.Dropout(0.1)(x)
        x = layers.MaxPooling2D(2)(x)  # 56->28
        
        # Residual Block 3: 128 -> 256 channels
        identity = layers.Conv2D(256, 1, padding='same', use_bias=False)(x)
        identity = layers.BatchNormalization()(identity)
        
        x = layers.Conv2D(256, 3, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(256, 3, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        
        x = layers.Add()([x, identity])  # Residual connection
        x = layers.ReLU()(x)
        x = layers.Dropout(0.15)(x)
        x = layers.MaxPooling2D(2)(x)  # 28->14
        
        # Residual Block 4: 256 -> 512 channels
        identity = layers.Conv2D(512, 1, padding='same', use_bias=False)(x)
        identity = layers.BatchNormalization()(identity)
        
        x = layers.Conv2D(512, 3, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(512, 3, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        
        x = layers.Add()([x, identity])  # Residual connection
        x = layers.ReLU()(x)
        x = layers.Dropout(0.15)(x)
        
        # Attention mechanism (more effective on GPU)
        if device_config['device_type'] == 'GPU':
            # Full attention mechanism for GPU
            attention = layers.GlobalAveragePooling2D()(x)
            attention = layers.Dense(512//16, activation='relu')(attention)
            attention = layers.Dense(512, activation='sigmoid')(attention)
            attention = layers.Reshape((1, 1, 512))(attention)
            x = layers.Multiply()([x, attention])
        
        # Global pooling and classifier
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.5)(x)
        
        # Classifier layers (adjust complexity based on device)
        if device_config['device_type'] == 'GPU':
            # More complex classifier for GPU
            x = layers.Dense(256, activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.3)(x)
            x = layers.Dense(128, activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.2)(x)
        else:
            # Simpler classifier for CPU
            x = layers.Dense(128, activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.3)(x)
        
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        model = models.Model(inputs, outputs)
    
    return model

# ===== TRAINING UTILITIES =====
def create_device_optimized_callbacks(model_save_path, device_config, patience=15, min_epochs=20):
    """Create callbacks optimized for detected device"""
    
    # Adjust patience based on device (GPU trains faster, can be more patient)
    if device_config['device_type'] == 'GPU':
        patience = patience
        min_epochs = min_epochs
    else:
        patience = max(10, patience - 5)  # Less patience for CPU
        min_epochs = max(15, min_epochs - 5)
    
    callbacks_list = [
        # Save best model
        keras.callbacks.ModelCheckpoint(
            filepath=f'{model_save_path}/best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        ),
        
        # Early stopping
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=patience,
            restore_best_weights=True,
            verbose=1,
            start_from_epoch=min_epochs
        ),
        
        # Learning rate scheduling (different strategies for GPU/CPU)
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=8 if device_config['device_type'] == 'GPU' else 5,
            min_lr=1e-7,
            verbose=1
        ),
        
        # CSV logger
        keras.callbacks.CSVLogger(
            f'{model_save_path}/training_log.csv'
        )
    ]
    
    return callbacks_list

def calculate_class_weights(data_dir, classes):
    """Calculate class weights for imbalanced datasets"""
    
    print("⚖️  Calculating class weights...")
    
    class_counts = {}
    total_samples = 0
    
    for class_name in classes:
        class_path = os.path.join(data_dir, class_name)
        if os.path.exists(class_path) and os.path.isdir(class_path):
            count = len([f for f in os.listdir(class_path) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            if count > 0:
                class_counts[class_name] = count
                total_samples += count
    
    if not class_counts:
        print("⚠️  No valid classes found, using equal weights")
        return None
    
    class_weights = {}
    num_classes = len(class_counts)
    
    print("📊 Class distribution and weights:")
    for i, (class_name, count) in enumerate(class_counts.items()):
        weight = total_samples / (num_classes * count)
        class_weights[i] = weight
        percentage = (count / total_samples) * 100
        print(f"   {class_name}: {count} images ({percentage:.1f}%) -> weight: {weight:.2f}")
    
    return class_weights

# ===== MAIN TRAINING FUNCTION =====
def train_smart_cnn():
    """Smart training that adapts to available hardware"""
    
    print("🚀 Starting Smart Device-Adaptive CNN Training")
    print("=" * 60)
    
    # Configuration that adapts to device
    config = {
        'data_dir': 'datasets',
        'model_save_path': 'models/saved_models',
        'epochs': 100 if DEVICE_CONFIG['device_type'] == 'GPU' else 80,  # Fewer epochs for CPU
        'learning_rate': 0.001,
        'img_size': 224,
        'validation_split': 0.2,
        'patience': 15 if DEVICE_CONFIG['device_type'] == 'GPU' else 10,
        'min_epochs': 20 if DEVICE_CONFIG['device_type'] == 'GPU' else 15
    }
    
    # Add device config to main config
    config.update(DEVICE_CONFIG)
    
    # Create save directory
    os.makedirs(config['model_save_path'], exist_ok=True)
    
    print(f"🎯 Smart Training Configuration:")
    print(f"   Device: {config['device_type']} ({config['device_name']})")
    print(f"   Batch size: {config['batch_size']} (optimized for {config['device_type']})")
    print(f"   Epochs: {config['epochs']}")
    print(f"   Learning rate: {config['learning_rate']}")
    print(f"   Mixed precision: {config['mixed_precision']}")
    print(f"   Memory optimized: {config['memory_optimized']}")
    
    # Create device-optimized data generators
    print(f"\n📊 Creating {config['device_type']}-optimized data generators...")
    try:
        train_generator, val_generator, valid_classes = create_smart_data_generators(
            config['data_dir'],
            DEVICE_CONFIG,
            config['img_size'],
            config['validation_split']
        )
    except Exception as e:
        print(f"❌ Error creating data generators: {e}")
        print("💡 Make sure your dataset directory structure is correct:")
        print("   datasets/")
        print("   ├── class1/ (with .jpg/.png images)")
        print("   ├── class2/ (with .jpg/.png images)")
        print("   └── ...")
        return None, 0
    
    classes = valid_classes
    num_classes = len(classes)
    
    if num_classes == 0:
        print("❌ No valid classes found!")
        return None, 0
    
    print(f"\n📋 Dataset Information:")
    print(f"   Classes: {num_classes}")
    print(f"   Training samples: {train_generator.samples}")
    print(f"   Validation samples: {val_generator.samples}")
    print(f"   Classes: {classes}")
    
    # Calculate class weights
    class_weights = calculate_class_weights(config['data_dir'], classes)
    
    # Create device-optimized model
    print(f"\n🧠 Creating {config['device_type']}-optimized model...")
    
    # Enable mixed precision for GPU if supported
    if config['mixed_precision'] and config['device_type'] == 'GPU':
        print("⚡ Enabling mixed precision for GPU acceleration...")
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
    
    model = create_device_optimized_model(num_classes, DEVICE_CONFIG, config['img_size'])
    
    # Print model info
    total_params = model.count_params()
    print(f"📊 Model Architecture:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Model layers: {len(model.layers)}")
    print(f"   Optimized for: {config['device_type']}")
    print(f"   Features: Residual connections + {('Attention + ' if config['device_type'] == 'GPU' else '')}Enhanced classifier")
    
    # Compile model with device-optimized settings
    if config['device_type'] == 'GPU':
        # GPU: Use SGD with Nesterov for faster convergence
        optimizer = keras.optimizers.SGD(
            learning_rate=config['learning_rate'],
            momentum=0.9,
            nesterov=True
        )
    else:
        # CPU: Use Adam for more stable convergence
        optimizer = keras.optimizers.Adam(
            learning_rate=config['learning_rate'] * 0.8  # Slightly lower LR for CPU
        )
    
    # Compile with mixed precision considerations
    if config['mixed_precision']:
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
    else:
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
    
    # Create device-optimized callbacks
    callbacks_list = create_device_optimized_callbacks(
        config['model_save_path'],
        DEVICE_CONFIG,
        config['patience'],
        config['min_epochs']
    )
    
    print(f"\n🎓 Starting training on {config['device_type']}...")
    print(f"🔧 Optimizations: Device-adaptive architecture, Smart batching, Optimized callbacks")
    
    start_time = time.time()
    
    # Training with device context
    try:
        with tf.device(config['device_name']):
            history = model.fit(
                train_generator,
                epochs=config['epochs'],
                validation_data=val_generator,
                callbacks=callbacks_list,
                class_weight=class_weights,
                verbose=1
            )
    except Exception as e:
        print(f"❌ Training failed: {e}")
        if config['device_type'] == 'GPU':
            print("💡 GPU training failed - you could try running again with CPU fallback")
        return None, 0
    
    # Training completed
    total_time = time.time() - start_time
    
    print(f"\n🎉 SMART TRAINING COMPLETED!")
    print("=" * 60)
    
    # Get best metrics
    if 'val_accuracy' in history.history and len(history.history['val_accuracy']) > 0:
        best_val_acc = max(history.history['val_accuracy']) * 100
        best_epoch = history.history['val_accuracy'].index(max(history.history['val_accuracy'])) + 1
        final_epochs = len(history.history['val_accuracy'])
    else:
        print("⚠️  No validation accuracy recorded")
        best_val_acc = 0
        best_epoch = 0
        final_epochs = 0
    
    print(f"🏆 Best validation accuracy: {best_val_acc:.2f}%")
    print(f"📅 Best epoch: {best_epoch}")
    print(f"🔄 Total epochs completed: {final_epochs}")
    print(f"⏱️  Total training time: {total_time/60:.1f} minutes")
    print(f"🎮 Device used: {config['device_type']}")
    print(f"📊 Training speed: {final_epochs/(total_time/60):.1f} epochs/minute")
    
    # Performance analysis
    if config['device_type'] == 'GPU':
        print(f"⚡ GPU Training Benefits:")
        print(f"   • Faster training with larger batches")
        print(f"   • Mixed precision acceleration")
        print(f"   • Advanced attention mechanisms")
    else:
        print(f"💻 CPU Training Optimizations:")
        print(f"   • Memory-efficient smaller batches")
        print(f"   • Simplified architecture for faster processing")
        print(f"   • Reduced augmentation overhead")
    
    # Calculate improvement
    if len(history.history['val_accuracy']) > 1:
        initial_acc = history.history['val_accuracy'][0] * 100
        improvement = best_val_acc - initial_acc
        print(f"📈 Accuracy improvement: +{improvement:.1f}% (from {initial_acc:.1f}% to {best_val_acc:.1f}%)")
    
    # Save comprehensive training info
    training_info = {
        'classes': classes,
        'num_classes': num_classes,
        'best_val_acc': float(best_val_acc),
        'best_epoch': int(best_epoch),
        'total_epochs': final_epochs,
        'config': config,
        'device_used': config['device_type'],
        'device_config': DEVICE_CONFIG,
        'model_type': f'Smart_Adaptive_CNN_{config["device_type"]}',
        'training_completed': True,
        'total_training_time_minutes': total_time / 60,
        'training_speed_epochs_per_minute': final_epochs / (total_time / 60),
        'optimization_features': [
            f'Device-adaptive architecture ({config["device_type"]})',
            'TRUE residual connections with dimension matching',
            f'{"Attention mechanism" if config["device_type"] == "GPU" else "Simplified processing"}',
            f'{"SGD with Nesterov" if config["device_type"] == "GPU" else "Adam optimizer"}',
            f'{"Mixed precision" if config["mixed_precision"] else "Standard precision"}',
            'Smart batch sizing',
            'Device-optimized callbacks',
            'Class weighting for imbalance'
        ]
    }
    
    # Save files
    print(f"\n💾 Saving training files...")
    
    try:
        # Save classes
        with open(f"{config['model_save_path']}/classes.json", 'w') as f:
            json.dump(classes, f, indent=2)
        
        # Save training info
        with open(f"{config['model_save_path']}/training_info.json", 'w') as f:
            json.dump(training_info, f, indent=2)
        
        # Save training history
        if 'val_accuracy' in history.history:
            history_dict = {
                'epochs': list(range(1, len(history.history['val_accuracy']) + 1)),
                'train_accuracy': [float(x) * 100 for x in history.history['accuracy']],
                'validation_accuracy': [float(x) * 100 for x in history.history['val_accuracy']],
                'train_loss': [float(x) for x in history.history['loss']],
                'validation_loss': [float(x) for x in history.history['val_loss']],
                'device_used': config['device_type']
            }
            
            with open(f"{config['model_save_path']}/training_history.json", 'w') as f:
                json.dump(history_dict, f, indent=2)
        
        # Create training plots
        if 'val_accuracy' in history.history and len(history.history['val_accuracy']) > 1:
            create_training_plots(history, config['model_save_path'], config['device_type'])
        
        print(f"💾 All files saved to: {config['model_save_path']}/")
        print(f"   • best_model.h5 (trained model)")
        print(f"   • classes.json (class names)")
        print(f"   • training_info.json (complete training details)")
        print(f"   • training_history.json (accuracy/loss curves)")
        print(f"   • training_curves.png (visualization)")
        print(f"   • training_log.csv (epoch-by-epoch log)")
        
    except Exception as e:
        print(f"⚠️  Error saving files: {e}")
    
    return model, best_val_acc

def create_training_plots(history, save_path, device_type):
    """Create and save training visualization plots"""
    
    try:
        plt.style.use('default')
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f'Training Progress ({device_type})', fontsize=16, fontweight='bold')
        
        epochs = range(1, len(history.history['accuracy']) + 1)
        
        # Accuracy plot
        ax1.plot(epochs, [x*100 for x in history.history['accuracy']], 'b-', label='Training Accuracy', linewidth=2)
        ax1.plot(epochs, [x*100 for x in history.history['val_accuracy']], 'r-', label='Validation Accuracy', linewidth=2)
        ax1.set_title(f'Model Accuracy ({device_type})', fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy (%)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Loss plot
        ax2.plot(epochs, history.history['loss'], 'b-', label='Training Loss', linewidth=2)
        ax2.plot(epochs, history.history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        ax2.set_title(f'Model Loss ({device_type})', fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Training visualization saved as training_curves.png")
    except Exception as e:
        print(f"⚠️  Could not create training plots: {e}")

def test_trained_model(model_path, classes_path):
    """Test the trained model"""
    try:
        # Load model
        model = keras.models.load_model(model_path)
        
        # Load classes
        with open(classes_path, 'r') as f:
            classes = json.load(f)
        
        print(f"✅ Model loaded successfully!")
        print(f"📋 Classes: {classes}")
        print(f"🧠 Model input shape: {model.input_shape}")
        print(f"🎯 Model output shape: {model.output_shape}")
        
        return True
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False

# ===== MAIN EXECUTION =====
if __name__ == "__main__":
    print("🌱 Smart Device-Adaptive CNN Training Script")
    print("🎯 Automatically optimizes for GPU or falls back to CPU")
    print("=" * 60)
    
    # Train the smart adaptive model
    model, final_accuracy = train_smart_cnn()
    
    if model is not None:
        print(f"\n🎯 FINAL RESULT: {final_accuracy:.1f}% accuracy achieved!")
        print(f"🎮 Training device: {DEVICE_CONFIG['device_type']}")
        
        if final_accuracy >= 50:
            print("🏆 EXCELLENT! Target achieved!")
        elif final_accuracy >= 45:
            print("✅ GREAT! Significant improvement!")
        else:
            print("📈 Good progress made!")
        
        # Test loading the saved model
        print(f"\n🧪 Testing saved model...")
        model_path = "models/saved_models/best_model.h5"
        classes_path = "models/saved_models/classes.json"
        
        if os.path.exists(model_path) and os.path.exists(classes_path):
            success = test_trained_model(model_path, classes_path)
            if success:
                print(f"🎉 SUCCESS! Model ready for deployment!")
                print(f"📁 Model location: {model_path}")
                print(f"📋 Classes file: {classes_path}")
            else:
                print(f"⚠️  Model saved but testing failed")
        else:
            print(f"❌ Model files not found")
        
        # Performance summary
        device_emoji = "🎮" if DEVICE_CONFIG['device_type'] == 'GPU' else "💻"
        print(f"\n{device_emoji} TRAINING SUMMARY:")
        print(f"   Device: {DEVICE_CONFIG['device_type']}")
        print(f"   Accuracy: {final_accuracy:.1f}%")
        print(f"   Batch size: {DEVICE_CONFIG['batch_size']}")
        print(f"   Mixed precision: {DEVICE_CONFIG['mixed_precision']}")
        
    else:
        print("❌ Training failed. Please check:")
        print("   • Dataset directory structure")
        print("   • Image file formats (.jpg, .png)")
        print("   • Sufficient disk space")
        print("   • TensorFlow installation")