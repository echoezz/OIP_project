import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
import numpy as np
import os
import json
import time
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from pathlib import Path

# Hide TensorFlow info messages and set memory growth
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

def clean_dataset_directory(data_dir):
    """Remove empty directories and return valid classes"""
    valid_classes = []
    removed_dirs = []
    
    print("🧹 Checking dataset directories...")
    
    for item in os.listdir(data_dir):
        item_path = os.path.join(data_dir, item)
        
        if os.path.isdir(item_path):
            # Count image files
            image_files = [f for f in os.listdir(item_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            if len(image_files) == 0:
                print(f"   ⚠️  Empty directory found: {item} (will be skipped)")
                removed_dirs.append(item)
            else:
                valid_classes.append(item)
                print(f"   ✅ {item}: {len(image_files)} images")
    
    if removed_dirs:
        print(f"📊 Found {len(removed_dirs)} empty directories: {removed_dirs}")
        print(f"   These will be ignored during training")
    
    print(f"✅ Valid classes: {len(valid_classes)}")
    return valid_classes

def create_data_generators(data_dir, batch_size=16, img_size=224, validation_split=0.2):
    """Create TensorFlow data generators with heavy augmentation"""
    
    # Check for valid classes first
    valid_classes = clean_dataset_directory(data_dir)
    
    if len(valid_classes) == 0:
        raise ValueError("No valid classes with images found in dataset directory!")
    
    # Heavy augmentation for training
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='reflect',
        validation_split=validation_split
    )
    
    # Light augmentation for validation
    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        validation_split=validation_split
    )
    
    # Create generators
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True,
        seed=42
    )
    
    val_generator = val_datagen.flow_from_directory(
        data_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False,
        seed=42
    )
    
    return train_generator, val_generator, valid_classes

def create_enhanced_pest_classifier_model(num_classes, img_size=224):
    """Create ENHANCED CNN model with FIXED residual connections"""
    
    inputs = layers.Input(shape=(img_size, img_size, 3))
    
    # Initial conv block
    x = layers.Conv2D(32, 3, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(2)(x)  # 224->112
    
    # Block 1: 32 -> 64 channels
    identity = layers.Conv2D(64, 1, padding='same', use_bias=False)(x)  # Match channels
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
    
    # Block 2: 64 -> 128 channels
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
    
    # Block 3: 128 -> 256 channels
    identity = layers.Conv2D(256, 1, padding='same', use_bias=False)(x)
    identity = layers.BatchNormalization()(identity)
    
    x = layers.Conv2D(256, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(256, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Add()([x, identity])  # Residual connection
    x = layers.ReLU()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.MaxPooling2D(2)(x)  # 28->14
    
    # Block 4: 256 -> 512 channels
    identity = layers.Conv2D(512, 1, padding='same', use_bias=False)(x)
    identity = layers.BatchNormalization()(identity)
    
    x = layers.Conv2D(512, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(512, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Add()([x, identity])  # Residual connection
    x = layers.ReLU()(x)
    x = layers.Dropout(0.1)(x)
    
    # Attention mechanism
    attention = layers.GlobalAveragePooling2D()(x)
    attention = layers.Dense(512//16, activation='relu')(attention)
    attention = layers.Dense(512, activation='sigmoid')(attention)
    attention = layers.Reshape((1, 1, 512))(attention)
    x = layers.Multiply()([x, attention])
    
    # Global pooling and classifier
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs, outputs)
    return model

def create_advanced_callbacks(model_save_path, patience=15, min_epochs=20):
    """Create advanced callbacks for training"""
    
    callbacks_list = [
        # Save best model
        keras.callbacks.ModelCheckpoint(
            filepath=f'{model_save_path}/best_model.keras',
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        ),
        
        # Early stopping with patience
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=patience,
            restore_best_weights=True,
            verbose=1,
            start_from_epoch=min_epochs
        ),
        
        # Reduce learning rate on plateau
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=8,
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
    
    # Get actual class distribution from filesystem
    class_counts = {}
    total_samples = 0
    
    for class_name in classes:
        class_path = os.path.join(data_dir, class_name)
        if os.path.exists(class_path) and os.path.isdir(class_path):
            count = len([f for f in os.listdir(class_path) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            if count > 0:  # Only include classes with images
                class_counts[class_name] = count
                total_samples += count
    
    if not class_counts:
        print("⚠️  No valid classes found, using equal weights")
        return None
    
    # Calculate weights using class names as keys
    class_weights = {}
    num_classes = len(class_counts)
    
    print("📊 Class distribution and weights:")
    for i, (class_name, count) in enumerate(class_counts.items()):
        weight = total_samples / (num_classes * count)
        class_weights[i] = weight  # TensorFlow uses integer indices
        percentage = (count / total_samples) * 100
        print(f"   {class_name}: {count} images ({percentage:.1f}%) -> weight: {weight:.2f}")
    
    # Check for severe imbalance
    weights = list(class_weights.values())
    if max(weights) / min(weights) > 5:
        print("⚠️  Severe class imbalance detected! Consider collecting more data.")
    
    return class_weights

def train_tensorflow_cnn():
    """Train CNN using TensorFlow/Keras"""
    
    print("🚀 Starting FIXED Enhanced TensorFlow CNN Training")
    print("=" * 60)
    
    # Configuration
    config = {
        'data_dir': 'datasets',
        'model_save_path': 'models/saved_models',
        'epochs': 100,
        'learning_rate': 0.001,
        'batch_size': 16,
        'img_size': 224,
        'validation_split': 0.2,
        'patience': 15,
        'min_epochs': 20
    }
    
    # Create save directory
    os.makedirs(config['model_save_path'], exist_ok=True)
    
    print(f"🖥️  Using TensorFlow {tf.__version__}")
    print(f"🔧 GPU Available: {len(tf.config.list_physical_devices('GPU')) > 0}")
    
    # Create data generators
    print("\n📊 Creating data generators...")
    try:
        train_generator, val_generator, valid_classes = create_data_generators(
            config['data_dir'],
            batch_size=config['batch_size'],
            img_size=config['img_size'],
            validation_split=config['validation_split']
        )
    except Exception as e:
        print(f"❌ Error creating data generators: {e}")
        return None, 0
    
    # Get classes and dataset info
    classes = valid_classes
    num_classes = len(classes)
    
    if num_classes == 0:
        print("❌ No valid classes found!")
        return None, 0
    
    print(f"\n🎯 FIXED Training Configuration:")
    print(f"   Classes: {num_classes}")
    print(f"   Training samples: {train_generator.samples}")
    print(f"   Validation samples: {val_generator.samples}")
    print(f"   Batch size: {config['batch_size']}")
    print(f"   Learning rate: {config['learning_rate']}")
    print(f"   🚀 FIXED: Residual connections with proper dimension matching")
    
    # Calculate class weights
    class_weights = calculate_class_weights(config['data_dir'], classes)
    
    # Create FIXED model
    print(f"\n🧠 Creating FIXED model with residual connections...")
    model = create_enhanced_pest_classifier_model(num_classes, config['img_size'])
    
    # Print model summary
    print(f"\n📋 FIXED Model Architecture:")
    total_params = model.count_params()
    print(f"   📊 Total parameters: {total_params:,}")
    print(f"   🧠 Model layers: {len(model.layers)}")
    print(f"   ✅ FIXED: Dimension matching in residual connections")
    
    # Compile model
    optimizer = keras.optimizers.SGD(
        learning_rate=config['learning_rate'],
        momentum=0.9,
        nesterov=True
    )
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Create callbacks
    callbacks_list = create_advanced_callbacks(
        config['model_save_path'],
        patience=config['patience'],
        min_epochs=config['min_epochs']
    )
    
    print(f"\n🎓 Starting FIXED Training...")
    
    start_time = time.time()
    
    # Train the model
    try:
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
        return None, 0
    
    # Training completed
    total_time = time.time() - start_time
    
    print(f"\n🎉 FIXED TRAINING COMPLETED!")
    print("=" * 60)
    
    # Get best metrics
    if 'val_accuracy' in history.history and len(history.history['val_accuracy']) > 0:
        best_val_acc = max(history.history['val_accuracy']) * 100
        best_epoch = history.history['val_accuracy'].index(max(history.history['val_accuracy'])) + 1
        final_epochs = len(history.history['val_accuracy'])
    else:
        best_val_acc = 0
        best_epoch = 0
        final_epochs = 0
    
    print(f"🏆 Best validation accuracy: {best_val_acc:.2f}%")
    print(f"📅 Best epoch: {best_epoch}")
    print(f"🔄 Total epochs completed: {final_epochs}")
    print(f"⏱️  Total training time: {total_time/60:.1f} minutes")
    
    # Save training info
    training_info = {
        'classes': classes,
        'num_classes': num_classes,
        'best_val_acc': float(best_val_acc),
        'best_epoch': int(best_epoch),
        'total_epochs': final_epochs,
        'config': config,
        'model_type': 'FIXED_Enhanced_TensorFlow_CNN_with_Residual',
        'training_completed': True,
        'total_training_time_minutes': total_time / 60,
        'optimization_features': [
            'FIXED residual connections with dimension matching',
            'Attention mechanism',
            'SGD with Nesterov momentum',
            'Enhanced data augmentation',
            'Class weighting for imbalance',
            'Early stopping optimization'
        ]
    }
    
    # Save files
    try:
        with open(f"{config['model_save_path']}/classes.json", 'w') as f:
            json.dump(classes, f, indent=2)
        
        with open(f"{config['model_save_path']}/training_info.json", 'w') as f:
            json.dump(training_info, f, indent=2)
            
        print(f"\n💾 FIXED model saved successfully!")
        
    except Exception as e:
        print(f"⚠️  Error saving files: {e}")
    
    return model, best_val_acc

def create_training_plots(history, save_path):
    """Create and save training visualization plots"""
    
    try:
        plt.style.use('default')
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Training Progress', fontsize=16, fontweight='bold')
        
        epochs = range(1, len(history.history['accuracy']) + 1)
        
        # Accuracy plot
        ax1.plot(epochs, [x*100 for x in history.history['accuracy']], 'b-', label='Training Accuracy', linewidth=2)
        ax1.plot(epochs, [x*100 for x in history.history['val_accuracy']], 'r-', label='Validation Accuracy', linewidth=2)
        ax1.set_title('Model Accuracy', fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy (%)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Loss plot
        ax2.plot(epochs, history.history['loss'], 'b-', label='Training Loss', linewidth=2)
        ax2.plot(epochs, history.history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        ax2.set_title('Model Loss', fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("📊 Training visualization saved as training_curves.png")
    except Exception as e:
        print(f"⚠️  Could not create training plots: {e}")

if __name__ == "__main__":
    print("🌱 FIXED Enhanced TensorFlow CNN Training Script")
    print("=" * 60)
    
    # Train the FIXED model
    model, final_accuracy = train_tensorflow_cnn()
    
    if model is not None:
        print(f"\n🎯 FIXED RESULT: {final_accuracy:.1f}% accuracy achieved!")
        
        if final_accuracy >= 50:
            print("🏆 EXCELLENT! Target achieved with FIXED architecture!")
        elif final_accuracy >= 45:
            print("✅ GREAT! Significant improvement with FIXED residual connections!")
        else:
            print("📈 Progress made, FIXED architecture should help!")
    else:
        print("❌ Training failed. Please check your dataset structure.")