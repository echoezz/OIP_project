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

def create_data_generators(data_dir, batch_size=8, img_size=224, validation_split=0.2):
    """Create TensorFlow data generators with heavy augmentation"""
    
    # Check for valid classes first
    valid_classes = clean_dataset_directory(data_dir)
    
    if len(valid_classes) == 0:
        raise ValueError("No valid classes with images found in dataset directory!")
    
    # Heavy augmentation for training
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=45,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.6, 1.4],
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

def create_pest_classifier_model(num_classes, img_size=224):
    """Create CNN model optimized for pest classification"""
    
    model = models.Sequential([
        # First Conv Block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 3)),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),
        
        # Second Conv Block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),
        
        # Third Conv Block
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),
        
        # Fourth Conv Block
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),
        
        # Fifth Conv Block
        layers.Conv2D(512, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),
        
        # Dense layers
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def create_advanced_callbacks(model_save_path, patience=25, min_epochs=30):
    """Create advanced callbacks for training"""
    
    callbacks_list = [
        # Save best model
        keras.callbacks.ModelCheckpoint(
            filepath=f'{model_save_path}/best_model.h5',
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
            patience=10,
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
    """Calculate class weights for imbalanced datasets (fixed version)"""
    
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
    
    print("🚀 Starting TensorFlow CNN Pest Classification Training")
    print("=" * 60)
    
    # Configuration
    config = {
        'data_dir': 'datasets',
        'model_save_path': 'models/saved_models',
        'epochs': 150,
        'learning_rate': 0.0003,
        'batch_size': 8,
        'img_size': 224,
        'validation_split': 0.2,
        'patience': 25,
        'min_epochs': 30
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
        print("💡 Make sure your dataset directory structure is correct:")
        print("   datasets/")
        print("   ├── class1/")
        print("   │   ├── image1.jpg")
        print("   │   └── image2.jpg")
        print("   ├── class2/")
        print("   │   └── image3.jpg")
        return None, 0
    
    # Get classes and dataset info
    classes = valid_classes
    num_classes = len(classes)
    
    if num_classes == 0:
        print("❌ No valid classes found!")
        return None, 0
    
    print(f"\n🎯 Training Configuration:")
    print(f"   Classes: {num_classes}")
    print(f"   Training samples: {train_generator.samples}")
    print(f"   Validation samples: {val_generator.samples}")
    print(f"   Batch size: {config['batch_size']}")
    print(f"   Image size: {config['img_size']}x{config['img_size']}")
    print(f"   Epochs: {config['epochs']}")
    print(f"   Learning rate: {config['learning_rate']}")
    
    print(f"\n📋 Classes found: {classes}")
    
    # Calculate class weights
    class_weights = calculate_class_weights(config['data_dir'], classes)
    
    # Create model
    print(f"\n🧠 Creating model...")
    model = create_pest_classifier_model(num_classes, config['img_size'])
    
    # Print model summary
    print(f"\n📋 Model Architecture:")
    total_params = model.count_params()
    print(f"   📊 Total parameters: {total_params:,}")
    print(f"   🧠 Model layers: {len(model.layers)}")
    print(f"   📥 Input shape: {model.input_shape}")
    print(f"   📤 Output shape: {model.output_shape}")
    
    # Compile model with advanced settings
    optimizer = keras.optimizers.AdamW(
        learning_rate=config['learning_rate'],
        weight_decay=1e-3
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
    
    print(f"\n🎓 Starting Training...")
    print(f"🔧 Features: Heavy augmentation, Class weighting, Early stopping, LR scheduling")
    
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
    
    print(f"\n🎉 TRAINING COMPLETED!")
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
    
    # Calculate improvement
    if len(history.history['val_accuracy']) > 1:
        initial_acc = history.history['val_accuracy'][0] * 100
        improvement = best_val_acc - initial_acc
        print(f"📈 Accuracy improvement: +{improvement:.1f}% (from {initial_acc:.1f}% to {best_val_acc:.1f}%)")
    
    # Final model analysis
    print(f"\n📋 Final Model Analysis:")
    print(f"   🎯 Target achieved: {'✅' if best_val_acc >= 50 else '❌'} (50%+ confidence)")
    print(f"   📊 Model performance: {'Excellent' if best_val_acc >= 70 else 'Good' if best_val_acc >= 50 else 'Needs improvement'}")
    
    if best_val_acc < 50:
        print(f"\n💡 Suggestions for improvement:")
        print(f"   • Train for more epochs")
        print(f"   • Adjust augmentation parameters")
        print(f"   • Try different learning rates")
        print(f"   • Consider transfer learning")
    
    # Generate predictions for classification report
    print(f"\n📊 Generating detailed classification report...")
    try:
        val_generator.reset()
        predictions = model.predict(val_generator, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = val_generator.classes
        
        # Classification report
        report = classification_report(
            true_classes, 
            predicted_classes, 
            target_names=classes, 
            output_dict=True,
            zero_division=0
        )
        
        print("📋 Per-class performance:")
        for class_name in classes:
            if class_name in report:
                precision = report[class_name].get('precision', 0)
                recall = report[class_name].get('recall', 0)
                f1 = report[class_name].get('f1-score', 0)
                print(f"   {class_name}: P={precision:.2f}, R={recall:.2f}, F1={f1:.2f}")
        
    except Exception as e:
        print(f"⚠️  Could not generate classification report: {e}")
        report = {}
    
    # Save comprehensive training info
    training_info = {
        'classes': classes,
        'num_classes': num_classes,
        'best_val_acc': float(best_val_acc),
        'best_epoch': int(best_epoch),
        'total_epochs': final_epochs,
        'config': config,
        'model_type': 'TensorFlowCNN',
        'training_completed': True,
        'total_training_time_minutes': total_time / 60,
        'optimization_features': [
            'Heavy data augmentation',
            'Class weighting for imbalance',
            'AdamW optimizer with weight decay',
            'Learning rate scheduling',
            'Early stopping with patience',
            'Batch normalization',
            'Dropout regularization'
        ]
    }
    
    if class_weights:
        training_info['class_weights'] = {str(k): float(v) for k, v in class_weights.items()}
    
    # Save files
    print(f"\n💾 Saving training files...")
    
    try:
        # Save classes
        with open(f"{config['model_save_path']}/classes.json", 'w') as f:
            json.dump(classes, f, indent=2)
        
        # Save training info
        with open(f"{config['model_save_path']}/training_info.json", 'w') as f:
            json.dump(training_info, f, indent=2)
        
        # Save classification report
        if report:
            with open(f"{config['model_save_path']}/classification_report.json", 'w') as f:
                json.dump(report, f, indent=2)
        
        # Save training history
        if 'val_accuracy' in history.history:
            history_dict = {
                'epochs': list(range(1, len(history.history['val_accuracy']) + 1)),
                'train_accuracy': [float(x) * 100 for x in history.history['accuracy']],
                'validation_accuracy': [float(x) * 100 for x in history.history['val_accuracy']],
                'train_loss': [float(x) for x in history.history['loss']],
                'validation_loss': [float(x) for x in history.history['val_loss']]
            }
            
            with open(f"{config['model_save_path']}/training_history.json", 'w') as f:
                json.dump(history_dict, f, indent=2)
        
        # Create training plots
        if 'val_accuracy' in history.history and len(history.history['val_accuracy']) > 1:
            create_training_plots(history, config['model_save_path'])
        
        print(f"\n💾 All files saved to: {config['model_save_path']}/")
        print(f"   • best_model.h5 (trained model)")
        print(f"   • classes.json (class names)")
        print(f"   • training_info.json (complete training details)")
        print(f"   • training_history.json (accuracy/loss curves)")
        print(f"   • classification_report.json (per-class metrics)")
        print(f"   • training_curves.png (visualization)")
        print(f"   • training_log.csv (epoch-by-epoch log)")
        
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

def test_tensorflow_model(model_path, classes_path):
    """Test the trained TensorFlow model"""
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

if __name__ == "__main__":
    print("🌱 TensorFlow CNN Training Script")
    print("=" * 60)
    
    # Train the model
    model, final_accuracy = train_tensorflow_cnn()
    
    if model is not None:
        # Test loading the saved model
        print(f"\n🧪 Testing saved model...")
        model_path = "models/saved_models/best_model.h5"
        classes_path = "models/saved_models/classes.json"
        
        if os.path.exists(model_path) and os.path.exists(classes_path):
            success = test_tensorflow_model(model_path, classes_path)
            if success:
                print(f"🎉 SUCCESS! Model ready for deployment!")
            else:
                print(f"⚠️  Model saved but testing failed")
        else:
            print(f"❌ Model files not found")
        
        print(f"\n🎯 FINAL RESULT: {final_accuracy:.1f}% accuracy achieved!")
        
        if final_accuracy >= 50:
            print("🏆 EXCELLENT! Target achieved!")
        elif final_accuracy >= 35:
            print("✅ GOOD! Significant improvement from baseline!")
        else:
            print("📈 Progress made, consider more training or data")
    else:
        print("❌ Training failed. Please check your dataset structure.")