# train.py - TensorFlow Version
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import json
import time
from datetime import datetime

# Import our TensorFlow modules
from pest_classifier import create_model, save_model, get_model_summary
from data_loader import get_balanced_data_loaders

class TrainingProgress(keras.callbacks.Callback):
    """Custom callback for detailed training progress"""
    
    def __init__(self):
        super().__init__()
        self.batch_losses = []
        
    def on_batch_end(self, batch, logs=None):
        loss = logs.get('loss', 0)
        acc = logs.get('accuracy', 0)
        self.batch_losses.append(loss)
        
        # Show progress every 50 batches
        if batch % 50 == 0 and batch > 0:
            smooth_loss = np.mean(self.batch_losses[-50:])
            print(f"    Batch {batch}: Loss={loss:.4f}, Smooth={smooth_loss:.4f}, Acc={acc*100:.1f}%")

def train_tensorflow_model():
    """Train TensorFlow Grayscale CNN"""
    
    print("🎯 TENSORFLOW TRAINING - GRAYSCALE CNN")
    print("=" * 60)
    
    # Configuration
    config = {
        'data_dir': 'datasets',
        'model_save_path': 'models/saved_models',
        'epochs': 50,
        'learning_rate': 0.001,
        'batch_size': 32,
        'patience': 10,
        'validation_split': 0.2
    }
    
    # Create directories
    os.makedirs(config['model_save_path'], exist_ok=True)
    
    # GPU setup
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
            print(f"🖥️  Using GPU: {physical_devices[0].name}")
        except:
            print("🖥️  GPU setup failed, using CPU")
    else:
        print("🖥️  Using CPU")
    
    # Load data
    try:
        train_gen, val_gen, classes = get_balanced_data_loaders(
            data_dir=config['data_dir'],
            batch_size=config['batch_size'],
            validation_split=config['validation_split']
        )
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return None, 0
    
    num_classes = len(classes)
    print(f"✅ Dataset loaded: {num_classes} classes")
    
    # Create model
    model = create_model(num_classes)
    get_model_summary(model)
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config['learning_rate']),
        loss=keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=config['patience'],
            restore_best_weights=True,
            verbose=1
        ),
        
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(config['model_save_path'], 'best_model.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        
        TrainingProgress()
    ]
    
    print(f"\n🚀 Starting TensorFlow training...")
    print(f"📊 Batch size: {config['batch_size']}")
    print(f"🎯 Learning rate: {config['learning_rate']}")
    print("=" * 60)
    
    start_time = time.time()
    
    # Train model
    history = model.fit(
        train_gen,
        epochs=config['epochs'],
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )
    
    total_time = time.time() - start_time
    best_val_acc = max(history.history['val_accuracy']) * 100
    
    print(f"\n🎉 TRAINING COMPLETED!")
    print("=" * 60)
    print(f"🏆 Best validation accuracy: {best_val_acc:.2f}%")
    print(f"⏱️  Total training time: {total_time/60:.1f} minutes")
    
    # Save training history
    history_dict = {
        'train_loss': history.history['loss'],
        'train_accuracy': [acc * 100 for acc in history.history['accuracy']],
        'val_loss': history.history['val_loss'],
        'val_accuracy': [acc * 100 for acc in history.history['val_accuracy']],
        'best_val_acc': float(best_val_acc),
        'classes': classes,
        'config': config,
        'framework': 'TensorFlow',
        'tf_version': tf.__version__,
        'total_epochs': len(history.history['loss']),
        'training_time_minutes': total_time / 60
    }
    
    history_path = os.path.join(config['model_save_path'], 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history_dict, f, indent=2)
    
    return model, best_val_acc

def main():
    """Main training function"""
    print("🚀 TENSORFLOW GRAYSCALE CNN TRAINING")
    print("🎯 Features:")
    print("   • RGB → Grayscale conversion")
    print("   • TensorFlow/Keras implementation")
    print("   • Built-in progress tracking")
    print("   • Automatic GPU acceleration")
    print("=" * 60)
    
    model, final_accuracy = train_tensorflow_model()
    
    if model is not None:
        print(f"\n🎯 FINAL RESULT: {final_accuracy:.2f}%")
        
        if final_accuracy >= 80:
            print("🏆 EXCELLENT!")
        elif final_accuracy >= 70:
            print("🥇 VERY GOOD!")
        elif final_accuracy >= 60:
            print("👍 GOOD!")
        else:
            print("📈 DECENT!")

if __name__ == "__main__":
    main()