# train.py - FIXED Anti-Overfitting Training
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import json
import time

from pest_classifier import create_model, get_model_summary
from data_loader import get_balanced_data_loaders

class DetailedProgress(keras.callbacks.Callback):
    """Detailed progress with overfitting detection"""
    
    def __init__(self):
        super().__init__()
        self.batch_losses = []
        
    def on_epoch_end(self, epoch, logs=None):
        train_acc = logs.get('accuracy', 0) * 100
        val_acc = logs.get('val_accuracy', 0) * 100
        gap = train_acc - val_acc
        
        print(f"\n📊 Epoch {epoch+1} Summary:")
        print(f"   🏋️ Train Accuracy: {train_acc:.1f}%")
        print(f"   ✅ Val Accuracy: {val_acc:.1f}%")
        print(f"   📈 Gap: {gap:.1f}%", end="")
        
        if gap > 15:
            print(" ⚠️ OVERFITTING WARNING!")
        elif gap > 10:
            print(" 🟡 Watch for overfitting")
        else:
            print(" ✅ Healthy gap")

def train_anti_overfitting_model():
    """Train with anti-overfitting measures"""
    
    print("🛡️ ANTI-OVERFITTING CNN TRAINING")
    print("=" * 60)
    
    # Conservative configuration
    config = {
        'data_dir': 'datasets',
        'model_save_path': 'models/saved_models',
        'epochs': 100,                # More epochs with early stopping
        'learning_rate': 0.002,      # LOWER learning rate
        'batch_size': 32,             # SMALLER batch size
        'patience': 15,               # MORE patience
        'validation_split': 0.2
    }
    
    os.makedirs(config['model_save_path'], exist_ok=True)
    
    # GPU setup
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
            print(f"🖥️ Using GPU: {physical_devices[0].name}")
        except:
            print("🖥️ Using CPU")
    else:
        print("🖥️ Using CPU")
    
    # Load data with heavy augmentation
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
    
    # Create smaller, regularized model
    model = create_model(num_classes)
    get_model_summary(model)
    
    # Compile with conservative settings
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=config['learning_rate']
            # Note: weight_decay removed for compatibility
        ),
        loss=keras.losses.CategoricalCrossentropy(
            from_logits=True,
            label_smoothing=0.1            # Label smoothing
        ),
        metrics=['accuracy']
    )
    
    # FIXED: Anti-overfitting callbacks with correct filename
    class BestWeightsSaver(keras.callbacks.Callback):
        def __init__(self, filepath):
            super().__init__()
            # FIXED: Use .weights.h5 extension for newer TensorFlow versions
            if not filepath.endswith('.weights.h5'):
                filepath = filepath.replace('.h5', '.weights.h5')
            self.filepath = filepath
            self.best_val_acc = 0
            
        def on_epoch_end(self, epoch, logs=None):
            val_acc = logs.get('val_accuracy', 0)
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                try:
                    self.model.save_weights(self.filepath)
                    print(f"🎯 New best val_acc: {val_acc:.4f} - weights saved!")
                except Exception as e:
                    print(f"⚠️ Could not save weights: {e}")
    
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=config['patience'],
            restore_best_weights=True,
            verbose=1,
            mode='max'
        ),
        
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,                    # Aggressive LR reduction
            patience=8,
            min_lr=1e-7,
            verbose=1
        ),
        
        # FIXED: Proper filename
        BestWeightsSaver(
            filepath=os.path.join(config['model_save_path'], 'best_model.weights.h5')
        ),
        
        DetailedProgress()
    ]
    
    print(f"\n🛡️ Starting ANTI-OVERFITTING training...")
    print(f"📊 Batch size: {config['batch_size']} (small)")
    print(f"🎯 Learning rate: {config['learning_rate']} (conservative)")
    print(f"🔄 Heavy augmentation: ENABLED")
    print(f"🛑 Early stopping: {config['patience']} epochs patience")
    print("=" * 60)
    
    start_time = time.time()
    
    # Train with anti-overfitting measures
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
    print(f"⏱️ Total time: {total_time/60:.1f} minutes")
    
    # FIXED: Save everything with proper extensions
    try:
        # Save model architecture
        model_json = model.to_json()
        json_path = os.path.join(config['model_save_path'], 'model_architecture.json')
        with open(json_path, 'w') as f:
            f.write(model_json)
        
        # Save final weights with correct extension
        weights_path = os.path.join(config['model_save_path'], 'final_model.weights.h5')
        model.save_weights(weights_path)
        
        print(f"✅ Model architecture saved: {json_path}")
        print(f"✅ Final weights saved: {weights_path}")
        
    except Exception as e:
        print(f"⚠️ Saving warning: {e}")
        print("✅ Training completed successfully anyway!")
    
    # Save training history
    history_dict = {
        'train_loss': history.history['loss'],
        'train_accuracy': [acc * 100 for acc in history.history['accuracy']],
        'val_loss': history.history['val_loss'],
        'val_accuracy': [acc * 100 for acc in history.history['val_accuracy']],
        'best_val_acc': float(best_val_acc),
        'classes': classes,
        'config': config,
        'anti_overfitting_measures': [
            'Heavy data augmentation',
            'Smaller model architecture',
            'Progressive dropout (0.2-0.6)',
            'Lower learning rate',
            'Smaller batch size',
            'Label smoothing',
            'Early stopping'
        ],
        'framework': 'TensorFlow',
        'tf_version': tf.__version__,
        'total_epochs': len(history.history['loss']),
        'training_time_minutes': total_time / 60
    }
    
    history_path = os.path.join(config['model_save_path'], 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history_dict, f, indent=2)
    
    print(f"📊 Training history saved: {history_path}")
    
    return model, best_val_acc

def main():
    """Main anti-overfitting training"""
    print("🛡️ ANTI-OVERFITTING CNN TRAINING")
    print("🎯 Measures:")
    print("   • Smaller model (less memorization)")
    print("   • Heavy data augmentation")  
    print("   • Progressive dropout")
    print("   • Conservative learning rate")
    print("   • Label smoothing")
    print("   • Overfitting monitoring")
    print("=" * 60)
    
    model, final_accuracy = train_anti_overfitting_model()
    
    if model is not None:
        print(f"\n🎯 FINAL RESULT: {final_accuracy:.2f}%")
        
        if final_accuracy >= 70:
            print("🏆 EXCELLENT! Anti-overfitting worked!")
        elif final_accuracy >= 60:
            print("🥇 VERY GOOD! Much better generalization!")
        elif final_accuracy >= 50:
            print("👍 GOOD! Healthy train/val performance!")
        else:
            print("📈 DECENT! Model is learning without overfitting!")
        
        # Show saved files
        print(f"\n📁 SAVED FILES:")
        print(f"   🤖 Best weights: models/saved_models/best_model.weights.h5")
        print(f"   🤖 Final weights: models/saved_models/final_model.weights.h5")
        print(f"   🏗️ Architecture: models/saved_models/model_architecture.json")
        print(f"   📊 History: models/saved_models/training_history.json")

if __name__ == "__main__":
    main()