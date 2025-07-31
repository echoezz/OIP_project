# pest_classifier.py - Anti-Overfitting Version
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class RegularizedCNN(keras.Model):
    """Smaller, heavily regularized CNN to prevent overfitting"""
    
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        
        # SMALLER feature extraction to prevent overfitting
        self.features = keras.Sequential([
            # Block 1 - Smaller filters
            layers.Conv2D(16, 3, padding='same', activation='relu', name='conv1'),
            layers.BatchNormalization(name='bn1'),
            layers.Dropout(0.2, name='drop1'),  # Early dropout
            layers.MaxPooling2D(2, name='pool1'),
            
            # Block 2
            layers.Conv2D(32, 3, padding='same', activation='relu', name='conv2'),
            layers.BatchNormalization(name='bn2'),
            layers.Dropout(0.3, name='drop2'),
            layers.MaxPooling2D(2, name='pool2'),
            
            # Block 3
            layers.Conv2D(64, 3, padding='same', activation='relu', name='conv3'),
            layers.BatchNormalization(name='bn3'),
            layers.Dropout(0.4, name='drop3'),
            layers.MaxPooling2D(2, name='pool3'),
            
            # Block 4 - Last conv block
            layers.Conv2D(128, 3, padding='same', activation='relu', name='conv4'),
            layers.BatchNormalization(name='bn4'),
            layers.Dropout(0.5, name='drop4'),
            layers.GlobalAveragePooling2D(name='global_pool')  # No more pooling
        ], name='feature_extractor')
        
        # MUCH SMALLER classifier to prevent overfitting
        self.classifier = keras.Sequential([
            layers.Dropout(0.6, name='dropout1'),              # Heavy dropout
            layers.Dense(256, activation='relu', name='fc1'),   # Smaller layer
            layers.Dropout(0.5, name='dropout2'),
            layers.Dense(128, activation='relu', name='fc2'),   # Smaller layer
            layers.Dropout(0.4, name='dropout3'),
            layers.Dense(num_classes, name='predictions')
        ], name='classifier')
    
    def call(self, x, training=None):
        x = self.features(x, training=training)
        x = self.classifier(x, training=training)
        return x

def create_model(num_classes):
    """Create anti-overfitting model"""
    model = RegularizedCNN(num_classes)
    model.build((None, 224, 224, 3))
    return model

def get_model_summary(model):
    """Get model summary"""
    print("\n" + "="*60)
    model.summary()
    print("="*60)
    
    total_params = model.count_params()
    
    summary_text = f"""
🤖 **Anti-Overfitting CNN for Pest Classification**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📋 **Model Type:** Regularized CNN (Anti-Overfitting)
🎯 **Classes:** {model.num_classes}
⚙️  **Parameters:** {total_params:,}
💾 **Size:** {total_params * 4 / 1024 / 1024:.1f} MB

🛡️ **Anti-Overfitting Features:**
   • Smaller model (less memorization capacity)
   • Heavy dropout at every layer
   • Batch normalization for stability
   • Global average pooling
   • Progressive dropout (0.2 → 0.6)

✅ **Framework:** TensorFlow {tf.__version__}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    
    print(summary_text)
    return summary_text

def save_model_safely(model, save_path):
    """Save model safely"""
    try:
        weights_path = save_path.replace('.h5', '_weights.h5')
        model.save_weights(weights_path)
        
        json_path = save_path.replace('.h5', '_architecture.json')
        model_json = model.to_json()
        with open(json_path, 'w') as f:
            f.write(model_json)
        
        print(f"✅ Model saved: {weights_path}")
        return True
    except Exception as e:
        print(f"❌ Save failed: {e}")
        return False

def load_model_from_weights(weights_path, num_classes):
    """Load model from weights"""
    model = create_model(num_classes)
    model.load_weights(weights_path)
    return model