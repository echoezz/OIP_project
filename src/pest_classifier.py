# pest_classifier.py - TensorFlow Version
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class GrayscaleCNN(keras.Model):
    """TensorFlow Grayscale CNN for Pest Classification - FIXED"""
    
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        
        # FIXED: RGB to Grayscale conversion using Lambda layer
        self.to_grayscale = layers.Lambda(
            lambda x: tf.reduce_sum(x * tf.constant([0.299, 0.587, 0.114]), axis=-1, keepdims=True),
            name='rgb_to_grayscale'
        )
        
        # Feature extraction
        self.features = keras.Sequential([
            # Block 1
            layers.Conv2D(32, 3, padding='same', activation='relu', name='conv1'),
            layers.BatchNormalization(name='bn1'),
            layers.MaxPooling2D(2, name='pool1'),
            
            # Block 2
            layers.Conv2D(64, 3, padding='same', activation='relu', name='conv2'),
            layers.BatchNormalization(name='bn2'),
            layers.MaxPooling2D(2, name='pool2'),
            
            # Block 3
            layers.Conv2D(128, 3, padding='same', activation='relu', name='conv3'),
            layers.BatchNormalization(name='bn3'),
            layers.MaxPooling2D(2, name='pool3'),
            
            # Block 4
            layers.Conv2D(256, 3, padding='same', activation='relu', name='conv4'),
            layers.BatchNormalization(name='bn4'),
            layers.MaxPooling2D(2, name='pool4'),
            
            # Block 5
            layers.Conv2D(512, 3, padding='same', activation='relu', name='conv5'),
            layers.BatchNormalization(name='bn5'),
            layers.GlobalAveragePooling2D(name='global_pool')
        ], name='feature_extractor')
        
        # Classifier
        self.classifier = keras.Sequential([
            layers.Dropout(0.5, name='dropout1'),
            layers.Dense(1024, activation='relu', name='fc1'),
            layers.Dropout(0.3, name='dropout2'),
            layers.Dense(512, activation='relu', name='fc2'),
            layers.Dropout(0.2, name='dropout3'),
            layers.Dense(num_classes, name='predictions')
        ], name='classifier')
    
    def call(self, x, training=None):
        # Convert RGB to grayscale using Lambda layer
        x = self.to_grayscale(x)
        
        # Extract features
        x = self.features(x, training=training)
        
        # Classify
        x = self.classifier(x, training=training)
        
        return x

def create_model(num_classes):
    """Create TensorFlow Grayscale CNN model - FIXED"""
    model = GrayscaleCNN(num_classes)
    
    # Build the model properly
    dummy_input = tf.random.normal((1, 224, 224, 3))
    _ = model(dummy_input, training=False)
    
    return model

def save_model(model, path):
    """Save TensorFlow model"""
    model.save(path)
    print(f"✅ Model saved to: {path}")

def load_model(path):
    """Load TensorFlow model"""
    return keras.models.load_model(path)

def get_model_summary(model):
    """Get detailed model summary"""
    print("\n" + "="*60)
    model.summary()
    print("="*60)
    
    total_params = model.count_params()
    
    summary_text = f"""
🤖 **TensorFlow Grayscale CNN for Pest Classification**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📋 **Model Architecture:** RGB→Grayscale + Standard CNN
🎯 **Number of Classes:** {model.num_classes}
⚙️  **Total Parameters:** {total_params:,}
💾 **Estimated Size:** {total_params * 4 / 1024 / 1024:.1f} MB

🏗️ **Architecture Details:**
   • RGB → Grayscale conversion (frozen weights)
   • 5 CNN blocks with BatchNorm + ReLU + MaxPool
   • Progressive filters: 32→64→128→256→512
   • Global Average Pooling (no flatten needed)
   • 3-layer classifier with dropout regularization

🎯 **Key Features:**
   • Grayscale preprocessing for efficiency
   • Batch normalization for stable training
   • Dropout for overfitting prevention  
   • Global pooling for translation invariance

✅ **Framework:** TensorFlow {tf.__version__}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    
    print(summary_text)
    return summary_text