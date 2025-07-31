# pest_classifier.py - Anti-Overfitting Version
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# PyTorch imports for PestClassifier
import torch
import torch.nn as nn
import torch.nn.functional as F

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


# PyTorch Model for Pest Classification
class DepthwiseSeparableConv(nn.Module):
    """Depthwise Separable Convolution - core of MobileNet efficiency"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        
        # Depthwise convolution
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size, 
            stride=stride, padding=padding, groups=in_channels, bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        
        # Pointwise convolution
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU6(inplace=True)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        return x

class InvertedResidual(nn.Module):
    """Inverted Residual Block with skip connection"""
    def __init__(self, in_channels, out_channels, stride=1, expand_ratio=6):
        super(InvertedResidual, self).__init__()
        
        self.stride = stride
        self.use_residual = stride == 1 and in_channels == out_channels
        
        hidden_dim = in_channels * expand_ratio
        
        layers = []
        
        # Expand
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True)
            ])
        
        # Depthwise
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride=stride, 
                     padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])
        
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        else:
            return self.conv(x)

class PestClassifier(nn.Module):
    """CustomModel from notebook - MobileNet-inspired architecture"""
    def __init__(self, num_classes=12):
        super(PestClassifier, self).__init__()
        
        # Initial convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True)
        )
        
        # MobileNet blocks
        self.features = nn.Sequential(
            InvertedResidual(32, 16, 1, 1),      # 112x112
            InvertedResidual(16, 24, 2, 6),      # 56x56
            InvertedResidual(24, 24, 1, 6),      # 56x56
            InvertedResidual(24, 32, 2, 6),      # 28x28
            InvertedResidual(32, 32, 1, 6),      # 28x28
            InvertedResidual(32, 32, 1, 6),      # 28x28
            InvertedResidual(32, 64, 2, 6),      # 14x14
            InvertedResidual(64, 64, 1, 6),      # 14x14
            InvertedResidual(64, 64, 1, 6),      # 14x14
            InvertedResidual(64, 64, 1, 6),      # 14x14
            InvertedResidual(64, 96, 1, 6),      # 14x14
            InvertedResidual(96, 96, 1, 6),      # 14x14
            InvertedResidual(96, 96, 1, 6),      # 14x14
            InvertedResidual(96, 160, 2, 6),     # 7x7
            InvertedResidual(160, 160, 1, 6),    # 7x7
            InvertedResidual(160, 160, 1, 6),    # 7x7
            InvertedResidual(160, 320, 1, 6),    # 7x7
        )
        
        # Final convolution - MUST match notebook exactly
        self.conv_last = nn.Sequential(
            nn.Conv2d(320, 1280, 1, bias=False),
            nn.BatchNorm2d(1280),
            nn.ReLU6(inplace=True)
        )
        
        # Classifier - MUST match notebook exactly
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.features(x)
        x = self.conv_last(x)
        x = self.classifier(x)
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