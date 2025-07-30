import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import numpy as np

class FocusedCNN(nn.Module):
    """Focused CNN designed for small datasets with attention mechanism"""
    
    def __init__(self, num_classes: int):
        super(FocusedCNN, self).__init__()
        self.num_classes = num_classes
        
        # Feature extraction with residual-style blocks
        self.conv_block1 = self._make_conv_block(3, 32)
        self.conv_block2 = self._make_conv_block(32, 64)
        self.conv_block3 = self._make_conv_block(64, 128)
        self.conv_block4 = self._make_conv_block(128, 256)
        
        # Attention mechanism for feature focusing
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(256, 256//16, 1),
            nn.ReLU(),
            nn.Conv2d(256//16, 256, 1),
            nn.Sigmoid()
        )
        
        # Global pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _make_conv_block(self, in_channels, out_channels):
        """Create a convolutional block with residual-style architecture"""
        return nn.Sequential(
            # First conv
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
            # Second conv (residual style)
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
            # Pooling and dropout
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25)
        )
    
    def _initialize_weights(self):
        """Initialize network weights with Xavier initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Feature extraction
        x1 = self.conv_block1(x)  # 112x112
        x2 = self.conv_block2(x1) # 56x56
        x3 = self.conv_block3(x2) # 28x28
        x4 = self.conv_block4(x3) # 14x14
        
        # Apply attention mechanism
        attention = self.attention(x4)
        x4 = x4 * attention
        
        # Global pooling and classification
        x = self.global_pool(x4)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x

class PestClassifier(nn.Module):
    """Main pest classifier using focused CNN"""
    
    def __init__(self, num_classes: int):
        super(PestClassifier, self).__init__()
        self.num_classes = num_classes
        self.model = FocusedCNN(num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def get_model_info(self) -> dict:
        """Get model information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': 'FocusedCNN',
            'num_classes': self.num_classes,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024)
        }

# Utility functions
def create_model(num_classes: int) -> PestClassifier:
    """Create a focused CNN pest classifier model"""
    return PestClassifier(num_classes)

def load_model(model_path: str, num_classes: int) -> PestClassifier:
    """Load a trained custom model"""
    model = PestClassifier(num_classes)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    return model

def save_model(model: PestClassifier, save_path: str):
    """Save model state dict"""
    torch.save(model.state_dict(), save_path)

def get_model_summary(model: PestClassifier) -> str:
    """Get a formatted model summary"""
    info = model.get_model_info()
    
    summary = f"""
🤖 **Focused CNN Pest Classifier Summary**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📋 **Model Architecture:** Focused CNN with Attention
🎯 **Number of Classes:** {info['num_classes']}
⚙️  **Total Parameters:** {info['total_parameters']:,}
🔧 **Trainable Parameters:** {info['trainable_parameters']:,}
💾 **Estimated Size:** {info['model_size_mb']:.1f} MB

🏗️ **Architecture Details:**
   • 4 Convolutional Blocks (32→64→128→256 channels)
   • Attention mechanism for feature focusing
   • Batch Normalization + Dropout for regularization
   • Global Average Pooling
   • 2 Fully Connected layers (256→128→classes)
   • ReLU activations throughout

🎯 **Optimized for Small Datasets**

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    return summary