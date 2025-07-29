import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class CustomCNN(nn.Module):
    
    def __init__(self, num_classes: int, input_channels: int = 3):
        super(CustomCNN, self).__init__()
        self.num_classes = num_classes
        
        # First Convolutional Block
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)  # 224x224 -> 112x112
        self.dropout1 = nn.Dropout2d(0.25)
        
        # Second Convolutional Block
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)  # 112x112 -> 56x56
        self.dropout2 = nn.Dropout2d(0.25)
        
        # Third Convolutional Block
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)  # 56x56 -> 28x28
        self.dropout3 = nn.Dropout2d(0.3)
        
        # Fourth Convolutional Block
        self.conv7 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(256)
        self.conv8 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)  # 28x28 -> 14x14
        self.dropout4 = nn.Dropout2d(0.3)
        
        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(256, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.dropout_fc1 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(512, 256)
        self.bn_fc2 = nn.BatchNorm1d(256)
        self.dropout_fc2 = nn.Dropout(0.3)
        
        # Output layer
        self.fc_out = nn.Linear(256, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # First block
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Second block
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Third block
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool3(x)
        x = self.dropout3(x)
        
        # Fourth block
        x = F.relu(self.bn7(self.conv7(x)))
        x = F.relu(self.bn8(self.conv8(x)))
        x = self.pool4(x)
        x = self.dropout4(x)
        
        # Global average pooling
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        
        # Fully connected layers
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout_fc1(x)
        
        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout_fc2(x)
        
        # Output
        x = self.fc_out(x)
        
        return x

class PestClassifier(nn.Module):
    """Main pest classifier using custom CNN"""
    
    def __init__(self, num_classes: int):
        super(PestClassifier, self).__init__()
        self.num_classes = num_classes
        self.model = CustomCNN(num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def get_model_info(self) -> dict:
        """Get model information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': 'CustomCNN',
            'num_classes': self.num_classes,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024)
        }

# Utility functions
def create_model(num_classes: int) -> PestClassifier:
    """Create a custom CNN pest classifier model"""
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
🤖 **Custom CNN Pest Classifier Summary**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📋 **Model Architecture:** Custom CNN (Built from Scratch)
🎯 **Number of Classes:** {info['num_classes']}
⚙️  **Total Parameters:** {info['total_parameters']:,}
🔧 **Trainable Parameters:** {info['trainable_parameters']:,}
💾 **Estimated Size:** {info['model_size_mb']:.1f} MB

🏗️ **Architecture Details:**
   • 4 Convolutional Blocks (32→64→128→256 channels)
   • Batch Normalization + Dropout for regularization
   • Global Average Pooling
   • 2 Fully Connected layers (512→256→classes)
   • ReLU activations throughout

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    return summary