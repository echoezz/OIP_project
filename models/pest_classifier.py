import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import numpy as np

class ResidualBlock(nn.Module):
    """Residual block with skip connection"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        # Main path (your existing convolutions)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection (THIS IS THE KEY!)
        self.skip_connection = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip_connection = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        # Keep your pooling and dropout
        self.pool = nn.MaxPool2d(2) if stride == 1 else nn.Identity()
        self.dropout = nn.Dropout2d(0.25)
        
    def forward(self, x):
        # Save input for skip connection
        identity = self.skip_connection(x)
        
        # Main path (your existing flow)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # ADD SKIP CONNECTION HERE! ⭐
        out += identity  # This is the magic!
        out = F.relu(out)
        
        # Apply pooling and dropout
        out = self.pool(out)
        out = self.dropout(out)
        
        return out

class FocusedCNN(nn.Module):
    """Enhanced Focused CNN with residual connections and dual attention"""
    
    def __init__(self, num_classes: int):
        super(FocusedCNN, self).__init__()
        self.num_classes = num_classes
        
        # Feature extraction with TRUE residual blocks
        self.conv_block1 = ResidualBlock(3, 32, stride=1)    # No downsampling first
        self.conv_block2 = ResidualBlock(32, 64, stride=2)   # Downsample here
        self.conv_block3 = ResidualBlock(64, 128, stride=2)  # Downsample here  
        self.conv_block4 = ResidualBlock(128, 256, stride=2) # Downsample here
        self.conv_block5 = ResidualBlock(256, 512, stride=2) # Added 5th block
        
        # Enhanced dual attention mechanism
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 512//16, 1),
            nn.ReLU(),
            nn.Conv2d(512//16, 512, 1),
            nn.Sigmoid()
        )
        
        # Spatial attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        
        # Enhanced classifier with more capacity
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize network weights with He initialization for ReLU"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Feature extraction with residual connections
        x1 = self.conv_block1(x)   # 224x224 -> 224x224
        x2 = self.conv_block2(x1)  # 224x224 -> 112x112
        x3 = self.conv_block3(x2)  # 112x112 -> 56x56
        x4 = self.conv_block4(x3)  # 56x56 -> 28x28
        x5 = self.conv_block5(x4)  # 28x28 -> 14x14
        
        # Apply dual attention mechanism
        # Channel attention
        channel_att = self.channel_attention(x5)
        x5 = x5 * channel_att
        
        # Spatial attention
        avg_pool = torch.mean(x5, dim=1, keepdim=True)
        max_pool, _ = torch.max(x5, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_pool, max_pool], dim=1)
        spatial_att = self.spatial_attention(spatial_input)
        x5 = x5 * spatial_att
        
        # Global pooling and classification
        x = self.global_pool(x5)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x

class PestClassifier(nn.Module):
    """Enhanced pest classifier using focused CNN with residual connections"""
    
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
            'model_name': 'Enhanced_FocusedCNN_with_Residual',
            'num_classes': self.num_classes,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024)
        }

# Advanced training utilities
class MixupLoss(nn.Module):
    """Mixup loss for better generalization"""
    def __init__(self, alpha=0.2):
        super().__init__()
        self.alpha = alpha
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, outputs, targets_a, targets_b, lam):
        return lam * self.criterion(outputs, targets_a) + (1 - lam) * self.criterion(outputs, targets_b)

def mixup_data(x, y, alpha=0.2):
    """Mixup augmentation for training"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam

class LabelSmoothingLoss(nn.Module):
    """Label smoothing for better generalization"""
    def __init__(self, num_classes, smoothing=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
    
    def forward(self, pred, target):
        pred = F.log_softmax(pred, dim=1)
        true_dist = torch.zeros_like(pred)
        true_dist.fill_(self.smoothing / (self.num_classes - 1))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=1))

# Utility functions
def create_model(num_classes: int) -> PestClassifier:
    """Create an enhanced focused CNN pest classifier model with residual connections"""
    return PestClassifier(num_classes)

def load_model(model_path: str, num_classes: int) -> PestClassifier:
    """Load a trained custom model"""
    model = PestClassifier(num_classes)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    return model

def save_model(model: PestClassifier, save_path: str):
    """Save model state dict"""
    torch.save(model.state_dict(), save_path)

def count_parameters(model: PestClassifier) -> tuple:
    """Count total and trainable parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def get_model_summary(model: PestClassifier) -> str:
    """Get a formatted model summary"""
    info = model.get_model_info()
    
    summary = f"""
🤖 **Enhanced Focused CNN with Residual Connections**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📋 **Model Architecture:** Enhanced Focused CNN + TRUE Residual Connections
🎯 **Number of Classes:** {info['num_classes']}
⚙️  **Total Parameters:** {info['total_parameters']:,}
🔧 **Trainable Parameters:** {info['trainable_parameters']:,}
💾 **Estimated Size:** {info['model_size_mb']:.1f} MB

🏗️ **Architecture Details:**
   • 5 TRUE Residual Blocks (32→64→128→256→512 channels)
   • Skip connections for gradient flow (ResNet-style)
   • Dual Attention: Channel + Spatial focusing
   • 3-layer enhanced classifier (512→256→128→classes)
   • He initialization optimized for ReLU
   • Advanced regularization (BatchNorm + Dropout)


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    return summary

# Test the model
if __name__ == "__main__":
    # Create a test model
    num_classes = 12  # Example for 12 pest classes
    model = create_model(num_classes)
    
    # Print model summary
    print(get_model_summary(model))
    
    # Test with dummy input
    x = torch.randn(1, 3, 224, 224)
    output = model(x)
    print(f"\n🧪 **Test Output Shape:** {output.shape}")
    print(f"✅ **Model working correctly!**")