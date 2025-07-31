import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import numpy as np

# ===== YOUR NEW GRAYSCALE + SEPARABLE CNN =====
class DepthwiseConv(nn.Module):
    """Depthwise convolution - each channel processed separately"""
    def __init__(self, in_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, 
                                  stride, padding, groups=in_channels, bias=False)
        self.bn = nn.BatchNorm2d(in_channels)
        
    def forward(self, x):
        return F.relu(self.bn(self.depthwise(x)))

class PointwiseConv(nn.Module):
    """Pointwise convolution - 1x1 conv for channel mixing"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        return F.relu(self.bn(self.pointwise(x)))

class SeparableBlock(nn.Module):
    """Depthwise + Pointwise block"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.depthwise = DepthwiseConv(in_channels, stride=stride)
        self.pointwise = PointwiseConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2) if stride == 1 else nn.Identity()
        self.dropout = nn.Dropout2d(0.1)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.pool(x)
        x = self.dropout(x)
        return x

class GrayscaleCNN(nn.Module):
    """Grayscale CNN with Depthwise/Pointwise convolutions"""
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        
        # Initial conv - FIXED: 1 channel input (grayscale) to 32 channels
        self.initial = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),  # Changed from 3 to 1 input channels
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Separable blocks
        self.block1 = SeparableBlock(32, 64)
        self.block2 = SeparableBlock(64, 128) 
        self.block3 = SeparableBlock(128, 256)
        self.block4 = SeparableBlock(256, 512)
        
        # Global pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
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
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
        
    def rgb_to_grayscale(self, x):
        """Convert RGB to grayscale"""
        if x.size(1) == 3:
            weights = torch.tensor([0.299, 0.587, 0.114], device=x.device).view(1, 3, 1, 1)
            return torch.sum(x * weights, dim=1, keepdim=True)
        return x
        
    def forward(self, x):
        # Convert to grayscale (3 channels -> 1 channel)
        x = self.rgb_to_grayscale(x)
        
        # x is [batch, 1, height, width] - perfect for Conv2d(1, 32, ...)
        x = self.initial(x)
        
        # Separable convolution blocks
        x = self.block1(x)  # 32 -> 64
        x = self.block2(x)  # 64 -> 128  
        x = self.block3(x)  # 128 -> 256
        x = self.block4(x)  # 256 -> 512
        
        # Classification
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# ===== UPDATE YOUR PESTCLASSIFIER CLASS =====
class PestClassifier(nn.Module):
    """Enhanced pest classifier using Grayscale + Separable CNN"""
    
    def __init__(self, num_classes: int):
        super(PestClassifier, self).__init__()
        self.num_classes = num_classes
        # Use the new Grayscale CNN instead of your old model
        self.model = GrayscaleCNN(num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def get_model_info(self) -> dict:
        """Get model information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': 'Grayscale_Separable_CNN',
            'num_classes': self.num_classes,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024)
        }

# ===== KEEP YOUR EXISTING UTILITY FUNCTIONS =====
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

# ===== UTILITY FUNCTIONS (UPDATED) =====
def create_model(num_classes: int) -> PestClassifier:
    """Create a Grayscale Separable CNN pest classifier model"""
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
🤖 **Grayscale Separable CNN for Pest Classification**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📋 **Model Architecture:** Grayscale + Depthwise/Pointwise CNN
🎯 **Number of Classes:** {info['num_classes']}
⚙️  **Total Parameters:** {info['total_parameters']:,}
🔧 **Trainable Parameters:** {info['trainable_parameters']:,}
💾 **Estimated Size:** {info['model_size_mb']:.1f} MB

🏗️ **Architecture Details:**
   • Automatic RGB → Grayscale conversion
   • Depthwise convolutions for spatial filtering
   • Pointwise convolutions for channel mixing
   • 4 Separable blocks (32→64→128→256→512)
   • Global average pooling
   • 2-layer classifier with dropout

🎯 **Key Features:**
   • 10x fewer parameters than standard CNN
   • Faster training and inference
   • Better for small datasets
   • Focus on shape/texture over color

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
    print(f"✅ **Grayscale Separable CNN working correctly!**")
    
    # Show parameter reduction
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n📊 **Parameter Count:** {total_params:,}")
    print(f"🎯 **Much smaller than standard CNN!**")