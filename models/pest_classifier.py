import torch
import torch.nn as nn
from torchvision import models
from typing import Optional

class PestClassifier(nn.Module):
    def __init__(self, num_classes: int, model_name: str = 'efficientnet_b0', pretrained: bool = True):
        super(PestClassifier, self).__init__()
        self.num_classes = num_classes
        self.model_name = model_name
        
        # Load pre-trained model
        if model_name == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            # Replace classifier
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, num_classes)
            )
        
        elif model_name == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, num_classes)
            )
        
        elif model_name == 'mobilenet_v3_small':
            self.backbone = models.mobilenet_v3_small(pretrained=pretrained)
            num_features = self.backbone.classifier[3].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Linear(num_features, 512),
                nn.Hardswish(),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.Hardswish(),
                nn.Dropout(0.2),
                nn.Linear(256, num_classes)
            )
        
        else:
            raise ValueError(f"Unsupported model: {model_name}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
    
    def get_model_info(self) -> dict:
        """Get model information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': self.model_name,
            'num_classes': self.num_classes,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024)  # Rough estimate
        }
    
    def freeze_backbone(self):
        """Freeze backbone parameters for transfer learning"""
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Unfreeze classifier layers
        if self.model_name == 'efficientnet_b0':
            for param in self.backbone.classifier.parameters():
                param.requires_grad = True
        elif self.model_name == 'resnet50':
            for param in self.backbone.fc.parameters():
                param.requires_grad = True
        elif self.model_name == 'mobilenet_v3_small':
            for param in self.backbone.classifier.parameters():
                param.requires_grad = True
    
    def unfreeze_all(self):
        """Unfreeze all parameters"""
        for param in self.parameters():
            param.requires_grad = True

# Additional utility functions for model management
def create_model(num_classes: int, model_name: str = 'efficientnet_b0') -> PestClassifier:
    """Create a pest classifier model"""
    return PestClassifier(num_classes, model_name)

def load_model(model_path: str, num_classes: int, model_name: str = 'efficientnet_b0') -> PestClassifier:
    """Load a trained model"""
    model = PestClassifier(num_classes, model_name)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    return model

def save_model(model: PestClassifier, save_path: str):
    """Save model state dict"""
    torch.save(model.state_dict(), save_path)

def get_model_summary(model: PestClassifier) -> str:
    """Get a formatted model summary"""
    info = model.get_model_info()
    
    summary = f"""
🤖 **Pest Classifier Model Summary**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📋 **Model Architecture:** {info['model_name']}
🎯 **Number of Classes:** {info['num_classes']}
⚙️  **Total Parameters:** {info['total_parameters']:,}
🔧 **Trainable Parameters:** {info['trainable_parameters']:,}
💾 **Estimated Size:** {info['model_size_mb']:.1f} MB

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    return summary