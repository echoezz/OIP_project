import os
import json
import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any

def ensure_directory(path: str) -> None:
    """Create directory if it doesn't exist"""
    os.makedirs(path, exist_ok=True)

def load_json(file_path: str) -> Dict[Any, Any]:
    """Load JSON file safely"""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: {file_path} not found")
        return {}
    except json.JSONDecodeError:
        print(f"Warning: Invalid JSON in {file_path}")
        return {}

def save_json(data: Dict[Any, Any], file_path: str) -> None:
    """Save data to JSON file"""
    ensure_directory(os.path.dirname(file_path))
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

def validate_image_file(file_path: str) -> bool:
    """Check if file is a valid JPG image"""
    if not os.path.exists(file_path):
        return False
    
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext not in valid_extensions:
        return False
    
    try:
        with Image.open(file_path) as img:
            img.verify()
        return True
    except:
        return False

def get_device() -> torch.device:
    """Get the best available device"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():  # Apple Silicon
        return torch.device('mps')
    else:
        return torch.device('cpu')

def format_confidence(confidence: float) -> str:
    """Format confidence score as percentage"""
    return f"{confidence * 100:.1f}%"

def crop_center(image: Image.Image, crop_size: tuple) -> Image.Image:
    """Crop image from center"""
    width, height = image.size
    crop_width, crop_height = crop_size
    
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    right = left + crop_width
    bottom = top + crop_height
    
    return image.crop((left, top, right, bottom))

def analyze_dataset_distribution(dataset_path: str) -> Dict[str, int]:
    """Analyze distribution of images across classes"""
    distribution = {}
    
    for class_name in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_name)
        if os.path.isdir(class_path):
            jpg_files = [f for f in os.listdir(class_path) 
                        if f.lower().endswith(('.jpg', '.jpeg'))]
            distribution[class_name] = len(jpg_files)
    
    return distribution

def print_training_progress(epoch: int, total_epochs: int, train_acc: float, 
                          val_acc: float, train_loss: float, val_loss: float):
    """Print formatted training progress"""
    print(f"Epoch [{epoch+1}/{total_epochs}]")
    print(f"  Train: Acc={train_acc:.2f}%, Loss={train_loss:.4f}")
    print(f"  Val:   Acc={val_acc:.2f}%, Loss={val_loss:.4f}")
    print("-" * 50)