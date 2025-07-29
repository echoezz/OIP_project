import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.pest_classifier import PestClassifier

class PestIdentifier:
    def __init__(self, model_path, classes_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load classes
        with open(classes_path, 'r') as f:
            self.classes = json.load(f)
        
        # Load model
        self.model = PestClassifier(len(self.classes))
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Supported image formats
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    
    def identify_pest(self, image_path_or_pil):
        """Identify pest from JPG or other image formats"""
        try:
            # Handle both file path and PIL image
            if isinstance(image_path_or_pil, str):
                # Check if file exists and has supported extension
                if not os.path.exists(image_path_or_pil):
                    return {
                        'pest_name': None,
                        'confidence': 0,
                        'success': False,
                        'error': 'Image file not found'
                    }
                
                file_ext = os.path.splitext(image_path_or_pil)[1].lower()
                if file_ext not in self.supported_formats:
                    return {
                        'pest_name': None,
                        'confidence': 0,
                        'success': False,
                        'error': f'Unsupported file format. Please use: {", ".join(self.supported_formats)}'
                    }
                
                image = Image.open(image_path_or_pil).convert('RGB')
            else:
                # PIL Image object from Gradio
                image = image_path_or_pil.convert('RGB')
            
            # Preprocess
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Predict
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                confidence, predicted = torch.max(probabilities, 0)
            
            pest_name = self.classes[predicted.item()]
            confidence_score = confidence.item()
            
            return {
                'pest_name': pest_name,
                'confidence': confidence_score,
                'success': True,
                'image_format': 'JPG/JPEG' if isinstance(image_path_or_pil, str) and image_path_or_pil.lower().endswith(('.jpg', '.jpeg')) else 'Other'
            }
        
        except Exception as e:
            return {
                'pest_name': None,
                'confidence': 0,
                'success': False,
                'error': str(e)
            }