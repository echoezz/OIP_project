import torch
from PIL import Image
import torchvision.transforms as transforms
import json
from typing import Dict, Any, List
import os
import numpy as np
import random

from models.pest_classifier import PestClassifier

class PestIdentifier:
    def __init__(self, model_path: str, classes_path: str, enable_tta: bool = True):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.classes = self.load_classes(classes_path)
        self.model = self.load_model(model_path)
        self.enable_tta = enable_tta
        self.confidence_threshold = 0.25  # Lowered threshold
        
        # Standard transform for single predictions
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # TTA transforms for enhanced predictions
        self.tta_transforms = self._create_tta_transforms()
    
    def load_classes(self, classes_path: str) -> list:
        """Load class names"""
        with open(classes_path, 'r') as f:
            return json.load(f)
    
    def load_model(self, model_path: str) -> PestClassifier:
        """Load the trained focused CNN model"""
        num_classes = len(self.classes)
        model = PestClassifier(num_classes)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model
    
    def _create_tta_transforms(self) -> List[transforms.Compose]:
        """Create test-time augmentation transforms"""
        return [
            # Original
            transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            # Horizontal flip
            transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            # Slight rotation
            transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            # Scale variation
            transforms.Compose([
                transforms.Resize((240, 240)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            # Color jitter
            transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            # Vertical flip (for insects)
            transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomVerticalFlip(p=1.0),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            # Slight zoom
            transforms.Compose([
                transforms.Resize((200, 200)),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            # Different crop
            transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        ]
    
    def predict_single(self, image: Image.Image) -> Dict[str, Any]:
        """Single prediction without TTA"""
        try:
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                confidence, predicted_idx = torch.max(probabilities, 0)
            
            pest_name = self.classes[predicted_idx.item()]
            confidence_score = confidence.item()
            
            return {
                'success': True,
                'pest_name': pest_name,
                'confidence': confidence_score,
                'method': 'single_prediction',
                'all_predictions': {
                    self.classes[i]: prob.item() 
                    for i, prob in enumerate(probabilities)
                }
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def predict_with_tta(self, image: Image.Image) -> Dict[str, Any]:
        """Test-time augmentation for enhanced predictions"""
        try:
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            all_predictions = []
            
            with torch.no_grad():
                # Apply each TTA transform
                for transform in self.tta_transforms:
                    try:
                        input_tensor = transform(image).unsqueeze(0).to(self.device)
                        outputs = self.model(input_tensor)
                        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                        all_predictions.append(probabilities.cpu().numpy())
                    except:
                        # Skip failed transforms
                        continue
                
                if not all_predictions:
                    # Fallback to single prediction
                    return self.predict_single(image)
                
                # Average all predictions
                avg_predictions = np.mean(all_predictions, axis=0)
                confidence = np.max(avg_predictions)
                predicted_idx = np.argmax(avg_predictions)
                
                pest_name = self.classes[predicted_idx]
                
                # Calculate prediction variance for confidence adjustment
                prediction_std = np.std([pred[predicted_idx] for pred in all_predictions])
                
                # Adjust confidence based on consistency across augmentations
                consistency_bonus = 1.0 - prediction_std
                adjusted_confidence = confidence * (0.8 + 0.2 * consistency_bonus)
                
                return {
                    'success': True,
                    'pest_name': pest_name,
                    'confidence': adjusted_confidence,
                    'method': 'TTA_enhanced',
                    'num_augmentations': len(all_predictions),
                    'prediction_consistency': consistency_bonus,
                    'all_predictions': {
                        self.classes[i]: float(prob) for i, prob in enumerate(avg_predictions)
                    }
                }
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def identify_pest(self, image: Image.Image) -> Dict[str, Any]:
        """Main identification method with enhanced confidence"""
        if self.enable_tta:
            result = self.predict_with_tta(image)
        else:
            result = self.predict_single(image)
        
        if not result['success']:
            return result
        
        # Enhanced confidence assessment
        confidence = result['confidence']
        pest_name = result['pest_name']
        
        # Add confidence level description
        if confidence >= 0.7:
            confidence_level = "High"
            confidence_desc = "Very confident identification"
        elif confidence >= 0.4:
            confidence_level = "Medium"
            confidence_desc = "Confident identification"
        elif confidence >= 0.25:
            confidence_level = "Low"
            confidence_desc = "Possible identification - consider additional angles"
        else:
            confidence_level = "Very Low"
            confidence_desc = "Uncertain identification - please try a clearer image"
            pest_name = "uncertain"
        
        result.update({
            'confidence_level': confidence_level,
            'confidence_description': confidence_desc,
            'meets_threshold': confidence >= self.confidence_threshold
        })
        
        return result
    
    def get_top_predictions(self, image: Image.Image, top_k: int = 3) -> Dict[str, Any]:
        """Get top K predictions with confidence scores"""
        result = self.identify_pest(image)
        
        if not result['success']:
            return result
        
        # Sort predictions by confidence
        all_preds = result['all_predictions']
        sorted_preds = sorted(all_preds.items(), key=lambda x: x[1], reverse=True)
        
        top_predictions = []
        for i, (pest_name, confidence) in enumerate(sorted_preds[:top_k]):
            top_predictions.append({
                'rank': i + 1,
                'pest_name': pest_name,
                'confidence': confidence,
                'confidence_percent': f"{confidence:.1%}"
            })
        
        result['top_predictions'] = top_predictions
        return result
    
    def batch_identify(self, images: List[Image.Image]) -> List[Dict[str, Any]]:
        """Identify multiple images efficiently"""
        results = []
        
        for i, image in enumerate(images):
            print(f"Processing image {i+1}/{len(images)}...")
            result = self.identify_pest(image)
            results.append(result)
        
        return results
    
    def set_confidence_threshold(self, threshold: float):
        """Adjust confidence threshold"""
        self.confidence_threshold = max(0.1, min(0.9, threshold))
        print(f"Confidence threshold set to {self.confidence_threshold:.2f}")
    
    def toggle_tta(self, enable: bool):
        """Enable or disable test-time augmentation"""
        self.enable_tta = enable
        method = "TTA enabled" if enable else "Single prediction only"
        print(f"Prediction method: {method}")