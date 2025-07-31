import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import json
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Import the PestClassifier from pest_classifier.py
from pest_classifier import PestClassifier

class PestIdentifier:
    """Enhanced Pest Identifier with TTA and confidence assessment"""
    
    def __init__(self, model_path: str, classes_path: str, device: str = None, enable_tta: bool = True):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.enable_tta = enable_tta
        self.confidence_threshold = 0.3  # Minimum confidence for reliable prediction
        
        print(f"🔧 Initializing PestIdentifier on {self.device}")
        
        # Load class names
        self.class_names = self._load_classes(classes_path)
        self.num_classes = len(self.class_names)
        
        # Load model
        self.model = self._load_model(model_path)
        
        # Define transforms
        self.base_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # TTA transforms
        self.tta_transforms = self._create_tta_transforms()
        
        print(f"✅ Model loaded successfully! Classes: {self.num_classes}, TTA: {enable_tta}")
    
    def _load_classes(self, classes_path: str) -> List[str]:
        """Load class names from JSON file"""
        try:
            with open(classes_path, 'r') as f:
                classes = json.load(f)
            
            if isinstance(classes, dict):
                # If it's a dict, extract the values or keys based on structure
                if 'classes' in classes:
                    return classes['classes']
                elif all(isinstance(k, str) and isinstance(v, int) for k, v in classes.items()):
                    # If it's {class_name: index}, sort by index
                    return [k for k, v in sorted(classes.items(), key=lambda x: x[1])]
                else:
                    return list(classes.keys())
            elif isinstance(classes, list):
                return classes
            else:
                raise ValueError(f"Unexpected classes format: {type(classes)}")
                
        except Exception as e:
            print(f"❌ Error loading classes: {e}")
            # Fallback to default classes if file doesn't exist
            return [
                'aphid', 'armyworm', 'beetle', 'bollworm', 'earthworm', 
                'grasshopper', 'mites', 'mosquito', 'sawfly', 'stem_borer'
            ]
    
    def _load_model(self, model_path: str) -> nn.Module:
        """Load the trained model with proper checkpoint handling"""
        try:
            # Create model instance
            model = PestClassifier(num_classes=self.num_classes)
            
            # Print model architecture for debugging
            print(f"🏗️ Model architecture:")
            print(f"   - Features: {len(list(model.features.children()))} layers")
            print(f"   - Classifier: {len(list(model.classifier.children()))} layers")
            print(f"   - Total parameters: {sum(p.numel() for p in model.parameters())}")
            
            # Load checkpoint
            print(f"📦 Loading checkpoint from {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Debug: Check checkpoint structure
            print(f"🔍 Checkpoint keys: {list(checkpoint.keys())}")
            
            # Check if it's a full checkpoint or just state_dict
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    # It's a full checkpoint
                    model_state_dict = checkpoint['model_state_dict']
                    print(f"📊 Checkpoint info: Epoch {checkpoint.get('epoch', 'N/A')}, "
                          f"Best Val Acc: {checkpoint.get('best_val_acc', 'N/A'):.3f}")
                else:
                    # It's just the state_dict
                    model_state_dict = checkpoint
            else:
                # Old format
                model_state_dict = checkpoint
            
            # Debug: Check state_dict keys
            print(f"🔍 State dict keys (first 10): {list(model_state_dict.keys())[:10]}")
            print(f"🔍 Expected model keys (first 10): {list(model.state_dict().keys())[:10]}")
            
            # Check if keys match
            model_keys = set(model.state_dict().keys())
            checkpoint_keys = set(model_state_dict.keys())
            missing_keys = model_keys - checkpoint_keys
            unexpected_keys = checkpoint_keys - model_keys
            
            if missing_keys:
                print(f"⚠️ Missing keys: {list(missing_keys)[:5]}...")
            if unexpected_keys:
                print(f"⚠️ Unexpected keys: {list(unexpected_keys)[:5]}...")
            
            # Load the state dict
            model.load_state_dict(model_state_dict)
            model.to(self.device)
            model.eval()
            
            print("✅ Model loaded and ready for inference")
            return model
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print(f"💡 Creating new model instance (for testing)")
            # Create a new model for testing
            model = PestClassifier(num_classes=self.num_classes)
            model.to(self.device)
            model.eval()
            print("⚠️ Using untrained model - predictions will be random!")
            return model
    
    def _create_tta_transforms(self) -> List[transforms.Compose]:
        """Create Test-Time Augmentation transforms"""
        if not self.enable_tta:
            return [self.base_transform]
        
        tta_list = [
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
            # Brightness adjustment
            transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ColorJitter(brightness=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            # Combined augmentations
            transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        ]
        
        return tta_list
    
    def _predict_single(self, image: Image.Image, transform: transforms.Compose) -> np.ndarray:
        """Make prediction on single transformed image"""
        try:
            # Transform image
            input_tensor = transform(image).unsqueeze(0).to(self.device)
            
            # Predict
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                probs_np = probabilities.cpu().numpy()[0]
                
                # Debug: Print raw outputs and probabilities
                print(f"🔍 Raw model outputs: {outputs.cpu().numpy()[0][:5]}...")  # First 5 values
                print(f"🔍 Probabilities: {probs_np[:5]}...")  # First 5 probabilities
                print(f"🔍 Max probability: {np.max(probs_np):.4f} at index {np.argmax(probs_np)}")
                print(f"🔍 Predicted class: {self.class_names[np.argmax(probs_np)]}")
                
                return probs_np
        except Exception as e:
            print(f"⚠️ Prediction error: {e}")
            # Return uniform distribution as fallback
            return np.ones(self.num_classes) / self.num_classes
    
    def _assess_confidence(self, confidence: float, method: str = 'standard') -> Tuple[str, str]:
        """Assess confidence level and provide description"""
        if method == 'TTA_enhanced':
            # Stricter thresholds for TTA
            if confidence >= 0.7:
                level = "High"
                desc = "Very confident identification with TTA consensus"
            elif confidence >= 0.4:
                level = "Medium"
                desc = "Moderately confident with TTA enhancement"
            else:
                level = "Low"
                desc = "Low confidence - consider retaking photo"
        else:
            # Standard thresholds
            if confidence >= 0.6:
                level = "High"
                desc = "Confident identification"
            elif confidence >= 0.3:
                level = "Medium"
                desc = "Reasonable confidence"
            else:
                level = "Low"
                desc = "Low confidence - image may be unclear"
        
        return level, desc
    
    def identify_pest(self, image: Image.Image) -> Dict:
        """Basic pest identification"""
        try:
            probabilities = self._predict_single(image, self.base_transform)
            predicted_class = np.argmax(probabilities)
            confidence = float(probabilities[predicted_class])
            
            pest_name = self.class_names[predicted_class]
            confidence_level, confidence_desc = self._assess_confidence(confidence)
            
            return {
                'success': True,
                'pest_name': pest_name,
                'confidence': confidence,
                'confidence_level': confidence_level,
                'confidence_description': confidence_desc,
                'meets_threshold': confidence >= self.confidence_threshold,
                'method': 'standard',
                'probabilities': probabilities.tolist()
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def identify_pest_with_tta(self, image: Image.Image) -> Dict:
        """Enhanced pest identification with Test-Time Augmentation"""
        if not self.enable_tta:
            return self.identify_pest(image)
        
        try:
            all_predictions = []
            
            # Get predictions from all TTA transforms
            for transform in self.tta_transforms:
                probs = self._predict_single(image, transform)
                all_predictions.append(probs)
            
            # Average predictions
            avg_probabilities = np.mean(all_predictions, axis=0)
            predicted_class = np.argmax(avg_probabilities)
            confidence = float(avg_probabilities[predicted_class])
            
            # Calculate prediction consistency
            individual_predictions = [np.argmax(pred) for pred in all_predictions]
            consistency = individual_predictions.count(predicted_class) / len(individual_predictions)
            
            pest_name = self.class_names[predicted_class]
            confidence_level, confidence_desc = self._assess_confidence(confidence, 'TTA_enhanced')
            
            return {
                'success': True,
                'pest_name': pest_name,
                'confidence': confidence,
                'confidence_level': confidence_level,
                'confidence_description': confidence_desc,
                'meets_threshold': confidence >= self.confidence_threshold,
                'method': 'TTA_enhanced',
                'probabilities': avg_probabilities.tolist(),
                'prediction_consistency': consistency,
                'num_augmentations': len(self.tta_transforms)
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def get_top_predictions(self, image: Image.Image, top_k: int = 3) -> Dict:
        """Get top-k predictions with enhanced analysis"""
        try:
            # Use TTA if enabled, otherwise standard
            if self.enable_tta:
                result = self.identify_pest_with_tta(image)
            else:
                result = self.identify_pest(image)
            
            if not result['success']:
                return result
            
            # Get top-k predictions
            probabilities = np.array(result['probabilities'])
            top_indices = np.argsort(probabilities)[::-1][:top_k]
            
            top_predictions = []
            for i, idx in enumerate(top_indices):
                top_predictions.append({
                    'rank': i + 1,
                    'pest_name': self.class_names[idx],
                    'confidence': float(probabilities[idx]),
                    'confidence_percent': f"{probabilities[idx]:.1%}"
                })
            
            # Add top predictions to result
            result['top_predictions'] = top_predictions
            
            return result
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def batch_predict(self, images: List[Image.Image]) -> List[Dict]:
        """Predict multiple images"""
        results = []
        for i, image in enumerate(images):
            print(f"🔍 Processing image {i+1}/{len(images)}")
            result = self.get_top_predictions(image)
            results.append(result)
        return results