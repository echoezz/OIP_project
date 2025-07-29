import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import sys
import json
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.pest_classifier import PestClassifier, get_model_summary
from src.data_loader import PestDataLoader

def train_custom_cnn():
    """Train custom CNN model on JPG pest images"""
    
    print("🚀 Starting Custom CNN Pest Classification Training")
    print("=" * 60)
    
    config = {
        'data_dir': 'datasets', # Directory containing training data
        'model_save_path': 'models/saved_models',
        'epochs': 30,  # Number of training epochs
        'learning_rate': 0.001, # Initial learning rate
        'batch_size': 32, # Batch size for training
        'img_size': 224 # Input image size (224x224 for ResNet-like models)
    }
    
    # Create save directory
    os.makedirs(config['model_save_path'], exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    
    # Data loaders
    data_loader = PestDataLoader(
        config['data_dir'], 
        batch_size=config['batch_size'],
        img_size=config['img_size']
    )
    
    train_loader, val_loader, classes = data_loader.get_data_loaders()
    num_classes = len(classes)
    
    print(f"\n🎯 Training Configuration:")
    print(f"   Classes: {num_classes}")
    print(f"   Epochs: {config['epochs']}")
    print(f"   Batch size: {config['batch_size']}")
    print(f"   Learning rate: {config['learning_rate']}")
    
    # Model - NO model_name parameter
    model = PestClassifier(num_classes).to(device)
    
    # Print model info
    print(f"\n{get_model_summary(model)}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    # Training loop
    best_val_acc = 0.0
    training_history = {'train_acc': [], 'val_acc': [], 'train_loss': [], 'val_loss': []}
    
    print(f"\n🎓 Starting Training...")
    start_time = time.time()
    
    for epoch in range(config['epochs']):
        print(f"\n📅 Epoch {epoch+1}/{config['epochs']}")
        print("-" * 40)
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_bar = tqdm(train_loader, desc='Training')
        for batch_idx, (images, labels) in enumerate(train_bar):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # Update progress bar
            current_acc = 100 * train_correct / train_total
            train_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{current_acc:.2f}%'
            })
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc='Validation')
            for images, labels in val_bar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        # Step scheduler
        scheduler.step()
        
        # Calculate metrics
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        # Save metrics
        training_history['train_acc'].append(train_acc)
        training_history['val_acc'].append(val_acc)
        training_history['train_loss'].append(avg_train_loss)
        training_history['val_loss'].append(avg_val_loss)
        
        # Print epoch results
        print(f"📊 Results:")
        print(f"   Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"   Val Loss: {avg_val_loss:.4f}   | Val Acc: {val_acc:.2f}%")
        print(f"   Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"{config['model_save_path']}/best_model.pth")
            print(f"🏆 New best model saved! Validation accuracy: {best_val_acc:.2f}%")
    
    # Training completed
    total_time = time.time() - start_time
    print(f"\n🎉 Training completed in {total_time/60:.1f} minutes!")
    print(f"🏆 Best validation accuracy: {best_val_acc:.2f}%")
    
    # Save training info
    training_info = {
        'classes': classes,
        'num_classes': num_classes,
        'best_val_acc': best_val_acc,
        'config': config,
        'training_history': training_history,
        'model_type': 'CustomCNN'
    }
    
    with open(f"{config['model_save_path']}/classes.json", 'w') as f:
        json.dump(classes, f)
    
    with open(f"{config['model_save_path']}/training_info.json", 'w') as f:
        json.dump(training_info, f, indent=2)
    
    print(f"💾 Model and training info saved to {config['model_save_path']}/")

if __name__ == "__main__":
    train_custom_cnn()