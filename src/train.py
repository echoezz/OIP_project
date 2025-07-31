import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import json
import os
from tqdm import tqdm

# Import your updated classifier
from pest_classifier import create_model, save_model, get_model_summary
from data_loader import get_balanced_data_loaders  # Your balanced data loader

def train_grayscale_separable_model():
    """Train the new Grayscale Separable CNN"""
    
    print("🎯 TRAINING GRAYSCALE SEPARABLE CNN")
    print("=" * 60)
    
    # Configuration (you can keep your existing config)
    config = {
        'data_dir': 'datasets',
        'model_save_path': 'models/saved_models',
        'epochs': 100,
        'learning_rate': 0.0008,
        'batch_size': 16,
        'patience': 15,
        'weight_decay': 1e-5
    }
    
    # Create save directory
    os.makedirs(config['model_save_path'], exist_ok=True)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    
    # Load data (use your balanced data loader)
    print("📊 Loading balanced dataset...")
    try:
        train_loader, val_loader, classes = get_balanced_data_loaders(
            data_dir=config['data_dir'],
            batch_size=config['batch_size'],
            auto_balance=True  # Use your class balancing
        )
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return None, 0
    
    num_classes = len(classes)
    print(f"✅ Dataset loaded: {num_classes} classes")
    
    # Create the NEW model (automatically uses Grayscale + Separable CNN)
    model = create_model(num_classes).to(device)
    
    # Print model info
    print(get_model_summary(model))
    
    # Optimizer (AdamW works better with the new architecture)
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['epochs'], eta_min=1e-6
    )
    
    # Loss function (Label smoothing works well with the new model)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Training tracking
    best_val_acc = 0.0
    patience_counter = 0
    train_losses = []
    val_accuracies = []
    
    print(f"\n🚀 Starting training for {config['epochs']} epochs...")
    start_time = time.time()
    
    for epoch in range(config['epochs']):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["epochs"]} [Train]')
        
        for batch_idx, (data, targets) in enumerate(train_bar):
            data, targets = data.to(device), targets.to(device)
            
            # Forward pass (model automatically converts RGB to grayscale)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            train_acc = 100. * train_correct / train_total
            train_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{train_acc:.2f}%'
            })
        
        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{config["epochs"]} [Val]')
            
            for data, targets in val_bar:
                data, targets = data.to(device), targets.to(device)
                outputs = model(data)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
                
                val_acc = 100. * val_correct / val_total
                val_bar.set_postfix({'Acc': f'{val_acc:.2f}%'})
        
        # Calculate epoch metrics
        epoch_train_loss = train_loss / len(train_loader)
        epoch_train_acc = 100. * train_correct / train_total
        epoch_val_acc = 100. * val_correct / val_total
        
        train_losses.append(epoch_train_loss)
        val_accuracies.append(epoch_val_acc)
        
        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print epoch results
        print(f"\nEpoch {epoch+1}/{config['epochs']}:")
        print(f"  Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.2f}%")
        print(f"  Val Acc: {epoch_val_acc:.2f}% | LR: {current_lr:.2e}")
        
        # Save best model
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            patience_counter = 0
            
            # Save model
            save_path = os.path.join(config['model_save_path'], 'best_grayscale_model.pth')
            save_model(model, save_path)
            print(f"  ✅ New best model saved! Accuracy: {best_val_acc:.2f}%")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config['patience']:
            print(f"\n⏹️  Early stopping triggered after {epoch+1} epochs")
            break
        
        print("-" * 60)
    
    # Training completed
    total_time = time.time() - start_time
    
    print(f"\n🎉 TRAINING COMPLETED!")
    print("=" * 60)
    print(f"🏆 Best validation accuracy: {best_val_acc:.2f}%")
    print(f"⏱️  Total training time: {total_time/60:.1f} minutes")
    print(f"💾 Best model saved at: {save_path}")
    
    # Save training history
    history = {
        'train_losses': train_losses,
        'val_accuracies': val_accuracies,
        'best_val_acc': best_val_acc,
        'classes': classes,
        'config': config,
        'model_type': 'Grayscale_Separable_CNN'
    }
    
    history_path = os.path.join(config['model_save_path'], 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"📊 Training history saved at: {history_path}")
    
    return model, best_val_acc

def main():
    """Main training function"""
    print("🚀 GRAYSCALE SEPARABLE CNN TRAINING")
    print("🎯 Automatic RGB → Grayscale conversion")
    print("⚡ 10x fewer parameters than standard CNN")
    print("🏃‍♂️ Much faster training")
    print("=" * 60)
    
    # Train the model
    model, final_accuracy = train_grayscale_separable_model()
    
    if model is not None:
        print(f"\n🎯 FINAL RESULT: {final_accuracy:.2f}%")
        
        if final_accuracy >= 70:
            print("🏆 EXCELLENT! Great performance!")
        elif final_accuracy >= 60:
            print("👍 GOOD! Solid performance!")
        elif final_accuracy >= 50:
            print("📈 DECENT! Room for improvement!")
        else:
            print("📊 BASELINE! Consider more data or tuning!")
        
        # Model info
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\n📊 Model Statistics:")
        print(f"   Parameters: {total_params:,}")
        print(f"   Architecture: Grayscale + Separable CNN")
        print(f"   Features: Depthwise + Pointwise convolutions")
        print(f"   Efficiency: ~10x faster than standard CNN")

if __name__ == "__main__":
    main()