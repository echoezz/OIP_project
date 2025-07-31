import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import json
import os
import sys
from tqdm import tqdm

# Add parent directory to path to find pest_classifier.py
sys.path.append('..')

# Now these imports will work
from pest_classifier import create_model, save_model, get_model_summary
from data_loader import get_balanced_data_loaders

class LossTracker:
    """Track smoothed loss for stable monitoring"""
    def __init__(self, window_size=50):
        self.losses = []
        self.window_size = window_size
    
    def update(self, loss):
        self.losses.append(loss)
        if len(self.losses) > self.window_size:
            self.losses.pop(0)
    
    def get_average(self):
        return sum(self.losses) / len(self.losses) if self.losses else 0

def train_grayscale_separable_model():
    """Train the new Grayscale Separable CNN with STABLE settings"""
    
    print("🎯 STABLE TRAINING - GRAYSCALE SEPARABLE CNN")
    print("=" * 60)
    
    # STABLE Configuration
    config = {
        'data_dir': 'datasets',
        'model_save_path': 'models/saved_models',
        'epochs': 100,
        'learning_rate': 0.0001,    # Lower LR for stability
        'batch_size': 32,           # Larger batch for stability
        'patience': 15,
        'weight_decay': 1e-4,       # Stronger regularization
        'validation_split': 0.2,
        'grad_clip_norm': 1.0       # Gradient clipping
    }
    
    # Create save directory
    os.makedirs(config['model_save_path'], exist_ok=True)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    
    # Load data
    print("📊 Loading balanced dataset...")
    try:
        train_loader, val_loader, classes = get_balanced_data_loaders(
            data_dir=config['data_dir'],
            batch_size=config['batch_size'],
            validation_split=config['validation_split'],
            auto_balance=True
        )
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        print(f"💡 Check that '{config['data_dir']}' exists and contains class folders")
        return None, 0
    
    num_classes = len(classes)
    print(f"✅ Dataset loaded: {num_classes} classes")
    print(f"📋 Classes: {classes}")
    
    # Create the model
    model = create_model(num_classes).to(device)
    
    # Print model info
    print(get_model_summary(model))
    
    # STABLE Optimizer
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # BETTER Scheduler - reduces LR when loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',
        factor=0.5,      # Reduce LR by half
        patience=5,      # Wait 5 epochs before reducing
        min_lr=1e-7
    )
    
    # STABLE Loss function
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    
    # Loss tracking for smooth monitoring
    loss_tracker = LossTracker(window_size=50)
    
    # Training tracking
    best_val_acc = 0.0
    patience_counter = 0
    train_losses = []
    val_accuracies = []
    
    print(f"\n🚀 Starting STABLE training for {config['epochs']} epochs...")
    print(f"📊 Batch size: {config['batch_size']}")
    print(f"🎯 Learning rate: {config['learning_rate']}")
    print(f"✂️  Gradient clipping: {config['grad_clip_norm']}")
    print("=" * 60)
    
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
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            
            # Backward pass WITH gradient clipping
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip_norm'])
            optimizer.step()
            
            # Statistics
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
            
            # Update loss tracker for smooth monitoring
            loss_tracker.update(loss.item())
            smooth_loss = loss_tracker.get_average()
            
            # Update progress bar
            train_acc = 100. * train_correct / train_total
            train_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Smooth': f'{smooth_loss:.4f}',
                'Acc': f'{train_acc:.2f}%',
                'LR': f'{optimizer.param_groups[0]["lr"]:.2e}'
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
        epoch_val_loss = val_loss / len(val_loader)
        
        train_losses.append(epoch_train_loss)
        val_accuracies.append(epoch_val_acc)
        
        # Update learning rate scheduler with validation loss
        scheduler.step(epoch_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print epoch results
        print(f"\nEpoch {epoch+1}/{config['epochs']}:")
        print(f"  Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.2f}%")
        print(f"  Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.2f}%")
        print(f"  Learning Rate: {current_lr:.2e}")
        
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
            print(f"📊 No improvement for {config['patience']} epochs")
            break
        
        # Show patience counter
        if patience_counter > 0:
            print(f"  ⏳ Patience: {patience_counter}/{config['patience']}")
        
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
        'best_val_acc': float(best_val_acc),
        'classes': classes,
        'config': config,
        'model_type': 'Grayscale_Separable_CNN',
        'total_epochs': epoch + 1,
        'training_time_minutes': total_time / 60
    }
    
    history_path = os.path.join(config['model_save_path'], 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"📊 Training history saved at: {history_path}")
    
    return model, best_val_acc

def main():
    """Main training function"""
    print("🚀 STABLE GRAYSCALE SEPARABLE CNN TRAINING")
    print("🎯 Features:")
    print("   • Automatic RGB → Grayscale conversion")
    print("   • 10x fewer parameters than standard CNN")
    print("   • Gradient clipping for stability")
    print("   • Adaptive learning rate")
    print("   • Smooth loss monitoring")
    print("=" * 60)
    
    # Train the model
    model, final_accuracy = train_grayscale_separable_model()
    
    if model is not None:
        print(f"\n🎯 FINAL RESULT: {final_accuracy:.2f}%")
        
        if final_accuracy >= 80:
            print("🏆 EXCELLENT! Outstanding performance!")
        elif final_accuracy >= 70:
            print("🥇 VERY GOOD! Great performance!")
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
        print(f"   Stability: Gradient clipping + adaptive LR")
        
        print(f"\n💡 Tips for better performance:")
        print(f"   • Add more diverse training data")
        print(f"   • Try different augmentations")
        print(f"   • Experiment with batch sizes")
        print(f"   • Consider ensemble methods")

if __name__ == "__main__":
    main()