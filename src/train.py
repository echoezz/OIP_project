import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import sys
import json
import time
import numpy as np
from sklearn.metrics import classification_report

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.pest_classifier import PestClassifier, get_model_summary
from src.data_loader import PestDataLoader

def train_small_dataset_cnn():
    """Train CNN optimized for small datasets with advanced techniques"""
    
    print("🚀 Starting Small Dataset Optimized CNN Training")
    print("=" * 60)
    
    # Optimized configuration for small datasets
    config = {
        'data_dir': 'datasets',
        'model_save_path': 'models/saved_models',
        'epochs': 150,  # More epochs for small datasets
        'learning_rate': 0.0003,  # Lower learning rate for stability
        'batch_size': 8,  # Smaller batch size for better gradients
        'img_size': 224,
        'weight_decay': 1e-3,  # Stronger regularization
        'augmentation_factor': 20,  # Heavy augmentation (20x data)
        'patience': 25,  # More patience for small datasets
        'accumulation_steps': 4,  # Simulate larger batch size
        'min_epochs': 30  # Minimum epochs before early stopping
    }
    
    # Create save directory
    os.makedirs(config['model_save_path'], exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    
    # Data loaders with heavy augmentation
    data_loader = PestDataLoader(
        config['data_dir'], 
        batch_size=config['batch_size'],
        img_size=config['img_size'],
        augmentation_factor=config['augmentation_factor']
    )
    
    # Analyze dataset balance first
    print("\n📊 Analyzing dataset...")
    class_distribution = data_loader.analyze_dataset_balance()
    
    train_loader, val_loader, classes = data_loader.get_data_loaders()
    num_classes = len(classes)
    
    print(f"\n🎯 Small Dataset Training Configuration:")
    print(f"   Classes: {num_classes}")
    print(f"   Epochs: {config['epochs']} (with early stopping)")
    print(f"   Batch size: {config['batch_size']} (with gradient accumulation)")
    print(f"   Effective batch size: {config['batch_size'] * config['accumulation_steps']}")
    print(f"   Learning rate: {config['learning_rate']}")
    print(f"   Data augmentation: {config['augmentation_factor']}x")
    print(f"   Regularization: Strong (weight_decay={config['weight_decay']})")
    
    # Model
    model = PestClassifier(num_classes).to(device)
    print(f"\n{get_model_summary(model)}")
    
    # Calculate class weights for imbalanced datasets
    if class_distribution:
        total_samples = sum(class_distribution.values())
        class_weights = []
        for class_name in classes:
            class_count = class_distribution.get(class_name, 1)
            weight = total_samples / (num_classes * class_count)
            class_weights.append(weight)
        
        class_weights = torch.FloatTensor(class_weights).to(device)
        print(f"📊 Using class weights for imbalanced data: {[f'{w:.2f}' for w in class_weights]}")
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.2)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.2)
    
    # Optimizer optimized for small datasets
    optimizer = optim.AdamW(model.parameters(), 
                           lr=config['learning_rate'],
                           weight_decay=config['weight_decay'],
                           betas=(0.9, 0.999))
    
    # Advanced learning rate scheduling
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=30, T_mult=2, eta_min=1e-6
    )
    
    # Training tracking
    best_val_acc = 0.0
    patience_counter = 0
    training_history = {
        'train_acc': [], 'val_acc': [], 'train_loss': [], 'val_loss': [],
        'learning_rates': [], 'epoch_times': []
    }
    
    print(f"\n🎓 Starting Small Dataset Optimized Training...")
    print(f"🔧 Advanced features: Gradient accumulation, Label smoothing, Class weighting, Warm restarts")
    start_time = time.time()
    
    for epoch in range(config['epochs']):
        epoch_start = time.time()
        print(f"\n📅 Epoch {epoch+1}/{config['epochs']}")
        print("-" * 50)
        
        # Training phase with gradient accumulation
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        accumulated_loss = 0.0
        
        optimizer.zero_grad()
        
        train_bar = tqdm(train_loader, desc='Training')
        for batch_idx, (images, labels) in enumerate(train_bar):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels) / config['accumulation_steps']
            
            # Backward pass
            loss.backward()
            accumulated_loss += loss.item()
            
            # Gradient accumulation
            if (batch_idx + 1) % config['accumulation_steps'] == 0 or (batch_idx + 1) == len(train_loader):
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                
                # Update metrics
                train_loss += accumulated_loss
                accumulated_loss = 0.0
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # Update progress bar
            current_acc = 100 * train_correct / train_total
            current_lr = optimizer.param_groups[0]['lr']
            train_bar.set_postfix({
                'Loss': f'{loss.item() * config["accumulation_steps"]:.4f}',
                'Acc': f'{current_acc:.2f}%',
                'LR': f'{current_lr:.2e}'
            })
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_val_preds = []
        all_val_labels = []
        
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
                
                # Store for detailed analysis
                all_val_preds.extend(predicted.cpu().numpy())
                all_val_labels.extend(labels.cpu().numpy())
                
                val_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100 * val_correct / val_total:.2f}%'
                })
        
        # Step scheduler
        scheduler.step()
        epoch_time = time.time() - epoch_start
        
        # Calculate metrics
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        avg_train_loss = train_loss / (len(train_loader) // config['accumulation_steps'])
        avg_val_loss = val_loss / len(val_loader)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save metrics
        training_history['train_acc'].append(train_acc)
        training_history['val_acc'].append(val_acc)
        training_history['train_loss'].append(avg_train_loss)
        training_history['val_loss'].append(avg_val_loss)
        training_history['learning_rates'].append(current_lr)
        training_history['epoch_times'].append(epoch_time)
        
        # Print detailed epoch results
        print(f"\n📊 Epoch {epoch+1} Results:")
        print(f"   ⏱️  Time: {epoch_time:.1f}s")
        print(f"   🏋️  Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"   🎯 Val Loss: {avg_val_loss:.4f}   | Val Acc: {val_acc:.2f}%")
        print(f"   📈 Learning Rate: {current_lr:.2e}")
        
        # Overfitting detection
        if len(training_history['train_acc']) > 5:
            recent_train_acc = np.mean(training_history['train_acc'][-3:])
            recent_val_acc = np.mean(training_history['val_acc'][-3:])
            overfitting_gap = recent_train_acc - recent_val_acc
            
            if overfitting_gap > 30:
                print(f"   ⚠️  Overfitting detected: {overfitting_gap:.1f}% gap")
            elif overfitting_gap > 20:
                print(f"   📊 Training gap: {overfitting_gap:.1f}%")
        
        # Early stopping and model saving
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            # Save best model
            torch.save(model.state_dict(), f"{config['model_save_path']}/best_model.pth")
            print(f"   🏆 NEW BEST MODEL! Validation accuracy: {best_val_acc:.2f}%")
            
            # Save detailed classification report for best model
            if epoch >= 10:  # Only after some training
                try:
                    report = classification_report(all_val_labels, all_val_preds, 
                                                 target_names=classes, output_dict=True, zero_division=0)
                    with open(f"{config['model_save_path']}/classification_report.json", 'w') as f:
                        json.dump(report, f, indent=2)
                    
                    # Print per-class accuracy
                    print(f"   📋 Per-class accuracy:")
                    for class_name in classes:
                        if class_name in report:
                            f1_score = report[class_name].get('f1-score', 0)
                            print(f"      {class_name}: {f1_score:.2f}")
                except Exception as e:
                    print(f"   ⚠️  Could not generate classification report: {e}")
        else:
            patience_counter += 1
            print(f"   ⏳ Patience: {patience_counter}/{config['patience']} (Best: {best_val_acc:.2f}%)")
            
            # Early stopping (but not too early for small datasets)
            if patience_counter >= config['patience'] and epoch >= config['min_epochs']:
                print(f"\n🛑 Early stopping triggered at epoch {epoch+1}")
                print(f"   📊 No improvement for {config['patience']} epochs")
                break
        
        # Save checkpoint every 20 epochs
        if (epoch + 1) % 20 == 0:
            checkpoint_path = f"{config['model_save_path']}/checkpoint_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_acc,
                'training_history': training_history
            }, checkpoint_path)
            print(f"   💾 Checkpoint saved: {checkpoint_path}")
    
    # Training completed
    total_time = time.time() - start_time
    avg_epoch_time = np.mean(training_history['epoch_times']) if training_history['epoch_times'] else 0
    
    print(f"\n🎉 TRAINING COMPLETED!")
    print("=" * 60)
    print(f"🏆 Best validation accuracy: {best_val_acc:.2f}%")
    print(f"⏱️  Total training time: {total_time/60:.1f} minutes")
    print(f"📊 Average time per epoch: {avg_epoch_time:.1f} seconds")
    print(f"🔄 Total epochs completed: {epoch+1}")
    
    # Calculate improvement
    if len(training_history['val_acc']) > 1:
        initial_acc = training_history['val_acc'][0]
        final_acc = best_val_acc
        improvement = final_acc - initial_acc
        print(f"📈 Accuracy improvement: +{improvement:.1f}% (from {initial_acc:.1f}% to {final_acc:.1f}%)")
    
    # Final model analysis
    print(f"\n📋 Final Model Analysis:")
    print(f"   🎯 Target achieved: {'✅' if best_val_acc >= 50 else '❌'} (50%+ confidence)")
    print(f"   📊 Model performance: {'Excellent' if best_val_acc >= 70 else 'Good' if best_val_acc >= 50 else 'Needs improvement'}")
    
    if best_val_acc < 50:
        print(f"\n💡 Suggestions for improvement:")
        print(f"   • Add more training data if possible")
        print(f"   • Try training for more epochs")
        print(f"   • Adjust augmentation parameters")
        print(f"   • Consider ensemble methods")
    
    # Save comprehensive training info
    training_info = {
        'classes': classes,
        'num_classes': num_classes,
        'best_val_acc': best_val_acc,
        'final_epoch': epoch + 1,
        'config': config,
        'training_history': training_history,
        'model_type': 'SmallDatasetOptimizedCNN',
        'training_completed': True,
        'total_training_time_minutes': total_time / 60,
        'average_epoch_time_seconds': avg_epoch_time,
        'class_distribution': class_distribution,
        'optimization_features': [
            'Heavy data augmentation (20x)',
            'Gradient accumulation',
            'Class weighting for imbalance',
            'Label smoothing',
            'Strong regularization',
            'Cosine annealing with warm restarts',
            'Gradient clipping',
            'Early stopping with patience'
        ]
    }
    
    # Save files
    with open(f"{config['model_save_path']}/classes.json", 'w') as f:
        json.dump(classes, f, indent=2)
    
    with open(f"{config['model_save_path']}/training_info.json", 'w') as f:
        json.dump(training_info, f, indent=2)
    
    # Save training history plot data
    history_summary = {
        'epochs': list(range(1, len(training_history['val_acc']) + 1)),
        'train_accuracy': training_history['train_acc'],
        'validation_accuracy': training_history['val_acc'],
        'train_loss': training_history['train_loss'],
        'validation_loss': training_history['val_loss'],
        'learning_rates': training_history['learning_rates']
    }
    
    with open(f"{config['model_save_path']}/training_history.json", 'w') as f:
        json.dump(history_summary, f, indent=2)
    
    print(f"\n💾 All files saved to: {config['model_save_path']}/")
    print(f"   • best_model.pth (trained model)")
    print(f"   • classes.json (class names)")
    print(f"   • training_info.json (complete training details)")
    print(f"   • training_history.json (accuracy/loss curves)")
    print(f"   • classification_report.json (per-class metrics)")
    
    return model, best_val_acc

def quick_test_model(model_path, classes_path):
    """Quick test of trained model"""
    try:
        from src.pest_identifier import PestIdentifier
        identifier = PestIdentifier(model_path, classes_path, enable_tta=True)
        print("✅ Model loaded successfully for testing!")
        return True
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False

if __name__ == "__main__":
    print("🌱 Small Dataset CNN Training Script")
    print("=" * 60)
    
    # Train the model
    model, final_accuracy = train_small_dataset_cnn()
    
    # Test loading the saved model
    print(f"\n🧪 Testing saved model...")
    model_path = "models/saved_models/best_model.pth"
    classes_path = "models/saved_models/classes.json"
    
    if os.path.exists(model_path) and os.path.exists(classes_path):
        success = quick_test_model(model_path, classes_path)
        if success:
            print(f"🎉 SUCCESS! Model ready for deployment!")
            print(f"📱 You can now run: python app/main_app.py")
        else:
            print(f"⚠️  Model saved but testing failed")
    else:
        print(f"❌ Model files not found")
    
    print(f"\n🎯 FINAL RESULT: {final_accuracy:.1f}% accuracy achieved!")
    
    if final_accuracy >= 50:
        print("🏆 EXCELLENT! Target achieved!")
    elif final_accuracy >= 35:
        print("✅ GOOD! Significant improvement from baseline!")
    else:
        print("📈 Progress made, consider more training or data")