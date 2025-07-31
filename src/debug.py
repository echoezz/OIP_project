import torch
import torch.nn as nn
import sys
sys.path.append('..')

from pest_classifier import create_model

def debug_current_model():
    """Debug your existing model to find the problem"""
    
    print("🔍 DEBUGGING YOUR CURRENT MODEL")
    print("=" * 50)
    
    # Create your current model
    model = create_model(12)
    
    # Test input (same as your real data)
    test_input = torch.randn(2, 3, 224, 224)  # Batch=2, RGB, 224x224
    print(f"✅ Input shape: {test_input.shape}")
    print(f"✅ Input range: [{test_input.min():.3f}, {test_input.max():.3f}]")
    
    try:
        # Forward pass
        print("\n🧪 Testing forward pass...")
        output = model(test_input)
        
        print(f"✅ Output shape: {output.shape}")
        print(f"✅ Expected shape: torch.Size([2, 12])")
        
        if output.shape != torch.Size([2, 12]):
            print(f"❌ WRONG OUTPUT SHAPE! Expected [2, 12], got {output.shape}")
            return False
        
        print(f"✅ Output range: [{output.min():.3f}, {output.max():.3f}]")
        print(f"✅ Output sample: {output[0][:5]}...")  # First 5 values
        
        # Check for problems
        if torch.isnan(output).any():
            print("❌ NaN VALUES IN OUTPUT!")
            return False
        
        if torch.isinf(output).any():
            print("❌ INFINITE VALUES IN OUTPUT!")
            return False
        
        if output.abs().max() > 100:
            print(f"❌ OUTPUT VALUES TOO LARGE: {output.abs().max()}")
            return False
        
        # Test loss calculation
        print("\n🧪 Testing loss calculation...")
        targets = torch.randint(0, 12, (2,))  # Random valid targets
        print(f"✅ Targets: {targets}")
        
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, targets)
        
        print(f"✅ Loss: {loss.item():.4f}")
        
        # Check loss sanity
        if loss.item() > 10:
            print(f"❌ LOSS TOO HIGH: {loss.item()}")
            print("💡 Expected loss for random 12-class: ~2.48")
            return False
        
        if torch.isnan(loss):
            print("❌ LOSS IS NaN!")
            return False
        
        print("✅ MODEL LOOKS GOOD!")
        return True
        
    except Exception as e:
        print(f"❌ MODEL FORWARD FAILED: {e}")
        print(f"💡 Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False

def debug_with_real_data():
    """Test with your actual data loader"""
    
    print("\n🔍 TESTING WITH REAL DATA")
    print("=" * 50)
    
    # Import your data loader
    from data_loader import get_balanced_data_loaders
    
    # Load just one batch
    train_loader, _, classes = get_balanced_data_loaders(
        data_dir='datasets',
        batch_size=4,  # Small batch for testing
        validation_split=0.2
    )
    
    model = create_model(len(classes))
    criterion = nn.CrossEntropyLoss()
    
    # Test first batch
    for batch_idx, (data, targets) in enumerate(train_loader):
        print(f"✅ Real data shape: {data.shape}")
        print(f"✅ Real data range: [{data.min():.3f}, {data.max():.3f}]")
        print(f"✅ Real targets: {targets}")
        print(f"✅ Target range: [{targets.min()}, {targets.max()}]")
        
        # Check data problems
        if torch.isnan(data).any():
            print("❌ NaN IN REAL DATA!")
            return False
        
        if targets.max() >= len(classes):
            print(f"❌ INVALID TARGET: {targets.max()} >= {len(classes)}")
            return False
        
        # Forward pass with real data
        try:
            output = model(data)
            loss = criterion(output, targets)
            
            print(f"✅ Real data output: {output.shape}")
            print(f"✅ Real data loss: {loss.item():.4f}")
            
            if loss.item() > 10:
                print(f"❌ REAL DATA LOSS TOO HIGH: {loss.item()}")
                return False
            
            print("✅ REAL DATA TEST PASSED!")
            return True
            
        except Exception as e:
            print(f"❌ REAL DATA FORWARD FAILED: {e}")
            return False
        
        break  # Only test first batch

if __name__ == "__main__":
    print("🚀 COMPREHENSIVE MODEL DEBUGGING")
    print("=" * 60)
    
    # Test 1: Model architecture
    model_ok = debug_current_model()
    
    if model_ok:
        # Test 2: Real data
        data_ok = debug_with_real_data()
        
        if data_ok:
            print("\n🎉 ALL TESTS PASSED!")
            print("✅ Your model and data are correct")
            print("💡 The problem might be in training loop or hyperparameters")
        else:
            print("\n❌ PROBLEM WITH REAL DATA")
    else:
        print("\n❌ PROBLEM WITH MODEL ARCHITECTURE")
        print("💡 Check your pest_classifier.py file")