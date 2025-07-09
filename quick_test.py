#!/usr/bin/env python3
"""
Quick test script to verify the environment is working.
This should run without errors if everything is set up correctly.
"""

def test_basic_imports():
    """Test basic Python imports."""
    print("Testing basic imports...")
    
    try:
        import sys
        print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
        
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        
        import torchvision
        print(f"✓ TorchVision {torchvision.__version__}")
        
        import numpy as np
        print(f"✓ NumPy {np.__version__}")
        
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_torch_functionality():
    """Test basic PyTorch functionality."""
    print("\nTesting PyTorch functionality...")
    
    try:
        import torch
        
        # Test tensor creation
        x = torch.randn(2, 3)
        print(f"✓ Tensor creation: {x.shape}")
        
        # Test basic operations
        y = x + 1
        print(f"✓ Basic operations: {y.mean().item():.3f}")
        
        # Test gradients
        x.requires_grad_(True)
        z = (x ** 2).sum()
        z.backward()
        print(f"✓ Gradients: {x.grad.shape}")
        
        return True
    except Exception as e:
        print(f"❌ PyTorch test failed: {e}")
        return False

def test_model_import():
    """Test importing our model."""
    print("\nTesting model import...")
    
    try:
        import sys
        import os
        
        # Add src directory to path if we're not in it
        if not os.path.exists('model.py'):
            if os.path.exists('src/model.py'):
                sys.path.insert(0, 'src')
            else:
                print("❌ Cannot find model.py")
                return False
        
        from model import create_model
        model = create_model()
        print(f"✓ Model created with {model.get_num_parameters():,} parameters")
        
        # Test forward pass
        import torch
        test_input = torch.randn(1, 1, 28, 28)
        output = model(test_input)
        print(f"✓ Forward pass: {test_input.shape} -> {output.shape}")
        
        return True
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False

def main():
    print("="*50)
    print(" QUICK ENVIRONMENT TEST")
    print("="*50)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("PyTorch Functionality", test_torch_functionality),
        ("Model Import", test_model_import),
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\n{name}:")
        print("-" * len(name))
        success = test_func()
        results.append(success)
    
    print("\n" + "="*50)
    print(" TEST RESULTS")
    print("="*50)
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"🎉 All tests passed! ({passed}/{total})")
        print("\nYour environment is working correctly!")
        print("You can now run:")
        print("  cd src")
        print("  python diagnostic.py")
    else:
        print(f"❌ Some tests failed. ({passed}/{total})")
        print("\nPlease run fix_env.bat to fix the environment.")

if __name__ == "__main__":
    main() 