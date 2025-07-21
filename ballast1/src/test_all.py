"""
Test script to verify all components of the no-hyperplanes project work correctly.
"""
import torch
import sys
import os

def test_basic_imports():
    """Test that all required modules can be imported."""
    print("Testing basic imports...")
    
    try:
        import torch
        import torchvision
        import numpy as np
        print(f"✓ PyTorch version: {torch.__version__}")
        print(f"✓ TorchVision version: {torchvision.__version__}")
        print(f"✓ NumPy version: {np.__version__}")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False


def test_model():
    """Test the MNIST model creation and basic functionality."""
    print("\nTesting model creation...")
    
    try:
        from model import create_model
        model = create_model()
        
        # Test forward pass
        test_input = torch.randn(2, 1, 28, 28)
        with torch.no_grad():
            output = model(test_input)
        
        assert output.shape == (2, 10), f"Expected output shape (2, 10), got {output.shape}"
        print("✓ Model creation and forward pass working")
        return True
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False


def test_model_loading():
    """Test loading a trained model if it exists."""
    print("\nTesting model loading...")
    
    model_path = 'mnist_net.pth'
    if not os.path.exists(model_path):
        print(f"⚠️  No trained model found at {model_path}")
        print("   Run 'python train.py' to train a model first")
        return True
    
    try:
        from model import create_model
        checkpoint = torch.load(model_path, map_location='cpu')
        model = create_model()
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print(f"✓ Model loaded successfully (accuracy: {checkpoint['test_accuracy']:.2f}%)")
        return True
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False


def test_diagnostic_basic():
    """Test basic diagnostic functionality."""
    print("\nTesting diagnostic functions...")
    
    try:
        from diagnostic import get_singfol_dim, is_in_dead_zone, HAS_TORCH_FUNC
        from model import create_model
        
        print(f"  torch.func available: {HAS_TORCH_FUNC}")
        
        # Create a simple model for testing
        model = create_model()
        model.eval()
        
        # Test with random input
        test_input = torch.randn(1, 1, 28, 28)
        
        # Test SingFolDIM computation
        singfol_dim = get_singfol_dim(model, test_input)
        dead_zone = is_in_dead_zone(model, test_input)
        
        print(f"  SingFolDIM: {singfol_dim}")
        print(f"  In dead zone: {dead_zone}")
        print("✓ Diagnostic functions working")
        return True
    except Exception as e:
        print(f"❌ Diagnostic test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("Running comprehensive tests for no-hyperplanes project")
    print("=" * 60)
    
    tests = [
        test_basic_imports,
        test_model,
        test_model_loading,
        test_diagnostic_basic,
    ]
    
    results = []
    for test in tests:
        result = test()
        results.append(result)
    
    print("\n" + "=" * 60)
    print("Test Results:")
    print(f"  Passed: {sum(results)}/{len(results)}")
    
    if all(results):
        print("🎉 All tests passed! The codebase is working correctly.")
    else:
        print("❌ Some tests failed. Check the output above for details.")
    
    return all(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 