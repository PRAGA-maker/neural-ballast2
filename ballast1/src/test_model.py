import torch
from model import create_model
import os


def test_model_creation():
    """Test that the model can be created and runs forward pass."""
    print("Testing model creation...")
    
    model = create_model()
    
    # Test forward pass with random input
    test_input = torch.randn(5, 1, 28, 28)  # Batch of 5 images
    
    with torch.no_grad():
        output = model(test_input)
    
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    # Check output dimensions
    assert output.shape == (5, 10), f"Expected output shape (5, 10), got {output.shape}"
    print("✓ Model creation test passed!")


def test_saved_model():
    """Test loading a saved model if it exists."""
    model_path = 'mnist_net.pth'  # Fixed path - model is in current directory
    
    if os.path.exists(model_path):
        print(f"\nTesting saved model from {model_path}...")
        
        # Load the saved model
        checkpoint = torch.load(model_path, map_location='cpu')
        
        model = create_model()
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print(f"✓ Model loaded successfully!")
        print(f"  Test accuracy: {checkpoint['test_accuracy']:.2f}%")
        print(f"  Parameters: {checkpoint['model_info']['parameters']:,}")
        print(f"  Epochs trained: {checkpoint['model_info']['epochs']}")
        
        # Test inference
        test_input = torch.randn(1, 1, 28, 28)
        with torch.no_grad():
            output = model(test_input)
            predicted_class = output.argmax(dim=1).item()
        
        print(f"  Test prediction: class {predicted_class}")
        print("✓ Saved model test passed!")
    else:
        print(f"\nNo saved model found at {model_path}")
        print("Run train.py first to create a trained model.")


if __name__ == "__main__":
    test_model_creation()
    test_saved_model()