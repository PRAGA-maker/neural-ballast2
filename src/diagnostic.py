import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import os
import numpy as np
from typing import Tuple

from model import create_model, MNISTNet

# Try to import torch.func for PyTorch 2.0+, fallback to manual implementation
try:
    import torch.func as func
    HAS_TORCH_FUNC = True
except ImportError:
    HAS_TORCH_FUNC = False
    print("torch.func not available. Using manual Jacobian computation.")


def compute_jacobian_manual(model, input_tensor):
    """
    Manual Jacobian computation using torch.autograd for compatibility.
    
    Args:
        model: The neural network model
        input_tensor: Input tensor [1, C, H, W]
        
    Returns:
        torch.Tensor: Jacobian matrix [output_dim, input_dim]
    """
    model.eval()
    input_tensor = input_tensor.detach().requires_grad_(True)
    
    # Forward pass
    output = model(input_tensor)
    output_dim = output.shape[1]  # Number of output classes
    
    # Flatten input for Jacobian computation
    input_flat = input_tensor.flatten()
    input_dim = input_flat.shape[0]
    
    # Initialize Jacobian matrix
    jacobian = torch.zeros(output_dim, input_dim)
    
    # Compute Jacobian row by row
    for i in range(output_dim):
        # Zero gradients
        if input_tensor.grad is not None:
            input_tensor.grad.zero_()
        
        # Create one-hot vector for current output
        grad_output = torch.zeros_like(output)
        grad_output[0, i] = 1.0
        
        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=output,
            inputs=input_tensor,
            grad_outputs=grad_output,
            retain_graph=True,
            create_graph=False
        )[0]
        
        # Store in Jacobian matrix
        jacobian[i, :] = gradients.flatten()
    
    return jacobian


def compute_jacobian_func(model, input_tensor):
    """
    Jacobian computation using torch.func for PyTorch 2.0+.
    
    Args:
        model: The neural network model
        input_tensor: Input tensor [1, C, H, W]
        
    Returns:
        torch.Tensor: Jacobian matrix [output_dim, input_dim]
    """
    model.eval()
    
    # Define a function that takes the flattened input
    def model_fn(x_flat):
        # Reshape back to image format
        x = x_flat.view(input_tensor.shape)
        return model(x).squeeze(0)  # Remove batch dimension
    
    # Flatten input
    input_flat = input_tensor.flatten()
    
    # Compute Jacobian
    jacobian = func.jacrev(model_fn)(input_flat)
    
    return jacobian


def get_singfol_dim(model: torch.nn.Module, input_tensor: torch.Tensor, threshold: float = 1e-6) -> int:
    """
    Calculate the Singular Foliation Dimension (SingFolDIM) of a neural network at a given input.
    
    This function computes the Jacobian of the model's output with respect to the input,
    then counts how many singular values are effectively zero (below the threshold).
    A higher count indicates the input is in a "dead zone" where the model's behavior
    becomes degenerate.
    
    Args:
        model: The neural network model (should be in eval mode)
        input_tensor: Input tensor, typically shape [1, C, H, W] for images
        threshold: Threshold below which singular values are considered zero
        
    Returns:
        int: Number of singular values below the threshold (the SingFolDIM)
    """
    # Ensure model is in evaluation mode
    model.eval()
    
    # Ensure input has correct shape and requires gradients
    if input_tensor.dim() == 3:
        input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension
    input_tensor = input_tensor.detach().requires_grad_(True)
    
    # Compute Jacobian matrix
    try:
        if HAS_TORCH_FUNC:
            jacobian_matrix = compute_jacobian_func(model, input_tensor)
        else:
            jacobian_matrix = compute_jacobian_manual(model, input_tensor)
    except Exception as e:
        print(f"Error computing Jacobian: {e}")
        return 0
    
    # Compute singular values
    try:
        singular_values = torch.linalg.svdvals(jacobian_matrix)
    except Exception as e:
        print(f"Error computing singular values: {e}")
        return 0
    
    # Count how many singular values are below the threshold
    num_zero_singular_values = torch.sum(singular_values < threshold).item()
    
    return num_zero_singular_values


def is_in_dead_zone(model: torch.nn.Module, input_tensor: torch.Tensor, dim_threshold: int = 1) -> bool:
    """
    Simple helper function to determine if an input is in a dead zone.
    
    Args:
        model: The neural network model
        input_tensor: Input tensor to test
        dim_threshold: Threshold for considering input in dead zone
        
    Returns:
        bool: True if the input is considered to be in a dead zone
    """
    singfol_dim = get_singfol_dim(model, input_tensor)
    return singfol_dim >= dim_threshold


def get_jacobian_info(model: torch.nn.Module, input_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """
    Get detailed information about the Jacobian computation for analysis.
    
    Args:
        model: The neural network model
        input_tensor: Input tensor
        
    Returns:
        tuple: (jacobian_matrix, singular_values, info_dict)
    """
    model.eval()
    
    # Ensure input has correct shape and requires gradients
    if input_tensor.dim() == 3:
        input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension
    input_tensor = input_tensor.detach().requires_grad_(True)
    
    # Compute Jacobian matrix
    try:
        if HAS_TORCH_FUNC:
            jacobian_matrix = compute_jacobian_func(model, input_tensor)
        else:
            jacobian_matrix = compute_jacobian_manual(model, input_tensor)
    except Exception as e:
        print(f"Error computing Jacobian: {e}")
        return torch.zeros(10, 784), torch.zeros(10), {}
    
    # Compute singular values
    try:
        singular_values = torch.linalg.svdvals(jacobian_matrix)
    except Exception as e:
        print(f"Error computing singular values: {e}")
        return jacobian_matrix, torch.zeros(jacobian_matrix.shape[0]), {}
    
    info_dict = {
        'jacobian_shape': jacobian_matrix.shape,
        'num_singular_values': len(singular_values),
        'max_singular_value': singular_values.max().item(),
        'min_singular_value': singular_values.min().item(),
        'mean_singular_value': singular_values.mean().item(),
        'std_singular_value': singular_values.std().item(),
        'torch_func_available': HAS_TORCH_FUNC,
    }
    
    return jacobian_matrix, singular_values, info_dict


if __name__ == '__main__':
    print("Testing SingFolDIM Diagnostic Function")
    print("=" * 50)
    
    # Print system info
    print(f"PyTorch version: {torch.__version__}")
    print(f"torch.func available: {HAS_TORCH_FUNC}")
    
    # Check if trained model exists
    model_path = 'mnist_net.pth'
    if not os.path.exists(model_path):
        print(f"❌ Trained model not found at {model_path}")
        print("Please run 'python train.py' first to train the model.")
        print("\nFor now, testing with a randomly initialized model...")
        model = create_model()
        model.eval()
    else:
        print(f"✅ Loading trained model from {model_path}")
        # Load the trained model
        checkpoint = torch.load(model_path, map_location='cpu')
        model = create_model()
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print(f"Model accuracy: {checkpoint['test_accuracy']:.2f}%")
    
    # Load MNIST test dataset
    print("\nLoading MNIST test dataset...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = torchvision.datasets.MNIST(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform
    )
    
    # Test on multiple samples
    print("\nTesting SingFolDIM on sample images...")
    print("-" * 40)
    
    for i in range(3):  # Reduced to 3 samples for faster testing
        # Get a sample from the test set
        sample_image, sample_label = test_dataset[i]
        
        # Add batch dimension
        input_tensor = sample_image.unsqueeze(0)  # Shape: [1, 1, 28, 28]
        
        # Get model prediction
        with torch.no_grad():
            output = model(input_tensor)
            predicted_class = output.argmax(dim=1).item()
            confidence = torch.softmax(output, dim=1).max().item()
        
        print(f"\nSample {i+1}:")
        print(f"  True label: {sample_label}")
        print(f"  Predicted: {predicted_class}")
        print(f"  Confidence: {confidence:.3f}")
        
        # Calculate SingFolDIM
        try:
            singfol_dim = get_singfol_dim(model, input_tensor)
            in_dead_zone = is_in_dead_zone(model, input_tensor)
            
            print(f"  SingFolDIM: {singfol_dim}")
            print(f"  In dead zone: {in_dead_zone}")
            
            # Get detailed Jacobian info
            jacobian_matrix, singular_values, info = get_jacobian_info(model, input_tensor)
            print(f"  Jacobian shape: {info['jacobian_shape']}")
            print(f"  Singular values range: [{info['min_singular_value']:.6f}, {info['max_singular_value']:.6f}]")
            print(f"  Mean singular value: {info['mean_singular_value']:.6f}")
            
        except Exception as e:
            print(f"  ❌ Error computing SingFolDIM: {str(e)}")
    
    print("\n" + "=" * 50)
    print("SingFolDIM diagnostic testing completed!")
    
    # Test with a simple threshold analysis
    print("\nTesting different thresholds...")
    sample_image, _ = test_dataset[0]
    input_tensor = sample_image.unsqueeze(0)
    
    thresholds = [1e-8, 1e-6, 1e-4, 1e-2]
    for threshold in thresholds:
        try:
            singfol_dim = get_singfol_dim(model, input_tensor, threshold=threshold)
            print(f"  Threshold {threshold:.0e}: SingFolDIM = {singfol_dim}")
        except Exception as e:
            print(f"  Threshold {threshold:.0e}: Error - {str(e)}") 