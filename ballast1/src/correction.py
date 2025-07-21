"""
Neural Ballast Correction Module
===============================

This module provides corrective nudging functionality for neural networks to escape dead zones.

The core idea is that when a neural network input falls into a "dead zone" (a region where
the Jacobian has many near-zero singular values), the model's behavior becomes unreliable.
By applying small, carefully controlled noise to nudge the input out of the dead zone,
we can restore reliable inference behavior.

Key Functions:
    apply_corrective_nudge: Main function to nudge inputs out of dead zones

Theory:
    Dead zones in ReLU networks correspond to regions where many neurons are inactive,
    leading to reduced model expressiveness. Small perturbations can move inputs to
    regions with better computational properties.

Author: Neural Ballast Project
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os

from model import create_model
from diagnostic import is_in_dead_zone


def apply_corrective_nudge(model: torch.nn.Module, 
                          bad_input_tensor: torch.Tensor, 
                          is_in_dead_zone_func, 
                          max_attempts: int = 10, 
                          sigma: float = 0.01) -> torch.Tensor:
    """
    Apply corrective nudge to push an input out of a neural network dead zone.
    
    This function iteratively adds small amounts of Gaussian noise to an input tensor
    until it finds a version that is no longer in a dead zone. The process is designed
    to be minimal and conservative, adding just enough noise to restore reliable
    model behavior without significantly altering the input's semantic content.
    
    Args:
        model (torch.nn.Module): The neural network model to test against
        bad_input_tensor (torch.Tensor): Input tensor currently in a dead zone
                                       Shape: [batch_size, channels, height, width]
        is_in_dead_zone_func (callable): Function that takes (model, input) and returns
                                        True if input is in dead zone, False otherwise
        max_attempts (int, optional): Maximum number of noise attempts. Defaults to 10.
        sigma (float, optional): Standard deviation for Gaussian noise. Smaller values
                               mean more conservative corrections. Defaults to 0.01.
    
    Returns:
        torch.Tensor: Corrected input tensor, hopefully no longer in dead zone.
                     Same shape as input. If correction fails, returns last attempt.
    
    Raises:
        None: Function never raises exceptions, but prints warnings for failed corrections
    
    Example:
        >>> model = create_model()
        >>> bad_input = torch.randn(1, 1, 28, 28)  # Some input in dead zone
        >>> corrected = apply_corrective_nudge(model, bad_input, is_in_dead_zone)
        >>> # corrected is now (hopefully) out of the dead zone
    
    Notes:
        - The function preserves the original input and works on detached copies
        - Noise is sampled fresh for each attempt to explore different directions
        - If all attempts fail, the last generated input is returned with a warning
        - The correction is purely additive (original + noise), preserving input structure
    """
    # Set model to evaluation mode to ensure consistent behavior
    model.eval()
    
    # Detach input to prevent gradient computation during noise generation
    # This ensures we don't interfere with any existing computational graph
    original_input = bad_input_tensor.detach()
    
    # Initialize variable to store the last attempt (in case all fail)
    nudged_input = original_input
    
    # Iteratively try different noise patterns
    for attempt in range(max_attempts):
        # Generate fresh Gaussian noise for this attempt
        # Using randn_like ensures noise has same shape and device as input
        noise = torch.randn_like(original_input) * sigma
        
        # Create candidate corrected input
        nudged_input = original_input + noise
        
        # Test if this nudged input escapes the dead zone
        if not is_in_dead_zone_func(model, nudged_input):
            print(f"✓ Corrective nudge successful on attempt {attempt + 1}")
            print(f"  Applied noise with σ={sigma}, ||noise||={noise.norm().item():.4f}")
            return nudged_input
    
    # If we reach here, all attempts failed
    print(f"⚠ Warning: Could not escape dead zone after {max_attempts} attempts")
    print(f"  Consider increasing sigma (current: {sigma}) or max_attempts")
    print(f"  Returning last attempt (may still be in dead zone)")
    
    return nudged_input


def batch_corrective_nudge(model: torch.nn.Module,
                          input_batch: torch.Tensor,
                          is_in_dead_zone_func,
                          **nudge_kwargs) -> torch.Tensor:
    """
    Apply corrective nudging to a batch of inputs, processing each individually.
    
    This function processes each input in a batch separately, applying corrective
    nudging only to those inputs that are detected to be in dead zones. Inputs
    not in dead zones are returned unchanged.
    
    Args:
        model (torch.nn.Module): The neural network model
        input_batch (torch.Tensor): Batch of input tensors to process
                                   Shape: [batch_size, channels, height, width]
        is_in_dead_zone_func (callable): Dead zone detection function
        **nudge_kwargs: Additional arguments passed to apply_corrective_nudge
    
    Returns:
        torch.Tensor: Batch with corrected inputs where needed
    
    Example:
        >>> batch = torch.randn(32, 1, 28, 28)  # Batch of 32 MNIST images
        >>> corrected_batch = batch_corrective_nudge(model, batch, is_in_dead_zone)
    """
    model.eval()
    corrected_inputs = []
    
    for i in range(input_batch.shape[0]):
        single_input = input_batch[i:i+1]  # Preserve batch dimension
        
        if is_in_dead_zone_func(model, single_input):
            # Apply correction to this input
            corrected_input = apply_corrective_nudge(
                model, single_input, is_in_dead_zone_func, **nudge_kwargs
            )
        else:
            # Input is already healthy, no correction needed
            corrected_input = single_input
        
        corrected_inputs.append(corrected_input)
    
    return torch.cat(corrected_inputs, dim=0)


if __name__ == '__main__':
    """
    Test script demonstrating the corrective nudge functionality.
    
    This script:
    1. Loads a trained MNIST model
    2. Searches for inputs in dead zones
    3. Applies corrective nudging
    4. Verifies the correction worked
    5. Shows prediction changes
    """
    print("🔧 Testing Neural Ballast Corrective Nudge Function")
    print("=" * 60)
    
    # Load the trained model
    model_path = 'mnist_net.pth'
    if not os.path.exists(model_path):
        print(f"❌ Error: Model file {model_path} not found!")
        print("Please run train.py first to create a trained model.")
        exit(1)
    
    print(f"📁 Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location='cpu')
    model = create_model()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"✓ Model loaded successfully (Test accuracy: {checkpoint['test_accuracy']:.2f}%)")
    
    # Load MNIST test set for finding problematic inputs
    print("\n📊 Loading MNIST test dataset...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST normalization
    ])
    
    test_dataset = torchvision.datasets.MNIST(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform
    )
    
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    print(f"✓ Loaded {len(test_dataset)} test samples")
    
    # Search for an input that is in a dead zone
    print("\n🔍 Searching for input in dead zone...")
    bad_input = None
    bad_input_label = None
    bad_input_idx = None
    
    for idx, (input_tensor, label) in enumerate(test_loader):
        if idx >= 100:  # Limit search to first 100 samples for efficiency
            break
            
        if is_in_dead_zone(model, input_tensor):
            bad_input = input_tensor
            bad_input_label = label.item()
            bad_input_idx = idx
            print(f"✓ Found dead zone input at index {idx} (true label: {bad_input_label})")
            break
    
    if bad_input is None:
        print("ℹ️  No dead zone input found in the first 100 samples.")
        print("This might indicate the model is working well, or we need to check more samples.")
        print("Try running with a larger search range or different inputs.")
        exit(0)
    
    # Apply corrective nudge to the problematic input
    print(f"\n🛠️  Applying corrective nudge to dead zone input...")
    nudged_input = apply_corrective_nudge(
        model=model, 
        bad_input_tensor=bad_input, 
        is_in_dead_zone_func=is_in_dead_zone,
        max_attempts=15,  # More attempts for thorough testing
        sigma=0.01
    )
    
    # Verify the correction was successful
    print(f"\n✅ Verifying the correction...")
    is_nudged_in_dead_zone = is_in_dead_zone(model, nudged_input)
    print(f"Original input in dead zone: True")
    print(f"Nudged input in dead zone: {is_nudged_in_dead_zone}")
    
    if not is_nudged_in_dead_zone:
        print("🎉 Success! The corrective nudge worked!")
        
        # Compare model predictions for original vs corrected inputs
        with torch.no_grad():
            original_output = model(bad_input)
            nudged_output = model(nudged_input)
            
            original_pred = original_output.argmax(dim=1).item()
            nudged_pred = nudged_output.argmax(dim=1).item()
            
            original_confidence = torch.softmax(original_output, dim=1).max().item()
            nudged_confidence = torch.softmax(nudged_output, dim=1).max().item()
            
            print(f"\n📈 Model prediction comparison:")
            print(f"  Original input  -> class {original_pred} (confidence: {original_confidence:.3f})")
            print(f"  Corrected input -> class {nudged_pred} (confidence: {nudged_confidence:.3f})")
            print(f"  True label: {bad_input_label}")
            
            if nudged_pred == bad_input_label and original_pred != bad_input_label:
                print("🎯 Ballast correction fixed a misclassification!")
            elif original_pred == bad_input_label and nudged_pred != bad_input_label:
                print("⚠️  Ballast correction changed a correct prediction")
            else:
                print("ℹ️  Prediction unchanged (both correct or both incorrect)")
    else:
        print("❌ The corrective nudge failed to escape the dead zone.")
        print("💡 Consider increasing sigma or max_attempts, or investigate the input further.")
    
    print(f"\n✨ Corrective nudge test complete!") 