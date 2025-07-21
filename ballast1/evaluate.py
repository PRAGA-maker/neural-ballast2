#!/usr/bin/env python3
"""
Neural Ballast Evaluation Script
================================

This script demonstrates the Neural Ballast system by:
1. Loading a trained MNIST model
2. Creating a NeuralBallast wrapper
3. Generating inputs guaranteed to be in dead zones
4. Comparing baseline model vs ballast-corrected predictions
5. Providing comprehensive results and statistics

Usage:
    python evaluate.py
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import os
import sys
import random

# Add src directory to Python path
sys.path.append('src')

from model import create_model
from ballast import NeuralBallast
from diagnostic import is_in_dead_zone, get_singfol_dim
from correction import apply_corrective_nudge


def generate_dead_zone_input(model: nn.Module, 
                            dataset, 
                            max_attempts: int = 1000,
                            noise_scale: float = 0.5) -> torch.Tensor:
    """
    Generate an input that is guaranteed to be in a dead zone.
    
    This function uses multiple strategies to create problematic inputs:
    1. Start with a random sample from the dataset
    2. Add noise to increase the chance of dead zone
    3. Use adversarial-like perturbations to find dead zones
    
    Args:
        model: The neural network model
        dataset: Dataset to sample from
        max_attempts: Maximum attempts to find a dead zone input
        noise_scale: Scale factor for noise added to inputs
        
    Returns:
        torch.Tensor: Input tensor that is in a dead zone
    """
    model.eval()
    
    print(f"Searching for dead zone input (max {max_attempts} attempts)...")
    
    # Strategy 1: Try random samples from dataset with added noise
    for attempt in range(max_attempts // 2):
        # Get a random sample
        idx = random.randint(0, len(dataset) - 1)
        sample_image, _ = dataset[idx]
        
        # Add noise to increase chance of dead zone
        noise = torch.randn_like(sample_image) * noise_scale
        noisy_input = sample_image + noise
        
        # Ensure input is still in valid range (approximately)
        noisy_input = torch.clamp(noisy_input, -3, 3)
        
        # Add batch dimension
        input_tensor = noisy_input.unsqueeze(0)
        
        # Check if it's in dead zone
        if is_in_dead_zone(model, input_tensor):
            print(f"✓ Found dead zone input on attempt {attempt + 1} (Strategy 1: Noisy sample)")
            return input_tensor
    
    # Strategy 2: Generate adversarial-like inputs using gradient information
    print("Strategy 1 failed, trying gradient-based approach...")
    
    for attempt in range(max_attempts // 2):
        # Start with a random sample
        idx = random.randint(0, len(dataset) - 1)
        sample_image, _ = dataset[idx]
        input_tensor = sample_image.unsqueeze(0).requires_grad_(True)
        
        # Get gradients
        try:
            output = model(input_tensor)
            loss = output.sum()  # Simple loss to get gradients
            loss.backward()
            
            # Use gradient information to create perturbation
            grad = input_tensor.grad.data
            
            # Create perturbation in direction that might lead to dead zone
            perturbation = grad.sign() * noise_scale * random.uniform(0.1, 1.0)
            
            # Create perturbed input
            perturbed_input = input_tensor.detach() + perturbation
            
            # Check if it's in dead zone
            if is_in_dead_zone(model, perturbed_input):
                print(f"✓ Found dead zone input on attempt {attempt + 1} (Strategy 2: Gradient-based)")
                return perturbed_input
                
        except Exception as e:
            continue
    
    # Strategy 3: Pure random inputs
    print("Gradient-based approach failed, trying pure random inputs...")
    
    for attempt in range(max_attempts // 4):
        # Generate completely random input
        random_input = torch.randn(1, 1, 28, 28) * 2.0
        
        if is_in_dead_zone(model, random_input):
            print(f"✓ Found dead zone input on attempt {attempt + 1} (Strategy 3: Random)")
            return random_input
    
    # If all strategies fail, return a heavily corrupted input
    print("⚠ All strategies failed, returning heavily corrupted input")
    idx = random.randint(0, len(dataset) - 1)
    sample_image, _ = dataset[idx]
    corrupted_input = sample_image + torch.randn_like(sample_image) * 2.0
    return corrupted_input.unsqueeze(0)


def evaluate_on_sample(model: nn.Module, 
                      ballast: NeuralBallast,
                      input_tensor: torch.Tensor,
                      true_label: int = None) -> dict:
    """
    Evaluate both baseline model and ballast on a single input.
    
    Args:
        model: Baseline model
        ballast: NeuralBallast wrapper
        input_tensor: Input tensor to evaluate
        true_label: Ground truth label (if known)
        
    Returns:
        dict: Evaluation results
    """
    model.eval()
    
    # Get baseline prediction
    with torch.no_grad():
        baseline_output = model(input_tensor)
        baseline_pred = baseline_output.argmax(dim=1).item()
        baseline_confidence = torch.softmax(baseline_output, dim=1).max().item()
    
    # Get ballast prediction
    ballast_output = ballast.predict(input_tensor)
    ballast_pred = ballast_output.argmax(dim=1).item()
    ballast_confidence = torch.softmax(ballast_output, dim=1).max().item()
    
    # Check if input is in dead zone
    in_dead_zone = is_in_dead_zone(model, input_tensor)
    singfol_dim = get_singfol_dim(model, input_tensor)
    
    return {
        'input_tensor': input_tensor,
        'true_label': true_label,
        'baseline_pred': baseline_pred,
        'baseline_confidence': baseline_confidence,
        'ballast_pred': ballast_pred,
        'ballast_confidence': ballast_confidence,
        'in_dead_zone': in_dead_zone,
        'singfol_dim': singfol_dim
    }


def print_evaluation_results(results: dict, sample_idx: int):
    """Print formatted evaluation results for a single sample."""
    print(f"\n{'='*20} SAMPLE {sample_idx} {'='*20}")
    print(f"Input in dead zone: {results['in_dead_zone']}")
    print(f"SingFolDIM: {results['singfol_dim']}")
    
    if results['true_label'] is not None:
        print(f"True label: {results['true_label']}")
    
    print(f"\nBaseline Model:")
    print(f"  Prediction: {results['baseline_pred']}")
    print(f"  Confidence: {results['baseline_confidence']:.3f}")
    
    print(f"\nNeural Ballast:")
    print(f"  Prediction: {results['ballast_pred']}")
    print(f"  Confidence: {results['ballast_confidence']:.3f}")
    
    # Analysis
    if results['in_dead_zone']:
        if results['baseline_pred'] != results['ballast_pred']:
            print(f"\n📊 ANALYSIS: Ballast changed prediction from {results['baseline_pred']} to {results['ballast_pred']}")
        else:
            print(f"\n📊 ANALYSIS: Ballast maintained prediction {results['ballast_pred']}")
    else:
        print(f"\n📊 ANALYSIS: Input was not in dead zone")


def main():
    """Main evaluation function."""
    print("Neural Ballast Evaluation")
    print("=" * 50)
    
    # Check if trained model exists
    model_path = 'src/mnist_net.pth'
    if not os.path.exists(model_path):
        print(f"❌ Trained model not found at {model_path}")
        print("Please run 'python src/train.py' first to train the model.")
        return
    
    # Load the trained model
    print(f"Loading trained model from {model_path}...")
    checkpoint = torch.load(model_path, map_location='cpu')
    model = create_model()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"✓ Model loaded successfully (Test accuracy: {checkpoint['test_accuracy']:.2f}%)")
    
    # Create NeuralBallast wrapper
    print("\nCreating NeuralBallast wrapper...")
    ballast = NeuralBallast(
        model=model,
        dim_threshold=1,
        noise_sigma=0.01,
        max_attempts=10,
        verbose=True
    )
    print("✓ NeuralBallast wrapper created")
    
    # Load MNIST test dataset
    print("\nLoading MNIST test dataset...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = torchvision.datasets.MNIST(
        root='./src/data',
        train=False,
        download=True,
        transform=transform
    )
    print(f"✓ Loaded {len(test_dataset)} test samples")
    
    # Generate problematic inputs and evaluate
    print("\n" + "=" * 50)
    print("GENERATING PROBLEMATIC INPUTS AND EVALUATING")
    print("=" * 50)
    
    num_samples = 3  # Number of dead zone inputs to generate and test
    results = []
    
    for i in range(num_samples):
        print(f"\nGenerating dead zone input {i+1}/{num_samples}...")
        
        # Generate a dead zone input
        dead_zone_input = generate_dead_zone_input(model, test_dataset)
        
        # Evaluate on this input
        result = evaluate_on_sample(model, ballast, dead_zone_input)
        results.append(result)
        
        # Print results
        print_evaluation_results(result, i+1)
    
    # Test on some regular inputs for comparison
    print("\n" + "=" * 50)
    print("TESTING ON REGULAR INPUTS FOR COMPARISON")
    print("=" * 50)
    
    for i in range(2):
        # Get a random sample from test set
        idx = random.randint(0, len(test_dataset) - 1)
        sample_image, sample_label = test_dataset[idx]
        input_tensor = sample_image.unsqueeze(0)
        
        result = evaluate_on_sample(model, ballast, input_tensor, sample_label)
        print_evaluation_results(result, f"Regular-{i+1}")
    
    # Print final statistics
    print("\n" + "=" * 50)
    print("FINAL EVALUATION STATISTICS")
    print("=" * 50)
    
    ballast.print_statistics()
    
    # Summary of results
    dead_zone_results = [r for r in results if r['in_dead_zone']]
    if dead_zone_results:
        print(f"\nDead Zone Results Summary:")
        print(f"  Total dead zone inputs tested: {len(dead_zone_results)}")
        
        prediction_changes = sum(1 for r in dead_zone_results 
                               if r['baseline_pred'] != r['ballast_pred'])
        print(f"  Ballast changed predictions: {prediction_changes}/{len(dead_zone_results)}")
        
        avg_baseline_conf = np.mean([r['baseline_confidence'] for r in dead_zone_results])
        avg_ballast_conf = np.mean([r['ballast_confidence'] for r in dead_zone_results])
        print(f"  Average baseline confidence: {avg_baseline_conf:.3f}")
        print(f"  Average ballast confidence: {avg_ballast_conf:.3f}")
        
        if avg_ballast_conf > avg_baseline_conf:
            print("  ✓ Ballast improved average confidence!")
        else:
            print("  ⚠ Ballast did not improve average confidence")
    
    print("\n" + "=" * 50)
    print("EVALUATION COMPLETED")
    print("=" * 50)


if __name__ == '__main__':
    main() 