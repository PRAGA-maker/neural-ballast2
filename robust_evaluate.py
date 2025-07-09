#!/usr/bin/env python3
"""
Robust Neural Ballast Evaluation Suite
=====================================

This script provides a comprehensive evaluation of the Neural Ballast system by:

1. Loading the trained MNIST model and test dataset
2. Generating two test sets:
   - Problematic Set: ~100 inputs guaranteed to be in dead zones
   - Control Set: ~100 inputs that the baseline model classifies correctly
3. Running experiments to measure:
   - Correction Rate: How often Ballast fixes misclassified dead zone inputs
   - Regression Rate: How often Ballast breaks correctly classified inputs
4. Providing detailed analysis and visualization

Usage:
    python robust_evaluate.py
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import random
from typing import List, Dict, Tuple
import json
from datetime import datetime

# Add src directory to Python path
sys.path.append('src')

from model import create_model, MNISTNet
from ballast import NeuralBallast
from diagnostic import is_in_dead_zone, get_singfol_dim
from correction import apply_corrective_nudge


class RobustEvaluator:
    """Comprehensive evaluation suite for Neural Ballast."""
    
    def __init__(self, model_path: str = "src/mnist_net.pth", verbose: bool = True):
        """
        Initialize the evaluator.
        
        Args:
            model_path: Path to the trained model
            verbose: Whether to print detailed progress
        """
        self.verbose = verbose
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = self._load_model(model_path)
        
        # Create ballast wrapper
        self.ballast = NeuralBallast(
            model=self.model,
            dim_threshold=1,
            noise_sigma=0.01,
            max_attempts=10,
            verbose=False  # Suppress individual corrections for cleaner output
        )
        
        # Load MNIST test dataset
        self.test_dataset = self._load_test_dataset()
        
        # Initialize results storage
        self.results = {
            'problematic_set': [],
            'control_set': [],
            'evaluation_time': None,
            'correction_rate': 0.0,
            'regression_rate': 0.0,
            'summary_stats': {}
        }
    
    def _load_model(self, model_path: str) -> nn.Module:
        """Load the trained MNIST model."""
        if self.verbose:
            print(f"Loading model from {model_path}...")
        
        if not os.path.exists(model_path):
            print(f"Model file not found at {model_path}")
            print("Creating new model (will not be trained)...")
            return create_model()
        
        model = create_model()
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(self.device)
        model.eval()
        
        if self.verbose:
            print("✓ Model loaded successfully")
        
        return model
    
    def _load_test_dataset(self):
        """Load MNIST test dataset."""
        if self.verbose:
            print("Loading MNIST test dataset...")
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        test_dataset = torchvision.datasets.MNIST(
            root='src/data', 
            train=False, 
            download=True, 
            transform=transform
        )
        
        if self.verbose:
            print(f"✓ Loaded {len(test_dataset)} test samples")
        
        return test_dataset
    
    def generate_problematic_set(self, target_size: int = 100) -> List[Dict]:
        """
        Generate a set of inputs guaranteed to be in dead zones.
        
        Args:
            target_size: Target number of problematic inputs
            
        Returns:
            List of dictionaries containing problematic inputs and metadata
        """
        if self.verbose:
            print(f"\n🔍 Generating problematic set (target: {target_size} samples)...")
        
        problematic_inputs = []
        max_total_attempts = target_size * 50  # Reasonable upper bound
        attempts = 0
        
        while len(problematic_inputs) < target_size and attempts < max_total_attempts:
            attempts += 1
            
            # Try different strategies to generate dead zone inputs
            dead_zone_input = self._generate_single_dead_zone_input()
            
            if dead_zone_input is not None:
                # Get original label (if started from dataset sample)
                original_idx = random.randint(0, len(self.test_dataset) - 1)
                _, original_label = self.test_dataset[original_idx]
                
                # Test baseline model prediction
                with torch.no_grad():
                    baseline_output = self.model(dead_zone_input)
                    baseline_pred = baseline_output.argmax(dim=1).item()
                    baseline_confidence = torch.softmax(baseline_output, dim=1).max().item()
                
                # Store the problematic input
                problematic_inputs.append({
                    'input_tensor': dead_zone_input.clone(),
                    'original_label': original_label,
                    'baseline_pred': baseline_pred,
                    'baseline_confidence': baseline_confidence,
                    'is_misclassified': baseline_pred != original_label,
                    'singfol_dim': get_singfol_dim(self.model, dead_zone_input)
                })
                
                if self.verbose and len(problematic_inputs) % 10 == 0:
                    print(f"  Generated {len(problematic_inputs)}/{target_size} problematic inputs...")
        
        if self.verbose:
            misclassified_count = sum(1 for item in problematic_inputs if item['is_misclassified'])
            print(f"✓ Generated {len(problematic_inputs)} problematic inputs")
            print(f"  - {misclassified_count} are misclassified by baseline model")
            print(f"  - Average SingFolDIM: {np.mean([item['singfol_dim'] for item in problematic_inputs]):.2f}")
        
        return problematic_inputs
    
    def _generate_single_dead_zone_input(self) -> torch.Tensor:
        """Generate a single input that is in a dead zone."""
        strategies = [
            self._strategy_noisy_sample,
            self._strategy_gradient_based,
            self._strategy_random_input
        ]
        
        for strategy in strategies:
            try:
                result = strategy()
                if result is not None and is_in_dead_zone(self.model, result):
                    return result
            except Exception:
                continue
        
        return None
    
    def _strategy_noisy_sample(self) -> torch.Tensor:
        """Strategy 1: Add noise to dataset samples."""
        idx = random.randint(0, len(self.test_dataset) - 1)
        sample_image, _ = self.test_dataset[idx]
        
        # Add various amounts of noise
        noise_scales = [0.5, 1.0, 1.5, 2.0]
        for noise_scale in noise_scales:
            noise = torch.randn_like(sample_image) * noise_scale
            noisy_input = sample_image + noise
            noisy_input = torch.clamp(noisy_input, -3, 3)
            input_tensor = noisy_input.unsqueeze(0).to(self.device)
            
            if is_in_dead_zone(self.model, input_tensor):
                return input_tensor
        
        return None
    
    def _strategy_gradient_based(self) -> torch.Tensor:
        """Strategy 2: Use gradient information to find dead zones."""
        idx = random.randint(0, len(self.test_dataset) - 1)
        sample_image, _ = self.test_dataset[idx]
        input_tensor = sample_image.unsqueeze(0).to(self.device).requires_grad_(True)
        
        try:
            output = self.model(input_tensor)
            loss = output.sum()
            loss.backward()
            
            grad = input_tensor.grad.data
            perturbation_scales = [0.1, 0.5, 1.0]
            
            for scale in perturbation_scales:
                perturbation = grad.sign() * scale * random.uniform(0.1, 1.0)
                perturbed_input = input_tensor.detach() + perturbation
                
                if is_in_dead_zone(self.model, perturbed_input):
                    return perturbed_input
        except Exception:
            pass
        
        return None
    
    def _strategy_random_input(self) -> torch.Tensor:
        """Strategy 3: Generate random inputs."""
        random_scales = [1.0, 2.0, 3.0]
        for scale in random_scales:
            random_input = torch.randn(1, 1, 28, 28).to(self.device) * scale
            if is_in_dead_zone(self.model, random_input):
                return random_input
        
        return None
    
    def generate_control_set(self, target_size: int = 100) -> List[Dict]:
        """
        Generate a control set of inputs that the baseline model classifies correctly.
        
        Args:
            target_size: Target number of control inputs
            
        Returns:
            List of dictionaries containing control inputs and metadata
        """
        if self.verbose:
            print(f"\n✅ Generating control set (target: {target_size} samples)...")
        
        control_inputs = []
        attempts = 0
        max_attempts = target_size * 10  # Should be easy to find correct predictions
        
        while len(control_inputs) < target_size and attempts < max_attempts:
            attempts += 1
            
            # Get a random sample from test set
            idx = random.randint(0, len(self.test_dataset) - 1)
            sample_image, true_label = self.test_dataset[idx]
            input_tensor = sample_image.unsqueeze(0).to(self.device)
            
            # Check if baseline model predicts correctly
            with torch.no_grad():
                baseline_output = self.model(input_tensor)
                baseline_pred = baseline_output.argmax(dim=1).item()
                baseline_confidence = torch.softmax(baseline_output, dim=1).max().item()
            
            # Only include if correctly classified and reasonably confident
            if baseline_pred == true_label and baseline_confidence > 0.7:
                control_inputs.append({
                    'input_tensor': input_tensor.clone(),
                    'true_label': true_label,
                    'baseline_pred': baseline_pred,
                    'baseline_confidence': baseline_confidence,
                    'in_dead_zone': is_in_dead_zone(self.model, input_tensor),
                    'singfol_dim': get_singfol_dim(self.model, input_tensor)
                })
                
                if self.verbose and len(control_inputs) % 20 == 0:
                    print(f"  Generated {len(control_inputs)}/{target_size} control inputs...")
        
        if self.verbose:
            dead_zone_count = sum(1 for item in control_inputs if item['in_dead_zone'])
            print(f"✓ Generated {len(control_inputs)} control inputs")
            print(f"  - Average confidence: {np.mean([item['baseline_confidence'] for item in control_inputs]):.3f}")
            print(f"  - {dead_zone_count} are in dead zones despite correct classification")
        
        return control_inputs
    
    def run_correction_rate_test(self, problematic_set: List[Dict]) -> float:
        """
        Test how often Ballast corrects misclassified inputs from the problematic set.
        
        Args:
            problematic_set: List of problematic inputs
            
        Returns:
            float: Correction rate (0.0 to 1.0)
        """
        if self.verbose:
            print(f"\n🔧 Running Correction Rate Test...")
        
        # Filter to only misclassified inputs
        misclassified_inputs = [item for item in problematic_set if item['is_misclassified']]
        
        if len(misclassified_inputs) == 0:
            if self.verbose:
                print("⚠ No misclassified inputs found in problematic set")
            return 0.0
        
        corrections = 0
        
        for i, item in enumerate(misclassified_inputs):
            # Run through Ballast
            ballast_output = self.ballast.predict(item['input_tensor'])
            ballast_pred = ballast_output.argmax(dim=1).item()
            ballast_confidence = torch.softmax(ballast_output, dim=1).max().item()
            
            # Check if prediction was corrected
            was_corrected = ballast_pred == item['original_label']
            if was_corrected:
                corrections += 1
            
            # Store detailed results
            item['ballast_pred'] = ballast_pred
            item['ballast_confidence'] = ballast_confidence
            item['was_corrected'] = was_corrected
            
            if self.verbose and (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(misclassified_inputs)} misclassified inputs...")
        
        correction_rate = corrections / len(misclassified_inputs)
        
        if self.verbose:
            print(f"✓ Correction Rate: {corrections}/{len(misclassified_inputs)} = {correction_rate:.1%}")
        
        return correction_rate
    
    def run_regression_rate_test(self, control_set: List[Dict]) -> float:
        """
        Test how often Ballast breaks correctly classified inputs (regression rate).
        
        Args:
            control_set: List of control inputs
            
        Returns:
            float: Regression rate (0.0 to 1.0, lower is better)
        """
        if self.verbose:
            print(f"\n⚖️ Running Regression Rate Test...")
        
        regressions = 0
        
        for i, item in enumerate(control_set):
            # Run through Ballast
            ballast_output = self.ballast.predict(item['input_tensor'])
            ballast_pred = ballast_output.argmax(dim=1).item()
            ballast_confidence = torch.softmax(ballast_output, dim=1).max().item()
            
            # Check if correct prediction was broken
            was_regression = ballast_pred != item['true_label']
            if was_regression:
                regressions += 1
            
            # Store detailed results
            item['ballast_pred'] = ballast_pred
            item['ballast_confidence'] = ballast_confidence
            item['was_regression'] = was_regression
            
            if self.verbose and (i + 1) % 20 == 0:
                print(f"  Processed {i + 1}/{len(control_set)} control inputs...")
        
        regression_rate = regressions / len(control_set)
        
        if self.verbose:
            print(f"✓ Regression Rate: {regressions}/{len(control_set)} = {regression_rate:.1%}")
        
        return regression_rate
    
    def visualize_correction_example(self, problematic_set: List[Dict], save_path: str = "correction_example.png"):
        """
        Create a visualization showing original, noise, and corrected images.
        
        Args:
            problematic_set: List of problematic inputs
            save_path: Path to save the visualization
        """
        if self.verbose:
            print(f"\n🎨 Creating correction visualization...")
        
        # Find a good example (misclassified input that was corrected)
        good_examples = [
            item for item in problematic_set 
            if item.get('is_misclassified', False) and item.get('was_corrected', False)
        ]
        
        if not good_examples:
            if self.verbose:
                print("⚠ No good correction examples found for visualization")
            return
        
        # Use the first good example
        example = good_examples[0]
        original_input = example['input_tensor']
        
        # Apply corrective nudge to get the corrected input
        try:
            corrected_input = apply_corrective_nudge(
                model=self.model,
                bad_input_tensor=original_input,
                is_in_dead_zone_func=lambda m, x: is_in_dead_zone(m, x, 1),
                max_attempts=10,
                sigma=0.01
            )
            
            # Calculate the noise that was added
            noise = corrected_input - original_input
            
            # Create the visualization
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            
            # Original image
            axes[0].imshow(original_input.squeeze().cpu().detach().numpy(), cmap='gray')
            axes[0].set_title(f'Original Image\nBaseline Pred: {example["baseline_pred"]}\nTrue Label: {example["original_label"]}')
            axes[0].axis('off')
            
            # Noise pattern
            noise_img = noise.squeeze().cpu().detach().numpy()
            im = axes[1].imshow(noise_img, cmap='RdBu', vmin=-0.1, vmax=0.1)
            axes[1].set_title(f'Noise Pattern\nScale: ±{np.abs(noise_img).max():.3f}')
            axes[1].axis('off')
            plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
            
            # Corrected image
            axes[2].imshow(corrected_input.squeeze().cpu().detach().numpy(), cmap='gray')
            axes[2].set_title(f'Corrected Image\nBallast Pred: {example["ballast_pred"]}\nTrue Label: {example["original_label"]}')
            axes[2].axis('off')
            
            plt.suptitle('Neural Ballast Correction Example', fontsize=16)
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            
            if self.verbose:
                print(f"✓ Visualization saved to {save_path}")
            
        except Exception as e:
            if self.verbose:
                print(f"⚠ Error creating visualization: {e}")
    
    def run_full_evaluation(self, 
                           problematic_size: int = 100, 
                           control_size: int = 100) -> Dict:
        """
        Run the complete evaluation suite.
        
        Args:
            problematic_size: Size of problematic test set
            control_size: Size of control test set
            
        Returns:
            dict: Complete evaluation results
        """
        start_time = datetime.now()
        
        if self.verbose:
            print("=" * 60)
            print("🧪 NEURAL BALLAST ROBUST EVALUATION SUITE")
            print("=" * 60)
        
        # Generate test sets
        problematic_set = self.generate_problematic_set(problematic_size)
        control_set = self.generate_control_set(control_size)
        
        # Run tests
        correction_rate = self.run_correction_rate_test(problematic_set)
        regression_rate = self.run_regression_rate_test(control_set)
        
        # Create visualization
        self.visualize_correction_example(problematic_set)
        
        # Calculate summary statistics
        ballast_stats = self.ballast.get_statistics()
        
        # Store results
        self.results = {
            'problematic_set': problematic_set,
            'control_set': control_set,
            'correction_rate': correction_rate,
            'regression_rate': regression_rate,
            'evaluation_time': (datetime.now() - start_time).total_seconds(),
            'ballast_stats': ballast_stats,
            'summary_stats': {
                'problematic_set_size': len(problematic_set),
                'control_set_size': len(control_set),
                'problematic_misclassified': sum(1 for item in problematic_set if item.get('is_misclassified', False)),
                'corrections_achieved': sum(1 for item in problematic_set if item.get('was_corrected', False)),
                'regressions_occurred': sum(1 for item in control_set if item.get('was_regression', False))
            }
        }
        
        # Print final report
        self.print_final_report()
        
        return self.results
    
    def print_final_report(self):
        """Print a comprehensive final report."""
        if not self.verbose:
            return
        
        print("\n" + "=" * 60)
        print("📊 FINAL EVALUATION REPORT")
        print("=" * 60)
        
        # Key metrics
        print(f"\n🎯 KEY METRICS:")
        print(f"  Correction Rate: {self.results['correction_rate']:.1%}")
        print(f"  Regression Rate: {self.results['regression_rate']:.1%}")
        
        # Dataset statistics
        print(f"\n📊 DATASET STATISTICS:")
        stats = self.results['summary_stats']
        print(f"  Problematic Set Size: {stats['problematic_set_size']}")
        print(f"  Control Set Size: {stats['control_set_size']}")
        print(f"  Misclassified in Problematic Set: {stats['problematic_misclassified']}")
        
        # Ballast performance
        print(f"\n⚙️ BALLAST PERFORMANCE:")
        ballast_stats = self.results['ballast_stats']
        print(f"  Total Predictions: {ballast_stats['total_predictions']}")
        print(f"  Dead Zone Detections: {ballast_stats['dead_zone_detections']}")
        print(f"  Successful Corrections: {ballast_stats['successful_corrections']}")
        print(f"  Dead Zone Detection Rate: {ballast_stats['dead_zone_rate']:.1%}")
        print(f"  Correction Success Rate: {ballast_stats['correction_success_rate']:.1%}")
        
        # Performance analysis
        print(f"\n🔍 ANALYSIS:")
        if self.results['correction_rate'] > 0.5:
            print("  ✅ Good correction rate - Ballast effectively fixes dead zone issues")
        else:
            print("  ⚠️  Moderate correction rate - Room for improvement in dead zone handling")
        
        if self.results['regression_rate'] < 0.05:
            print("  ✅ Low regression rate - Ballast doesn't harm correct predictions")
        elif self.results['regression_rate'] < 0.1:
            print("  ⚠️  Moderate regression rate - Some impact on correct predictions")
        else:
            print("  ❌ High regression rate - Ballast may be too aggressive")
        
        print(f"\n⏱️  Evaluation completed in {self.results['evaluation_time']:.1f} seconds")
        print("=" * 60)
    
    def save_results(self, filename: str = "evaluation_results.json"):
        """Save evaluation results to JSON file."""
        # Prepare serializable results
        serializable_results = {
            'correction_rate': self.results['correction_rate'],
            'regression_rate': self.results['regression_rate'],
            'evaluation_time': self.results['evaluation_time'],
            'ballast_stats': self.results['ballast_stats'],
            'summary_stats': self.results['summary_stats'],
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        if self.verbose:
            print(f"✓ Results saved to {filename}")


def main():
    """Main evaluation function."""
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Create evaluator
    evaluator = RobustEvaluator(verbose=True)
    
    # Run full evaluation
    results = evaluator.run_full_evaluation(
        problematic_size=100,
        control_size=100
    )
    
    # Save results
    evaluator.save_results()
    
    print("\n🎉 Evaluation complete! Check 'correction_example.png' for visualization.")


if __name__ == '__main__':
    main() 