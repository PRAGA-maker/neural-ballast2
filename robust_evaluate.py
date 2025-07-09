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
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import json
import matplotlib.pyplot as plt
import time
from pathlib import Path

# Import our modules
from src.model import MNISTNet
from src.ballast import NeuralBallast
from src.curvature_attack import FisherInformationAttack, generate_curvature_attacks

def load_model_and_data():
    """Load the trained model and MNIST test dataset."""
    print("Loading model from src/mnist_net.pth...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = MNISTNet()
    try:
        checkpoint = torch.load('src/mnist_net.pth', map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.to(device)
        model.eval()
        print(f"Created MNISTNet with {sum(p.numel() for p in model.parameters()):,} parameters")
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return None, None, None
    
    # Load MNIST test dataset
    print("Loading MNIST test dataset...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = torchvision.datasets.MNIST(
        root='src/data', train=False, download=True, transform=transform
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=True
    )
    
    print(f"✓ Loaded {len(test_dataset)} test samples")
    return model, test_loader, device

def generate_curvature_based_problematic_set(model, ballast, test_loader, device, target_size=100):
    """Generate problematic inputs using curvature-based attacks."""
    print("🔬 CURVATURE-BASED PROBLEMATIC INPUT GENERATION")
    print("=" * 60)
    
    # Initialize Fisher Information Attack
    fim_attack = FisherInformationAttack(model, device)
    
    problematic_inputs = []
    misclassified_count = 0
    total_attempts = 0
    
    attack_methods = [
        ("OSSA", lambda x: fim_attack.one_step_spectral_attack(x, epsilon=0.15)),
        ("TSSA", lambda x: fim_attack.two_step_spectral_attack(x, epsilon=0.2)),
        ("Curvature", lambda x: fim_attack.curvature_enhanced_attack(x, epsilon=0.25)),
        ("Targeted", lambda x: fim_attack.targeted_dead_zone_attack(x, ballast, epsilon=0.3))
    ]
    
    method_stats = {method: {'attempts': 0, 'dead_zones': 0, 'misclassified': 0} 
                   for method, _ in attack_methods}
    
    print(f"Generating {target_size} problematic inputs using curvature attacks...")
    
    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(test_loader):
            if len(problematic_inputs) >= target_size:
                break
                
            data, targets = data.to(device), targets.to(device)
            
            # Original prediction
            original_output = model(data)
            original_pred = torch.argmax(original_output, dim=1)
            
            # Skip if already misclassified
            if original_pred != targets:
                continue
            
            # Try each attack method
            for method_name, attack_func in attack_methods:
                if len(problematic_inputs) >= target_size:
                    break
                    
                try:
                    total_attempts += 1
                    method_stats[method_name]['attempts'] += 1
                    
                    # Generate adversarial example
                    if method_name == "Targeted":
                        adv_input = attack_func(data)
                        if adv_input is None:
                            continue
                    else:
                        adv_input = attack_func(data)
                    
                    # Test with ballast
                    ballast_output = ballast(adv_input)
                    ballast_pred = torch.argmax(ballast_output, dim=1)
                    ballast_conf = F.softmax(ballast_output, dim=1).max()
                    
                    # Check if it's a dead zone (high SingFolDIM or low confidence)
                    is_dead_zone = False
                    singfoldim = 0
                    
                    if hasattr(ballast, 'last_singfoldim'):
                        singfoldim = ballast.last_singfoldim
                        is_dead_zone = singfoldim > 0
                    
                    # Alternative dead zone detection: low confidence
                    if ballast_conf < 0.7:
                        is_dead_zone = True
                    
                    if is_dead_zone:
                        method_stats[method_name]['dead_zones'] += 1
                    
                    # Check if misclassified
                    is_misclassified = ballast_pred != targets
                    if is_misclassified:
                        method_stats[method_name]['misclassified'] += 1
                        misclassified_count += 1
                    
                    # Add to problematic set if dead zone or misclassified
                    if is_dead_zone or is_misclassified:
                        problematic_inputs.append({
                            'input': adv_input.cpu(),
                            'true_label': targets.cpu(),
                            'baseline_pred': original_pred.cpu(),
                            'ballast_pred': ballast_pred.cpu(),
                            'ballast_conf': ballast_conf.cpu(),
                            'singfoldim': singfoldim,
                            'is_dead_zone': is_dead_zone,
                            'is_misclassified': is_misclassified,
                            'method': method_name,
                            'original_input': data.cpu()
                        })
                        
                        if len(problematic_inputs) % 10 == 0:
                            print(f"  Generated {len(problematic_inputs)}/{target_size} problematic inputs...")
                            
                            # Show method effectiveness
                            for m, stats in method_stats.items():
                                if stats['attempts'] > 0:
                                    dz_rate = stats['dead_zones'] / stats['attempts'] * 100
                                    mis_rate = stats['misclassified'] / stats['attempts'] * 100
                                    print(f"    {m}: {stats['attempts']} attempts, "
                                          f"{dz_rate:.1f}% dead zones, {mis_rate:.1f}% misclassified")
                
                except Exception as e:
                    print(f"    ⚠ Attack {method_name} failed: {e}")
                    continue
    
    # Calculate statistics
    avg_singfoldim = np.mean([p['singfoldim'] for p in problematic_inputs if p['singfoldim'] > 0]) if any(p['singfoldim'] > 0 for p in problematic_inputs) else 0
    
    print(f"✓ Generated {len(problematic_inputs)} problematic inputs")
    print(f"  - {misclassified_count} are misclassified by baseline model")
    print(f"  - Average SingFolDIM: {avg_singfoldim:.3f}")
    
    # Print final method statistics
    print("\n📊 ATTACK METHOD EFFECTIVENESS:")
    for method, stats in method_stats.items():
        if stats['attempts'] > 0:
            dz_rate = stats['dead_zones'] / stats['attempts'] * 100
            mis_rate = stats['misclassified'] / stats['attempts'] * 100
            print(f"  {method:10}: {stats['attempts']:3d} attempts, "
                  f"{dz_rate:5.1f}% dead zones, {mis_rate:5.1f}% misclassified")
    
    return problematic_inputs

def evaluate_correction_rate(ballast, problematic_set, device):
    """Evaluate how often Neural Ballast corrects misclassified inputs."""
    print("🔧 Running Correction Rate Test...")
    
    if not problematic_set:
        print("⚠ No misclassified inputs found in problematic set")
        return 0.0, {}
    
    misclassified_inputs = [p for p in problematic_set if p['is_misclassified']]
    
    if not misclassified_inputs:
        print("⚠ No misclassified inputs found in problematic set")
        return 0.0, {}
    
    corrections = 0
    total = len(misclassified_inputs)
    method_corrections = {}
    
    print(f"  Testing {total} misclassified inputs for correction...")
    
    with torch.no_grad():
        for i, item in enumerate(misclassified_inputs):
            # Test Neural Ballast correction
            ballast_input = item['input'].to(device)
            true_label = item['true_label'].to(device)
            
            ballast_output = ballast(ballast_input)
            ballast_pred = torch.argmax(ballast_output, dim=1)
            
            if ballast_pred == true_label:
                corrections += 1
                method = item['method']
                method_corrections[method] = method_corrections.get(method, 0) + 1
            
            if (i + 1) % 20 == 0:
                print(f"    Processed {i + 1}/{total} misclassified inputs...")
    
    correction_rate = corrections / total if total > 0 else 0.0
    print(f"✓ Correction Rate: {corrections}/{total} = {correction_rate:.1%}")
    
    # Show corrections by method
    if method_corrections:
        print("  Corrections by attack method:")
        for method, count in method_corrections.items():
            method_total = sum(1 for p in misclassified_inputs if p['method'] == method)
            if method_total > 0:
                rate = count / method_total * 100
                print(f"    {method}: {count}/{method_total} ({rate:.1f}%)")
    
    return correction_rate, method_corrections

def create_curvature_correction_visualization(problematic_set, device):
    """Create visualization showing curvature attack corrections."""
    print("🎨 Creating curvature-based correction visualization...")
    
    # Find good examples for each attack method
    method_examples = {}
    for item in problematic_set:
        method = item['method']
        if method not in method_examples and item['is_dead_zone']:
            method_examples[method] = item
    
    if not method_examples:
        print("⚠ No good correction examples found for visualization")
        return False
    
    # Create visualization
    fig, axes = plt.subplots(2, len(method_examples), figsize=(4*len(method_examples), 8))
    if len(method_examples) == 1:
        axes = axes.reshape(-1, 1)
    
    for col, (method, item) in enumerate(method_examples.items()):
        original = item['original_input'].squeeze().numpy()
        adversarial = item['input'].squeeze().numpy()
        
        # Denormalize for display
        original = original * 0.3081 + 0.1307
        adversarial = adversarial * 0.3081 + 0.1307
        
        # Original image
        axes[0, col].imshow(original, cmap='gray')
        axes[0, col].set_title(f'{method}\nOriginal (pred: {item["baseline_pred"].item()})')
        axes[0, col].axis('off')
        
        # Adversarial image
        axes[1, col].imshow(adversarial, cmap='gray')
        axes[1, col].set_title(f'Attack Result\n(conf: {item["ballast_conf"]:.3f}, SFD: {item["singfoldim"]:.1f})')
        axes[1, col].axis('off')
    
    plt.tight_layout()
    plt.savefig('curvature_attack_examples.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Curvature attack examples saved as 'curvature_attack_examples.png'")
    return True

def main():
    print("=" * 60)
    print("🧪 NEURAL BALLAST CURVATURE-ENHANCED EVALUATION SUITE")
    print("=" * 60)
    
    start_time = time.time()
    
    # Load model and data
    model, test_loader, device = load_model_and_data()
    if model is None:
        return
    
    # Create Neural Ballast wrapper
    ballast = NeuralBallast(model, noise_scale=0.01, threshold=0.1)
    ballast.to(device)
    ballast.eval()
    
    print("\n" + "=" * 60)
    print("🔍 Generating problematic set using curvature attacks...")
    print("=" * 60)
    
    # Generate problematic inputs using curvature-based attacks
    problematic_set = generate_curvature_based_problematic_set(
        model, ballast, test_loader, device, target_size=100
    )
    
    print("\n" + "=" * 60)
    print("🔧 EVALUATION PHASE")
    print("=" * 60)
    
    # Evaluate correction rate
    correction_rate, method_corrections = evaluate_correction_rate(ballast, problematic_set, device)
    
    # Generate control set (correctly classified inputs)
    print("\n✅ Generating control set (target: 100 samples)...")
    control_set = []
    with torch.no_grad():
        for data, targets in test_loader:
            if len(control_set) >= 100:
                break
            
            data, targets = data.to(device), targets.to(device)
            output = model(data)
            pred = torch.argmax(output, dim=1)
            conf = F.softmax(output, dim=1).max()
            
            if pred == targets and conf > 0.9:  # High confidence correct predictions
                control_set.append({
                    'input': data.cpu(),
                    'label': targets.cpu(),
                    'confidence': conf.cpu()
                })
            
            if len(control_set) % 20 == 0 and len(control_set) > 0:
                print(f"  Generated {len(control_set)}/100 control inputs...")
    
    avg_control_conf = np.mean([item['confidence'].item() for item in control_set])
    print(f"✓ Generated {len(control_set)} control inputs")
    print(f"  - Average confidence: {avg_control_conf:.3f}")
    
    # Evaluate regression rate
    print("\n⚖️ Running Regression Rate Test...")
    regression_count = 0
    dead_zones_in_control = 0
    
    with torch.no_grad():
        for i, item in enumerate(control_set):
            data = item['input'].to(device)
            label = item['label'].to(device)
            
            ballast_output = ballast(data)
            ballast_pred = torch.argmax(ballast_output, dim=1)
            
            if ballast_pred != label:
                regression_count += 1
            
            # Check for dead zones in control set
            if hasattr(ballast, 'last_singfoldim') and ballast.last_singfoldim > 0:
                dead_zones_in_control += 1
            
            if (i + 1) % 20 == 0:
                print(f"  Processed {i + 1}/{len(control_set)} control inputs...")
    
    regression_rate = regression_count / len(control_set) if control_set else 0.0
    print(f"✓ Regression Rate: {regression_count}/{len(control_set)} = {regression_rate:.1%}")
    
    # Create visualizations
    create_curvature_correction_visualization(problematic_set, device)
    
    # Ballast performance statistics
    total_predictions = len(problematic_set) + len(control_set)
    total_dead_zones = sum(1 for p in problematic_set if p['is_dead_zone']) + dead_zones_in_control
    successful_corrections = sum(1 for p in problematic_set 
                                if p['is_misclassified'] and p['ballast_pred'] == p['true_label'])
    
    dead_zone_rate = total_dead_zones / total_predictions if total_predictions > 0 else 0.0
    correction_success_rate = successful_corrections / sum(1 for p in problematic_set if p['is_misclassified']) if any(p['is_misclassified'] for p in problematic_set) else 0.0
    
    # Final evaluation report
    print("\n" + "=" * 60)
    print("📊 FINAL CURVATURE-ENHANCED EVALUATION REPORT")
    print("=" * 60)
    
    print("🎯 KEY METRICS:")
    print(f"  Correction Rate: {correction_rate:.1%}")
    print(f"  Regression Rate: {regression_rate:.1%}")
    
    print("📊 DATASET STATISTICS:")
    print(f"  Problematic Set Size: {len(problematic_set)}")
    print(f"  Control Set Size: {len(control_set)}")
    print(f"  Misclassified in Problematic Set: {sum(1 for p in problematic_set if p['is_misclassified'])}")
    
    print("⚙️ BALLAST PERFORMANCE:")
    print(f"  Total Predictions: {total_predictions}")
    print(f"  Dead Zone Detections: {total_dead_zones}")
    print(f"  Successful Corrections: {successful_corrections}")
    print(f"  Dead Zone Detection Rate: {dead_zone_rate:.1%}")
    print(f"  Correction Success Rate: {correction_success_rate:.1%}")
    
    print("🔍 ANALYSIS:")
    if correction_rate >= 0.7:
        print("  ✅ High correction rate - Ballast effectively handles problematic inputs")
    elif correction_rate >= 0.4:
        print("  ⚠️  Moderate correction rate - Room for improvement in dead zone handling")
    else:
        print("  ❌ Low correction rate - Ballast struggles with curvature-based attacks")
    
    if regression_rate <= 0.05:
        print("  ✅ Low regression rate - Ballast doesn't harm correct predictions")
    elif regression_rate <= 0.15:
        print("  ⚠️  Moderate regression rate - Some interference with correct predictions")
    else:
        print("  ❌ High regression rate - Ballast significantly harms correct predictions")
    
    elapsed_time = time.time() - start_time
    print(f"⏱️  Evaluation completed in {elapsed_time:.1f} seconds")
    
    # Save results
    results = {
        'curvature_enhanced_evaluation': True,
        'correction_rate': correction_rate,
        'regression_rate': regression_rate,
        'problematic_set_size': len(problematic_set),
        'control_set_size': len(control_set),
        'misclassified_count': sum(1 for p in problematic_set if p['is_misclassified']),
        'dead_zone_count': total_dead_zones,
        'successful_corrections': successful_corrections,
        'method_corrections': method_corrections,
        'ballast_stats': {
            'total_predictions': total_predictions,
            'dead_zone_detections': total_dead_zones,
            'dead_zone_rate': dead_zone_rate,
            'correction_success_rate': correction_success_rate
        },
        'evaluation_time': elapsed_time,
        'attack_methods_used': ['OSSA', 'TSSA', 'Curvature', 'Targeted']
    }
    
    with open('curvature_evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("=" * 60)
    print("✓ Curvature-enhanced evaluation results saved to 'curvature_evaluation_results.json'")
    print("🎉 Evaluation complete! Check 'curvature_attack_examples.png' for attack visualizations.")

if __name__ == "__main__":
    main() 