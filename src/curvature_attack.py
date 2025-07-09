"""
Curvature-Based Adversarial Attack Module

Implementation inspired by Eliot Tron's "Canonical foliations of neural networks: 
application to robustness" and the CurvNetAttack approach.

This module implements Fisher Information Metric (FIM) based attacks that exploit
the geometric properties of neural networks to generate inputs more likely to 
trigger dead zones.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class FisherInformationAttack:
    """
    Implements curvature-based adversarial attacks using Fisher Information Metric.
    
    Based on the Two-Step Spectral Attack (TSSA) approach that uses Riemannian
    geometry to find inputs that exploit neural network vulnerabilities.
    """
    
    def __init__(self, model: torch.nn.Module, device: str = 'cpu'):
        """
        Initialize the Fisher Information Attack.
        
        Args:
            model: PyTorch model to attack
            device: Device to run computations on
        """
        self.model = model
        self.device = device
        self.model.eval()
    
    def compute_fisher_information_matrix(self, x: torch.Tensor, 
                                        y_pred: torch.Tensor) -> torch.Tensor:
        """
        Compute the Fisher Information Matrix for input x.
        
        Args:
            x: Input tensor
            y_pred: Model predictions (probabilities)
            
        Returns:
            Fisher Information Matrix
        """
        x.requires_grad_(True)
        
        # Clear any existing gradients
        if x.grad is not None:
            x.grad.zero_()
        
        # Compute log probabilities
        log_probs = torch.log(y_pred + 1e-8)
        
        # Initialize FIM
        input_dim = x.numel()
        fim = torch.zeros(input_dim, input_dim, device=self.device)
        
        # Compute gradients for each class
        for i in range(y_pred.shape[-1]):
            # Clear gradients
            if x.grad is not None:
                x.grad.zero_()
            
            # Compute gradient of log probability w.r.t. input
            log_probs[0, i].backward(retain_graph=True)
            grad = x.grad.view(-1)
            
            # Add to Fisher Information Matrix
            fim += y_pred[0, i] * torch.outer(grad, grad)
        
        return fim
    
    def compute_fim_eigendecomposition(self, fim: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute eigendecomposition of Fisher Information Matrix.
        
        Args:
            fim: Fisher Information Matrix
            
        Returns:
            Tuple of (eigenvalues, eigenvectors)
        """
        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(fim)
            return eigenvalues, eigenvectors
        except Exception as e:
            logger.warning(f"Eigendecomposition failed: {e}, using SVD fallback")
            u, s, vh = torch.linalg.svd(fim)
            return s, u
    
    def one_step_spectral_attack(self, x: torch.Tensor, 
                                epsilon: float = 0.1) -> torch.Tensor:
        """
        Implement One-Step Spectral Attack (OSSA) using largest FIM eigenvector.
        
        Args:
            x: Input tensor
            epsilon: Attack budget (L2 norm constraint)
            
        Returns:
            Adversarial example
        """
        x = x.clone().detach().requires_grad_(True)
        
        with torch.enable_grad():
            # Forward pass
            output = self.model(x)
            y_pred = F.softmax(output, dim=-1)
            
            # Compute Fisher Information Matrix
            fim = self.compute_fisher_information_matrix(x, y_pred)
            
            # Eigendecomposition
            eigenvalues, eigenvectors = self.compute_fim_eigendecomposition(fim)
            
            # Get direction of largest eigenvalue
            max_idx = torch.argmax(eigenvalues)
            attack_direction = eigenvectors[:, max_idx].view(x.shape)
            
            # Normalize and scale by epsilon
            attack_direction = attack_direction / (torch.norm(attack_direction) + 1e-8)
            perturbation = epsilon * attack_direction
            
            # Choose sign that reduces confidence in original prediction
            original_class = torch.argmax(y_pred)
            
            # Test both directions
            x_pos = x + perturbation
            x_neg = x - perturbation
            
            output_pos = self.model(x_pos)
            output_neg = self.model(x_neg)
            
            prob_pos = F.softmax(output_pos, dim=-1)[0, original_class]
            prob_neg = F.softmax(output_neg, dim=-1)[0, original_class]
            
            # Choose direction that reduces original class probability most
            if prob_neg < prob_pos:
                return x_neg.detach()
            else:
                return x_pos.detach()
    
    def two_step_spectral_attack(self, x: torch.Tensor, 
                                epsilon: float = 0.1,
                                step1_ratio: float = 0.6) -> torch.Tensor:
        """
        Implement Two-Step Spectral Attack (TSSA) that accounts for curvature.
        
        Args:
            x: Input tensor
            epsilon: Total attack budget
            step1_ratio: Fraction of budget for first step
            
        Returns:
            Adversarial example
        """
        epsilon1 = epsilon * step1_ratio
        epsilon2 = epsilon * (1 - step1_ratio)
        
        # First step: Standard spectral attack
        x1 = self.one_step_spectral_attack(x, epsilon1)
        
        # Second step: Account for curvature at new position
        x1 = x1.clone().detach().requires_grad_(True)
        
        with torch.enable_grad():
            # Compute FIM at new position
            output1 = self.model(x1)
            y_pred1 = F.softmax(output1, dim=-1)
            fim1 = self.compute_fisher_information_matrix(x1, y_pred1)
            
            # Eigendecomposition at new position
            eigenvalues1, eigenvectors1 = self.compute_fim_eigendecomposition(fim1)
            
            # Get direction of largest eigenvalue at new position
            max_idx1 = torch.argmax(eigenvalues1)
            attack_direction1 = eigenvectors1[:, max_idx1].view(x.shape)
            
            # Normalize and scale
            attack_direction1 = attack_direction1 / (torch.norm(attack_direction1) + 1e-8)
            
            # Ensure we don't exceed total budget
            current_perturbation = x1 - x
            remaining_budget = epsilon - torch.norm(current_perturbation)
            
            if remaining_budget > 0:
                step2_magnitude = min(epsilon2, remaining_budget.item())
                perturbation2 = step2_magnitude * attack_direction1
                
                # Test both directions for second step
                original_class = torch.argmax(F.softmax(self.model(x), dim=-1))
                
                x2_pos = x1 + perturbation2
                x2_neg = x1 - perturbation2
                
                output_pos = self.model(x2_pos)
                output_neg = self.model(x2_neg)
                
                prob_pos = F.softmax(output_pos, dim=-1)[0, original_class]
                prob_neg = F.softmax(output_neg, dim=-1)[0, original_class]
                
                if prob_neg < prob_pos:
                    return x2_neg.detach()
                else:
                    return x2_pos.detach()
            else:
                return x1.detach()
    
    def curvature_enhanced_attack(self, x: torch.Tensor,
                                epsilon: float = 0.1,
                                num_steps: int = 10,
                                step_size: float = 0.01) -> torch.Tensor:
        """
        Multi-step attack that follows the curvature of the loss landscape.
        
        Args:
            x: Input tensor
            epsilon: Total attack budget
            num_steps: Number of iterative steps
            step_size: Step size for each iteration
            
        Returns:
            Adversarial example
        """
        x_adv = x.clone().detach()
        
        for step in range(num_steps):
            x_adv = x_adv.clone().detach().requires_grad_(True)
            
            with torch.enable_grad():
                # Forward pass
                output = self.model(x_adv)
                y_pred = F.softmax(output, dim=-1)
                
                # Compute FIM
                fim = self.compute_fisher_information_matrix(x_adv, y_pred)
                
                # Get attack direction
                eigenvalues, eigenvectors = self.compute_fim_eigendecomposition(fim)
                max_idx = torch.argmax(eigenvalues)
                attack_direction = eigenvectors[:, max_idx].view(x.shape)
                
                # Normalize
                attack_direction = attack_direction / (torch.norm(attack_direction) + 1e-8)
                
                # Update
                perturbation = step_size * attack_direction
                x_adv = x_adv + perturbation
                
                # Project to epsilon ball
                total_perturbation = x_adv - x
                if torch.norm(total_perturbation) > epsilon:
                    total_perturbation = epsilon * total_perturbation / torch.norm(total_perturbation)
                    x_adv = x + total_perturbation
        
        return x_adv.detach()
    
    def targeted_dead_zone_attack(self, x: torch.Tensor,
                                 ballast_model,
                                 epsilon: float = 0.1,
                                 max_iterations: int = 50) -> Optional[torch.Tensor]:
        """
        Specifically target inputs that trigger dead zones in Neural Ballast.
        
        Args:
            x: Input tensor
            ballast_model: Neural Ballast model to attack
            epsilon: Attack budget
            max_iterations: Maximum iterations
            
        Returns:
            Input that triggers dead zone, or None if not found
        """
        logger.info(f"Targeting dead zone attack on input shape {x.shape}")
        
        for iteration in range(max_iterations):
            # Try different attack strategies
            if iteration < 20:
                # Use TSSA for first 20 iterations
                x_adv = self.two_step_spectral_attack(x, epsilon * (0.5 + 0.5 * iteration / 20))
            else:
                # Use curvature-enhanced attack for remaining iterations
                x_adv = self.curvature_enhanced_attack(x, epsilon * (0.8 + 0.2 * iteration / max_iterations))
            
            # Check if this triggers a dead zone
            try:
                with torch.no_grad():
                    ballast_output = ballast_model(x_adv)
                    
                    # Check SingFolDIM and dead zone status
                    if hasattr(ballast_model, 'last_singfoldim'):
                        singfoldim = ballast_model.last_singfoldim
                        if singfoldim > 0:  # Found a dead zone
                            logger.info(f"Found dead zone at iteration {iteration}, SingFolDIM: {singfoldim}")
                            return x_adv
                    
                    # Alternative: check if prediction changed dramatically
                    original_output = self.model(x)
                    original_pred = torch.argmax(original_output)
                    new_pred = torch.argmax(ballast_output)
                    
                    original_conf = F.softmax(original_output, dim=-1).max()
                    new_conf = F.softmax(ballast_output, dim=-1).max()
                    
                    # Look for low confidence predictions (potential dead zones)
                    if new_conf < 0.6 and new_conf < original_conf * 0.7:
                        logger.info(f"Found potential dead zone at iteration {iteration}, confidence: {new_conf:.3f}")
                        return x_adv
                        
            except Exception as e:
                logger.warning(f"Error in dead zone check at iteration {iteration}: {e}")
                continue
        
        logger.warning(f"No dead zone found after {max_iterations} iterations")
        return None


def generate_curvature_attacks(model, ballast_model, test_loader, 
                             num_samples: int = 50,
                             device: str = 'cpu') -> Dict[str, Any]:
    """
    Generate curvature-based adversarial attacks on test samples.
    
    Args:
        model: Base neural network model
        ballast_model: Neural Ballast wrapped model
        test_loader: DataLoader with test samples
        num_samples: Number of samples to attack
        device: Device to run on
        
    Returns:
        Dictionary with attack results
    """
    attack = FisherInformationAttack(model, device)
    results = {
        'successful_attacks': [],
        'failed_attacks': [],
        'dead_zone_inputs': [],
        'attack_stats': {
            'ossa_success': 0,
            'tssa_success': 0,
            'targeted_success': 0,
            'total_attempts': 0
        }
    }
    
    logger.info(f"Starting curvature-based attack generation on {num_samples} samples")
    
    samples_processed = 0
    for batch_idx, (data, targets) in enumerate(test_loader):
        if samples_processed >= num_samples:
            break
            
        data = data.to(device)
        targets = targets.to(device)
        
        for i in range(data.shape[0]):
            if samples_processed >= num_samples:
                break
                
            x = data[i:i+1]  # Single sample
            target = targets[i:i+1]
            
            results['attack_stats']['total_attempts'] += 1
            
            # Try One-Step Spectral Attack
            try:
                x_ossa = attack.one_step_spectral_attack(x, epsilon=0.1)
                ossa_output = ballast_model(x_ossa)
                ossa_pred = torch.argmax(ossa_output)
                
                if ossa_pred != target:
                    results['attack_stats']['ossa_success'] += 1
                    results['successful_attacks'].append({
                        'method': 'OSSA',
                        'original': x.cpu(),
                        'adversarial': x_ossa.cpu(),
                        'original_pred': target.cpu(),
                        'adv_pred': ossa_pred.cpu()
                    })
            except Exception as e:
                logger.warning(f"OSSA failed for sample {samples_processed}: {e}")
            
            # Try Two-Step Spectral Attack
            try:
                x_tssa = attack.two_step_spectral_attack(x, epsilon=0.15)
                tssa_output = ballast_model(x_tssa)
                tssa_pred = torch.argmax(tssa_output)
                
                if tssa_pred != target:
                    results['attack_stats']['tssa_success'] += 1
                    results['successful_attacks'].append({
                        'method': 'TSSA',
                        'original': x.cpu(),
                        'adversarial': x_tssa.cpu(),
                        'original_pred': target.cpu(),
                        'adv_pred': tssa_pred.cpu()
                    })
            except Exception as e:
                logger.warning(f"TSSA failed for sample {samples_processed}: {e}")
            
            # Try Targeted Dead Zone Attack
            try:
                x_targeted = attack.targeted_dead_zone_attack(x, ballast_model, epsilon=0.2)
                if x_targeted is not None:
                    results['attack_stats']['targeted_success'] += 1
                    results['dead_zone_inputs'].append({
                        'original': x.cpu(),
                        'dead_zone': x_targeted.cpu(),
                        'target': target.cpu()
                    })
            except Exception as e:
                logger.warning(f"Targeted attack failed for sample {samples_processed}: {e}")
            
            samples_processed += 1
            
            if samples_processed % 10 == 0:
                logger.info(f"Processed {samples_processed}/{num_samples} samples")
    
    # Calculate success rates
    total = results['attack_stats']['total_attempts']
    if total > 0:
        results['attack_stats']['ossa_rate'] = results['attack_stats']['ossa_success'] / total
        results['attack_stats']['tssa_rate'] = results['attack_stats']['tssa_success'] / total
        results['attack_stats']['targeted_rate'] = results['attack_stats']['targeted_success'] / total
    
    logger.info(f"Attack generation complete. Found {len(results['dead_zone_inputs'])} dead zone inputs")
    return results 