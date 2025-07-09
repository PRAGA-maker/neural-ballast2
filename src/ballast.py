import torch
import torch.nn as nn
from typing import Union, Optional

from .model import create_model, MNISTNet
from .diagnostic import is_in_dead_zone, get_singfol_dim
from .correction import apply_corrective_nudge


class NeuralBallast:
    """
    Neural Ballast wrapper class that detects and corrects dead zones in neural networks.
    
    This wrapper encapsulates a neural network model and provides dead zone detection
    and correction functionality. When an input is detected to be in a dead zone,
    the wrapper applies corrective nudging to move the input out of the dead zone
    before passing it to the underlying model.
    """
    
    def __init__(self, 
                 model: nn.Module,
                 dim_threshold: int = 1,
                 noise_sigma: float = 0.01,
                 max_attempts: int = 10,
                 verbose: bool = True):
        """
        Initialize the NeuralBallast wrapper.
        
        Args:
            model: The neural network model to wrap
            dim_threshold: Threshold for determining if input is in dead zone
            noise_sigma: Standard deviation for Gaussian noise used in correction
            max_attempts: Maximum attempts to find a healthy input during correction
            verbose: Whether to print diagnostic messages
        """
        self.model = model
        self.dim_threshold = dim_threshold
        self.noise_sigma = noise_sigma
        self.max_attempts = max_attempts
        self.verbose = verbose
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Statistics tracking
        self.total_predictions = 0
        self.dead_zone_detections = 0
        self.successful_corrections = 0
        
    def predict(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Main prediction method that applies ballast correction when needed.
        
        Args:
            input_tensor: Input tensor to make predictions on
            
        Returns:
            torch.Tensor: Model output (logits)
        """
        self.total_predictions += 1
        
        # Check if input is in dead zone
        if is_in_dead_zone(self.model, input_tensor, self.dim_threshold):
            self.dead_zone_detections += 1
            
            if self.verbose:
                # Get detailed information about the dead zone
                singfol_dim = get_singfol_dim(self.model, input_tensor)
                print(f"Dead zone detected! SingFolDIM: {singfol_dim}")
                print("Applying neural ballast correction...")
            
            # Apply corrective nudge
            try:
                nudged_input = apply_corrective_nudge(
                    model=self.model,
                    bad_input_tensor=input_tensor,
                    is_in_dead_zone_func=lambda m, x: is_in_dead_zone(m, x, self.dim_threshold),
                    max_attempts=self.max_attempts,
                    sigma=self.noise_sigma
                )
                
                # Verify correction was successful
                if not is_in_dead_zone(self.model, nudged_input, self.dim_threshold):
                    self.successful_corrections += 1
                    if self.verbose:
                        print("✓ Ballast correction successful!")
                    
                    # Use the corrected input
                    with torch.no_grad():
                        return self.model(nudged_input)
                else:
                    if self.verbose:
                        print("⚠ Ballast correction failed, using original input")
                    
            except Exception as e:
                if self.verbose:
                    print(f"⚠ Error during ballast correction: {e}")
                    print("Using original input")
        
        # Use original input (either no dead zone detected or correction failed)
        with torch.no_grad():
            return self.model(input_tensor)
    
    def __call__(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Allow the wrapper to be called like a function.
        
        Args:
            input_tensor: Input tensor to make predictions on
            
        Returns:
            torch.Tensor: Model output (logits)
        """
        return self.predict(input_tensor)
    
    def get_statistics(self) -> dict:
        """
        Get statistics about ballast operations.
        
        Returns:
            dict: Statistics including total predictions, dead zone detections, etc.
        """
        return {
            'total_predictions': self.total_predictions,
            'dead_zone_detections': self.dead_zone_detections,
            'successful_corrections': self.successful_corrections,
            'dead_zone_rate': self.dead_zone_detections / max(self.total_predictions, 1),
            'correction_success_rate': self.successful_corrections / max(self.dead_zone_detections, 1)
        }
    
    def reset_statistics(self):
        """Reset all statistics counters."""
        self.total_predictions = 0
        self.dead_zone_detections = 0
        self.successful_corrections = 0
    
    def print_statistics(self):
        """Print formatted statistics about ballast operations."""
        stats = self.get_statistics()
        print("\n" + "=" * 50)
        print("NEURAL BALLAST STATISTICS")
        print("=" * 50)
        print(f"Total predictions: {stats['total_predictions']}")
        print(f"Dead zone detections: {stats['dead_zone_detections']}")
        print(f"Successful corrections: {stats['successful_corrections']}")
        print(f"Dead zone rate: {stats['dead_zone_rate']:.2%}")
        print(f"Correction success rate: {stats['correction_success_rate']:.2%}")
        print("=" * 50)
    
    def configure(self, **kwargs):
        """
        Update configuration parameters.
        
        Args:
            **kwargs: Configuration parameters to update
        """
        if 'dim_threshold' in kwargs:
            self.dim_threshold = kwargs['dim_threshold']
        if 'noise_sigma' in kwargs:
            self.noise_sigma = kwargs['noise_sigma']
        if 'max_attempts' in kwargs:
            self.max_attempts = kwargs['max_attempts']
        if 'verbose' in kwargs:
            self.verbose = kwargs['verbose']


def create_ballast_wrapper(model: Optional[nn.Module] = None, **kwargs) -> NeuralBallast:
    """
    Factory function to create a NeuralBallast wrapper.
    
    Args:
        model: Pre-trained model to wrap (if None, creates a new model)
        **kwargs: Additional configuration parameters
        
    Returns:
        NeuralBallast: Configured ballast wrapper
    """
    if model is None:
        model = create_model()
    
    return NeuralBallast(model, **kwargs)


if __name__ == '__main__':
    # Simple test of the NeuralBallast wrapper
    print("Testing NeuralBallast Wrapper")
    print("=" * 50)
    
    # Create a simple test model
    model = create_model()
    
    # Create ballast wrapper
    ballast = NeuralBallast(model, verbose=True)
    
    # Test with a random input
    test_input = torch.randn(1, 1, 28, 28)
    
    print(f"Input shape: {test_input.shape}")
    output = ballast.predict(test_input)
    print(f"Output shape: {output.shape}")
    print(f"Predicted class: {output.argmax(dim=1).item()}")
    
    # Print statistics
    ballast.print_statistics() 