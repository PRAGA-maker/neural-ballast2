import torch
import torch.nn as nn
from typing import Optional

from model import MNISTNet
from diagnostic import is_in_dead_zone
from correction import apply_corrective_nudge


class NeuralStethoscope:
    """
    A wrapper class that encapsulates neural network dead zone detection and correction.
    
    This class wraps around a trained neural network model and provides intelligent
    prediction capabilities that can detect when an input is in a "dead zone" 
    (where the model's behavior becomes degenerate) and apply corrective nudging
    to obtain more reliable predictions.
    """
    
    def __init__(self, 
                 model: nn.Module, 
                 dim_threshold: int = 1,
                 noise_sigma: float = 0.01,
                 max_nudge_attempts: int = 10,
                 svd_threshold: float = 1e-6,
                 verbose: bool = True):
        """
        Initialize the NeuralStethoscope wrapper.
        
        Args:
            model: The trained neural network model to wrap
            dim_threshold: Threshold for considering input in dead zone (default: 1)
            noise_sigma: Standard deviation for Gaussian noise in corrective nudging (default: 0.01)
            max_nudge_attempts: Maximum attempts to find healthy input during nudging (default: 10)
            svd_threshold: Threshold for singular value decomposition in dead zone detection (default: 1e-6)
            verbose: Whether to print diagnostic messages (default: True)
        """
        self.model = model
        self.dim_threshold = dim_threshold
        self.noise_sigma = noise_sigma
        self.max_nudge_attempts = max_nudge_attempts
        self.svd_threshold = svd_threshold
        self.verbose = verbose
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Statistics tracking
        self.total_predictions = 0
        self.dead_zone_detections = 0
        self.successful_corrections = 0
        
        if self.verbose:
            print("🔬 NeuralStethoscope initialized with:")
            print(f"   - Dead zone threshold: {dim_threshold}")
            print(f"   - Noise sigma: {noise_sigma}")
            print(f"   - Max nudge attempts: {max_nudge_attempts}")
            print(f"   - SVD threshold: {svd_threshold}")
    
    def predict(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Make a prediction with intelligent dead zone detection and correction.
        
        This is the main public-facing method. It:
        1. Checks if the input is in a dead zone
        2. If in a dead zone, applies corrective nudging
        3. Returns the model's prediction on the (possibly corrected) input
        
        Args:
            input_tensor: Input tensor to make prediction on
            
        Returns:
            torch.Tensor: Model output (logits)
        """
        self.total_predictions += 1
        
        # Ensure input tensor has the right shape and doesn't require gradients for prediction
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension
        
        # Check if input is in dead zone
        if self._is_in_dead_zone_with_threshold(input_tensor):
            self.dead_zone_detections += 1
            
            if self.verbose:
                print("⚠️  Dead zone detected! Applying corrective nudge...")
            
            # Apply corrective nudging
            nudged_input = self._apply_corrective_nudge(input_tensor)
            
            # Check if nudging was successful
            if not self._is_in_dead_zone_with_threshold(nudged_input):
                self.successful_corrections += 1
                if self.verbose:
                    print("✅ Corrective nudge successful!")
                final_input = nudged_input
            else:
                if self.verbose:
                    print("⚠️  Corrective nudge failed, using nudged input anyway...")
                final_input = nudged_input
        else:
            # Input is healthy, use it directly
            final_input = input_tensor
        
        # Make prediction with the final input
        with torch.no_grad():
            output = self.model(final_input)
        
        return output
    
    def _is_in_dead_zone_with_threshold(self, input_tensor: torch.Tensor) -> bool:
        """
        Check if input is in dead zone using the configured threshold.
        
        Args:
            input_tensor: Input tensor to check
            
        Returns:
            bool: True if input is in dead zone
        """
        return is_in_dead_zone(self.model, input_tensor, dim_threshold=self.dim_threshold)
    
    def _apply_corrective_nudge(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply corrective nudging to move input out of dead zone.
        
        Args:
            input_tensor: Input tensor in dead zone
            
        Returns:
            torch.Tensor: Nudged input tensor
        """
        return apply_corrective_nudge(
            model=self.model,
            bad_input_tensor=input_tensor,
            is_in_dead_zone_func=self._is_in_dead_zone_with_threshold,
            max_attempts=self.max_nudge_attempts,
            sigma=self.noise_sigma
        )
    
    def get_statistics(self) -> dict:
        """
        Get statistics about the stethoscope's performance.
        
        Returns:
            dict: Dictionary containing performance statistics
        """
        if self.total_predictions == 0:
            return {
                'total_predictions': 0,
                'dead_zone_detections': 0,
                'successful_corrections': 0,
                'dead_zone_rate': 0.0,
                'correction_success_rate': 0.0
            }
        
        dead_zone_rate = self.dead_zone_detections / self.total_predictions
        correction_success_rate = (self.successful_corrections / self.dead_zone_detections 
                                 if self.dead_zone_detections > 0 else 0.0)
        
        return {
            'total_predictions': self.total_predictions,
            'dead_zone_detections': self.dead_zone_detections,
            'successful_corrections': self.successful_corrections,
            'dead_zone_rate': dead_zone_rate,
            'correction_success_rate': correction_success_rate
        }
    
    def print_statistics(self):
        """Print a summary of the stethoscope's performance."""
        stats = self.get_statistics()
        
        print("\n📊 NeuralStethoscope Performance Summary:")
        print("=" * 50)
        print(f"Total predictions made: {stats['total_predictions']}")
        print(f"Dead zone detections: {stats['dead_zone_detections']}")
        print(f"Successful corrections: {stats['successful_corrections']}")
        print(f"Dead zone rate: {stats['dead_zone_rate']:.2%}")
        print(f"Correction success rate: {stats['correction_success_rate']:.2%}")
        print("=" * 50)
    
    def reset_statistics(self):
        """Reset performance statistics."""
        self.total_predictions = 0
        self.dead_zone_detections = 0
        self.successful_corrections = 0
        
        if self.verbose:
            print("📊 Statistics reset!")


if __name__ == "__main__":
    # Simple test of the NeuralStethoscope wrapper
    print("🔬 Testing NeuralStethoscope Wrapper")
    print("=" * 50)
    
    # Import additional modules for testing
    import os
    import torchvision
    import torchvision.transforms as transforms
    from model import create_model
    
    # Load trained model
    model_path = '../mnist_net.pth'
    if not os.path.exists(model_path):
        print(f"❌ Model file {model_path} not found!")
        print("Please run train.py first to create a trained model.")
        exit(1)
    
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location='cpu')
    model = create_model()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"✅ Model loaded (Test accuracy: {checkpoint['test_accuracy']:.2f}%)")
    
    # Create NeuralStethoscope wrapper
    stethoscope = NeuralStethoscope(
        model=model,
        dim_threshold=1,
        noise_sigma=0.01,
        max_nudge_attempts=10,
        verbose=True
    )
    
    # Load test data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = torchvision.datasets.MNIST(
        root='../data', 
        train=False, 
        download=True, 
        transform=transform
    )
    
    # Test on a few samples
    print("\n🧪 Testing on sample inputs...")
    for i in range(5):
        sample_image, sample_label = test_dataset[i]
        
        # Get predictions
        baseline_output = model(sample_image.unsqueeze(0))
        stethoscope_output = stethoscope.predict(sample_image)
        
        baseline_pred = baseline_output.argmax(dim=1).item()
        stethoscope_pred = stethoscope_output.argmax(dim=1).item()
        
        print(f"\nSample {i+1} (True label: {sample_label}):")
        print(f"  Baseline prediction: {baseline_pred}")
        print(f"  Stethoscope prediction: {stethoscope_pred}")
        print(f"  Match: {'✅' if baseline_pred == stethoscope_pred else '❌'}")
    
    # Print final statistics
    stethoscope.print_statistics() 