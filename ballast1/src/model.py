import torch
import torch.nn as nn
import torch.nn.functional as F


class MNISTNet(nn.Module):
    """
    A simple CNN for MNIST classification.
    
    Architecture:
    - Conv2d(1, 32, 3) -> ReLU -> MaxPool2d(2)
    - Conv2d(32, 64, 3) -> ReLU -> MaxPool2d(2)
    - Flatten -> Linear(64*5*5, 128) -> ReLU -> Dropout(0.5)
    - Linear(128, 10)
    """
    
    def __init__(self):
        super(MNISTNet, self).__init__()
        # First convolutional layer: 1 input channel, 32 output channels, 3x3 kernel
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        # Second convolutional layer: 32 input channels, 64 output channels, 3x3 kernel
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        
        # Max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        # After 2 conv layers with padding=1 and 2 maxpool layers: 28x28 -> 14x14 -> 7x7
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)  # 10 classes for MNIST
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # First conv block: Conv -> ReLU -> MaxPool
        x = self.pool(F.relu(self.conv1(x)))
        
        # Second conv block: Conv -> ReLU -> MaxPool
        x = self.pool(F.relu(self.conv2(x)))
        
        # Flatten for fully connected layers
        x = x.view(-1, 64 * 7 * 7)
        
        # First fully connected layer with ReLU and dropout
        x = self.dropout(F.relu(self.fc1(x)))
        
        # Output layer (no activation, will be handled by loss function)
        x = self.fc2(x)
        
        return x

    def get_num_parameters(self):
        """Return the total number of parameters in the model."""
        return sum(p.numel() for p in self.parameters())


def create_model():
    """Factory function to create and return a new MNIST model instance."""
    model = MNISTNet()
    print(f"Created MNISTNet with {model.get_num_parameters():,} parameters")
    return model


if __name__ == "__main__":
    # Test the model creation and forward pass
    model = create_model()
    
    # Test with a random input tensor (batch_size=1, channels=1, height=28, width=28)
    test_input = torch.randn(1, 1, 28, 28)
    output = model(test_input)
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output: {output}") 