import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import os
from datetime import datetime

from model import create_model


def get_data_loaders(batch_size=64):
    """
    Load and prepare MNIST dataset with train/test splits.
    
    Args:
        batch_size (int): Batch size for data loaders
        
    Returns:
        tuple: (train_loader, test_loader)
    """
    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])
    
    # Download and load training data
    train_dataset = torchvision.datasets.MNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    
    # Download and load test data
    test_dataset = torchvision.datasets.MNIST(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    return train_loader, test_loader


def train_model(model, train_loader, criterion, optimizer, device):
    """
    Train the model for one epoch.
    
    Args:
        model: The neural network model
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to run training on
        
    Returns:
        float: Average training loss for the epoch
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # Print progress every 100 batches
        if batch_idx % 100 == 0:
            print(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}, '
                  f'Acc: {100.*correct/total:.2f}%')
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def evaluate_model(model, test_loader, criterion, device):
    """
    Evaluate the model on test data.
    
    Args:
        model: The neural network model
        test_loader: DataLoader for test data
        criterion: Loss function
        device: Device to run evaluation on
        
    Returns:
        tuple: (test_loss, test_accuracy)
    """
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    test_loss = test_loss / len(test_loader)
    test_acc = 100. * correct / total
    
    return test_loss, test_acc


def main():
    """Main training function."""
    print("Starting MNIST CNN Training")
    print("=" * 50)
    
    # Training hyperparameters
    EPOCHS = 8
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("\nLoading MNIST dataset...")
    train_loader, test_loader = get_data_loaders(BATCH_SIZE)
    
    # Create model
    print("\nCreating model...")
    model = create_model()
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(f"\nTraining Configuration:")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"  Optimizer: Adam")
    print(f"  Loss Function: CrossEntropyLoss")
    
    # Training loop
    print("\nStarting training...")
    print("=" * 50)
    
    best_test_acc = 0.0
    training_history = {
        'train_loss': [], 'train_acc': [],
        'test_loss': [], 'test_acc': []
    }
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        print("-" * 30)
        
        # Train for one epoch
        train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, device)
        
        # Evaluate on test set
        test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)
        
        # Store history
        training_history['train_loss'].append(train_loss)
        training_history['train_acc'].append(train_acc)
        training_history['test_loss'].append(test_loss)
        training_history['test_acc'].append(test_acc)
        
        # Print epoch results
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
        
        # Save best model
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            print(f"New best test accuracy: {best_test_acc:.2f}%")
    
    # Save the trained model
    model_save_path = 'mnist_net.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'test_accuracy': best_test_acc,
        'training_history': training_history,
        'model_info': {
            'architecture': 'MNISTNet',
            'parameters': model.get_num_parameters(),
            'epochs': EPOCHS,
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE
        }
    }, model_save_path)
    
    print("\n" + "=" * 50)
    print("Training completed!")
    print(f"Best test accuracy: {best_test_acc:.2f}%")
    print(f"Model saved to: {model_save_path}")
    
    return model, training_history


if __name__ == "__main__":
    model, history = main() 