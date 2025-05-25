"""
Neural network models and training

This module contains the neural network model architecture and the trainer.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import List, Optional
from tqdm import tqdm
import gc


class Net(nn.Module):
    """
    Feedforward neural network for binary classification
    """
    
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int = 1, activation: str = 'relu', dropout_rate: float = 0.0):
        """
        Initialize feedforward network
        
        Args:
            input_size: Number of input features
            hidden_sizes: List of hidden layer sizes
            output_size: Number of output units (default: 1)
            activation: Activation function ('relu', 'tanh', 'silu')
            dropout_rate: Dropout rate for regularization
        """
        super(Net, self).__init__()
        
        if activation.lower() == 'relu':
            self.activation = nn.ReLU()
        elif activation.lower() == 'tanh':
            self.activation = nn.Tanh()
        elif activation.lower() == 'silu':
            self.activation = nn.SiLU()
        else:
            raise ValueError(f'Activation {activation} is not supported.')
        
        # Construct layers
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(self.activation)
            
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))
        layers.append(nn.Sigmoid())
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network"""
        return self.net(x)


class Trainer:
    """
    Neural network trainer with early stopping
    """
    
    def __init__(self, 
                 max_epochs: int = 100,
                 patience: int = 5,
                 learning_rate: float = 0.01,
                 momentum: float = 0.0,
                 weight_decay: float = 0.0,
                 device: Optional[torch.device] = None):
        """
        Initialize trainer
        
        Args:
            max_epochs: Maximum training epochs
            patience: Early stopping patience
            learning_rate: Learning rate for optimizer
            momentum: Momentum for SGD optimizer
            weight_decay: Weight decay for regularization
            device: Device to train on (CPU/GPU)
        """
        self.max_epochs = max_epochs
        self.patience = patience
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def train_model(self, 
                   model: nn.Module,
                   train_loader: DataLoader,
                   val_loader: DataLoader,
                   verbose: int = 0) -> tuple:
        """
        Train the model with early stopping
        
        Args:
            model: Neural network model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            verbose: Verbosity level (0=silent, 1=minimal, 2=detailed)
            
        Returns:
            Tuple of (trained_model, metrics_dict)
        """
        # Move model to device
        model.to(self.device)
        model.double()  # Use double precision for stability
        
        # Setup optimizer and loss function
        if self.weight_decay > 0:
            if self.momentum > 0:
                optimizer = optim.SGD(model.parameters(), lr=self.learning_rate, 
                                    momentum=self.momentum, weight_decay=self.weight_decay)
            else:
                optimizer = optim.SGD(model.parameters(), lr=self.learning_rate, 
                                    weight_decay=self.weight_decay)
        elif self.momentum > 0:
            optimizer = optim.SGD(model.parameters(), lr=self.learning_rate, momentum=self.momentum)
        else:
            optimizer = optim.SGD(model.parameters(), lr=self.learning_rate)
        
        criterion = nn.BCELoss()
        mse_loss = nn.MSELoss()
        
        # Training metrics
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        train_mse = []
        val_mse = []
        
        # Early stopping variables
        best_val_loss = float('inf')
        patience_counter = 0
        best_model = None
        
        try:
            epoch_range = tqdm(range(self.max_epochs)) if verbose == 2 else range(self.max_epochs)
            for epoch in epoch_range:
                # Training phase
                model.train()
                batch_loss = 0.0
                batch_mse = 0.0
                correct = 0
                total = 0
                
                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    
                    # Perform forward pass
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    # Perform backward pass
                    loss.backward()
                    optimizer.step()
                    
                    batch_loss += loss.item() * inputs.size(0)
                    batch_mse += mse_loss(outputs, labels).item() * inputs.size(0)
                    
                    predicted = (outputs > 0.5).float()
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                
                # Calculate training metrics
                epoch_train_loss = batch_loss / len(train_loader.dataset)
                epoch_train_acc = correct / total
                epoch_train_mse = batch_mse / len(train_loader.dataset)
                
                train_losses.append(epoch_train_loss)
                train_accuracies.append(epoch_train_acc)
                train_mse.append(epoch_train_mse)
                
                # Validation phase
                val_loss, val_acc, val_mse_metric = self._validate_model(model, val_loader, criterion, mse_loss)
                
                val_losses.append(val_loss)
                val_accuracies.append(val_acc)
                val_mse.append(val_mse_metric)
                
                # Verbose logging
                if verbose == 2:
                    self.logger.info(f'Epoch {epoch+1}/{self.max_epochs} - '
                                   f'Train Loss: {epoch_train_loss:.4f}, Train MSE: {epoch_train_mse:.4f}, Train Acc: {epoch_train_acc:.4f} - '
                                   f'Val Loss: {val_loss:.4f}, Val MSE: {val_mse_metric:.4f}, Val Acc: {val_acc:.4f}')
                elif verbose == 1 and ((epoch+1) % 5 == 0 or epoch == 0):
                    self.logger.info(f'Epoch {epoch+1}/{self.max_epochs} - '
                                   f'Train Loss: {epoch_train_loss:.4f}, Train MSE: {epoch_train_mse:.4f}, Train Acc: {epoch_train_acc:.4f} - '
                                   f'Val Loss: {val_loss:.4f}, Val MSE: {val_mse_metric:.4f}, Val Acc: {val_acc:.4f}')
                
                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model = model.state_dict().copy()
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        if verbose >= 1:
                            self.logger.info(f'Early stopping triggered at epoch {epoch+1}')
                        break
            
            # Load best model
            if best_model is not None:
                model.load_state_dict(best_model)
            
            return (model, {
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_accuracies': train_accuracies,
                'val_accuracies': val_accuracies,
                'train_mse': train_mse,
                'val_mse': val_mse,
                'best_val_loss': best_val_loss
            })
            
        except Exception as e:
            self.logger.error(f"Error during training: {e}")
            raise
        finally:
            # Clean up memory
            del optimizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def _validate_model(self, 
                       model: nn.Module,
                       val_loader: DataLoader,
                       criterion: nn.Module,
                       mse_loss: nn.Module) -> tuple:
        """Validate the model and return validation metrics"""
        model.eval()
        batch_loss = 0.0
        batch_mse = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)
                
                loss = criterion(outputs, labels)
                batch_loss += loss.item() * inputs.size(0)
                
                mse = mse_loss(outputs, labels)
                batch_mse += mse.item() * inputs.size(0)
                
                predicted = (outputs > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss = batch_loss / len(val_loader.dataset)
        val_acc = correct / total
        val_mse_metric = batch_mse / len(val_loader.dataset)
        
        return val_loss, val_acc, val_mse_metric
    
    def evaluate_model(self, 
                      model: nn.Module,
                      test_loader: DataLoader) -> dict:
        """
        Evaluate model performance on test data
        
        Args:
            model: Trained model
            test_loader: Test data loader
            
        Returns:
            Dictionary with evaluation metrics
        """
        model.eval()
        criterion = nn.BCELoss()
        mse_loss = nn.MSELoss()
        
        total_loss = 0.0
        total_mse = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                total_loss += loss.item() * batch_x.size(0)
                
                mse = mse_loss(outputs, batch_y)
                total_mse += mse.item() * batch_x.size(0)
                
                # Calculate accuracy
                predictions = (outputs > 0.5).float()
                correct_predictions += (predictions == batch_y).sum().item()
                total_predictions += batch_y.size(0)
        
        avg_loss = total_loss / len(test_loader.dataset)
        avg_mse = total_mse / len(test_loader.dataset)
        accuracy = correct_predictions / total_predictions
        
        return {
            'loss': avg_loss,
            'mse': avg_mse,
            'accuracy': accuracy,
            'total_samples': total_predictions
        } 