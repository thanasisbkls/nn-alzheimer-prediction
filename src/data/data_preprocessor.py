"""
Data preprocessor for feature engineering and scaling

This module contains the data preprocessor for the Alzheimer's disease dataset.
"""

import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from typing import Tuple, List


class DataPreprocessor:
    """Handles data preprocessing for machine learning models"""
    
    def __init__(self, batch_size: int = 32):
        self.batch_size = batch_size
        self._encoders = {}
        self._scalers = {}
    
    def preprocess_fold_data(self, 
                           X_train: pd.DataFrame, 
                           X_val: pd.DataFrame,
                           y_train: pd.Series, 
                           y_val: pd.Series,
                           categorical_features: List[str], 
                           numerical_features: List[str]) -> Tuple[DataLoader, DataLoader, int]:
        """
        Preprocess data for a single fold
        
        Args:
            X_train: Training features
            X_val: Validation features
            y_train: Training targets
            y_val: Validation targets
            categorical_features: List of categorical feature names
            numerical_features: List of numerical feature names
            
        Returns:
            Tuple of (train_loader, val_loader, input_size)
        """
        X_train_processed = X_train.copy()
        X_val_processed = X_val.copy()
        
        # Process categorical features
        X_train_processed, X_val_processed = self._encode_categorical_features(
            X_train_processed, X_val_processed, categorical_features
        )
        
        # Process numerical features
        if numerical_features:
            X_train_processed, X_val_processed = self._scale_numerical_features(
                X_train_processed, X_val_processed, numerical_features
            )
        
        # Convert to PyTorch tensors and create data loaders
        train_loader, val_loader = self._create_data_loaders(
            X_train_processed, X_val_processed, y_train, y_val
        )
        
        input_size = X_train_processed.shape[1]
        return train_loader, val_loader, input_size
    
    def _encode_categorical_features(self, 
                                   X_train: pd.DataFrame, 
                                   X_val: pd.DataFrame,
                                   categorical_features: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Encode categorical features using one-hot encoding"""
        X_train_processed = X_train.copy()
        X_val_processed = X_val.copy()
        
        for feature in categorical_features:
            if feature not in X_train.columns:
                continue
                
            # Create or get encoder
            if feature not in self._encoders:
                encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
                encoder.fit(X_train[[feature]])
                self._encoders[feature] = encoder
            else:
                encoder = self._encoders[feature]
            
            # Transform data
            train_encoded = encoder.transform(X_train[[feature]])
            val_encoded = encoder.transform(X_val[[feature]])
            
            # Create feature names
            categories = list(encoder.categories_[0])
            if len(categories) > 1:
                feature_names = [f"{feature}_{cat}" for cat in categories[1:]]
                
                # Replace original feature with encoded features
                train_encoded_df = pd.DataFrame(train_encoded, columns=feature_names, index=X_train.index)
                val_encoded_df = pd.DataFrame(val_encoded, columns=feature_names, index=X_val.index)
                
                X_train_processed = X_train_processed.drop(columns=[feature])
                X_val_processed = X_val_processed.drop(columns=[feature])
                X_train_processed = pd.concat([X_train_processed, train_encoded_df], axis=1)
                X_val_processed = pd.concat([X_val_processed, val_encoded_df], axis=1)
        
        return X_train_processed, X_val_processed
    
    def _scale_numerical_features(self, 
                                X_train: pd.DataFrame, 
                                X_val: pd.DataFrame,
                                numerical_features: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Scale numerical features using standardization and min-max scaling"""
        X_train_processed = X_train.copy()
        X_val_processed = X_val.copy()
        
        # Filter features that actually exist
        existing_numerical = [f for f in numerical_features if f in X_train.columns]
        
        if existing_numerical:
            # Create scalers
            centering_scaler = StandardScaler(with_std=False)  # Only center, do not standardize
            minmax_scaler = MinMaxScaler()
            
            centering_scaler.fit(X_train[existing_numerical])
            minmax_scaler.fit(X_train[existing_numerical])
            
            # Transform training data: first center, then min-max scale
            X_train_processed[existing_numerical] = centering_scaler.transform(
                X_train[existing_numerical]
            )
            X_train_processed[existing_numerical] = minmax_scaler.transform(
                X_train_processed[existing_numerical]
            )
            
            # Transform validation data: first center, then min-max scale
            X_val_processed[existing_numerical] = centering_scaler.transform(
                X_val[existing_numerical]
            )
            X_val_processed[existing_numerical] = minmax_scaler.transform(
                X_val_processed[existing_numerical]
            )
            
            # Store scalers for potential reuse
            self._scalers['centering'] = centering_scaler
            self._scalers['minmax'] = minmax_scaler
        
        return X_train_processed, X_val_processed
    
    def _create_data_loaders(self, 
                           X_train: pd.DataFrame, 
                           X_val: pd.DataFrame,
                           y_train: pd.Series, 
                           y_val: pd.Series) -> Tuple[DataLoader, DataLoader]:
        """Create PyTorch data loaders from preprocessed data"""
        # Convert to tensors
        X_train_tensor = torch.tensor(X_train.values, dtype=torch.float64)
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.float64).view(-1, 1)
        X_val_tensor = torch.tensor(X_val.values, dtype=torch.float64)
        y_val_tensor = torch.tensor(y_val.values, dtype=torch.float64).view(-1, 1)
        
        # Create datasets
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        
        # Create loaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True)
        
        return train_loader, val_loader
    
    def reset_preprocessors(self) -> None:
        """Reset stored preprocessors (useful for new folds)"""
        self._encoders.clear()
        self._scalers.clear() 