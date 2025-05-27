"""
Data preprocessor for feature engineering and scaling

This module contains the data preprocessor for the Alzheimer's disease dataset.
"""

import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from typing import List, Optional


class DataPreprocessor:
    """Handles data preprocessing for machine learning models using fixed weights approach"""
    
    def __init__(self, batch_size: int = 32):
        self.batch_size = batch_size
        self._encoders = {}
        self._scalers = {}
        self._fitted = False  # Track if preprocessors have been fitted
    
    def fit_preprocessors(self, 
                         X_train: pd.DataFrame,
                         categorical_features: List[str], 
                         numerical_features: List[str]) -> None:
        """
        Fit preprocessors on training data
        
        Args:
            X_train: Training features to fit preprocessors on
            categorical_features: List of categorical feature names
            numerical_features: List of numerical feature names
        """
        # Reset any existing preprocessors
        self._encoders.clear()
        self._scalers.clear()
        
        # Fit categorical encoders
        for feature in categorical_features:
            encoder = OneHotEncoder(sparse_output=False, drop='first')
            encoder.fit(X_train[[feature]])
            self._encoders[feature] = encoder    
        
        # Fit numerical scalers
        centering_scaler = StandardScaler(with_std=False)  # Only center, do not standardize
        centering_scaler.fit(X_train[numerical_features])
        self._scalers['centering'] = centering_scaler

        # Apply centering and create DataFrame to maintain feature names for MinMaxScaler
        centered_data = centering_scaler.transform(X_train[numerical_features])
        centered_df = pd.DataFrame(centered_data, columns=numerical_features, index=X_train.index)
        
        minmax_scaler = MinMaxScaler()
        minmax_scaler.fit(centered_df)  # Fit on DataFrame to preserve feature names
        self._scalers['minmax'] = minmax_scaler    
            
        self._fitted = True
    
    def transform_data(self, 
                      X_data: pd.DataFrame,
                      categorical_features: List[str], 
                      numerical_features: List[str],
                      selected_features: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Transform data using fitted preprocessors with optional feature selection
        
        Args:
            X_data: Data to transform
            categorical_features: List of all categorical feature names
            numerical_features: List of all numerical feature names
            selected_features: List of original features to process (for GA individuals)
            
        Returns:
            Transformed DataFrame with only selected and processed features
        """
        if not self._fitted:
            raise ValueError("Preprocessors must be fitted before transforming data. Call fit_preprocessors() first.")
        
        # Determine which features to actually process
        if selected_features is not None:
            # Filter to only selected features for GA evaluation
            active_categorical = [f for f in categorical_features if f in selected_features]
            active_numerical = [f for f in numerical_features if f in selected_features]
            
            # Extract only selected features from input data
            selected_columns = [col for col in selected_features if col in X_data.columns]
            X_subset = X_data[selected_columns].copy()
        else:
            # Process all features for reference model training
            active_categorical = categorical_features
            active_numerical = numerical_features
            X_subset = X_data.copy()
        
        processed_features = []
        
        # Apply categorical encoding to selected categorical features
        if active_categorical: 
            for feature in active_categorical:
                if feature in self._encoders and feature in X_subset.columns:
                    encoder = self._encoders[feature]
                    
                    # Transform data
                    encoded = encoder.transform(X_subset[[feature]])
                    
                    # Create feature names (accounting for drop='first')
                    categories = list(encoder.categories_[0])
                    feature_names = [f"{feature}_{cat}" for cat in categories[1:]]  # Skip first category
                        
                    # Create encoded DataFrame
                    encoded_df = pd.DataFrame(encoded, columns=feature_names, index=X_data.index)
                    processed_features.append(encoded_df)

        
        # Apply numerical scaling to selected numerical features
        if active_numerical:
            # Extract selected numerical features
            numerical_data = X_subset[active_numerical]
            
            if len(active_numerical) > 0 and 'centering' in self._scalers and 'minmax' in self._scalers:
                # Get the features that the scalers were fitted on
                fitted_features = list(self._scalers['centering'].feature_names_in_)
                
                # Create temporary data with all fitted features for proper scaling
                # Ensure proper feature names are maintained to avoid sklearn warnings
                temp_data = pd.DataFrame(index=X_data.index, columns=fitted_features, dtype=float)
                
                # Fill with selected features' data
                for feature in active_numerical:
                    if feature in fitted_features:
                        temp_data[feature] = numerical_data[feature]

                # Scalers expect all features to be present in the data! 
                # Fill unselected features with their training mean (centering target)
                centering_means = self._scalers['centering'].mean_
                for i, feature in enumerate(fitted_features):
                    if feature not in active_numerical:
                        temp_data[feature] = centering_means[i]
                
                # Apply centering transformation - maintain feature names
                temp_centered = self._scalers['centering'].transform(temp_data)
                temp_centered_df = pd.DataFrame(temp_centered, columns=fitted_features, index=X_data.index)
                
                # Apply min-max scaling - maintain feature names
                temp_scaled = self._scalers['minmax'].transform(temp_centered_df)
                temp_scaled_df = pd.DataFrame(temp_scaled, columns=fitted_features, index=X_data.index)
                
                # Extract only the selected numerical features
                selected_numerical_processed = temp_scaled_df[active_numerical]
                processed_features.append(selected_numerical_processed)
        
       
        X_processed = pd.concat(processed_features, axis=1)
        
        return X_processed
    
    def create_data_loader(self, 
                          X_data: pd.DataFrame, 
                          y_data: pd.Series,
                          shuffle: bool = True) -> DataLoader:
        """
        Create a single data loader from preprocessed data
        
        Args:
            X_data: Preprocessed feature data
            y_data: Target data
            shuffle: Whether to shuffle the data
            
        Returns:
            DataLoader
        """
        # Convert to tensors
        X_tensor = torch.tensor(X_data.values, dtype=torch.float64)
        y_tensor = torch.tensor(y_data.values, dtype=torch.float64).view(-1, 1)
        
        # Create dataset and loader
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)
        
        return loader