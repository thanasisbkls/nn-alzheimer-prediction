"""
Data loader for Alzheimer's disease dataset

This module contains the data loader for the Alzheimer's disease dataset.
"""

import pandas as pd
from typing import Tuple, List


class AlzheimerDataLoader:
    """Loads and performs basic cleaning of Alzheimer's disease dataset"""
    
    def __init__(self):
        self._numerical_features = [
            'Age', 'BMI', 'AlcoholConsumption', 'PhysicalActivity', 'DietQuality',
            'SleepQuality', 'SystolicBP', 'DiastolicBP', 'CholesterolTotal',
            'CholesterolLDL', 'CholesterolHDL', 'CholesterolTriglycerides',
            'MMSE', 'FunctionalAssessment', 'ADL'
        ]
        
        self._categorical_features = [
            'Gender', 'Ethnicity', 'EducationLevel', 'Smoking', 'FamilyHistoryAlzheimers',
            'CardiovascularDisease', 'Diabetes', 'Depression', 'HeadInjury', 'Hypertension',
            'MemoryComplaints', 'BehavioralProblems', 'Confusion', 'Disorientation',
            'PersonalityChanges', 'DifficultyCompletingTasks', 'Forgetfulness'
        ]
    
    def load_data(self, filepath: str) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
        """
        Load and perform basic cleaning of the dataset
        
        Args:
            filepath: Path to the CSV file
            
        Returns:
            Tuple of (features_df, target_series, categorical_features, numerical_features)
        """
        # Load data
        data = pd.read_csv(filepath)
        
        # Remove ID columns
        data = data.drop(columns=['PatientID', 'DoctorInCharge'], errors='ignore')
        
        # Separate features and target
        if 'Diagnosis' not in data.columns:
            raise ValueError("Target column 'Diagnosis' not found in dataset")
        
        X = data.drop(columns=['Diagnosis'])
        y = data['Diagnosis']
        
        # Get the numerical and categorical features separately
        numerical_features = [f for f in self._numerical_features if f in X.columns]
        categorical_features = [f for f in self._categorical_features if f in X.columns]
        
        return X, y, categorical_features, numerical_features
    
    def get_feature_info(self) -> dict:
        """Get information about expected feature types"""
        return {
            'numerical_features': self._numerical_features.copy(),
            'categorical_features': self._categorical_features.copy(),
            'total_expected_features': len(self._numerical_features) + len(self._categorical_features)
        } 