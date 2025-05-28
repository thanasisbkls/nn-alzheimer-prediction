#!/usr/bin/env python3
"""
Base Reporter Module

Common utilities and base class for all reporting modules.
Provides shared report formatting, file handling, and utility functions.
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types and other non-serializable objects"""
    def default(self, obj):
        import numpy as np
  
        # Handle numpy types
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        
        # Handle PyTorch tensors
        elif hasattr(obj, 'detach') and hasattr(obj, 'cpu'):  # PyTorch tensor
            return obj.detach().cpu().numpy().tolist()
        
        # Handle PyTorch models and other complex objects
        elif hasattr(obj, '__class__'):
            class_name = obj.__class__.__name__
            if 'torch' in str(type(obj)).lower() or 'model' in class_name.lower():
                return f"<{class_name} object - not serializable>"
            elif 'preprocessor' in class_name.lower():
                return f"<{class_name} object - not serializable>"
        
        # Last resort: try to convert to string
        try:
            return str(obj)
        except:
            return f"<{type(obj).__name__} object - not serializable>"


class BaseReporter(ABC):
    """
    Base class for all reporters with common reporting utilities
    """
    
    def __init__(self, results_dir: Path):
        """
        Initialize base reporter
        
        Args:
            results_dir: Directory to save reports and results
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
    def write_section_header(self, f, title: str, level: int = 1):
        """
        Write a formatted section header
        
        Args:
            f: File handle
            title: Section title
            level: Header level (1 for main, 2 for subsection)
        """
        if level == 1:
            f.write(f"\n{title}\n")
            f.write("=" * len(title) + "\n\n")
        elif level == 2:
            f.write(f"\n{title}\n")
            f.write("-" * len(title) + "\n")
        else:
            f.write(f"\n{title}:\n")
    
    def write_bullet_point(self, f, text: str, indent: int = 0):
        """Write a formatted bullet point"""
        f.write("  " * indent + f"- {text}\n")
    
    def write_numbered_point(self, f, number: int, text: str, indent: int = 0):
        """Write a formatted numbered point"""
        f.write("  " * indent + f"{number}. {text}\n")
    
    def format_percentage(self, value: float) -> str:
        """Format a value as percentage"""
        return f"{value:.1%}"
    
    def format_number(self, value: float, decimals: int = 4) -> str:
        """Format a number with specified decimals"""
        return f"{value:.{decimals}f}"
    
    def save_json_results(self, data: Dict[str, Any], filename: str, 
                         timestamp: bool = True) -> Path:
        """
        Save results as JSON file
        
        Args:
            data: Data to save
            filename: Base filename
            timestamp: Whether to add timestamp to filename
            
        Returns:
            Path to saved file
        """
        if timestamp:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{filename}_{ts}.json"
        
        filepath = self.results_dir / filename
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, cls=NumpyEncoder)
        
        return filepath
    
    def save_csv_summary(self, df: pd.DataFrame, filename: str, 
                        timestamp: bool = True) -> Path:
        """
        Save DataFrame as CSV file
        
        Args:
            df: DataFrame to save
            filename: Base filename
            timestamp: Whether to add timestamp to filename
            
        Returns:
            Path to saved file
        """
        if timestamp:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{filename}_{ts}.csv"
        
        filepath = self.results_dir / filename
        df.to_csv(filepath, index=False)
        
        return filepath
    
    def create_summary_table_string(self, headers: list, rows: list, 
                                   col_widths: Optional[list] = None) -> str:
        """
        Create a formatted table string
        
        Args:
            headers: List of column headers
            rows: List of row data (each row is a list)
            col_widths: Optional list of column widths
            
        Returns:
            Formatted table string
        """
        if col_widths is None:
            col_widths = [max(len(str(item)) for item in [header] + [row[i] for row in rows]) 
                         for i, header in enumerate(headers)]
        
        # Create header row
        header_row = "".join(f"{header:<{col_widths[i]}}" for i, header in enumerate(headers))
        separator = "-" * len(header_row)
        
        # Create data rows
        data_rows = []
        for row in rows:
            data_row = "".join(f"{str(row[i]):<{col_widths[i]}}" for i in range(len(row)))
            data_rows.append(data_row)
        
        return "\n".join([header_row, separator] + data_rows)
    
    @abstractmethod
    def generate_report(self, *args, **kwargs) -> Path:
        """Generate the main report (to be implemented by subclasses)"""
        pass 