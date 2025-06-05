#!/usr/bin/env python3
"""
Base Visualizer Module

Common utilities and base class for all visualization modules.
Provides shared plotting configuration, styling, and utility functions.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List
from abc import ABC, abstractmethod


class BaseVisualizer(ABC):
    """
    Base class for all visualizers with common plotting utilities
    """
    
    def __init__(self, plots_dir: Path, style: str = 'seaborn-v0_8'):
        """
        Initialize base visualizer
        
        Args:
            plots_dir: Directory to save plots
            style: Matplotlib style to use
        """
        self.plots_dir = Path(plots_dir)
        self.plots_dir.mkdir(exist_ok=True)
        
        # Set plotting style
        self.setup_plotting_style(style)
        
    def setup_plotting_style(self, style: str = 'seaborn-v0_8'):
        """Setup consistent plotting style across all visualizations"""
        plt.style.use(style)
        sns.set_palette("husl")
        
        # Common plot parameters with better spacing for text
        plt.rcParams.update({
            'figure.figsize': (10, 6),
            'font.size': 11,
            'axes.titlesize': 13,
            'axes.labelsize': 11,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 15,
            'axes.titlepad': 20,  # Add padding below titles
            'axes.labelpad': 10,  # Add padding for axis labels
            'figure.autolayout': True  # Enable automatic layout adjustment
        })
    
    def save_plot(self, filename: str, dpi: int = 300, bbox_inches: str = 'tight'):
        """
        Save current plot to file
        
        Args:
            filename: Name of the file (without path)
            dpi: Resolution for saved plot
            bbox_inches: Bounding box configuration
        """
        filepath = self.plots_dir / filename
        plt.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches)
        plt.close()
    
    def create_subplot_grid(self, rows: int, cols: int, figsize: Optional[Tuple[int, int]] = None,
                           suptitle: Optional[str] = None) -> Tuple[plt.Figure, np.ndarray]:
        """
        Create a subplot grid with consistent styling
        
        Args:
            rows: Number of rows
            cols: Number of columns
            figsize: Figure size tuple
            suptitle: Super title for the figure
            
        Returns:
            Figure and axes array
        """
        if figsize is None:
            figsize = (5 * cols, 4 * rows)
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        
        if suptitle:
            fig.suptitle(suptitle, fontsize=16)
        
        # Ensure axes is always a numpy array for consistent indexing
        if rows == 1 and cols == 1:
            axes = np.array([axes])
        elif rows == 1 or cols == 1:
            axes = np.array(axes).reshape(rows, cols)
        
        return fig, axes
    
    def add_value_labels_to_bars(self, ax: plt.Axes, bars, format_str: str = '{:.4f}',
                                offset: float = 0.01, ha: str = 'center', va: str = 'bottom'):
        """
        Add value labels to bar chart
        
        Args:
            ax: Matplotlib axes
            bars: Bar container from bar plot
            format_str: Format string for values
            offset: Vertical offset for labels
            ha: Horizontal alignment
            va: Vertical alignment
        """
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + offset,
                   format_str.format(height), ha=ha, va=va)
    
    def setup_grid(self, ax: plt.Axes, alpha: float = 0.3):
        """Add grid to plot with consistent styling"""
        ax.grid(True, alpha=alpha)
    
    def get_color_palette(self, n_colors: int) -> List[str]:
        """Get a consistent color palette"""
        return sns.color_palette("husl", n_colors)
    
    @abstractmethod
    def generate_all_plots(self, *args, **kwargs):
        """Generate all plots for this visualizer (to be implemented by subclasses)"""
        pass 