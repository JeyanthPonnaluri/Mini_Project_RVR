"""
Experiment management and logging for reproducible research.
Handles configuration saving, result logging, and experiment tracking.
"""

import json
import os
import hashlib
from datetime import datetime
from typing import Dict, Any
import pandas as pd
import numpy as np


class ExperimentManager:
    """
    Manages experiment configuration, logging, and reproducibility.
    """
    
    def __init__(self, base_dir: str = 'reports/version5'):
        """
        Initialize experiment manager.
        
        Parameters:
        -----------
        base_dir : str
            Base directory for experiment outputs
        """
        self.base_dir = base_dir
        self.experiment_id = None
        self.experiment_dir = None
        self.config = {}
        self.results = {}
        
    def create_experiment(self, config: Dict[str, Any]) -> str:
        """
        Create new experiment with timestamped directory.
        
        Parameters:
        -----------
        config : Dict[str, Any]
            Experiment configuration
            
        Returns:
        --------
        str
            Experiment ID
        """
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Generate experiment hash from config
        config_str = json.dumps(config, sort_keys=True)
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        
        # Create experiment ID
        self.experiment_id = f"exp_{timestamp}_{config_hash}"
        
        # Create experiment directory
        self.experiment_dir = os.path.join(self.base_dir, self.experiment_id)
        os.makedirs(self.experiment_dir, exist_ok=True)
        os.makedirs(os.path.join(self.experiment_dir, 'plots'), exist_ok=True)
        
        # Save config
        self.config = config
        self.config['experiment_id'] = self.experiment_id
        self.config['timestamp'] = timestamp
        self.config['config_hash'] = config_hash
        
        config_path = os.path.join(self.experiment_dir, 'config.json')
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, default=str)
        
        print(f"\n{'='*70}")
        print(f"EXPERIMENT CREATED")
        print(f"{'='*70}")
        print(f"Experiment ID: {self.experiment_id}")
        print(f"Directory: {self.experiment_dir}")
        print(f"Config Hash: {config_hash}")
        print(f"{'='*70}\n")
        
        return self.experiment_id
    
    def log_results(self, results: Dict[str, Any]):
        """
        Log experiment results.
        
        Parameters:
        -----------
        results : Dict[str, Any]
            Results dictionary
        """
        self.results.update(results)
        
        # Save results as JSON
        results_path = os.path.join(self.experiment_dir, 'results.json')
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"Results logged to {results_path}")
    
    def save_dataframe(self, df: pd.DataFrame, name: str):
        """
        Save DataFrame to experiment directory.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame to save
        name : str
            File name (without extension)
        """
        csv_path = os.path.join(self.experiment_dir, f'{name}.csv')
        df.to_csv(csv_path, index=False)
        print(f"DataFrame saved to {csv_path}")
    
    def get_plot_path(self, plot_name: str) -> str:
        """
        Get path for saving plot.
        
        Parameters:
        -----------
        plot_name : str
            Plot file name (with extension)
            
        Returns:
        --------
        str
            Full path for plot
        """
        return os.path.join(self.experiment_dir, 'plots', plot_name)
    
    def generate_summary_report(self):
        """
        Generate text summary report of experiment.
        """
        report_path = os.path.join(self.experiment_dir, 'summary_report.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("EXPERIMENT SUMMARY REPORT\n")
            f.write("="*70 + "\n\n")
            
            # Configuration
            f.write("CONFIGURATION:\n")
            f.write("-"*70 + "\n")
            for key, value in self.config.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
            
            # Results
            f.write("RESULTS:\n")
            f.write("-"*70 + "\n")
            for key, value in self.results.items():
                if isinstance(value, (int, float, str, bool)):
                    f.write(f"{key}: {value}\n")
                elif isinstance(value, (list, np.ndarray)):
                    if len(value) <= 10:
                        f.write(f"{key}: {value}\n")
                    else:
                        f.write(f"{key}: [array of length {len(value)}]\n")
            f.write("\n")
            
            f.write("="*70 + "\n")
            f.write(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*70 + "\n")
        
        print(f"Summary report saved to {report_path}")
    
    @staticmethod
    def load_experiment(experiment_dir: str) -> Dict[str, Any]:
        """
        Load experiment configuration and results.
        
        Parameters:
        -----------
        experiment_dir : str
            Path to experiment directory
            
        Returns:
        --------
        Dict[str, Any]
            Dictionary with 'config' and 'results'
        """
        config_path = os.path.join(experiment_dir, 'config.json')
        results_path = os.path.join(experiment_dir, 'results.json')
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        results = {}
        if os.path.exists(results_path):
            with open(results_path, 'r', encoding='utf-8') as f:
                results = json.load(f)
        
        return {'config': config, 'results': results}


def set_global_seed(seed: int):
    """
    Set global random seed for reproducibility.
    
    Parameters:
    -----------
    seed : int
        Random seed
    """
    np.random.seed(seed)
    # If using other libraries, set their seeds too
    # torch.manual_seed(seed) if using PyTorch
    # tf.random.set_seed(seed) if using TensorFlow
    print(f"Global seed set to: {seed}")
