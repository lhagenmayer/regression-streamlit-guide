"""
Data Splitter Service.
Handles data splitting logic for preview and training.
"""
import numpy as np
from typing import Tuple, Dict, Any
from src.core.domain.value_objects import SplitConfig, SplitStats

class DataSplitterService:
    """Service to handle data splitting calculations."""
    
    def preview_split(self, y: np.ndarray, config: SplitConfig) -> SplitStats:
        """
        Preview the statistics of a split without actually copying data.
        Returns indices stats.
        """
        n_samples = len(y)
        indices = np.arange(n_samples)
        
        np.random.seed(config.seed)
        
        if config.stratify:
            # Simple Stratified Split implementation
            # Group indices by class
            classes, y_indices = np.unique(y, return_inverse=True)
            train_indices = []
            test_indices = []
            
            for cls_idx in range(len(classes)):
                cls_mask = (y_indices == cls_idx)
                cls_samples = indices[cls_mask]
                np.random.shuffle(cls_samples)
                
                n_cls = len(cls_samples)
                n_train_cls = int(n_cls * config.train_size)
                
                train_indices.extend(cls_samples[:n_train_cls])
                test_indices.extend(cls_samples[n_train_cls:])
                
            train_indices = np.array(train_indices)
            test_indices = np.array(test_indices)
            
        else:
            # Random Split
            np.random.shuffle(indices)
            n_train = int(n_samples * config.train_size)
            train_indices = indices[:n_train]
            test_indices = indices[n_train:]
            
        # Calculate Distributions
        y_train = y[train_indices]
        y_test = y[test_indices]
        
        unique_train, counts_train = np.unique(y_train, return_counts=True)
        unique_test, counts_test = np.unique(y_test, return_counts=True)
        
        dist_train = dict(zip(unique_train, counts_train))
        dist_test = dict(zip(unique_test, counts_test))
        
        # Normalize keys to standard python types if numpy
        dist_train = {k.item() if hasattr(k, "item") else k: int(v) for k, v in dist_train.items()}
        dist_test = {k.item() if hasattr(k, "item") else k: int(v) for k, v in dist_test.items()}
        
        return SplitStats(
            train_count=len(train_indices),
            test_count=len(test_indices),
            train_distribution=dist_train,
            test_distribution=dist_test
        )

    def split_data(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        config: SplitConfig
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split arrays into train and test sets based on config.
        Returns: X_train, X_test, y_train, y_test
        """
        n_samples = len(y)
        indices = np.arange(n_samples)
        
        np.random.seed(config.seed)
        
        train_indices = []
        test_indices = []
        
        if config.stratify:
            classes, y_indices = np.unique(y, return_inverse=True)
            for cls_idx in range(len(classes)):
                cls_mask = (y_indices == cls_idx)
                cls_samples = indices[cls_mask]
                np.random.shuffle(cls_samples)
                
                n_cls = len(cls_samples)
                n_train_cls = int(n_cls * config.train_size)
                
                train_indices.extend(cls_samples[:n_train_cls])
                test_indices.extend(cls_samples[n_train_cls:])
                
            train_indices = np.array(train_indices)
            test_indices = np.array(test_indices)
            
            # Shuffle the resulting sets to avoid class ordering
            np.random.shuffle(train_indices)
            np.random.shuffle(test_indices)
            
        else:
            np.random.shuffle(indices)
            n_train = int(n_samples * config.train_size)
            train_indices = indices[:n_train]
            test_indices = indices[n_train:]
            
        return (
            X[train_indices], 
            X[test_indices], 
            y[train_indices], 
            y[test_indices]
        )
