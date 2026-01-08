"""
Step 2: CALCULATE (Classification)

This module implements classification algorithms from scratch
using only NumPy, ensuring transparency and educational value.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional

from ...core.domain import (
    IClassificationService,
    ClassificationResult,
    ClassificationMetrics,
    Result
)
from ...config import get_logger

logger = get_logger(__name__)


class ClassificationServiceImpl(IClassificationService):
    """
    Implementation of classification algorithms (Logistic, KNN).
    
    Principles:
    1. Pure NumPy implementation (educational)
    2. Vectorized operations (clean code)
    3. Detailed metrics calculation
    """
    
    # =========================================================================
    # LOGISTIC REGRESSION (Binary)
    # =========================================================================
    
    def train_logistic(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        learning_rate: float = 0.01,
        iterations: int = 1000
    ) -> ClassificationResult:
        """
        Train binary logistic regression using Gradient Descent.
        
        Model: p = sigmoid(X * w + b)
        Loss: Binary Cross Entropy
        """
        n_samples, n_features = X.shape
        
        # Initialize weights and bias
        # Using simplified standardization for stability during training
        # (Ideally this should be handled by a pre-processor service)
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0) + 1e-8
        X_scaled = (X - X_mean) / X_std
        
        weights = np.zeros(n_features)
        bias = 0.0
        
        # Gradient Descent
        loss_history = []
        
        for _ in range(iterations):
            # 1. Linear model
            z = np.dot(X_scaled, weights) + bias
            
            # 2. Sigmoid activation
            predictions = 1 / (1 + np.exp(-z))
            
            # 3. Gradients
            # dL/dw = (1/N) * X.T * (y_pred - y)
            error = predictions - y
            dw = (1 / n_samples) * np.dot(X_scaled.T, error)
            db = (1 / n_samples) * np.sum(error)
            
            # 4. Updates
            weights -= learning_rate * dw
            bias -= learning_rate * db
            
            # Log loss occasionally
            if _ % 100 == 0:
                loss = -np.mean(y * np.log(predictions + 1e-15) + (1-y) * np.log(1-predictions + 1e-15))
                loss_history.append(loss)
        
        # Final predictions
        z_final = np.dot(X_scaled, weights) + bias
        probs = 1 / (1 + np.exp(-z_final))
        preds = (probs >= 0.5).astype(int)
        
        # Unscale weights for interpretability (approximate)
        # w_orig = w_scaled / std
        # b_orig = b_scaled - sum(w_scaled * mean / std)
        real_weights = weights / X_std
        real_bias = bias - np.sum(weights * X_mean / X_std)
        
        metrics = self.calculate_metrics(y, preds, probs)
        
        return ClassificationResult(
            classes=[0, 1],
            predictions=preds,
            probabilities=probs,
            metrics=metrics,
            model_params={
                "coefficients": real_weights.tolist(),
                "intercept": float(real_bias),
                "iterations": iterations,
                "learning_rate": learning_rate,
                "loss_history": loss_history
            },
        )
    
    # =========================================================================
    # K-NEAREST NEIGHBORS (Multi-class)
    # =========================================================================
    
    def train_knn(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        k: int = 3
    ) -> ClassificationResult:
        """
        Train KNN model.
        
        Note: KNN is "lazy", so training just stores data.
        To make this output a Result immediately useful for visualization,
        we compute predictions on the Training set (or could be Test set if passed).
        Here we return training performance for consistency.
        """
        n_samples = len(y)
        classes = np.unique(y)
        n_classes = len(classes)
        
        # Standardize for distance calculation
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0) + 1e-8
        X_scaled = (X - X_mean) / X_std
        
        # Predict for all points (Leave-One-Out style ideally, but standard fit-predict here)
        preds = np.zeros(n_samples, dtype=int)
        probs = np.zeros((n_samples, n_classes))
        
        # Vectorized distance calculation is heavy for large N, 
        # but fine for our small datasets (<1000 samples)
        for i in range(n_samples):
            # 1. Calculate distances to all POINTS
            # Euclidean: sqrt(sum((x - y)^2))
            diff = X_scaled - X_scaled[i]
            dists = np.sqrt(np.sum(diff**2, axis=1))
            
            # 2. Find k nearest (excluding self if strictly LOO, but standard sklearn approach includes self in fit-predict)
            # We'll use standard approach: find k neighbors including self (k=1 will maximize accuracy to 1.0)
            nearest_indices = np.argsort(dists)[:k]
            nearest_labels = y[nearest_indices]
            
            # 3. Vote
            counts = np.bincount(nearest_labels, minlength=n_classes)
            preds[i] = np.argmax(counts)
            probs[i] = counts / k
            
        metrics = self.calculate_metrics(y, preds, probs[:, 1] if n_classes==2 else probs)
        
        return ClassificationResult(
            classes=classes.tolist(),
            predictions=preds,
            probabilities=probs,
            metrics=metrics,
            model_params={
                "k": k,
                "X_train": X,
                "y_train": y,
                "X_mean": X_mean,
                "X_std": X_std
            },
        )

    # =========================================================================
    # METRICS
    # =========================================================================

    def calculate_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        y_prob: np.ndarray
    ) -> ClassificationMetrics:
        """Calculate comprehensive classification metrics."""
        n = len(y_true)
        if n == 0:
            return ClassificationMetrics(0, 0, 0, 0, np.zeros((2,2)))
            
        # 1. Confusion Matrix
        # Handle both binary and multiclass by treating macro averages or specific class of interest
        # For simplicity in this version, we assume Binary or handle Multiclass via weighted
        classes = np.unique(y_true)
        is_binary = len(classes) <= 2
        
        if is_binary:
            # Binary: 0=Negative, 1=Positive
            tp = np.sum((y_true == 1) & (y_pred == 1))
            tn = np.sum((y_true == 0) & (y_pred == 0))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))
            
            cm = np.array([[tn, fp], [fn, tp]])
            
            accuracy = (tp + tn) / n
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            # Simple AUC approximation (trapezoidal) using sorted probabilities
            if len(np.unique(y_true)) > 1:
                # Basic AUC implementation
                order = np.argsort(y_prob)[::-1]
                y_true_sorted = y_true[order]
                tpr = np.cumsum(y_true_sorted) / np.sum(y_true_sorted)
                fpr = np.cumsum(1 - y_true_sorted) / np.sum(1 - y_true_sorted)
                auc = np.trapezoid(tpr, fpr)
            else:
                auc = 0.5
                
        else:
            # Multiclass: Macro averaging
            n_classes = len(classes)
            precisions, recalls = [], []
            cm = np.zeros((n_classes, n_classes), dtype=int)
            
            for i, c_true in enumerate(classes):
                for j, c_pred in enumerate(classes):
                    cm[i, j] = np.sum((y_true == c_true) & (y_pred == c_pred))
            
            accuracy = np.trace(cm) / n
            
            for i in range(n_classes):
                tp = cm[i, i]
                fp = np.sum(cm[:, i]) - tp
                fn = np.sum(cm[i, :]) - tp
                
                p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                precisions.append(p)
                recalls.append(r)
            
            precision = np.mean(precisions)
            recall = np.mean(recalls)
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            auc = None # Multiclass AUC is complex (One-vs-Rest), skipping for now
            
        return ClassificationMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            confusion_matrix=cm,
            auc=auc
        )

    # =========================================================================
    # PREDICTION & EVALUATION
    # =========================================================================

    def predict_logistic(
        self,
        X: np.ndarray,
        params: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict using Logistic Regression params."""
        weights = np.array(params["coefficients"])
        bias = params["intercept"]
        
        if X.shape[1] != len(weights):
             raise ValueError(f"Feature mismatch: X has {X.shape[1]}, model has {len(weights)}")
             
        # Standardize X using assumption of similar scale 
        # (Limitations of stateless service without stored scaler)
        # Ideally parameters should include scaler stats.
        # In current train_logistic, we unscaled weights, so they work on RAW X 
        # IF X has same scale. But we used Z-score.
        # Wait, if we use real_weights = weights / X_std, then:
        # y = w_scaled * x_scaled + b_scaled
        #   = w_scaled * (x - mean)/std + b_scaled
        #   = (w_scaled/std) * x - (w_scaled*mean/std) + b_scaled
        #   = real_weights * x + real_bias
        # So yes, real_weights and real_bias work on RAW X.
        
        # However, for stability, we might want to scale again? 
        # But we computed real_weights precisely to avoid storing scaler.
        # So we can just dot product.
        
        z = np.dot(X, weights) + bias
        probs = 1 / (1 + np.exp(-z))
        preds = (probs >= 0.5).astype(int)
        
        return preds, probs

    def predict_knn(
        self,
        X: np.ndarray,
        params: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict using KNN params."""
        k = params["k"]
        X_train = np.array(params["X_train"])
        y_train = np.array(params["y_train"])
        X_train_mean = np.array(params["X_mean"])
        X_train_std = np.array(params["X_std"])
        
        # Scale input using training stats
        X_scaled = (X - X_train_mean) / X_train_std
        # Note: X_train in params is ORIGINAL. We need to scale it too for distance calc.
        X_train_scaled = (X_train - X_train_mean) / X_train_std
        
        n_samples = len(X)
        n_classes = len(np.unique(y_train))
        
        preds = np.zeros(n_samples, dtype=int)
        probs = np.zeros((n_samples, n_classes))
        
        # Optimization: matrix operations for distances? vectorization
        # dists = sqrt(|x|^2 + |y|^2 - 2xy)
        # For education we stick to simple loop or semi-vectorized.
        
        for i in range(n_samples):
            diff = X_train_scaled - X_scaled[i]
            dists = np.sqrt(np.sum(diff**2, axis=1))
            
            nearest_indices = np.argsort(dists)[:k]
            nearest_labels = y_train[nearest_indices]
            
            counts = np.bincount(nearest_labels, minlength=n_classes)
            preds[i] = np.argmax(counts)
            probs[i] = counts / k
            
        return preds, probs[:, 1] if n_classes==2 else probs

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        params: Dict[str, Any],
        method: str
    ) -> ClassificationMetrics:
        """Evaluate model on new data."""
        if method == "knn":
            preds, probs = self.predict_knn(X, params)
        else:
            preds, probs = self.predict_logistic(X, params)
            
        return self.calculate_metrics(y, preds, probs)
