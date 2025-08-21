"""
Supervised Learning Utilities
============================

Elegant utility functions for supervised learning extension with
model evaluation, data processing, and performance analysis.

Key Features:
- Model evaluation and validation utilities
- Feature engineering helpers
- Performance comparison tools
- Data preprocessing utilities
"""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from utils.path_utils import ensure_project_root
ensure_project_root()

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from utils.print_utils import print_info, print_warning, print_error


def evaluate_model_performance(model, test_data: pd.DataFrame) -> Dict[str, float]:
    """Evaluate model performance on test dataset.
    
    Args:
        model: Trained ML model with predict method
        test_data: Test dataset with features and target
        
    Returns:
        Dictionary containing performance metrics
    """
    try:
        # Prepare test features
        X_test = []
        y_test = []
        
        move_to_idx = {"UP": 0, "DOWN": 1, "LEFT": 2, "RIGHT": 3}
        
        for _, row in test_data.iterrows():
            # Create game state from row
            game_state = {
                "snake_positions": eval(row.get("snake_positions", "[[5,5]]")),
                "apple_position": eval(row.get("apple_position", "[3,3]")),
                "grid_size": int(row.get("grid_size", 10))
            }
            
            # Extract features using model's method
            features = model.extract_features(game_state)
            move = row.get("move", "UP")
            
            if move in move_to_idx:
                X_test.append(features)
                y_test.append(move_to_idx[move])
        
        if not X_test:
            return {"error": "No valid test samples"}
        
        X_test = np.array(X_test)
        y_test = np.array(y_test)
        
        # Get predictions
        predictions = []
        for i in range(len(X_test)):
            game_state = {
                "snake_positions": [[5, 5]],  # Dummy state for prediction
                "apple_position": [3, 3],
                "grid_size": 10
            }
            # This is a simplified prediction - in practice would use actual game state
            pred = model.predict(game_state)
            pred_idx = move_to_idx.get(pred, 0)
            predictions.append(pred_idx)
        
        predictions = np.array(predictions)
        
        # Calculate metrics
        accuracy = np.mean(predictions == y_test)
        
        # Per-class accuracy
        class_accuracies = {}
        idx_to_move = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}
        
        for move_idx, move_name in idx_to_move.items():
            mask = y_test == move_idx
            if np.sum(mask) > 0:
                class_acc = np.mean(predictions[mask] == y_test[mask])
                class_accuracies[move_name] = class_acc
        
        return {
            "overall_accuracy": accuracy,
            "class_accuracies": class_accuracies,
            "total_samples": len(y_test)
        }
        
    except Exception as e:
        print_error(f"Error evaluating model: {e}")
        return {"error": str(e)}


def compare_with_heuristics(model_results: Dict[str, Any], heuristic_results: Dict[str, Any]) -> Dict[str, Any]:
    """Compare supervised learning model with heuristic algorithms.
    
    Args:
        model_results: Results from supervised learning model
        heuristic_results: Results from heuristic algorithms
        
    Returns:
        Comparison analysis dictionary
    """
    comparison = {
        "model_vs_heuristic": {},
        "strengths": [],
        "weaknesses": [],
        "recommendations": []
    }
    
    try:
        # Compare key metrics
        model_score = model_results.get("average_score", 0)
        heuristic_score = heuristic_results.get("average_score", 0)
        
        model_steps = model_results.get("average_steps", 0)
        heuristic_steps = heuristic_results.get("average_steps", 0)
        
        comparison["model_vs_heuristic"] = {
            "score_ratio": model_score / heuristic_score if heuristic_score > 0 else 0,
            "steps_ratio": model_steps / heuristic_steps if heuristic_steps > 0 else 0,
            "model_better_score": model_score > heuristic_score,
            "model_better_efficiency": (model_score / model_steps) > (heuristic_score / heuristic_steps) if model_steps > 0 and heuristic_steps > 0 else False
        }
        
        # Generate insights
        if comparison["model_vs_heuristic"]["model_better_score"]:
            comparison["strengths"].append("Higher average score than heuristics")
        else:
            comparison["weaknesses"].append("Lower average score than heuristics")
            comparison["recommendations"].append("Consider more training data or feature engineering")
        
        if comparison["model_vs_heuristic"]["model_better_efficiency"]:
            comparison["strengths"].append("Better score-to-steps efficiency")
        else:
            comparison["weaknesses"].append("Lower efficiency than heuristics")
            comparison["recommendations"].append("Focus on training for efficient gameplay")
        
    except Exception as e:
        comparison["error"] = str(e)
    
    return comparison


def extract_dataset_statistics(dataset_path: str) -> Dict[str, Any]:
    """Extract comprehensive statistics from dataset.
    
    Args:
        dataset_path: Path to CSV dataset file
        
    Returns:
        Dictionary containing dataset statistics
    """
    try:
        df = pd.read_csv(dataset_path)
        
        stats = {
            "total_samples": len(df),
            "features": len(df.columns) - 1,  # Exclude target column
            "file_size_mb": Path(dataset_path).stat().st_size / (1024 * 1024)
        }
        
        # Move distribution
        if 'move' in df.columns:
            move_counts = df['move'].value_counts()
            stats["move_distribution"] = move_counts.to_dict()
            stats["most_common_move"] = move_counts.index[0]
            stats["move_balance"] = move_counts.min() / move_counts.max()
        
        # Grid size distribution
        if 'grid_size' in df.columns:
            grid_sizes = df['grid_size'].unique()
            stats["grid_sizes"] = grid_sizes.tolist()
        
        # Score distribution
        if 'score' in df.columns:
            stats["score_stats"] = {
                "mean": df['score'].mean(),
                "std": df['score'].std(),
                "min": df['score'].min(),
                "max": df['score'].max()
            }
        
        return stats
        
    except Exception as e:
        return {"error": str(e)}


def validate_dataset_quality(dataset_path: str) -> Dict[str, Any]:
    """Validate dataset quality for ML training.
    
    Args:
        dataset_path: Path to CSV dataset file
        
    Returns:
        Validation results dictionary
    """
    try:
        df = pd.read_csv(dataset_path)
        
        validation = {
            "is_valid": True,
            "issues": [],
            "warnings": [],
            "recommendations": []
        }
        
        # Check required columns
        required_columns = ['move', 'snake_positions', 'apple_position', 'grid_size']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            validation["is_valid"] = False
            validation["issues"].append(f"Missing required columns: {missing_columns}")
        
        # Check data quality
        if len(df) < 100:
            validation["warnings"].append("Small dataset - consider generating more data")
            validation["recommendations"].append("Generate at least 1000 samples for robust training")
        
        # Check move distribution
        if 'move' in df.columns:
            move_counts = df['move'].value_counts()
            if move_counts.min() / move_counts.max() < 0.1:
                validation["warnings"].append("Imbalanced move distribution")
                validation["recommendations"].append("Ensure balanced representation of all moves")
        
        # Check for missing values
        missing_values = df.isnull().sum().sum()
        if missing_values > 0:
            validation["warnings"].append(f"{missing_values} missing values found")
            validation["recommendations"].append("Clean dataset by handling missing values")
        
        return validation
        
    except Exception as e:
        return {
            "is_valid": False,
            "error": str(e),
            "issues": [f"Failed to load dataset: {e}"]
        }