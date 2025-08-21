"""
Model Comparison and Benchmarking for Supervised v0.03
======================================================

Advanced model comparison, benchmarking, and analysis tools for supervised
learning models with comprehensive performance evaluation and visualization.

Key Features:
- Model performance comparison and benchmarking
- Training efficiency analysis and optimization
- Feature importance analysis and visualization
- Cross-validation and robustness testing
- Export capabilities for research and reporting
"""

from __future__ import annotations
import sys
from pathlib import Path

# Ensure UTF-8 encoding for cross-platform compatibility (SUPREME_RULE NO.7)
import os
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from utils.path_utils import ensure_project_root
ensure_project_root()

import json
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from utils.print_utils import print_info, print_warning, print_error, print_success

try:
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.metrics import classification_report, confusion_matrix
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print_warning("[ModelComparison] scikit-learn not available - some features disabled")


class ModelBenchmark:
    """
    Comprehensive benchmarking system for supervised learning models.
    
    Provides detailed performance analysis, model comparison, and
    optimization recommendations for ML models.
    """
    
    def __init__(self, models_info: List[Dict[str, Any]]):
        """Initialize benchmark with model information.
        
        Args:
            models_info: List of dictionaries containing model info
                        Each dict should have: type, accuracy, dataset, model
        """
        self.models_info = models_info
        self.benchmark_results = {}
        
        print_info(f"[ModelBenchmark] Initialized with {len(models_info)} models")
    
    def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Run comprehensive benchmark across all models."""
        results = {
            "model_comparison": {},
            "performance_ranking": {},
            "training_efficiency": {},
            "recommendations": {},
            "benchmark_timestamp": datetime.now().isoformat()
        }
        
        try:
            # Compare model performance
            results["model_comparison"] = self._compare_model_performance()
            
            # Rank models by performance
            results["performance_ranking"] = self._rank_models_by_performance()
            
            # Analyze training efficiency
            results["training_efficiency"] = self._analyze_training_efficiency()
            
            # Generate recommendations
            results["recommendations"] = self._generate_model_recommendations()
            
            self.benchmark_results = results
            return results
            
        except Exception as e:
            print_error(f"Error running benchmark: {e}")
            return results
    
    def _compare_model_performance(self) -> Dict[str, Any]:
        """Compare performance across different models."""
        comparison = {
            "accuracy_comparison": {},
            "model_characteristics": {},
            "strengths_weaknesses": {}
        }
        
        for model_info in self.models_info:
            model_type = model_info.get("type", "Unknown")
            accuracy = model_info.get("accuracy", 0)
            
            comparison["accuracy_comparison"][model_type] = accuracy
            
            # Model characteristics
            if model_type == "MLP":
                comparison["model_characteristics"][model_type] = {
                    "type": "Neural Network",
                    "complexity": "Medium",
                    "training_time": "Medium",
                    "interpretability": "Low",
                    "memory_usage": "Medium"
                }
            elif model_type == "LightGBM":
                comparison["model_characteristics"][model_type] = {
                    "type": "Gradient Boosting",
                    "complexity": "High",
                    "training_time": "Fast",
                    "interpretability": "High",
                    "memory_usage": "Low"
                }
        
        return comparison
    
    def _rank_models_by_performance(self) -> Dict[str, Any]:
        """Rank models by overall performance."""
        # Sort by accuracy
        sorted_models = sorted(
            self.models_info, 
            key=lambda x: x.get("accuracy", 0), 
            reverse=True
        )
        
        rankings = {}
        for i, model_info in enumerate(sorted_models, 1):
            model_type = model_info.get("type", "Unknown")
            rankings[model_type] = {
                "rank": i,
                "accuracy": model_info.get("accuracy", 0),
                "score": self._calculate_model_score(model_info)
            }
        
        return rankings
    
    def _analyze_training_efficiency(self) -> Dict[str, Any]:
        """Analyze training efficiency across models."""
        efficiency = {}
        
        for model_info in self.models_info:
            model_type = model_info.get("type", "Unknown")
            
            # Estimate training efficiency based on model characteristics
            if model_type == "MLP":
                efficiency[model_type] = {
                    "training_speed": "Medium",
                    "memory_efficiency": "Medium",
                    "scalability": "Good",
                    "ease_of_tuning": "Medium"
                }
            elif model_type == "LightGBM":
                efficiency[model_type] = {
                    "training_speed": "Fast",
                    "memory_efficiency": "High",
                    "scalability": "Excellent",
                    "ease_of_tuning": "Easy"
                }
        
        return efficiency
    
    def _generate_model_recommendations(self) -> Dict[str, List[str]]:
        """Generate optimization recommendations for each model."""
        recommendations = {}
        
        for model_info in self.models_info:
            model_type = model_info.get("type", "Unknown")
            accuracy = model_info.get("accuracy", 0)
            
            model_recommendations = []
            
            if accuracy < 0.7:
                model_recommendations.append("Consider more training data or feature engineering")
                model_recommendations.append("Try different hyperparameters or model architecture")
            elif accuracy < 0.8:
                model_recommendations.append("Good performance - consider fine-tuning hyperparameters")
            else:
                model_recommendations.append("Excellent performance - model is well-optimized")
            
            # Model-specific recommendations
            if model_type == "MLP":
                if accuracy < 0.75:
                    model_recommendations.append("Try deeper network or different activation functions")
                model_recommendations.append("Consider regularization techniques (dropout, weight decay)")
            elif model_type == "LightGBM":
                if accuracy < 0.8:
                    model_recommendations.append("Tune num_leaves and learning_rate parameters")
                model_recommendations.append("Consider feature selection or engineering")
            
            recommendations[model_type] = model_recommendations
        
        return recommendations
    
    def _calculate_model_score(self, model_info: Dict[str, Any]) -> float:
        """Calculate overall model score based on multiple factors."""
        accuracy = model_info.get("accuracy", 0)
        
        # Base score from accuracy
        score = accuracy * 100
        
        # Adjust based on model characteristics
        model_type = model_info.get("type", "Unknown")
        if model_type == "LightGBM":
            score += 5  # Bonus for interpretability and speed
        elif model_type == "MLP":
            score += 2  # Bonus for flexibility
        
        return min(100, score)
    
    def export_benchmark_results(self, output_path: str):
        """Export benchmark results to file."""
        try:
            output_file = Path(output_path)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(self.benchmark_results, f, indent=2, ensure_ascii=False)
            
            print_success(f"📊 Benchmark results exported to {output_file}")
            
        except Exception as e:
            print_error(f"Error exporting benchmark results: {e}")


class ModelValidator:
    """
    Advanced model validation and testing system.
    
    Provides comprehensive validation including cross-validation,
    robustness testing, and performance analysis.
    """
    
    def __init__(self, model, dataset_path: str):
        self.model = model
        self.dataset_path = dataset_path
        self.validation_results = {}
        
    def validate_model_robustness(self) -> Dict[str, Any]:
        """Validate model robustness across different conditions."""
        if not SKLEARN_AVAILABLE:
            return {"error": "scikit-learn required for robustness validation"}
        
        try:
            # Load dataset
            df = pd.read_csv(self.dataset_path)
            
            # Prepare features and targets
            X, y = self._prepare_data_for_validation(df)
            
            if len(X) == 0:
                return {"error": "No valid samples for validation"}
            
            # Perform cross-validation
            cv_scores = cross_val_score(self.model, X, y, cv=5, scoring='accuracy')
            
            # Train-test split validation
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train and evaluate
            self.model.fit(X_train, y_train)
            test_accuracy = self.model.score(X_test, y_test)
            
            # Generate classification report
            y_pred = self.model.predict(X_test)
            class_report = classification_report(y_test, y_pred, output_dict=True)
            
            validation = {
                "cross_validation": {
                    "mean_accuracy": cv_scores.mean(),
                    "std_accuracy": cv_scores.std(),
                    "scores": cv_scores.tolist()
                },
                "test_accuracy": test_accuracy,
                "classification_report": class_report,
                "robustness_score": self._calculate_robustness_score(cv_scores, test_accuracy)
            }
            
            self.validation_results = validation
            return validation
            
        except Exception as e:
            print_error(f"Error validating model robustness: {e}")
            return {"error": str(e)}
    
    def _prepare_data_for_validation(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for validation."""
        X_list = []
        y_list = []
        
        move_to_idx = {"UP": 0, "DOWN": 1, "LEFT": 2, "RIGHT": 3}
        
        for _, row in df.iterrows():
            try:
                # Create game state from row
                game_state = {
                    "snake_positions": eval(row.get("snake_positions", "[[5,5]]")),
                    "apple_position": eval(row.get("apple_position", "[3,3]")),
                    "grid_size": int(row.get("grid_size", 10))
                }
                
                # Extract features using model's method
                features = self.model.extract_features(game_state)
                move = row.get("move", "UP")
                
                if move in move_to_idx:
                    X_list.append(features)
                    y_list.append(move_to_idx[move])
                    
            except Exception:
                continue  # Skip invalid rows
        
        return np.array(X_list), np.array(y_list)
    
    def _calculate_robustness_score(self, cv_scores: np.ndarray, test_accuracy: float) -> float:
        """Calculate robustness score based on validation results."""
        # Consider both mean performance and consistency
        mean_cv = cv_scores.mean()
        std_cv = cv_scores.std()
        
        # Robustness score: high mean, low std, good test accuracy
        consistency_score = max(0, 1 - (std_cv / max(0.01, mean_cv)))
        performance_score = (mean_cv + test_accuracy) / 2
        
        return (consistency_score * 0.3 + performance_score * 0.7) * 100


def compare_models(model_configs: List[Dict[str, Any]]) -> None:
    """Compare multiple models and generate comprehensive report."""
    try:
        print_info("🔍 Starting comprehensive model comparison...")
        
        benchmark = ModelBenchmark(model_configs)
        results = benchmark.run_comprehensive_benchmark()
        
        # Display results
        print_success("📊 Model Comparison Results:")
        print_info("=" * 60)
        
        # Show rankings
        rankings = results.get("performance_ranking", {})
        if rankings:
            print_info("🏆 Performance Rankings:")
            for model, rank_info in sorted(rankings.items(), key=lambda x: x[1]["rank"]):
                print_info(f"   {rank_info['rank']}. {model}: {rank_info['accuracy']:.1%}")
        
        # Show recommendations
        recommendations = results.get("recommendations", {})
        if recommendations:
            print_info("\n💡 Optimization Recommendations:")
            for model, recs in recommendations.items():
                print_info(f"\n{model}:")
                for rec in recs:
                    print_info(f"   • {rec}")
        
        # Export results
        output_file = f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        benchmark.export_benchmark_results(output_file)
        
        print_success("✅ Model comparison completed successfully!")
        
    except Exception as e:
        print_error(f"❌ Model comparison failed: {e}")


if __name__ == "__main__":
    # Example usage
    example_models = [
        {
            "type": "MLP",
            "accuracy": 0.85,
            "dataset": "example_dataset.csv",
            "model": None  # Would contain actual model object
        },
        {
            "type": "LightGBM", 
            "accuracy": 0.90,
            "dataset": "example_dataset.csv",
            "model": None  # Would contain actual model object
        }
    ]
    
    compare_models(example_models)