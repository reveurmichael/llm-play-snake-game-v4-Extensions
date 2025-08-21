"""
Supervised Learning Extension v0.03
===================================

This extension demonstrates supervised learning approaches for Snake Game AI,
including Multi-Layer Perceptron (MLP) and LightGBM models trained on
heuristic-generated datasets.

Key Features:
- MLP neural network with PyTorch
- LightGBM gradient boosting
- Dataset loading from heuristics-v0.04
- Model training and evaluation
- Performance comparison with heuristics
"""

__version__ = "0.03"
__author__ = "Snake Game AI Project"

# Export main classes
from .game_manager import SupervisedGameManager
from .game_logic import SupervisedGameLogic
from .models import MLPModel, LightGBMModel

__all__ = [
    "SupervisedGameManager",
    "SupervisedGameLogic", 
    "MLPModel",
    "LightGBMModel",
]