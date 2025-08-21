"""
MLP Agent for Supervised Learning - KISS Edition
===============================================

Multi-Layer Perceptron agent using PyTorch neural networks for intelligent
Snake game decision making with fail-fast validation and clean architecture.

Design Philosophy:
- KISS: Keep It Simple, no unnecessary fallbacks
- Fail Fast: Immediate validation of dependencies and state
- SSOT: Single Source of Truth for all agent behavior
- Clean Code: Minimal, focused, and maintainable
"""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import time
from typing import Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
from .base_agent import BaseSupervisedAgent
from utils.print_utils import print_info, print_warning, print_error


class MLPModel(nn.Module):
    """
    Clean MLP model for move prediction with fail-fast validation.
    
    Architecture: Input -> Hidden Layers -> ReLU -> Dropout -> Output
    Fail-Fast: Validates input dimensions and parameters immediately
    """
    
    def __init__(self, input_size: int = 20, hidden_sizes: list = None, num_classes: int = 4):
        """Initialize MLP with fail-fast validation."""
        super().__init__()
        
        # Fail-fast validation
        if input_size <= 0:
            raise ValueError(f"[SSOT] Input size must be positive, got {input_size}")
        if num_classes <= 0:
            raise ValueError(f"[SSOT] Number of classes must be positive, got {num_classes}")
        
        if hidden_sizes is None:
            hidden_sizes = [64, 32]
        
        if not hidden_sizes or any(size <= 0 for size in hidden_sizes):
            raise ValueError(f"[SSOT] All hidden sizes must be positive, got {hidden_sizes}")
        
        layers = []
        prev_size = input_size
        
        # Build hidden layers
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass with fail-fast input validation."""
        if x is None:
            raise ValueError("[SSOT] Input tensor cannot be None")
        return self.network(x)


class MLPAgent(BaseSupervisedAgent):
    """
    MLP agent for supervised learning with fail-fast validation.
    
    Single Responsibility: Handles MLP model inference for Snake game
    Fail-Fast: Validates all inputs and state immediately
    SSOT: Centralized move prediction logic
    """
    
    description = "Multi-Layer Perceptron agent using PyTorch neural networks"
    
    def __init__(self, model_path: Optional[str] = None):
        """Initialize MLP agent with fail-fast validation."""
        super().__init__(model_path)
        
        # Fail-fast: Check PyTorch availability
        if not torch.cuda.is_available():
            print_info("CUDA not available, using CPU")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.move_mapping = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}
        
        if model_path:
            if not self.load_model(model_path):
                raise RuntimeError(f"[SSOT] Failed to load model from {model_path}")
    
    @property
    def agent_name(self) -> str:
        """Agent name following SSOT principle."""
        return "MLP"
    
    def load_model(self, model_path: str) -> bool:
        """
        Load trained MLP model with fail-fast validation.
        
        Returns:
            bool: True if successful, False otherwise (fail-fast)
        """
        model_path = Path(model_path)
        
        # Fail-fast: Check file existence
        if not model_path.exists():
            print_error(f"[SSOT] Model file not found: {model_path}")
            return False
        
        try:
            # Load model state
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Fail-fast: Validate checkpoint structure
            if "model_state_dict" not in checkpoint:
                raise ValueError("[SSOT] Invalid checkpoint: missing model_state_dict")
            
            # Get model configuration
            model_config = checkpoint.get("model_config", {})
            input_size = model_config.get("input_size", 20)
            hidden_sizes = model_config.get("hidden_sizes", [64, 32])
            num_classes = model_config.get("num_classes", 4)
            
            # Create and load model
            self.model = MLPModel(input_size, hidden_sizes, num_classes)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.to(self.device)
            self.model.eval()
            
            self.is_loaded = True
            self.model_path = str(model_path)
            
            print_info(f"✅ Loaded MLP model from {model_path}")
            return True
            
        except Exception as e:
            print_error(f"[SSOT] Failed to load MLP model: {e}")
            return False
    
    def predict_move(self, game_state: Dict[str, Any]) -> str:
        """
        Predict next move using MLP model with fail-fast validation.
        
        Args:
            game_state: Current game state dictionary
            
        Returns:
            str: Predicted move (UP, DOWN, LEFT, RIGHT)
            
        Raises:
            RuntimeError: If model not loaded or prediction fails
        """
        # Fail-fast: Check model availability
        if not self.is_loaded or self.model is None:
            raise RuntimeError("[SSOT] Model not loaded - cannot predict")
        
        # Fail-fast: Validate game state
        if not game_state or not isinstance(game_state, dict):
            raise ValueError("[SSOT] Game state must be non-empty dictionary")
        
        start_time = time.time()
        
        try:
            # Extract features with fail-fast validation
            features = self.extract_features(game_state)
            
            # Fail-fast: Validate features
            if features is None or len(features) == 0:
                raise ValueError("[SSOT] Feature extraction failed")
            
            # Convert to tensor
            features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(features_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
            
            # Fail-fast: Validate prediction
            if predicted_class not in self.move_mapping:
                raise ValueError(f"[SSOT] Invalid prediction class: {predicted_class}")
            
            predicted_move = self.move_mapping[predicted_class]
            
            # Update performance stats
            self.prediction_count += 1
            self.total_prediction_time += time.time() - start_time
            
            return predicted_move
            
        except Exception as e:
            self.prediction_count += 1
            self.total_prediction_time += time.time() - start_time
            raise RuntimeError(f"[SSOT] MLP prediction failed: {e}")
    
    def get_prediction_confidence(self, game_state: Dict[str, Any]) -> Dict[str, float]:
        """
        Get prediction confidence scores with fail-fast validation.
        
        Args:
            game_state: Current game state dictionary
            
        Returns:
            Dict[str, float]: Confidence scores for each move
            
        Raises:
            RuntimeError: If model not loaded or confidence calculation fails
        """
        # Fail-fast: Check model availability
        if not self.is_loaded or self.model is None:
            raise RuntimeError("[SSOT] Model not loaded - cannot get confidence")
        
        # Fail-fast: Validate game state
        if not game_state or not isinstance(game_state, dict):
            raise ValueError("[SSOT] Game state must be non-empty dictionary")
        
        try:
            # Extract features
            features = self.extract_features(game_state)
            features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            
            # Get probabilities
            with torch.no_grad():
                outputs = self.model(features_tensor)
                probabilities = torch.softmax(outputs, dim=1).squeeze().cpu().numpy()
            
            # Map to moves
            confidence = {}
            for i, move in self.move_mapping.items():
                confidence[move] = float(probabilities[i])
            
            return confidence
            
        except Exception as e:
            raise RuntimeError(f"[SSOT] Confidence calculation failed: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive model information with fail-fast validation.
        
        Returns:
            Dict[str, Any]: Model information and statistics
        """
        info = {
            "agent_name": self.agent_name,
            "model_type": "Multi-Layer Perceptron",
            "framework": "PyTorch",
            "device": str(self.device),
            "is_loaded": self.is_loaded,
            "model_path": self.model_path
        }
        
        if self.model is not None:
            # Count parameters
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            info.update({
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
                "model_size_mb": total_params * 4 / (1024 * 1024)  # Approximate size in MB
            })
        
        return info