"""
MLP Agent for Supervised Learning
=================================

Multi-Layer Perceptron agent that uses PyTorch-trained neural networks
for intelligent Snake game decision making.

Key Features:
- PyTorch MLP model inference
- Feature extraction and preprocessing
- Move prediction with confidence scoring
- Performance tracking and optimization
"""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import time
from typing import Dict, Any, Optional
from .base_agent import BaseSupervisedAgent

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    # Fallback numpy-like functionality
    class np:
        @staticmethod
        def array(data):
            return data
# Use simple print instead of utils to avoid numpy dependency issues
def print_info(msg): print(f"[INFO] {msg}")
def print_warning(msg): print(f"[WARNING] {msg}")  
def print_error(msg): print(f"[ERROR] {msg}")

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print_warning("[MLP Agent] PyTorch not available")
    # Create dummy classes to prevent import errors
    class torch:
        class device:
            def __init__(self, name): pass
        @staticmethod
        def cuda():
            class cuda:
                @staticmethod
                def is_available(): return False
            return cuda()  # Return instance, not class
        @staticmethod
        def load(path, map_location=None): return {}
        @staticmethod
        def FloatTensor(data): return data
        @staticmethod
        def no_grad():
            class NoGrad:
                def __enter__(self): return self
                def __exit__(self, *args): pass
            return NoGrad()
        @staticmethod
        def softmax(x, dim=None): return x
        @staticmethod
        def argmax(x, dim=None): return 0
    
    class nn:
        class Module:
            def __init__(self): pass
            def eval(self): pass
            def to(self, device): return self
            def load_state_dict(self, state): pass
        class Linear:
            def __init__(self, *args): pass
        class ReLU:
            def __init__(self): pass
        class Dropout:
            def __init__(self, *args): pass
        class Sequential:
            def __init__(self, *args): pass


if TORCH_AVAILABLE:
    class MLPModel(nn.Module):
        """Simple MLP model for move prediction."""
        
        def __init__(self, input_size: int = 20, hidden_sizes: list = None, num_classes: int = 4):
            super().__init__()
        
        if hidden_sizes is None:
            hidden_sizes = [64, 32]
        
        layers = []
        prev_size = input_size
        
        # Hidden layers
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
            return self.network(x)
else:
    # Fallback MLPModel when PyTorch is not available
    class MLPModel:
        def __init__(self, *args, **kwargs):
            pass
        def eval(self):
            pass
        def to(self, device):
            return self
        def load_state_dict(self, state):
            pass


class MLPAgent(BaseSupervisedAgent):
    """MLP agent for supervised learning."""
    
    description = "Multi-Layer Perceptron agent using PyTorch neural networks"
    
    def __init__(self, model_path: Optional[str] = None):
        super().__init__(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.move_mapping = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}
        
        if model_path:
            self.load_model(model_path)
    
    @property
    def agent_name(self) -> str:
        return "MLP"
    
    def load_model(self, model_path: str) -> bool:
        """Load trained MLP model from file."""
        if not TORCH_AVAILABLE:
            print_error("PyTorch not available, cannot load MLP model")
            return False
        
        try:
            model_path = Path(model_path)
            if not model_path.exists():
                print_error(f"Model file not found: {model_path}")
                return False
            
            # Load model state
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Create model with saved architecture
            model_config = checkpoint.get("model_config", {})
            input_size = model_config.get("input_size", 20)
            hidden_sizes = model_config.get("hidden_sizes", [64, 32])
            num_classes = model_config.get("num_classes", 4)
            
            self.model = MLPModel(input_size, hidden_sizes, num_classes)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.to(self.device)
            self.model.eval()
            
            self.is_loaded = True
            self.model_path = str(model_path)
            
            print_info(f"✅ Loaded MLP model from {model_path}")
            return True
            
        except Exception as e:
            print_error(f"Failed to load MLP model: {e}")
            return False
    
    def predict_move(self, game_state: Dict[str, Any]) -> str:
        """Predict next move using MLP model."""
        if not self.is_loaded or self.model is None:
            print_warning("Model not loaded, returning random move")
            return "UP"
        
        start_time = time.time()
        
        try:
            # Extract features
            features = self.extract_features(game_state)
            
            # Convert to tensor
            features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(features_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
            
            # Map to move
            predicted_move = self.move_mapping.get(predicted_class, "UP")
            
            # Update stats
            self.prediction_count += 1
            self.total_prediction_time += time.time() - start_time
            
            return predicted_move
            
        except Exception as e:
            print_warning(f"MLP prediction failed: {e}")
            self.prediction_count += 1
            self.total_prediction_time += time.time() - start_time
            return "UP"
    
    def get_prediction_confidence(self, game_state: Dict[str, Any]) -> Dict[str, float]:
        """Get prediction confidence scores for all moves."""
        if not self.is_loaded or self.model is None:
            return {"UP": 0.25, "DOWN": 0.25, "LEFT": 0.25, "RIGHT": 0.25}
        
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
            print_warning(f"Confidence calculation failed: {e}")
            return {"UP": 0.25, "DOWN": 0.25, "LEFT": 0.25, "RIGHT": 0.25}
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
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