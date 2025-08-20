"""
Supervised Learning Models
=========================

Implementation of ML models for Snake Game AI including MLP and LightGBM.

Design Philosophy:
- Factory pattern for model creation
- Common interface for all models
- Support for training on heuristic-generated datasets
- Feature engineering for game state representation
"""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
import numpy as np
import pandas as pd
from utils.print_utils import print_info, print_warning, print_error

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print_warning("[Models] PyTorch not available, MLP model will be disabled")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print_warning("[Models] LightGBM not available, LightGBM model will be disabled")


class BaseModel(ABC):
    """Base class for all supervised learning models."""
    
    def __init__(self, model_type: str):
        self.model_type = model_type
        self.is_trained = False
        self.accuracy = 0.0
    
    @abstractmethod
    def train(self, dataset_path: Optional[str] = None) -> float:
        """Train the model on dataset.
        
        Args:
            dataset_path: Path to training dataset
            
        Returns:
            Training accuracy
        """
        pass
    
    @abstractmethod
    def predict(self, game_state: Dict[str, Any]) -> str:
        """Predict next move for given game state.
        
        Args:
            game_state: Current game state dictionary
            
        Returns:
            Predicted move (UP, DOWN, LEFT, RIGHT)
        """
        pass
    
    def extract_features(self, game_state: Dict[str, Any]) -> np.ndarray:
        """Extract features from game state for ML model.
        
        Features extracted (16 total, grid-size agnostic):
        1-4: Head position (normalized x, y, distance to walls)
        5-8: Apple position (normalized x, y, distance from head)  
        9-12: Movement directions (up, down, left, right available)
        13-16: Snake body proximity (danger in 4 directions)
        
        Args:
            game_state: Game state dictionary
            
        Returns:
            Feature vector as numpy array
        """
        features = []
        
        # Get game state components
        snake_positions = game_state.get("snake_positions", [])
        apple_position = game_state.get("apple_position", [0, 0])
        grid_size = game_state.get("grid_size", 10)
        
        if not snake_positions:
            return np.zeros(16)  # Return zero features if no snake
        
        head_pos = snake_positions[-1]  # Head is last element
        head_x, head_y = head_pos[0], head_pos[1]
        apple_x, apple_y = apple_position[0], apple_position[1]
        
        # Features 1-4: Head position and wall distances (normalized)
        features.extend([
            head_x / grid_size,  # Normalized head x
            head_y / grid_size,  # Normalized head y
            head_x / grid_size,  # Distance to left wall (normalized)
            (grid_size - head_x - 1) / grid_size,  # Distance to right wall (normalized)
        ])
        
        # Features 5-8: Apple position and distance from head
        apple_distance = abs(head_x - apple_x) + abs(head_y - apple_y)  # Manhattan distance
        features.extend([
            apple_x / grid_size,  # Normalized apple x
            apple_y / grid_size,  # Normalized apple y
            apple_distance / (2 * grid_size),  # Normalized manhattan distance
            1.0 if apple_x > head_x else 0.0,  # Apple to the right
        ])
        
        # Features 9-12: Movement directions available
        body_positions = set(tuple(pos) for pos in snake_positions[:-1])  # Exclude head
        
        # Check each direction for validity
        directions = {
            'UP': (head_x, head_y - 1),
            'DOWN': (head_x, head_y + 1), 
            'LEFT': (head_x - 1, head_y),
            'RIGHT': (head_x + 1, head_y)
        }
        
        for direction, (new_x, new_y) in directions.items():
            # Check bounds and body collision
            valid = (
                0 <= new_x < grid_size and 
                0 <= new_y < grid_size and 
                (new_x, new_y) not in body_positions
            )
            features.append(1.0 if valid else 0.0)
        
        # Features 13-16: Snake body proximity (danger detection)
        for direction, (new_x, new_y) in directions.items():
            # Check if there's a body part in this direction within 2 steps
            danger = 0.0
            for step in range(1, 3):  # Check 1-2 steps ahead
                check_x = head_x + step * (new_x - head_x)
                check_y = head_y + step * (new_y - head_y)
                if (check_x, check_y) in body_positions:
                    danger = 1.0 - (step - 1) * 0.5  # Closer danger = higher value
                    break
            features.append(danger)
        
        return np.array(features, dtype=np.float32)


class MLPModel(BaseModel):
    """Multi-Layer Perceptron model using PyTorch."""
    
    def __init__(self, dataset_path: Optional[str] = None, verbose: bool = False):
        super().__init__("MLP")
        self.dataset_path = dataset_path
        self.verbose = verbose
        self.model = None
        self.move_to_idx = {"UP": 0, "DOWN": 1, "LEFT": 2, "RIGHT": 3}
        self.idx_to_move = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}
        
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for MLP model")
        
        # Create neural network
        self.model = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 4)  # 4 output classes for moves
        )
        
        print_info(f"[MLPModel] Initialized MLP model")
    
    def train(self, dataset_path: Optional[str] = None) -> float:
        """Train MLP model on dataset.
        
        Args:
            dataset_path: Path to CSV dataset file
            
        Returns:
            Training accuracy
        """
        if not TORCH_AVAILABLE:
            print_error("[MLPModel] PyTorch not available")
            return 0.0
        
        dataset_path = dataset_path or self.dataset_path
        if not dataset_path:
            print_warning("[MLPModel] No dataset path provided")
            return 0.0
        
        try:
            # Load dataset
            print_info(f"[MLPModel] Loading dataset from {dataset_path}")
            df = pd.read_csv(dataset_path)
            
            if len(df) == 0:
                print_warning("[MLPModel] Empty dataset")
                return 0.0
            
            # Prepare features and labels
            X_list = []
            y_list = []
            
            for _, row in df.iterrows():
                # Create game state from row
                game_state = {
                    "snake_positions": eval(row.get("snake_positions", "[[5,5]]")),
                    "apple_position": eval(row.get("apple_position", "[3,3]")),
                    "grid_size": int(row.get("grid_size", 10))
                }
                
                # Extract features
                features = self.extract_features(game_state)
                move = row.get("move", "UP")
                
                if move in self.move_to_idx:
                    X_list.append(features)
                    y_list.append(self.move_to_idx[move])
            
            if len(X_list) == 0:
                print_warning("[MLPModel] No valid training samples")
                return 0.0
            
            X = np.array(X_list)
            y = np.array(y_list)
            
            print_info(f"[MLPModel] Training on {len(X)} samples")
            
            # Convert to PyTorch tensors
            X_tensor = torch.FloatTensor(X)
            y_tensor = torch.LongTensor(y)
            
            # Create data loader
            dataset = TensorDataset(X_tensor, y_tensor)
            dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
            
            # Setup training
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(self.model.parameters(), lr=0.001)
            
            # Training loop
            self.model.train()
            epochs = 50
            
            for epoch in range(epochs):
                total_loss = 0
                correct = 0
                total = 0
                
                for batch_X, batch_y in dataloader:
                    optimizer.zero_grad()
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += batch_y.size(0)
                    correct += (predicted == batch_y).sum().item()
                
                if self.verbose and (epoch + 1) % 10 == 0:
                    accuracy = 100 * correct / total
                    print_info(f"[MLPModel] Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}, Accuracy: {accuracy:.2f}%")
            
            # Calculate final accuracy
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(X_tensor)
                _, predicted = torch.max(outputs, 1)
                self.accuracy = (predicted == y_tensor).float().mean().item()
            
            self.is_trained = True
            print_info(f"[MLPModel] Training completed with accuracy: {self.accuracy:.3f}")
            return self.accuracy
            
        except Exception as e:
            print_error(f"[MLPModel] Training failed: {e}")
            return 0.0
    
    def predict(self, game_state: Dict[str, Any]) -> str:
        """Predict next move using trained MLP model.
        
        Args:
            game_state: Current game state
            
        Returns:
            Predicted move
        """
        if not TORCH_AVAILABLE or not self.model:
            return "UP"  # Default move
        
        try:
            # Extract features
            features = self.extract_features(game_state)
            
            # Convert to tensor
            features_tensor = torch.FloatTensor(features).unsqueeze(0)
            
            # Get prediction
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(features_tensor)
                _, predicted = torch.max(outputs, 1)
                predicted_idx = predicted.item()
            
            return self.idx_to_move.get(predicted_idx, "UP")
            
        except Exception as e:
            print_warning(f"[MLPModel] Prediction failed: {e}")
            return "UP"  # Default move
    
    def predict_proba(self, game_state: Dict[str, Any]) -> float:
        """Get prediction confidence.
        
        Args:
            game_state: Current game state
            
        Returns:
            Confidence score (0-1)
        """
        if not TORCH_AVAILABLE or not self.model:
            return 0.0
        
        try:
            features = self.extract_features(game_state)
            features_tensor = torch.FloatTensor(features).unsqueeze(0)
            
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(features_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence = torch.max(probabilities).item()
            
            return confidence
            
        except Exception as e:
            return 0.0


class LightGBMModel(BaseModel):
    """LightGBM gradient boosting model."""
    
    def __init__(self, dataset_path: Optional[str] = None, verbose: bool = False):
        super().__init__("LightGBM")
        self.dataset_path = dataset_path
        self.verbose = verbose
        self.model = None
        self.move_to_idx = {"UP": 0, "DOWN": 1, "LEFT": 2, "RIGHT": 3}
        self.idx_to_move = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}
        
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM is required for LightGBM model")
        
        print_info(f"[LightGBMModel] Initialized LightGBM model")
    
    def train(self, dataset_path: Optional[str] = None) -> float:
        """Train LightGBM model on dataset.
        
        Args:
            dataset_path: Path to CSV dataset file
            
        Returns:
            Training accuracy
        """
        if not LIGHTGBM_AVAILABLE:
            print_error("[LightGBMModel] LightGBM not available")
            return 0.0
        
        dataset_path = dataset_path or self.dataset_path
        if not dataset_path:
            print_warning("[LightGBMModel] No dataset path provided")
            return 0.0
        
        try:
            # Load dataset
            print_info(f"[LightGBMModel] Loading dataset from {dataset_path}")
            df = pd.read_csv(dataset_path)
            
            if len(df) == 0:
                print_warning("[LightGBMModel] Empty dataset")
                return 0.0
            
            # Prepare features and labels
            X_list = []
            y_list = []
            
            for _, row in df.iterrows():
                # Create game state from row
                game_state = {
                    "snake_positions": eval(row.get("snake_positions", "[[5,5]]")),
                    "apple_position": eval(row.get("apple_position", "[3,3]")),
                    "grid_size": int(row.get("grid_size", 10))
                }
                
                # Extract features
                features = self.extract_features(game_state)
                move = row.get("move", "UP")
                
                if move in self.move_to_idx:
                    X_list.append(features)
                    y_list.append(self.move_to_idx[move])
            
            if len(X_list) == 0:
                print_warning("[LightGBMModel] No valid training samples")
                return 0.0
            
            X = np.array(X_list)
            y = np.array(y_list)
            
            print_info(f"[LightGBMModel] Training on {len(X)} samples")
            
            # Create LightGBM dataset
            train_data = lgb.Dataset(X, label=y)
            
            # Set parameters
            params = {
                'objective': 'multiclass',
                'num_class': 4,
                'metric': 'multi_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.1,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': 1 if self.verbose else -1
            }
            
            # Train model
            self.model = lgb.train(
                params,
                train_data,
                num_boost_round=100,
                valid_sets=[train_data],
                callbacks=[lgb.early_stopping(10)] if not self.verbose else None
            )
            
            # Calculate accuracy
            predictions = self.model.predict(X)
            predicted_classes = np.argmax(predictions, axis=1)
            self.accuracy = np.mean(predicted_classes == y)
            
            self.is_trained = True
            print_info(f"[LightGBMModel] Training completed with accuracy: {self.accuracy:.3f}")
            return self.accuracy
            
        except Exception as e:
            print_error(f"[LightGBMModel] Training failed: {e}")
            return 0.0
    
    def predict(self, game_state: Dict[str, Any]) -> str:
        """Predict next move using trained LightGBM model.
        
        Args:
            game_state: Current game state
            
        Returns:
            Predicted move
        """
        if not LIGHTGBM_AVAILABLE or not self.model:
            return "UP"  # Default move
        
        try:
            # Extract features
            features = self.extract_features(game_state)
            
            # Get prediction
            prediction = self.model.predict([features])
            predicted_idx = np.argmax(prediction[0])
            
            return self.idx_to_move.get(predicted_idx, "UP")
            
        except Exception as e:
            print_warning(f"[LightGBMModel] Prediction failed: {e}")
            return "UP"  # Default move
    
    def predict_proba(self, game_state: Dict[str, Any]) -> float:
        """Get prediction confidence.
        
        Args:
            game_state: Current game state
            
        Returns:
            Confidence score (0-1)
        """
        if not LIGHTGBM_AVAILABLE or not self.model:
            return 0.0
        
        try:
            features = self.extract_features(game_state)
            prediction = self.model.predict([features])
            confidence = np.max(prediction[0])
            return confidence
            
        except Exception as e:
            return 0.0


def create_model(model_type: str, dataset_path: Optional[str] = None, verbose: bool = False) -> BaseModel:
    """Factory function to create ML models.
    
    Args:
        model_type: Type of model to create ("MLP" or "LightGBM")
        dataset_path: Path to training dataset
        verbose: Enable verbose training output
        
    Returns:
        Initialized model instance
        
    Raises:
        ValueError: If model_type is not supported
    """
    model_type = model_type.upper()
    
    if model_type == "MLP":
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for MLP model but not available")
        return MLPModel(dataset_path, verbose)
    
    elif model_type == "LIGHTGBM":
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM is required for LightGBM model but not available")
        return LightGBMModel(dataset_path, verbose)
    
    else:
        available_models = []
        if TORCH_AVAILABLE:
            available_models.append("MLP")
        if LIGHTGBM_AVAILABLE:
            available_models.append("LightGBM")
        
        raise ValueError(f"Unknown model type: {model_type}. Available: {available_models}")