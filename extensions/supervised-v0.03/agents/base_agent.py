"""
Base Supervised Learning Agent
=============================

Base class for all supervised learning agents that use trained ML models
to make intelligent decisions in the Snake game.

Key Features:
- Model inference for move prediction
- Feature extraction from game state
- JSON data generation (no explanations needed)
- Performance tracking
"""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import time
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from utils.print_utils import print_info, print_warning, print_error


class BaseSupervisedAgent(ABC):
    """Base class for supervised learning agents."""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.model = None
        self.is_loaded = False
        self.prediction_count = 0
        self.total_prediction_time = 0.0
        self.feature_extraction_time = 0.0
        
    @property
    @abstractmethod
    def agent_name(self) -> str:
        """Return the agent name."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Return agent description."""
        pass
    
    @abstractmethod
    def load_model(self, model_path: str) -> bool:
        """Load trained model from file."""
        pass
    
    @abstractmethod
    def predict_move(self, game_state: Dict[str, Any]) -> str:
        """Predict next move based on game state."""
        pass
    
    def extract_features(self, game_state: Dict[str, Any]) -> np.ndarray:
        """
        Extract features from game state.
        
        Standard feature extraction for Snake game:
        - Snake head position (normalized)
        - Food position (normalized)
        - Snake body positions (relative to head)
        - Wall distances in 4 directions
        - Body collision distances in 4 directions
        """
        start_time = time.time()
        
        try:
            # Get basic game info
            snake = game_state.get("snake", [])
            food = game_state.get("food", [0, 0])
            grid_size = game_state.get("grid_size", 20)
            
            if not snake:
                return np.zeros(20)  # Return zero features if invalid state
            
            head = snake[0]
            
            # Normalize positions
            head_x, head_y = head[0] / grid_size, head[1] / grid_size
            food_x, food_y = food[0] / grid_size, food[1] / grid_size
            
            # Distance to food
            food_dist_x = food_x - head_x
            food_dist_y = food_y - head_y
            
            # Wall distances (normalized)
            wall_up = head_y / grid_size
            wall_down = (grid_size - 1 - head_y) / grid_size
            wall_left = head_x / grid_size
            wall_right = (grid_size - 1 - head_x) / grid_size
            
            # Body collision distances in 4 directions
            body_up = self._get_body_distance(head, snake[1:], (0, -1), grid_size)
            body_down = self._get_body_distance(head, snake[1:], (0, 1), grid_size)
            body_left = self._get_body_distance(head, snake[1:], (-1, 0), grid_size)
            body_right = self._get_body_distance(head, snake[1:], (1, 0), grid_size)
            
            # Snake length (normalized)
            snake_length = len(snake) / (grid_size * grid_size)
            
            # Direction features (current direction if available)
            direction_features = self._extract_direction_features(game_state)
            
            # Combine all features
            features = np.array([
                head_x, head_y,           # Head position
                food_x, food_y,           # Food position  
                food_dist_x, food_dist_y, # Distance to food
                wall_up, wall_down,       # Wall distances
                wall_left, wall_right,
                body_up, body_down,       # Body collision distances
                body_left, body_right,
                snake_length,             # Snake length
                *direction_features       # Direction features (5 values)
            ])
            
            self.feature_extraction_time += time.time() - start_time
            return features
            
        except Exception as e:
            print_warning(f"Feature extraction failed: {e}")
            self.feature_extraction_time += time.time() - start_time
            return np.zeros(20)
    
    def _get_body_distance(self, head: List[int], body: List[List[int]], 
                          direction: Tuple[int, int], grid_size: int) -> float:
        """Get normalized distance to body in given direction."""
        x, y = head
        dx, dy = direction
        distance = 0
        
        while 0 <= x + dx < grid_size and 0 <= y + dy < grid_size:
            x += dx
            y += dy
            distance += 1
            
            if [x, y] in body:
                return distance / grid_size
        
        return 1.0  # No collision in this direction
    
    def _extract_direction_features(self, game_state: Dict[str, Any]) -> List[float]:
        """Extract direction-related features."""
        # Get previous move if available
        last_move = game_state.get("last_move", "UP")
        
        # One-hot encode direction
        directions = ["UP", "DOWN", "LEFT", "RIGHT"]
        direction_features = [1.0 if last_move == d else 0.0 for d in directions]
        
        # Add a bias feature
        direction_features.append(1.0)
        
        return direction_features
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get agent performance statistics."""
        avg_prediction_time = (self.total_prediction_time / self.prediction_count 
                              if self.prediction_count > 0 else 0.0)
        avg_feature_time = (self.feature_extraction_time / self.prediction_count 
                           if self.prediction_count > 0 else 0.0)
        
        return {
            "agent_name": self.agent_name,
            "predictions_made": self.prediction_count,
            "total_prediction_time": self.total_prediction_time,
            "average_prediction_time": avg_prediction_time,
            "feature_extraction_time": self.feature_extraction_time,
            "average_feature_time": avg_feature_time,
            "predictions_per_second": (self.prediction_count / self.total_prediction_time 
                                     if self.total_prediction_time > 0 else 0.0)
        }
    
    def reset_stats(self):
        """Reset performance statistics."""
        self.prediction_count = 0
        self.total_prediction_time = 0.0
        self.feature_extraction_time = 0.0