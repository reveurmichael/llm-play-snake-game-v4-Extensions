"""
Supervised Learning Game Manager
==============================

Session management for supervised learning models (MLP, LightGBM) trained on
heuristic-generated datasets.

Design Philosophy:
- Extends BaseGameManager with minimal code
- Uses supervised learning models for move prediction
- Leverages heuristics-generated datasets for training
- Demonstrates clean inheritance from base classes

Design Patterns:
- Template Method: Inherits base session management structure
- Factory Pattern: Uses SupervisedGameLogic for game logic
- Strategy Pattern: Pluggable ML models (MLP, LightGBM)
"""

from __future__ import annotations
import sys
import os
from pathlib import Path

# Fix UTF-8 encoding issues on Windows
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

# Ensure project root is set and properly configured
from utils.path_utils import ensure_project_root
ensure_project_root()

import argparse
import time
from datetime import datetime
from typing import Optional, List, Dict, Any
import json

# Import from project root using absolute imports
from utils.print_utils import print_info, print_warning, print_success, print_error
from core.game_manager import BaseGameManager
from extensions.common import EXTENSIONS_LOGS_DIR

# Import supervised-specific components
from .game_logic import SupervisedGameLogic
from .models import create_model


class SupervisedGameManager(BaseGameManager):
    """
    Multi-model session manager for supervised learning v0.03.
    
    Demonstrates clean inheritance from BaseGameManager with minimal
    extension-specific code. Uses the template method pattern extensively.
    
    Design Patterns:
    - Template Method: Inherits base session management structure  
    - Factory Pattern: Uses SupervisedGameLogic for game logic
    - Strategy Pattern: Pluggable ML models (MLP, LightGBM)
    """

    # Use supervised learning game logic
    GAME_LOGIC_CLS = SupervisedGameLogic

    def __init__(self, args: argparse.Namespace) -> None:
        """Initialize supervised learning game manager.
        
        Args:
            args: Command line arguments namespace
        """
        super().__init__(args)

        # Supervised learning specific attributes
        self.model_type: str = getattr(args, "model", "MLP")
        self.dataset_path: Optional[str] = getattr(args, "dataset", None)
        self.model = None
        self.verbose: bool = getattr(args, "verbose", False)
        
        # Session statistics for summary
        self.session_start_time: datetime = datetime.now()
        self.prediction_times: List[float] = []
        self.model_accuracy: float = 0.0

        print_info(f"[SupervisedGameManager] Initialized for {self.model_type}")

    def initialize(self) -> None:
        """Initialize the supervised learning manager."""
        # Setup logging directory
        self._setup_logging()
        
        # Load and train model
        self._setup_model()
        
        # Setup base game components
        self.setup_game()
        
        # Configure game with model
        if isinstance(self.game, SupervisedGameLogic) and self.model:
            self.game.set_model(self.model)
            # Ensure grid_size is set correctly
            if hasattr(self.game.game_state, "grid_size"):
                self.game.game_state.grid_size = self.args.grid_size

        print_info(f"[SupervisedGameManager] Initialization complete for {self.model_type}")

    def _setup_logging(self) -> None:
        """Setup logging directory for supervised learning extension."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        grid_size = getattr(self.args, "grid_size", 10)

        # Follow standardized dataset folder structure
        dataset_folder = f"supervised_v0.03_{timestamp}"
        base_dir = os.path.join(
            EXTENSIONS_LOGS_DIR, "datasets", f"grid-size-{grid_size}", dataset_folder
        )

        # Model-specific subdirectory
        self.log_dir = os.path.join(base_dir, self.model_type.lower())

        # Create directories
        os.makedirs(self.log_dir, exist_ok=True)

    def _setup_model(self) -> None:
        """Setup and train the supervised learning model."""
        try:
            # Create model using factory pattern
            self.model = create_model(
                model_type=self.model_type,
                dataset_path=self.dataset_path,
                verbose=self.verbose
            )
            
            # Train model if dataset is provided
            if self.dataset_path:
                print_info(f"[SupervisedGameManager] Training {self.model_type} model...")
                self.model_accuracy = self.model.train()
                print_success(f"[SupervisedGameManager] Model trained with accuracy: {self.model_accuracy:.3f}")
            else:
                print_warning("[SupervisedGameManager] No dataset provided, using untrained model")
                
        except Exception as e:
            print_error(f"[SupervisedGameManager] Failed to setup model: {e}")
            raise

    def run(self) -> None:
        """Run the supervised learning session."""
        print_success("✅ 🚀 Starting supervised learning v0.03 session...")
        print_info(f"📊 Target games: {self.args.max_games}")
        print_info(f"🧠 Model: {self.model_type}")
        if self.model_accuracy > 0:
            print_info(f"🎯 Model accuracy: {self.model_accuracy:.3f}")
        print_info("")

        # Run games using base class template method
        for game_id in range(1, self.args.max_games + 1):
            print_info(f"🎮 Game {game_id}")
            
            # Run single game
            game_duration = self._execute_single_game()
            
            # Finalize game using base class method
            self.finalize_game(game_duration)
            
            # Display results using base class method
            self.display_game_results(game_duration)
            
            # Check if we should continue
            if game_id < self.args.max_games:
                print_info("")  # Spacer between games

        # Save session summary
        self._save_session_summary()

        print_success("✅ ✅ Supervised learning v0.03 session completed!")
        print_info(f"🎮 Games played: {len(self.game_scores)}")
        print_info(f"🏆 Total score: {self.total_score}")
        print_info(f"📈 Average score: {self.total_score / len(self.game_scores) if self.game_scores else 0:.1f}")
        print_success("✅ Supervised learning v0.03 execution completed successfully!")
        if hasattr(self, "log_dir") and self.log_dir:
            print_info(f"📂 Logs: {self.log_dir}")

    def _execute_single_game(self) -> float:
        """Execute a single game using supervised learning model."""
        start_time = time.time()

        # Initialize game
        self.game.reset()

        # Game loop
        steps = 0
        while not self.game.game_over:
            steps += 1

            # Start new round
            self.start_new_round(f"{self.model_type} prediction")

            # Get current game state
            game_state = self.game.get_state_snapshot()

            # Get move prediction from model
            prediction_start = time.time()
            move = self.game.get_next_planned_move()
            prediction_time = time.time() - prediction_start
            self.prediction_times.append(prediction_time)

            # Validate move
            if move == "NO_PATH_FOUND" or not move:
                print_warning(f"[SupervisedGameManager] Model returned invalid move: {move}")
                self.game.game_state.record_game_end("NO_PATH_FOUND")
                break

            # Apply move
            self.game.make_move(move)

            # Update display if GUI is enabled
            if hasattr(self.game, "update_display"):
                self.game.update_display()

            # Check limits using base class limits manager
            if self.limits_manager.should_end_game(steps, "MAX_STEPS"):
                print_info(f"[SupervisedGameManager] Max steps reached: {steps}")
                self.game.game_state.record_game_end("MAX_STEPS_REACHED")
                break

        return time.time() - start_time

    def _add_task_specific_game_data(self, game_data: Dict[str, Any], game_duration: float) -> None:
        """Add supervised learning specific game data."""
        game_data["model_type"] = self.model_type
        game_data["model_accuracy"] = self.model_accuracy
        game_data["avg_prediction_time"] = (
            sum(self.prediction_times) / len(self.prediction_times) 
            if self.prediction_times else 0.0
        )

    def _display_task_specific_results(self, game_duration: float) -> None:
        """Display supervised learning specific results."""
        if self.prediction_times:
            avg_prediction_time = sum(self.prediction_times) / len(self.prediction_times)
            print_info(f"🧠 Model: {self.model_type}, Avg prediction time: {avg_prediction_time:.4f}s")

    def _save_session_summary(self) -> None:
        """Save session summary with supervised learning specific data."""
        session_duration = (datetime.now() - self.session_start_time).total_seconds()

        summary = {
            "session_timestamp": self.session_start_time.strftime("%Y%m%d_%H%M%S"),
            "model_type": self.model_type,
            "model_accuracy": self.model_accuracy,
            "total_games": len(self.game_scores),
            "total_score": self.total_score,
            "average_score": (
                self.total_score / len(self.game_scores) if self.game_scores else 0
            ),
            "total_steps": self.total_steps,
            "total_rounds": self.total_rounds,
            "session_duration_seconds": round(session_duration, 2),
            "avg_prediction_time": (
                sum(self.prediction_times) / len(self.prediction_times) 
                if self.prediction_times else 0.0
            ),
            "game_scores": self.game_scores,
            "round_counts": self.round_counts,
            "configuration": {
                "grid_size": getattr(self.args, "grid_size", 10),
                "max_games": getattr(self.args, "max_games", 1),
                "verbose": getattr(self.args, "verbose", False),
                "dataset_path": self.dataset_path,
            },
        }

        # Save summary to file
        summary_file = os.path.join(self.log_dir, "summary.json")
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        # Display summary
        print_info(f"🧠 Model: {self.model_type}")
        print_info(f"🎯 Model accuracy: {self.model_accuracy:.3f}")
        print_info(f"🎮 Total games: {len(self.game_scores)}")
        print_info(f"🏆 Total score: {self.total_score}")
        print_info(f"📈 Scores: {self.game_scores}")
        print_info(f"📊 Average score: {summary['average_score']:.1f}")
        print_info(f"⚡ Avg prediction time: {summary['avg_prediction_time']:.4f}s")