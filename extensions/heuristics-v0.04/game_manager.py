"""
Heuristic Game Manager v0.04
============================

Streamlined session management for multi-algorithm heuristic agents with
comprehensive dataset generation capabilities.

This module demonstrates clean extension architecture by inheriting from
BaseGameManager and focusing purely on heuristics-specific functionality.
"""

from __future__ import annotations
import sys
import os
from pathlib import Path

# Ensure UTF-8 encoding for cross-platform compatibility (SUPREME_RULE NO.7)
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

# Simple path management - KISS principle
import os
os.chdir(Path(__file__).resolve().parents[3])  # Change to project root

import argparse
import time
from datetime import datetime
from typing import Optional, List, Dict, Any
import json
import os
import copy

# Simple logging - KISS principle, avoid utils dependency
def print_info(msg): print(f"[INFO] {msg}")
def print_warning(msg): print(f"[WARNING] {msg}")
def print_success(msg): print(f"[SUCCESS] {msg}")
def print_error(msg): print(f"[ERROR] {msg}")
from core.game_manager import BaseGameManager
from extensions.common import EXTENSIONS_LOGS_DIR
from config.game_constants import END_REASON_MAP

# Import heuristic-specific components using relative imports
from game_logic import HeuristicGameLogic
from agents import get_available_algorithms, DEFAULT_ALGORITHM

# Import dataset generation utilities for automatic updates
from dataset_generator import DatasetGenerator

# Import BFSAgent for SSOT utilities
from extensions.common.utils.game_state_utils import to_serializable
from heuristics_utils import (
    calculate_manhattan_distance,
    calculate_valid_moves_ssot,
    bfs_pathfind,
)

# Import state management for robust pre/post state separation
from state_management import StateManager, validate_explanation_head_consistency

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


class HeuristicGameManager(BaseGameManager):
    """
    Elegant multi-algorithm session manager for heuristic pathfinding.
    
    Supports BFS, A*, DFS, Hamiltonian algorithms with automatic dataset
    generation (CSV/JSONL) and comprehensive state management.
    
    Key Features:
    - Multi-algorithm support via strategy pattern
    - Automatic dataset updates after each game
    - Robust pre/post-move state validation
    - Streamlined architecture leveraging BaseGameManager
    """

    # Use heuristic-specific game logic
    GAME_LOGIC_CLS = HeuristicGameLogic

    def __init__(self, args: argparse.Namespace, agent: Any) -> None:
        """
        Initialize heuristic game manager with fail-fast validation.
        
        Args:
            args: Command line arguments namespace
            agent: Heuristic agent instance (required)
            
        Raises:
            ValueError: If required arguments are missing or invalid (fail-fast)
            TypeError: If agent doesn't have required methods (fail-fast)
        """
        # Fail-fast: Validate arguments
        if not args:
            raise ValueError("[SSOT] Arguments namespace is required")
        
        if not hasattr(args, 'algorithm'):
            raise ValueError("[SSOT] Algorithm argument is required")
        
        # Fail-fast: Validate agent
        if agent is None:
            raise ValueError("[SSOT] Heuristic agent is required")
        
        if not hasattr(agent, 'algorithm_name'):
            raise TypeError("[SSOT] Agent must have algorithm_name attribute")
        
        if not hasattr(agent, 'find_path'):
            raise TypeError("[SSOT] Agent must have find_path method")
        
        super().__init__(args)

        # Heuristic-specific configuration with validation
        self.algorithm_name: str = getattr(args, "algorithm", DEFAULT_ALGORITHM)
        
        # Validate algorithm name
        from heuristic_config import validate_algorithm_name
        validate_algorithm_name(self.algorithm_name)
        
        self.agent: Any = agent
        self.verbose: bool = getattr(args, "verbose", False)
        
        # Extension-specific tracking
        self.game_rounds: List[int] = []
        self.dataset_generator: Optional[DatasetGenerator] = None

        print_info(f"[HeuristicGameManager] Initialized for {self.algorithm_name}")

    def initialize(self) -> None:
        """Initialize heuristic game manager components."""
        # Setup logging directory
        self._setup_logging()

        # Setup agent
        self._setup_agent()

        # Initialize dataset generator for automatic updates
        self._setup_dataset_generator()

        # Setup base game components
        self.setup_game()

        # Configure game with agent
        if isinstance(self.game, HeuristicGameLogic) and self.agent:
            self.game.set_agent(self.agent)
            # Ensure grid_size is set correctly
            if hasattr(self.game.game_state, "grid_size"):
                self.game.game_state.grid_size = self.args.grid_size

        print_info(
            f"[HeuristicGameManager] Initialization complete for {self.algorithm_name}"
        )

    def _setup_logging(self) -> None:
        """Setup logging directory using streamlined base class approach."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        grid_size = getattr(self.args, "grid_size", 10)

        # Follow standardized dataset folder structure
        dataset_folder = f"heuristics_v0.04_{timestamp}"
        base_dir = os.path.join(
            EXTENSIONS_LOGS_DIR, "datasets", f"grid-size-{grid_size}", dataset_folder
        )

        # Algorithm-specific subdirectory (all files for one run live here)
        self.log_dir = os.path.join(base_dir, self.algorithm_name.lower())

        # Use base class directory creation with error handling
        self.create_log_directory()
    
    def _create_extension_subdirectories(self) -> None:
        """Create heuristics-specific subdirectories."""
        # Create subdirectories for organized heuristics data
        subdirs = ["datasets", "states", "explanations"]
        for subdir in subdirs:
            try:
                os.makedirs(os.path.join(self.log_dir, subdir), exist_ok=True)
            except Exception:
                pass  # Non-critical if subdirectories can't be created

    def _setup_agent(self) -> None:
        """Validate and configure the heuristic agent."""
        try:
            # Validate agent is provided
            if self.agent is None:
                raise RuntimeError(f"Agent required for {self.algorithm_name}")

            # Validate agent algorithm matches requested
            provided_name = getattr(self.agent, "algorithm_name", None)
            if provided_name and provided_name.upper() != self.algorithm_name.upper():
                raise RuntimeError(f"Agent algorithm mismatch: {provided_name} != {self.algorithm_name}")

            if self.verbose:
                print_info(f"🏭 Using {self.agent.__class__.__name__} for {self.algorithm_name}")
        except Exception:
            raise

    def _setup_dataset_generator(self) -> None:
        """Setup dataset generator for CSV/JSONL output."""
        self.dataset_generator = DatasetGenerator(
            self.algorithm_name, Path(self.log_dir), agent=self.agent
        )
        
        # Initialize output files
        self.dataset_generator._open_csv()
        self.dataset_generator._open_jsonl()
        
        print_info("[HeuristicGameManager] Dataset generator ready")

    def run(self) -> None:
        """Run heuristic game session with streamlined base class management."""
        # Use the fully streamlined base class approach
        self.run_game_session()
    
    def _display_session_start(self) -> None:
        """Display heuristics-specific session start information."""
        super()._display_session_start()
        print_info(f"🧠 Algorithm: {self.algorithm_name}")
    
    def _display_session_completion(self) -> None:
        """Display heuristics-specific session completion."""
        super()._display_session_completion()
        print_success("✅ Heuristics v0.04 execution completed!")


    
    def _initialize_game_specific_rounds(self) -> None:
        """Initialize heuristics-specific rounds data."""
        # Initialize state manager for robust pre/post state separation
        self.state_manager = StateManager()
        
        # Validate initial game state
        initial_raw_state = self.game.get_state_snapshot()
        initial_pre_state = self.state_manager.create_pre_move_state(initial_raw_state)

        # Fail-fast: Validate initial game state
        if not initial_pre_state.get_snake_positions():
            print_error(
                f"[FAIL-FAST] Initial game state has no snake positions: {initial_raw_state}"
            )
            raise RuntimeError(
                "[SSOT] Initial game state has no snake positions - game reset failed"
            )

        # Ensure round manager is available
        if not (
            hasattr(self.game.game_state, "round_manager")
            and self.game.game_state.round_manager
        ):
            raise RuntimeError("[SSOT] Round manager missing after game reset.")
    
    def _process_game_state_before_move(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Process game state before move using heuristics state management."""
        # Create immutable pre-move state from current game state
        pre_state = self.state_manager.create_pre_move_state(game_state)

        # Store pre-move state in round data
        if (
            hasattr(self.game.game_state, "round_manager")
            and self.game.game_state.round_manager
        ):
            round_num = self.game.game_state.round_manager.round_buffer.number
            round_data = (
                self.game.game_state.round_manager._get_or_create_round_data(
                    round_num
                )
            )
            round_data["game_state"] = dict(
                pre_state.game_state
            )  # Convert back to dict for storage

        # Store pre-state for validation
        self.current_pre_state = pre_state
        
        # Return state dict for agent compatibility
        return dict(pre_state.game_state)
    
    def _get_next_move(self, game_state: Dict[str, Any]) -> str:
        """Get next move from heuristics agent with explanation."""
        # Fail-fast: Ensure game logic has required method
        if not hasattr(self.game, "get_next_planned_move_with_state"):
            raise RuntimeError(
                "[SSOT] Game logic missing get_next_planned_move_with_state method - required for SSOT compliance"
            )

        # Get move and explanation using processed state
        move, explanation = self.game.get_next_planned_move_with_state(
            game_state, return_explanation=True
        )

        # --- FAIL-FAST: VALIDATE EXPLANATION HEAD CONSISTENCY ---
        if not validate_explanation_head_consistency(self.current_pre_state, explanation):
            raise RuntimeError(
                "[SSOT] FAIL-FAST: Explanation head position does not match pre-move state"
            )

        return move
    
    def _validate_move_custom(self, move: str, game_state: Dict[str, Any]) -> bool:
        """Validate move using heuristics-specific validation."""
        # Use centralized SSOT validation
        valid_moves = calculate_valid_moves_ssot(game_state)
        
        if move == "NO_PATH_FOUND":
            if valid_moves:
                head = self.current_pre_state.get_head_position()
                print_error(
                    f"[SSOT VIOLATION] Agent returned 'NO_PATH_FOUND' but valid moves exist: {valid_moves} for head {head}"
                )
                raise RuntimeError(
                    f"SSOT violation: agent returned 'NO_PATH_FOUND' but valid moves exist: {valid_moves}"
                )
            # Record final game state
            if (
                hasattr(self.game.game_state, "round_manager")
                and self.game.game_state.round_manager
            ):
                round_num = self.game.game_state.round_manager.round_buffer.number
                round_data = (
                    self.game.game_state.round_manager._get_or_create_round_data(
                        round_num
                    )
                )
                round_data["game_state"] = copy.deepcopy(
                    self.game.get_state_snapshot()
                )
            self.game.game_state.record_game_end("NO_PATH_FOUND")
            return False

        if move not in valid_moves:
            raise RuntimeError(
                f"SSOT violation: agent move '{move}' not in valid moves {valid_moves}"
            )
        
        return True
    
    def _process_game_state_after_move(self, game_state: Dict[str, Any]) -> None:
        """Process game state after move using heuristics post-move validation."""
        # Create post-move state from game state after move
        post_state = self.state_manager.create_post_move_state(
            self.current_pre_state, 
            self.game.get_last_move() if hasattr(self.game, 'get_last_move') else "UNKNOWN",
            game_state
        )

        # --- POST-MOVE VALIDATION ---
        # Check if there are any valid moves left after move
        post_valid_moves = calculate_valid_moves_ssot(dict(post_state.game_state))
        if not post_valid_moves:
            print_error(
                "[DEBUG] No valid moves left after move. Ending game as TRAPPED/NO_PATH_FOUND."
            )
            self.game.game_state.record_game_end("NO_PATH_FOUND")
            return

        # Check if apple is reachable from new post-move head position
        post_head = post_state.get_head_position()
        post_apple = post_state.get_apple_position()
        post_snake_positions = post_state.get_snake_positions()
        obstacles = set(tuple(p) for p in post_snake_positions[:-1])

        # Simple BFS pathfinding implementation
        path_to_apple = bfs_pathfind(
            post_head, post_apple, obstacles, post_state.get_grid_size()
        )
        if path_to_apple is None:
            print_error(
                "Apple unreachable after move. Ending game as NO_PATH_FOUND."
            )
            self.game.game_state.record_game_end("NO_PATH_FOUND")
    
    def _finalize_game_specific_rounds(self) -> None:
        """Finalize heuristics-specific rounds data with validation."""
        # --- FAIL-FAST: Ensure explanations, metrics, and moves are aligned ---
        explanations = getattr(self.game.game_state, "move_explanations", [])
        metrics = getattr(self.game.game_state, "move_metrics", [])
        dataset_game_states = self.game.game_state.generate_game_summary().get(
            "dataset_game_states", {}
        )
        # Count pre-move states for rounds 1..N (clean Task-0 pattern)
        n_states = len(
            [k for k in dataset_game_states.keys() if str(k).isdigit() and int(k) >= 1]
        )
        n_expl = len(explanations)
        n_metrics = len(metrics)
        if not (n_expl == n_metrics == n_states):
            print_error("[SSOT] FAIL-FAST: Misalignment detected after game!")
            print_error(
                f"[SSOT] Explanations: {n_expl}, Metrics: {n_metrics}, Pre-move states (rounds 1+): {n_states}"
            )
            print_error(
                f"[SSOT] dataset_game_states keys: {list(dataset_game_states.keys())}"
            )
            raise RuntimeError(
                f"[SSOT] Misalignment: explanations={n_expl}, metrics={n_metrics}, pre-move states (rounds 1+): {n_states}"
            )


    
    def _add_task_specific_game_data(self, game_data: Dict[str, Any], game_duration: float) -> None:
        """Add heuristics-specific game data."""
        # Add algorithm name for heuristics
        game_data["algorithm"] = self.algorithm_name
        
        # Add explanations and metrics for dataset generation (v0.04 enhancement)
        game_data["move_explanations"] = getattr(
            self.game.game_state, "move_explanations", []
        )
        game_data["move_metrics"] = getattr(self.game.game_state, "move_metrics", []
        )
    
    def _finalize_task_specific(self, game_data: Dict[str, Any], game_duration: float) -> None:
        """Add heuristics-specific finalization - update datasets."""
        # Update datasets automatically (heuristics-specific feature)
        self._update_datasets_incrementally([game_data])
    
    def _update_task_specific_stats(self, game_duration: float) -> None:
        """Update heuristics-specific session statistics."""
        # Track game rounds for heuristics
        self.game_rounds.append(self.round_count)



    def _add_task_specific_summary_data(self, summary: Dict[str, Any]) -> None:
        """Add heuristics-specific data to session summary."""
        summary["algorithm"] = self.algorithm_name
        summary["round_counts"] = self.game_rounds
    
    def _display_task_specific_summary(self, summary: Dict[str, Any]) -> None:
        """Display heuristics-specific summary information."""
        print_info(f"🧠 Algorithm: {self.algorithm_name}")
        print_info(f"📈 Game scores: {summary['game_scores']}")
        print_info(f"🔢 Round counts: {self.game_rounds}")
    
    def _finalize_session(self) -> None:
        """Finalize heuristics session - close dataset files."""
        # Close dataset generator files
        if self.dataset_generator:
            if self.dataset_generator._csv_writer:
                self.dataset_generator._csv_writer[1].close()
                print_success("CSV dataset saved")
            if self.dataset_generator._jsonl_fh:
                self.dataset_generator._jsonl_fh.close()
                print_success("JSONL dataset saved")



    def _update_datasets_incrementally(self, games_data: List[Dict[str, Any]]) -> None:
        """Update datasets incrementally after each game."""
        if not self.dataset_generator:
            return

        for game_data in games_data:
            # game_count is already incremented
            game_data["game_number"] = self.game_count
            self.dataset_generator._process_single_game(game_data)

    def _configure_controller(self) -> None:
        """Configure the game controller for heuristics-specific needs."""
        if self.game_controller:
            # Add any heuristics-specific controller configuration here
            pass
