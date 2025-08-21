# TODO: In the attached "extensions/heuristics-v0.04/game_manager.py", I have a bunch of TODOs. Please go through them and fix them. Basically, you might want to make writting game_manager.py for extensions much easier (not only for heuristics-v0.04, but also for other extensions). However, keep in mind that writing jsonl files is specific to heuristics-v0.04. Hence, state_management.py (PRE/POST move states) is specific to heuristics-v0.04. For this time, you are allowed to adjust Task0 codebase. But, don't change any functionality of Task0 and heuristics-v0.04. Attached md files can be useful for you, though some of them are outdated. You might want to update core.md file after you are finished.


from __future__ import annotations
import sys
import os
from pathlib import Path

# Fix UTF-8 encoding issues on Windows
# This ensures that all subprocesses and file operations use UTF-8
# All file operations (CSV, JSONL, JSON) in v0.04 use UTF-8 encoding for cross-platform compatibility
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

"""
Heuristic Game Manager 
----------------

Session management for multi-algorithm heuristic agents.

Evolution from v0.01: This module demonstrates how to extend the simple
proof-of-concept to support multiple algorithms using factory patterns.
Shows natural software progression while maintaining the same base architecture.

Design Philosophy:
- Extends BaseGameManager (inherits all generic session management)
- Uses HeuristicGameLogic for game mechanics
- Factory pattern for algorithm selection (v0.02 enhancement)
- No LLM dependencies (no token stats, no continuation mode)
- Simplified logging (no Task-0 replay compatibility as requested)

Evolution from v0.03: Adds language-rich move explanations and JSONL dataset generation while retaining multi-algorithm flexibility.

v0.04 Enhancement: Supports incremental JSONL/CSV dataset updates after each game
to provide real-time dataset growth visibility.

Design Patterns:
- Template Method: Inherits base session management structure
- Factory Pattern: Uses HeuristicGameLogic for game logic
- Strategy Pattern: Pluggable heuristic algorithms
- Observer Pattern: Game state changes trigger dataset updates
"""

# Ensure project root is set and properly configured
from utils.path_utils import ensure_project_root

ensure_project_root()

import argparse
import time
from datetime import datetime
from typing import Optional, List, Dict, Any
import json
import os
import copy

# Import from project root using absolute imports
from utils.print_utils import print_info, print_warning, print_success, print_error
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

# Type alias for any heuristic agent (from agents package)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

# JSON serialization moved to BFSAgent for SSOT compliance


class HeuristicGameManager(BaseGameManager):
    """
    Multi-algorithm session manager for heuristics v0.04.

    Evolution from v0.01:
    - Factory pattern for algorithm selection (was: hardcoded BFS)
    - Support for 7 different heuristic algorithms
    - Improved error handling and verbose mode
    - Simplified logging without Task-0 replay compatibility

    v0.04 Enhancement:
    - Automatic JSONL/CSV/summary.json updates after each game
    - Real-time dataset growth visibility
    - No optional parameters - updates always happen

    Design Patterns:
    - Template Method: Inherits base session management structure
    - Factory Pattern: Uses HeuristicGameLogic for game logic
    - Strategy Pattern: Pluggable heuristic algorithms (v0.02 enhancement)
    - Abstract Factory: Algorithm creation based on configuration
    - Observer Pattern: Game state changes trigger dataset updates
    """

    # Use heuristic-specific game logic
    GAME_LOGIC_CLS = HeuristicGameLogic

    def __init__(self, args: argparse.Namespace, agent: Any) -> None:
        """Initialize heuristic game manager with automatic dataset update capabilities.
        
        Args:
            args: Command line arguments namespace
            agent: Required agent instance (SSOT enforcement)
        """
        super().__init__(args)

        # Heuristic-specific attributes
        self.algorithm_name: str = getattr(args, "algorithm", DEFAULT_ALGORITHM)
        # Shared agent instance (SSOT). Must be provided.
        self.agent: Any = agent
        self.verbose: bool = getattr(args, "verbose", False)

        # Heuristics-specific session data (base class handles common stats)
        self.game_rounds: List[int] = []

        # Dataset update tracking (always enabled)
        self.dataset_generator: Optional[DatasetGenerator] = None

        print_info(f"[HeuristicGameManager] Initialized for {self.algorithm_name}")

    def initialize(self) -> None:
        """Initialize the game manager with automatic dataset update capabilities."""
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
        """Setup logging directory for **extension mode**.

        CRITICAL: All heuristic extensions write their outputs under:

            ROOT/logs/extensions/datasets/grid-size-N/<extension>_v<version>_<timestamp>/

        This follows the standardized dataset folder structure defined in
        docs/extensions-guideline/datasets-folder.md for consistency across
        all extensions and grid sizes.

        Directory pattern:
            logs/extensions/datasets/grid-size-{N}/heuristics_v0.04_{timestamp}/
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        grid_size = getattr(self.args, "grid_size", 10)

        # Follow standardized dataset folder structure
        # Reference: docs/extensions-guideline/datasets-folder.md
        dataset_folder = f"heuristics_v0.04_{timestamp}"
        base_dir = os.path.join(
            EXTENSIONS_LOGS_DIR, "datasets", f"grid-size-{grid_size}", dataset_folder
        )

        # Algorithm-specific subdirectory (all files for one run live here)
        self.log_dir = os.path.join(base_dir, self.algorithm_name.lower())

        # Create directories
        os.makedirs(self.log_dir, exist_ok=True)

    def _setup_agent(self) -> None:
        """
        Factory method to create appropriate agent based on algorithm selection.
        """
        try:
            # Require agent to be provided (SSOT enforcement)
            if self.agent is None:
                raise RuntimeError(
                    f"Agent is required for HeuristicGameManager. Algorithm '{self.algorithm_name}' needs an agent instance."
                )

            # Validate that the provided agent matches requested algorithm
            provided_name = getattr(self.agent, "algorithm_name", None)
            if provided_name and provided_name.upper() != self.algorithm_name.upper():
                raise RuntimeError(
                    f"Provided agent algorithm '{provided_name}' does not match requested '{self.algorithm_name}'."
                )

            if not self.agent:
                available_algorithms = get_available_algorithms()
                raise ValueError(
                    f"Unknown algorithm: {self.algorithm_name}. Available: {available_algorithms}"
                )

            if self.verbose:
                print_info(
                    f"🏭 Using {self.agent.__class__.__name__} for {self.algorithm_name}"
                )
        except Exception:
            raise

    def _setup_dataset_generator(self) -> None:
        """Setup dataset generator for automatic updates."""
        # Pass current agent instance to allow agent-level control over prompt/completion formatting
        self.dataset_generator = DatasetGenerator(
            self.algorithm_name, Path(self.log_dir), agent=self.agent
        )

        # Open CSV and JSONL files for writing
        self.dataset_generator._open_csv()
        self.dataset_generator._open_jsonl()

        print_info(
            "[HeuristicGameManager] Dataset generator initialized for automatic updates"
        )

    def run(self) -> None:
        """Run the heuristic game session with automatic dataset updates."""
        # Start session tracking using base class
        self.start_session()
        
        print_success("✅ 🚀 Starting heuristics v0.04 session...")
        print_info(f"📊 Target games: {self.args.max_games}")
        print_info(f"🧠 Algorithm: {self.algorithm_name}")
        print_info("")

        # Run games
        for game_id in range(1, self.args.max_games + 1):
            print_info(f"🎮 Game {game_id}")
            # Run single game
            game_duration = self._run_single_game()
            # Finalize game and update datasets
            self._finalize_game(game_duration)
            # Display results
            self.display_game_results(game_duration)
            # Check if we should continue
            if game_id < self.args.max_games:
                print_info("")  # Spacer between games

        # End session and generate comprehensive summary
        self.end_session()

        print_success("✅ ✅ Heuristics v0.04 session completed!")
        if hasattr(self, "log_dir") and self.log_dir:
            print_info(f"📂 Logs: {self.log_dir}")

    def _run_single_game(self) -> float:
        """Run a single game and return its duration using robust state management."""
        start_time = time.time()

        # Initialize game
        self.game.reset()

        # Initialize state manager for robust pre/post state separation
        state_manager = StateManager()

        # Validate initial game state (no round creation yet - rounds start with actual moves)
        initial_raw_state = self.game.get_state_snapshot()
        initial_pre_state = state_manager.create_pre_move_state(initial_raw_state)

        # Fail-fast: Validate initial game state
        if not initial_pre_state.get_snake_positions():
            print_error(
                f"[FAIL-FAST] Initial game state has no snake positions: {initial_raw_state}"
            )
            raise RuntimeError(
                "[SSOT] Initial game state has no snake positions - game reset failed"
            )

        # Ensure round manager is available but don't create any rounds yet
        # Rounds will be created starting from Round 1 when the first move is made
        if not (
            hasattr(self.game.game_state, "round_manager")
            and self.game.game_state.round_manager
        ):
            raise RuntimeError("[SSOT] Round manager missing after game reset.")

        # Game loop with robust state management
        steps = 0
        while not self.game.game_over:
            steps += 1

            # Sync previous round's data before starting a new round
            if (
                hasattr(self.game.game_state, "round_manager")
                and self.game.game_state.round_manager
                and self.round_count > 0
            ):
                self.game.game_state.round_manager.sync_round_data()

            # Start new round for each move
            self.start_new_round(f"{self.algorithm_name} pathfinding")

            # --- ROBUST PRE-MOVE STATE MANAGEMENT ---
            # Create immutable pre-move state from current game state
            raw_pre_state = self.game.get_state_snapshot()
            pre_state = state_manager.create_pre_move_state(raw_pre_state)

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

            # --- AGENT DECISION MAKING WITH IMMUTABLE STATE ---
            # Extract state dict for agent compatibility (safe because original was deep-copied)
            agent_state_dict = dict(pre_state.game_state)

            # Fail-fast: Ensure game logic has required method
            if not hasattr(self.game, "get_next_planned_move_with_state"):
                raise RuntimeError(
                    "[SSOT] Game logic missing get_next_planned_move_with_state method - required for SSOT compliance"
                )

            # Get move and explanation using immutable pre-move state
            move, explanation = self.game.get_next_planned_move_with_state(
                agent_state_dict, return_explanation=True
            )

            # --- FAIL-FAST: VALIDATE EXPLANATION HEAD CONSISTENCY ---
            if not validate_explanation_head_consistency(pre_state, explanation):
                raise RuntimeError(
                    "[SSOT] FAIL-FAST: Explanation head position does not match pre-move state"
                )

            # --- FAIL-FAST SSOT VALIDATION ---
            # Validate move against pre-move state using centralized utilities
            head = pre_state.get_head_position()
            body_positions = pre_state.get_snake_positions()
            manhattan_distance = calculate_manhattan_distance(
                dict(pre_state.game_state)
            )
            valid_moves = calculate_valid_moves_ssot(dict(pre_state.game_state))

            if move == "NO_PATH_FOUND":
                if valid_moves:
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
                break

            if move not in valid_moves:
                raise RuntimeError(
                    f"SSOT violation: agent move '{move}' not in valid moves {valid_moves}"
                )

            # --- APPLY MOVE AND CREATE POST-MOVE STATE ---
            # Apply move to game logic
            self.game.make_move(move)

            # Create post-move state from game state after move
            raw_post_state = self.game.get_state_snapshot()
            post_state = state_manager.create_post_move_state(
                pre_state, move, raw_post_state
            )

            # --- POST-MOVE VALIDATION ---
            # Check if there are any valid moves left after move
            post_valid_moves = calculate_valid_moves_ssot(dict(post_state.game_state))
            if not post_valid_moves:
                print_error(
                    "[DEBUG] No valid moves left after move. Ending game as TRAPPED/NO_PATH_FOUND."
                )
                self.game.game_state.record_game_end("NO_PATH_FOUND")
                break

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
                break

            # Update display if GUI is enabled
            if hasattr(self.game, "update_display"):
                self.game.update_display()

            # Check max steps after move execution using limits manager
            if self.limits_manager.should_end_game(steps, "MAX_STEPS"):
                print_error("[DEBUG] Max steps reached. Current game state:")
                print_error(f"[DEBUG] Head: {self.game.head_position}")
                print_error(f"[DEBUG] Snake: {self.game.snake_positions}")
                print_error(f"[DEBUG] Apple: {self.game.apple_position}")
                print_error(f"[DEBUG] Score: {self.game.game_state.score}")
                print_error(f"[DEBUG] Steps: {self.game.game_state.steps}")
                print_error(f"[DEBUG] Game over: {self.game.game_over}")
                print_error(
                    f"[DEBUG] Game end reason: {getattr(self.game.game_state, 'game_end_reason', 'None')}"
                )

                # Record the final move before ending the game
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

                self.game.game_state.record_game_end("MAX_STEPS_REACHED")
                break

        # Calculate duration
        game_duration = time.time() - start_time

        # --- FAIL-FAST: Ensure final step is recorded ---
        final_steps = self.game.game_state.steps
        final_rounds = (
            self.game.game_state.round_manager.round_count
            if hasattr(self.game.game_state, "round_manager")
            else 0
        )
        if final_steps != steps:
            print_error(
                f"[SSOT] FAIL-FAST: Step count mismatch! Game state shows {final_steps} steps but loop executed {steps} steps"
            )
            raise RuntimeError(
                f"[SSOT] Step count mismatch: game_state.steps={final_steps}, loop_steps={steps}"
            )

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

        return game_duration

    def _finalize_game(self, game_duration: float) -> None:
        """Finalize game and update datasets automatically."""
        # Use base class finalize_game method
        self.finalize_game(game_duration)
    
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



    # TODO: this one seems to be heuristics-v0.04 specific. Hence is not shared.
    def _update_datasets_incrementally(self, games_data: List[Dict[str, Any]]) -> None:
        """Update datasets incrementally after each game."""
        if not self.dataset_generator:
            return

        for game_data in games_data:
            # game_count is already incremented
            game_data["game_number"] = self.game_count
            self.dataset_generator._process_single_game(game_data)

    def setup_game(self) -> None:
        """Create game logic and optional GUI interface with correct grid size."""
        # Get grid size from command line arguments
        grid_size = getattr(self.args, "grid_size", 10)

        # Use the specified game logic class with correct grid size
        self.game = self.GAME_LOGIC_CLS(grid_size=grid_size, use_gui=self.use_gui)

        # Attach GUI if visual mode is requested
        if self.use_gui:
            # Lazy import keeps headless extensions free of pygame.
            from gui.game_gui import GameGUI  # noqa: WPS433 – intentional local import

            gui = GameGUI()
            # Ensure GUI pixel scaling matches the *actual* game grid size
            if hasattr(self.game, "grid_size"):
                gui.resize(self.game.grid_size)  # auto-adjust cell size & grid lines
            self.game.set_gui(gui)
