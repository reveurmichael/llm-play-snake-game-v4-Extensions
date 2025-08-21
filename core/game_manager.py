"""Session management for Snake game tasks (0-5).

This module implements a clean, future-proof architecture where:
- BaseGameManager provides all generic functionality for Tasks 1-5
- GameManager (Task-0) adds only LLM-specific features

Design Philosophy:
- Tasks 0-5 inherit BaseGameManager directly
- Each task gets exactly what it needs, nothing more
- Clean separation of concerns, no historical baggage

Testability & head-less execution
---------------------
Now *pygame* is **lazy-loaded**.  We only import/initialise
it inside :pyclass:`BaseGameManager` **when** the caller explicitly requests a
GUI session (``use_gui=True``, default for Task-0).  This brings two major
benefits:

1. **Head-less CI pipelines** – unit-tests can exercise the full planning &
   game-logic stack on platforms where SDL/pygame is unavailable.
2. **Lower coupling / faster import time** – every non-visual extension
   (heuristics, RL, dataset generation, …) remains free of the heavyweight
   dependency.

The pattern uses ``importlib.import_module("pygame")`` and stores the module
on :pyattr:`BaseGameManager._pygame`.  All downstream code gates GUI calls via
``self.use_gui`` **and** ``self._pygame is not None``.
"""

from __future__ import annotations

import os
import importlib
from typing import TYPE_CHECKING, List, Optional, Dict, Any
from collections import defaultdict

from colorama import Fore

# Core components - all future-ready
from core.game_logic import BaseGameLogic, GameLogic
from core.game_loop import run_game_loop
from config.ui_constants import TIME_DELAY, TIME_TICK
from config.game_constants import MAX_STEPS_ALLOWED

# LLM components - only for Task-0
from llm.client import LLMClient

# Utilities - organized by purpose
from core.game_stats_manager import GameStatsManager
# Continuation utilities imported locally to avoid circular dependency
from core.game_manager_helper import GameManagerHelper

# Agent protocol for all tasks
from core.game_agents import BaseAgent

if TYPE_CHECKING:
    import argparse


# -------------------
# BASE CLASS FOR ALL TASKS (0-5) - Pure Generic Implementation
# -------------------


class BaseGameManager:
    """Generic session manager for all Snake game tasks.
    
    This class contains ONLY attributes and methods that are useful
    across Tasks 1-5. No LLM-specific code, no legacy patterns.
    
    Perfect for:
    - Task-1 (Heuristics): BFS, A*, Hamiltonian cycles
    - Task-2 (Supervised): Neural network training on game data  
    - Task-3 (Reinforcement): DQN, PPO, actor-critic agents
    - Task-4 (LLM Fine-tuning): Custom fine-tuned models
    - Task-5 (Distillation): Model compression techniques
    """

    # Factory hook - subclasses specify their game logic type
    GAME_LOGIC_CLS = BaseGameLogic

    def __init__(self, args: "argparse.Namespace") -> None:
        """Initialize generic session state for any task type."""
        self.args = args

        # -------------------
        # Core session metrics (used by ALL tasks)
        # -------------------
        self.game_count: int = 0
        self.round_count: int = 1
        self.total_score: int = 0
        self.total_steps: int = 0
        self.total_rounds: int = 0

        # Per-game data tracking
        self.game_scores: List[int] = []
        self.round_counts: List[int] = []
        self.current_game_moves: List[str] = []

        # Error tracking (generic across all algorithms)
        self.valid_steps: int = 0
        self.invalid_reversals: int = 0
        self.consecutive_invalid_reversals: int = 0
        self.consecutive_no_path_found: int = 0
        self.no_path_found_steps: int = 0
        self.last_no_path_found: bool = False

        # -------------------
        # Game limits management (used by ALL tasks)
        # -------------------
        from core.game_limits_manager import create_limits_manager
        self.limits_manager = create_limits_manager(args)

        # -------------------
        # Game state management (used by ALL tasks)
        # -------------------
        self.game: Optional[BaseGameLogic] = None
        self.game_active: bool = True
        self.need_new_plan: bool = True
        self.running: bool = True
        self._first_plan: bool = True  # Track first planning cycle for round management

        # -------------------
        # Visualization & timing (used by ALL tasks)
        # -------------------
        self.use_gui: bool = not getattr(args, "no_gui", False)
        self.pause_between_moves: float = getattr(args, "pause_between_moves", 0.0)
        self.auto_advance: bool = getattr(args, "auto_advance", False)

        # Lazy-load pygame ONLY when GUI is requested.
        # This keeps head-less extensions (heuristics, RL, …) completely
        # free of the heavyweight SDL dependency and avoids opening any
        # graphical window when ``use_gui`` is False.

        self._pygame = None  # type: ignore[assignment]

        if self.use_gui:
            try:
                # Import inside the branch so that *headless* runs never even
                # attempt to import pygame.
                self._pygame = importlib.import_module("pygame")
                self.clock = self._pygame.time.Clock()  # type: ignore[attr-defined]
                self.time_delay = TIME_DELAY
                self.time_tick = TIME_TICK
            except ModuleNotFoundError as exc:  # pragma: no cover – dev machines without pygame
                raise RuntimeError(
                    "GUI mode requested but pygame is not installed. "
                    "Install it or re-run with --no-gui."
                ) from exc
        else:
            # Headless – initialise dummies so the rest of the code can rely on them.
            self.clock = None
            self.time_delay = 0
            self.time_tick = 0

        # -------------------
        # Logging infrastructure (used by ALL tasks)
        # -------------------
        self.log_dir: Optional[str] = None

        # -------------------
        # Session management (used by ALL tasks)
        # -------------------
        self.session_start_time: Optional["datetime"] = None
        self.game_steps: List[int] = []

    # -------------------
    # CORE LIFECYCLE METHODS - All tasks implement these
    # -------------------

    def initialize(self) -> None:
        """Initialize the task-specific components.
        
        Override in subclasses to set up:
        - Logging directories
        - Models/algorithms  
        - Dataset connections
        - Agent configurations
        """
        raise NotImplementedError("Subclasses must implement initialize()")

    def run(self) -> None:
        """Execute the main task workflow.
        
        Override in subclasses to implement:
        - Training loops (RL, Supervised)
        - Evaluation protocols (Heuristics)  
        - Fine-tuning pipelines (LLM tasks)
        """
        raise NotImplementedError("Subclasses must implement run()")

    # -------------------
    # GENERIC GAME SETUP - Reusable across all tasks
    # -------------------

    def setup_game(self) -> None:
        """Create game logic and optional GUI interface."""
        # Use the specified game logic class (BaseGameLogic by default)
        self.game = self.GAME_LOGIC_CLS(use_gui=self.use_gui)

        # Attach GUI if visual mode is requested
        if self.use_gui:
            # Lazy import keeps headless extensions free of pygame.
            from gui.game_gui import GameGUI  # noqa: WPS433 – intentional local import
            gui = GameGUI()
            # Ensure GUI pixel scaling matches the *actual* game grid size
            if hasattr(self.game, "grid_size"):
                gui.resize(self.game.grid_size)  # auto-adjust cell size & grid lines
            self.game.set_gui(gui)

    def get_pause_between_moves(self) -> float:
        """Get pause duration between moves.
        
        Returns:
            Pause time in seconds (0.0 for no-GUI mode)
        """
        return self.pause_between_moves if self.use_gui else 0.0


    # -------------------
    # ROUND MANAGEMENT - Generic for all planning-based tasks
    # -------------------

    def start_new_round(self, reason: str = "") -> None:
        """Begin a new planning round.
        
        All tasks use rounds to track planning cycles:
        - Heuristics: Each path-finding attempt
        - RL: Each action selection
        - LLM: Each prompt/response cycle
        """
        if not self.game or not hasattr(self.game, "game_state"):
            return

        game_state = self.game.game_state
        apple_pos = getattr(self.game, "apple_position", None)
        game_state.round_manager.start_new_round(apple_pos)

        # Sync public counter
        self.round_count = game_state.round_manager.round_count
        game_state.round_manager.sync_round_data()

        # Console feedback for long experiments
        if reason:
            print(Fore.BLUE + f"📊 Round {self.round_count} started ({reason})")
        else:
            print(Fore.BLUE + f"📊 Round {self.round_count} started")

    def increment_round(self, reason: str = "") -> None:
        """Increment the round counter and flush the current round data.
        
        This method is called to finish the current round and start a new one.
        Generic across all tasks that use planning rounds.
        
        Args:
            reason: Optional description of why the round is incrementing
        """
        if not self.game or not hasattr(self.game, "game_state"):
            return

        # Prevent double-increment during certain conditions
        if hasattr(self, "awaiting_plan") and self.awaiting_plan:
            return

        game_state = self.game.game_state
        
        # Flush current round data before incrementing
        game_state.round_manager.flush_buffer()
        
        # Update session-level round tracking
        self.round_counts.append(self.round_count)
        
        # Start new round with current apple position
        apple_pos = getattr(self.game, "apple_position", None)
        game_state.round_manager.start_new_round(apple_pos)
        
        # Sync public counter
        self.round_count = game_state.round_manager.round_count
        game_state.round_manager.sync_round_data()

        # Console feedback for long experiments  
        if reason:
            print(Fore.BLUE + f"📊 Round {self.round_count} started ({reason})")
        else:
            print(Fore.BLUE + f"📊 Round {self.round_count} incremented")

    def finish_round(self, reason: str = "") -> None:  # noqa: D401
        """Finalize the **current** round without starting a new one.

        This helper is called by :pyclass:`core.game_loop.GameLoop` when the
        pre-computed *plan* has been fully executed (i.e. our move queue is
        empty).  Its sole responsibility is to **persist** whatever is left in
        the volatile :class:`core.game_rounds.RoundManager` buffer so that JSON
        outputs and in-memory statistics remain consistent.

        Crucially, the method deliberately **does not** bump
        :pyattr:`round_count` – that is handled at the *beginning* of the next
        planning cycle via :meth:`increment_round`.  This design keeps filenames
        like ``game_2_round_5_prompt.txt`` in perfect sync with the data stored
        in ``rounds_data`` while avoiding the off-by-one issues observed in the
        legacy procedural loop.

        Args:
            reason: Optional text explaining why the round is being closed.
                Primarily useful for verbose logging during long experiments.
        """
        if not self.game or not hasattr(self.game, "game_state"):
            return

        # Flush any pending data for the *current* round.
        game_state = self.game.game_state
        game_state.round_manager.flush_buffer()
        game_state.round_manager.sync_round_data()

        if reason:
            print(Fore.BLUE + f"📁 Round {self.round_count} finished ({reason})")
        else:
            # Lightweight marker; keeps console noise minimal during large runs.
            print(Fore.BLUE + f"📁 Round {self.round_count} finished")

    # -------------------
    # LOGGING INFRASTRUCTURE - Used by all tasks
    # -------------------

    def setup_logging(self, base_dir: str, task_name: str) -> None:
        """Set up logging directory structure.
        
        Args:
            base_dir: Base logs directory (e.g., "logs/")
            task_name: Task identifier (e.g., "heuristics", "rl", "llm")
        """
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = os.path.join(base_dir, f"{task_name}_{timestamp}")
        os.makedirs(self.log_dir, exist_ok=True)

    def save_session_summary(self) -> None:
        """Save comprehensive session summary with extension customization."""
        if not self.log_dir:
            return
            
        # Generate comprehensive session summary
        summary = self.generate_session_summary()
        
        # Save to file
        self.save_session_summary_to_file(summary)
        
        # Display to console
        self.display_session_summary(summary)
    
    def generate_session_summary(self) -> Dict[str, Any]:
        """Generate comprehensive session summary with extension hooks.
        
        Returns:
            Dictionary containing session summary data
        """
        from datetime import datetime
        
        # Calculate session duration
        session_duration = 0.0
        if hasattr(self, 'session_start_time'):
            session_duration = (datetime.now() - self.session_start_time).total_seconds()
        
        # Calculate derived statistics
        total_steps = sum(getattr(self, 'game_steps', []))
        total_rounds = sum(getattr(self, 'round_counts', []))
        
        # Base session summary structure
        summary = {
            "session_timestamp": getattr(self, 'session_start_time', datetime.now()).strftime("%Y%m%d_%H%M%S"),
            "total_games": len(self.game_scores),
            "total_score": self.total_score,
            "average_score": self.total_score / len(self.game_scores) if self.game_scores else 0.0,
            "total_steps": total_steps,
            "total_rounds": total_rounds,
            "session_duration_seconds": round(session_duration, 2),
            "score_per_step": self.total_score / total_steps if total_steps > 0 else 0.0,
            "score_per_round": self.total_score / total_rounds if total_rounds > 0 else 0.0,
            "game_scores": self.game_scores,
            "game_steps": getattr(self, 'game_steps', []),
            "round_counts": self.round_counts,
            "configuration": self._get_base_configuration(),
        }
        
        # Hook for extensions to add task-specific summary data
        self._add_task_specific_summary_data(summary)
        
        return summary
    
    def _get_base_configuration(self) -> Dict[str, Any]:
        """Get base configuration data for session summary.
        
        Returns:
            Dictionary containing base configuration
        """
        return {
            "grid_size": getattr(self.args, "grid_size", 10),
            "max_games": getattr(self.args, "max_games", 1),
            "max_steps": getattr(self.args, "max_steps", 1000),
            "use_gui": not getattr(self.args, "no_gui", False),
            "verbose": getattr(self.args, "verbose", False),
        }
    
    def _add_task_specific_summary_data(self, summary: Dict[str, Any]) -> None:
        """Hook for extensions to add task-specific summary data.
        
        Override in subclasses to add extension-specific fields to session summary.
        
        Args:
            summary: Session summary dictionary to modify
        """
        # Base implementation does nothing - extensions override this
        pass
    
    def save_session_summary_to_file(self, summary: Dict[str, Any]) -> None:
        """Save session summary to JSON file.
        
        Args:
            summary: Session summary dictionary to save
        """
        import json
        
        summary_file = os.path.join(self.log_dir, "summary.json")
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
    
    def display_session_summary(self, summary: Dict[str, Any]) -> None:
        """Display session summary to console.
        
        Args:
            summary: Session summary dictionary to display
        """
        from utils.print_utils import print_info, print_success
        
        print_success("📊 Session Summary")
        print_info("=" * 50)
        print_info(f"🎮 Total games: {summary['total_games']}")
        print_info(f"🏆 Total score: {summary['total_score']}")
        print_info(f"📈 Average score: {summary['average_score']:.1f}")
        print_info(f"👣 Total steps: {summary['total_steps']}")
        print_info(f"🔄 Total rounds: {summary['total_rounds']}")
        print_info(f"⏱️  Session duration: {summary['session_duration_seconds']:.1f}s")
        print_info(f"⚡ Score per step: {summary['score_per_step']:.3f}")
        print_info(f"🎯 Score per round: {summary['score_per_round']:.3f}")
        
        # Hook for extensions to add task-specific display
        self._display_task_specific_summary(summary)
        
        print_info("=" * 50)
    
    def _display_task_specific_summary(self, summary: Dict[str, Any]) -> None:
        """Hook for extensions to display task-specific summary information.
        
        Override in subclasses to add extension-specific summary display.
        
        Args:
            summary: Session summary dictionary
        """
        # Base implementation does nothing - extensions override this
        pass

    def start_session(self) -> None:
        """Initialize session tracking and logging.
        
        Call this at the beginning of any extension's run() method.
        """
        from datetime import datetime
        
        if not self.session_start_time:
            self.session_start_time = datetime.now()
        
        # Hook for extensions to add session initialization
        self._initialize_session()
    
    def _initialize_session(self) -> None:
        """Hook for extensions to initialize session-specific data.
        
        Override in subclasses to add extension-specific session initialization.
        """
        # Base implementation does nothing - extensions override this
        pass
    
    def end_session(self) -> None:
        """Finalize session and generate summary.
        
        Call this at the end of any extension's run() method.
        """
        # Save comprehensive session summary
        self.save_session_summary()
        
        # Hook for extensions to add session cleanup
        self._finalize_session()
    
    def _finalize_session(self) -> None:
        """Hook for extensions to finalize session-specific data.
        
        Override in subclasses to add extension-specific session cleanup.
        """
        # Base implementation does nothing - extensions override this
        pass

    def track_game_completion(self, game_steps: int) -> None:
        """Track completion of a single game.
        
        Args:
            game_steps: Number of steps taken in the completed game
        """
        self.game_steps.append(game_steps)

    # -------------------
    # GENERIC GAME DATA MANAGEMENT - Used by all extensions
    # -------------------

    def generate_game_data(self, game_duration: float) -> Dict[str, Any]:
        """Generate game data for logging and dataset generation.
        
        This method provides a base implementation that extensions can override
        to add task-specific data while maintaining consistency.
        
        Args:
            game_duration: Duration of the game in seconds
            
        Returns:
            Dictionary containing game data
        """
        if not self.game or not hasattr(self.game, 'game_state'):
            raise RuntimeError("Game not initialized or missing game_state")
            
        # Get base game summary from game state
        game_data = self.game.game_state.generate_game_summary()
        
        # Add common metadata
        game_data["game_number"] = self.game_count
        game_data["duration_seconds"] = round(game_duration, 2)
        
        # Hook for extensions to add task-specific data
        self._add_task_specific_game_data(game_data, game_duration)
        
        return game_data
    
    def _add_task_specific_game_data(self, game_data: Dict[str, Any], game_duration: float) -> None:
        """Hook for extensions to add task-specific game data.
        
        Override in subclasses to add extension-specific fields to game data.
        
        Args:
            game_data: Game data dictionary to modify
            game_duration: Duration of the game in seconds
        """
        # Base implementation does nothing - extensions override this
        pass

    def save_game_data(self, game_data: Dict[str, Any]) -> None:
        """Save individual game data to JSON file.
        
        Args:
            game_data: Dictionary containing game data to save
        """
        if not self.log_dir:
            return
            
        import json
        from extensions.common.utils.game_state_utils import to_serializable
        
        # Use game_count for consistent numbering (games start at 1)
        game_file = os.path.join(self.log_dir, f"game_{self.game_count}.json")
        with open(game_file, "w", encoding="utf-8") as f:
            json.dump(to_serializable(game_data), f, indent=2)

    def display_game_results(self, game_duration: float) -> None:
        """Display game results to console.
        
        Args:
            game_duration: Duration of the game in seconds
        """
        if not self.game or not hasattr(self.game, 'game_state'):
            return
            
        from utils.print_utils import print_info
        
        print_info(
            f"📊 Score: {self.game.game_state.score}, "
            f"Steps: {self.game.game_state.steps}, "
            f"Duration: {game_duration:.2f}s"
        )
        
        # Hook for extensions to add task-specific display
        self._display_task_specific_results(game_duration)
    
    def _display_task_specific_results(self, game_duration: float) -> None:
        """Hook for extensions to display task-specific results.
        
        Override in subclasses to add extension-specific result display.
        
        Args:
            game_duration: Duration of the game in seconds
        """
        # Base implementation does nothing - extensions override this
        pass

    def determine_game_end_reason(self) -> str:
        """Determine why the game ended using uniform END_REASON_MAP.
        
        Returns:
            Canonical end reason key from END_REASON_MAP
        """
        from config.game_constants import END_REASON_MAP
        
        if not self.game or not hasattr(self.game, 'game_state'):
            return "UNKNOWN"
            
        game_state = self.game.game_state
        
        # Check if game state has explicit end reason
        if hasattr(game_state, "game_end_reason") and game_state.game_end_reason:
            raw_reason = game_state.game_end_reason
        else:
            # Fallback logic based on game state
            if game_state.steps >= getattr(game_state, 'max_steps', MAX_STEPS_ALLOWED):
                raw_reason = "MAX_STEPS_REACHED"
            elif hasattr(game_state, 'max_score') and game_state.score >= game_state.max_score:
                raw_reason = "MAX_SCORE_REACHED"
            else:
                raw_reason = "UNKNOWN"
        
        # Validate against END_REASON_MAP
        if raw_reason not in END_REASON_MAP:
            from utils.print_utils import print_warning
            print_warning(f"Unknown end reason '{raw_reason}', defaulting to 'UNKNOWN'")
            return "UNKNOWN"
            
        return raw_reason

    def finalize_game(self, game_duration: float) -> None:
        """Finalize game processing including data generation and saving.
        
        This method provides a complete workflow for game finalization that
        extensions can use or override as needed.
        
        Args:
            game_duration: Duration of the game in seconds
        """
        # Increment game count
        self.game_count += 1
        
        # Set game number in game state for consistency
        if self.game and hasattr(self.game, 'game_state'):
            self.game.game_state.game_number = self.game_count
        
        # Generate and save game data
        game_data = self.generate_game_data(game_duration)
        self.save_game_data(game_data)
        
        # Update session statistics
        self.update_session_stats(game_duration)
        
        # Hook for extensions to add custom finalization
        self._finalize_task_specific(game_data, game_duration)
    
    def _finalize_task_specific(self, game_data: Dict[str, Any], game_duration: float) -> None:
        """Hook for extensions to add task-specific finalization.
        
        Override in subclasses to add extension-specific finalization logic.
        
        Args:
            game_data: Generated game data
            game_duration: Duration of the game in seconds
        """
        # Base implementation does nothing - extensions override this
        pass

    def update_session_stats(self, game_duration: float) -> None:
        """Update session-level statistics.
        
        Args:
            game_duration: Duration of the game in seconds
        """
        if not self.game or not hasattr(self.game, 'game_state'):
            return
            
        # Update core statistics
        self.total_score += self.game.game_state.score
        self.total_steps += self.game.game_state.steps
        self.total_rounds += self.round_count
        
        # Update per-game tracking
        self.game_scores.append(self.game.game_state.score)
        self.round_counts.append(self.round_count)
        
        # Track game completion for session management
        self.track_game_completion(self.game.game_state.steps)
        
        # Hook for extensions to add custom stats
        self._update_task_specific_stats(game_duration)
    
    def _update_task_specific_stats(self, game_duration: float) -> None:
        """Hook for extensions to update task-specific statistics.
        
        Override in subclasses to add extension-specific statistics.
        
        Args:
            game_duration: Duration of the game in seconds
        """
        # Base implementation does nothing - extensions override this
        pass

    # -------------------
    # COMPREHENSIVE ROUNDS MANAGEMENT - Used by all extensions
    # -------------------

    def _initialize_game_rounds(self) -> None:
        """Initialize rounds tracking for the current game.
        
        This method sets up round management and prepares for round-by-round tracking.
        """
        # Reset round counter for new game
        self.round_count = 1
        
        # Hook for extensions to initialize game-specific rounds data
        self._initialize_game_specific_rounds()
    
    def _initialize_game_specific_rounds(self) -> None:
        """Hook for extensions to initialize game-specific rounds data.
        
        Override in subclasses to add extension-specific rounds initialization.
        """
        # Base implementation does nothing - extensions override this
        pass
    
    def _finalize_game_rounds(self) -> None:
        """Finalize rounds tracking for the completed game.
        
        This method ensures all round data is properly saved and synchronized.
        """
        # Ensure final round data is flushed
        if self.game and hasattr(self.game, 'game_state') and hasattr(self.game.game_state, 'round_manager'):
            self.game.game_state.round_manager.flush_buffer()
            self.game.game_state.round_manager.sync_round_data()
        
        # Hook for extensions to finalize game-specific rounds data
        self._finalize_game_specific_rounds()
    
    def _finalize_game_specific_rounds(self) -> None:
        """Hook for extensions to finalize game-specific rounds data.
        
        Override in subclasses to add extension-specific rounds finalization.
        """
        # Base implementation does nothing - extensions override this
        pass

    def execute_game_step(self, step_description: str = "") -> bool:
        """Execute a single game step with automatic rounds management.
        
        This method provides a template for executing individual game steps
        with automatic round tracking and state management.
        
        Args:
            step_description: Optional description of the step being executed
            
        Returns:
            True if game should continue, False if game should end
        """
        # Check if game is over
        if not self.game or self.game.game_over:
            return False
        
        # Start new round for this step
        self.start_new_round(step_description or f"Step {self.round_count}")
        
        # Get current game state
        game_state = self.game.get_state_snapshot()
        
        # Hook for extensions to process game state before move
        processed_state = self._process_game_state_before_move(game_state)
        
        # Get move decision from extension
        move = self._get_next_move(processed_state)
        
        # Validate move
        if not self._validate_move(move, processed_state):
            return False
        
        # Apply move
        try:
            self.game.make_move(move)
        except Exception as e:
            from utils.print_utils import print_warning
            print_warning(f"[BaseGameManager] Move application failed: {e}")
            return False
        
        # Update display if GUI is enabled
        if hasattr(self.game, "update_display"):
            self.game.update_display()
        
        # Hook for extensions to process game state after move
        self._process_game_state_after_move(self.game.get_state_snapshot())
        
        # Check game limits
        if self._should_end_game():
            return False
        
        return True
    
    def _process_game_state_before_move(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Hook for extensions to process game state before move decision.
        
        Args:
            game_state: Current game state dictionary
            
        Returns:
            Processed game state dictionary
        """
        # Base implementation returns state unchanged
        return game_state
    
    def _get_next_move(self, game_state: Dict[str, Any]) -> str:
        """Hook for extensions to determine the next move.
        
        Extensions must override this method to implement their move decision logic.
        
        Args:
            game_state: Current game state dictionary
            
        Returns:
            Next move as string (UP, DOWN, LEFT, RIGHT)
        """
        raise NotImplementedError("Subclasses must implement _get_next_move()")
    
    def _validate_move(self, move: str, game_state: Dict[str, Any]) -> bool:
        """Validate the proposed move against the current game state.
        
        Args:
            move: Proposed move
            game_state: Current game state
            
        Returns:
            True if move is valid, False otherwise
        """
        # Check for invalid moves
        if move in ["NO_PATH_FOUND", "INVALID", None, ""]:
            return False
        
        # Hook for extensions to add custom validation
        return self._validate_move_custom(move, game_state)
    
    def _validate_move_custom(self, move: str, game_state: Dict[str, Any]) -> bool:
        """Hook for extensions to add custom move validation.
        
        Args:
            move: Proposed move
            game_state: Current game state
            
        Returns:
            True if move is valid, False otherwise
        """
        # Base implementation accepts all non-invalid moves
        return True
    
    def _process_game_state_after_move(self, game_state: Dict[str, Any]) -> None:
        """Hook for extensions to process game state after move is applied.
        
        Args:
            game_state: Game state after move application
        """
        # Base implementation does nothing - extensions override this
        pass
    
    def _should_end_game(self) -> bool:
        """Check if the game should end based on various conditions.
        
        Returns:
            True if game should end, False otherwise
        """
        # Check basic game over condition
        if self.game and self.game.game_over:
            return True
        
        # Check limits using limits manager
        if hasattr(self, 'limits_manager') and self.limits_manager:
            steps = getattr(self.game.game_state, 'steps', 0) if self.game else 0
            if self.limits_manager.should_end_game(steps, "MAX_STEPS"):
                if self.game:
                    self.game.game_state.record_game_end("MAX_STEPS_REACHED")
                return True
        
        # Hook for extensions to add custom end conditions
        return self._should_end_game_custom()
    
    def _should_end_game_custom(self) -> bool:
        """Hook for extensions to add custom game end conditions.
        
        Returns:
            True if game should end based on extension-specific conditions
        """
        # Base implementation doesn't add any conditions
        return False

    def run_single_game(self) -> float:
        """Run a single game and return its duration.
        
        This method provides a comprehensive template for running individual games
        with automatic rounds management and state tracking.
        
        Returns:
            Duration of the game in seconds
        """
        import time
        
        start_time = time.time()
        
        # Initialize game
        if self.game:
            self.game.reset()
        
        # Initialize rounds tracking
        self._initialize_game_rounds()
        
        # Template method - extensions implement game-specific logic
        self._execute_game_loop()
        
        # Finalize rounds tracking
        self._finalize_game_rounds()
        
        return time.time() - start_time
    
    def _execute_game_loop(self) -> None:
        """Execute the main game loop.
        
        Extensions can override this method to implement custom game loops,
        or use the default step-by-step execution with _get_next_move().
        """
        # Default implementation: step-by-step execution
        while self.execute_game_step():
            pass  # Continue until game ends

    def run_game_session(self) -> None:
        """Run a complete game session with automatic session management.
        
        This method provides a complete template for running game sessions
        that extensions can use with minimal customization.
        """
        # Start session tracking
        self.start_session()
        
        # Display session start information
        self._display_session_start()
        
        # Run games
        for game_id in range(1, self.args.max_games + 1):
            from utils.print_utils import print_info
            print_info(f"🎮 Game {game_id}")
            
            # Run single game
            game_duration = self.run_single_game()
            
            # Finalize game
            self.finalize_game(game_duration)
            
            # Display results
            self.display_game_results(game_duration)
            
            # Add spacer between games
            if game_id < self.args.max_games:
                print_info("")
        
        # End session
        self.end_session()
        
        # Display completion message
        self._display_session_completion()
    
    def _display_session_start(self) -> None:
        """Display session start information.
        
        Extensions can override this to customize session start display.
        """
        from utils.print_utils import print_success, print_info
        
        extension_name = self.__class__.__name__.replace("GameManager", "")
        print_success(f"✅ 🚀 Starting {extension_name} session...")
        print_info(f"📊 Target games: {self.args.max_games}")
        print_info("")
    
    def _display_session_completion(self) -> None:
        """Display session completion information.
        
        Extensions can override this to customize session completion display.
        """
        from utils.print_utils import print_success, print_info
        
        extension_name = self.__class__.__name__.replace("GameManager", "")
        print_success(f"✅ ✅ {extension_name} session completed!")
        if hasattr(self, "log_dir") and self.log_dir:
            print_info(f"📂 Logs: {self.log_dir}")


# -------------------
# TASK-0 SPECIFIC CLASS - LLM Snake Game
# -------------------


class GameManager(BaseGameManager):
    """LLM-powered Snake game manager (Task-0).
    
    Extends BaseGameManager with LLM-specific functionality:
    - Language model clients
    - Prompt/response logging  
    - Token usage tracking
    - LLM-specific error handling
    
    This is the ONLY class that should import LLM modules.
    """

    # Use LLM-capable game logic
    GAME_LOGIC_CLS = GameLogic

    def __init__(
        self, args: "argparse.Namespace", agent: Optional[BaseAgent] = None
    ) -> None:
        """Initialize LLM-specific session."""
        super().__init__(args)

        # -------------------
        # LLM-specific counters and state
        # -------------------
        self.empty_steps: int = 0
        self.something_is_wrong_steps: int = 0
        self.consecutive_empty_steps: int = 0
        self.consecutive_something_is_wrong: int = 0
        self.awaiting_plan: bool = False
        self.skip_empty_this_tick: bool = False

        # -------------------
        # LLM performance tracking
        # -------------------
        self.time_stats: defaultdict[str, int] = defaultdict(int)
        self.token_stats: dict[str, defaultdict[str, int]] = {
            "primary": defaultdict(int),
            "secondary": defaultdict(int),
        }

        # -------------------
        # LLM infrastructure
        # -------------------
        self.llm_client: Optional[LLMClient] = None
        self.parser_provider: Optional[str] = None
        self.parser_model: Optional[str] = None
        self.agent: Optional[BaseAgent] = agent

        # LLM-specific logging directories
        self.prompts_dir: Optional[str] = None
        self.responses_dir: Optional[str] = None

    def initialize(self) -> None:
        """Initialize LLM clients and logging infrastructure."""
        helper = GameManagerHelper()
        helper.initialize_game_manager(self)

    def setup_logging(self, base_dir: str, task_name: str = "llm") -> None:
        """Set up LLM-specific logging directories."""
        super().setup_logging(base_dir, task_name)
        if self.log_dir:
            # Create LLM-specific directories using centralized utilities
            from llm.log_utils import ensure_llm_directories
            self.prompts_dir, self.responses_dir = ensure_llm_directories(self.log_dir)
            # Convert Path objects to plain strings for JSON serialisation
            self.prompts_dir = str(self.prompts_dir)
            self.responses_dir = str(self.responses_dir)

    def create_llm_client(self, provider: str, model: Optional[str] = None) -> LLMClient:
        """Create LLM client for the specified provider."""
        return LLMClient(provider=provider, model=model)

    def run(self) -> None:
        """Execute LLM game session."""
        try:
            # For continuation mode, we need to set up LLM clients but skip other initialization
            if getattr(self.args, "is_continuation", False):
                # Set up LLM clients for continuation mode
                from utils.initialization_utils import setup_llm_clients, initialize_game_state
                setup_llm_clients(self)
                initialize_game_state(self)
            else:
                # Full initialization for new sessions
                self.initialize()

            # Run game loop until completion
            while self.game_count < self.args.max_games and self.running:
                run_game_loop(self)
                if self.game_count >= self.args.max_games:
                    break

        finally:
            # Cleanup and reporting
            # Graceful SDL shutdown (only if we ever initialised it)
            if self.use_gui and self._pygame and self._pygame.get_init():
                self._pygame.quit()
            self.report_final_statistics()

    def process_events(self) -> None:
        """Handle pygame events and user input."""
        helper = GameManagerHelper()
        helper.process_events(self)

    def report_final_statistics(self) -> None:
        """Generate comprehensive LLM session report."""
        if self.game_count == 0:
            return

        # Update session metadata
        self.save_session_summary()

        # Compile LLM-specific statistics
        stats_info = {
            "log_dir": self.log_dir,
            "game_count": self.game_count,
            "total_score": self.total_score,
            "total_steps": self.total_steps,
            "game_scores": self.game_scores,
            "empty_steps": self.empty_steps,
            "something_is_wrong_steps": self.something_is_wrong_steps,
            "valid_steps": self.valid_steps,
            "invalid_reversals": self.invalid_reversals,
            "game": self.game,
            "time_stats": self.time_stats,
            "token_stats": self.token_stats,
            "round_counts": self.round_counts,
            "total_rounds": self.total_rounds,
            "max_games": self.args.max_games,
            "no_path_found_steps": self.no_path_found_steps,
        }

        # Generate report and mark session complete
        helper = GameManagerHelper()
        helper.report_final_statistics(stats_info)
        self.running = False

    # -------------------
    # CONTINUATION SUPPORT - LLM-specific feature
    # -------------------

    def continue_from_session(self, log_dir: str, start_game_number: int) -> None:
        """Resume LLM session from previous checkpoint."""
        from utils.continuation_utils import setup_continuation_session, handle_continuation_game_state

        print(Fore.GREEN + f"🔄 Resuming LLM session from: {log_dir}")
        print(Fore.GREEN + f"🔄 Starting at game: {start_game_number}")

        setup_continuation_session(self, log_dir, start_game_number)

        # Initialize LLM clients with health check
        from utils.initialization_utils import setup_llm_clients, enforce_launch_sleep
        setup_llm_clients(self)
        enforce_launch_sleep(self.args)

        # Restore game state
        handle_continuation_game_state(self)
        self.run()

    @classmethod
    def continue_from_directory(cls, args: "argparse.Namespace") -> "GameManager":
        """Factory method for creating continuation sessions."""
        from utils.continuation_utils import continue_from_directory
        return continue_from_directory(cls, args)
