"""
Example Simple Extension
=======================

This demonstrates how incredibly simple it is to create new extensions
using the enhanced BaseGameManager architecture.

Total lines of extension-specific code: ~20 lines
"""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from utils.path_utils import ensure_project_root
ensure_project_root()

import argparse
from typing import Dict, Any
from core.game_manager import BaseGameManager
from core.game_logic import BaseGameLogic
from utils.print_utils import print_info


class SimpleGameManager(BaseGameManager):
    """
    Ultra-simple example extension showing minimal code required.
    
    This extension just makes random moves but demonstrates the full
    session management, statistics, and file generation capabilities
    inherited from BaseGameManager.
    """
    
    # Use base game logic (or create custom if needed)
    GAME_LOGIC_CLS = BaseGameLogic
    
    def run(self) -> None:
        """Run the simple game session."""
        # That's it! One line gets you full session management
        self.run_game_session()
    
    def _execute_game_loop(self) -> None:
        """Execute simple random-move game loop."""
        import random
        
        steps = 0
        while not self.game.game_over and steps < 100:
            steps += 1
            
            # Make random move
            moves = ["UP", "DOWN", "LEFT", "RIGHT"]
            move = random.choice(moves)
            
            try:
                self.game.make_move(move)
            except:
                break  # Invalid move, end game
    
    def _display_session_start(self) -> None:
        """Customize session start display."""
        super()._display_session_start()
        print_info("🎲 Making random moves...")
    
    def _add_task_specific_summary_data(self, summary: Dict[str, Any]) -> None:
        """Add extension-specific summary data."""
        summary["strategy"] = "random_moves"
    
    def _display_task_specific_summary(self, summary: Dict[str, Any]) -> None:
        """Display extension-specific summary."""
        print_info("🎲 Strategy: Random moves")


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple Extension Example")
    parser.add_argument("--max_games", type=int, default=3, help="Number of games")
    parser.add_argument("--grid_size", type=int, default=10, help="Grid size")
    parser.add_argument("--no_gui", action="store_true", help="Run without GUI")
    
    args = parser.parse_args()
    
    # Create and run manager
    manager = SimpleGameManager(args)
    manager.initialize()
    manager.run()