"""
Example Streamlined Extension
============================

This demonstrates the ultimate simplicity possible with the enhanced BaseGameManager
that includes comprehensive controller integration and core functions.

Total lines of extension-specific code: ~12 lines
Gets automatically:
- Game controller integration
- JSON file saving/loading
- Directory management with subdirectories
- Session management with hooks
- Rounds management
- Statistics tracking
- Error handling
- GUI integration (optional)
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
from utils.print_utils import print_info


class StreamlinedGameManager(BaseGameManager):
    """
    Ultra-streamlined extension showing the absolute minimum code needed.
    
    This extension demonstrates:
    - Automatic controller integration
    - Automatic JSON file management
    - Automatic directory creation with subdirectories
    - Automatic session and rounds management
    - Custom algorithm with minimal code
    """
    
    def run(self) -> None:
        """Run the streamlined game session - one line gets everything!"""
        self.run_game_session()
    
    def _get_next_move(self, game_state: Dict[str, Any]) -> str:
        """Implement a simple wall-following algorithm."""
        # Get snake head position
        snake_positions = game_state.get("snake_positions", [])
        if not snake_positions:
            return "UP"
        
        head = snake_positions[-1]
        head_x, head_y = head[0], head[1]
        
        # Simple wall-following: try right, then up, then left, then down
        moves = ["RIGHT", "UP", "LEFT", "DOWN"]
        grid_size = game_state.get("grid_size", 10)
        
        for move in moves:
            new_x, new_y = head_x, head_y
            if move == "RIGHT":
                new_x += 1
            elif move == "UP":
                new_y -= 1
            elif move == "LEFT":
                new_x -= 1
            elif move == "DOWN":
                new_y += 1
            
            # Check bounds and body collision
            if (0 <= new_x < grid_size and 0 <= new_y < grid_size and
                [new_x, new_y] not in snake_positions[:-1]):
                return move
        
        return "UP"  # Fallback
    
    def _create_extension_subdirectories(self) -> None:
        """Create custom subdirectories - automatically handled!"""
        # This gets called automatically by base class
        subdirs = ["algorithms", "analysis", "results"]
        for subdir in subdirs:
            try:
                import os
                os.makedirs(os.path.join(self.log_dir, subdir), exist_ok=True)
            except Exception:
                pass
    
    def _add_task_specific_summary_data(self, summary: Dict[str, Any]) -> None:
        """Add custom summary data - automatically saved to JSON!"""
        summary["algorithm"] = "wall_following"
        summary["strategy"] = "right_hand_rule"
    
    def _display_task_specific_summary(self, summary: Dict[str, Any]) -> None:
        """Display custom summary - automatically called!"""
        print_info("🧱 Algorithm: Wall Following (Right-hand rule)")


# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Streamlined Extension Example")
    parser.add_argument("--max_games", type=int, default=3, help="Number of games")
    parser.add_argument("--grid_size", type=int, default=10, help="Grid size")
    parser.add_argument("--no_gui", action="store_true", help="Run without GUI")
    
    args = parser.parse_args()
    
    # Create and run manager - everything else is automatic!
    manager = StreamlinedGameManager(args)
    manager.initialize()
    manager.run()