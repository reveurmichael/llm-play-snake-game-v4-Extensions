#!/usr/bin/env python3
"""
Supervised Learning Extension Main Script
========================================

Command-line interface for running supervised learning models on Snake Game AI.

Usage:
    python main.py --model MLP --dataset /path/to/dataset.csv --max_games 5
    python main.py --model LightGBM --dataset /path/to/dataset.csv --grid_size 15
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
from utils.print_utils import print_info, print_error, print_success
from game_manager import SupervisedGameManager


def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="🧠 Supervised Learning Snake Game AI v0.03",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train MLP model and play games
  python main.py --model MLP --dataset path/to/heuristic_dataset.csv --max_games 5
  
  # Train LightGBM with custom parameters
  python main.py --model LightGBM --dataset data.csv --grid_size 15 --verbose
  
  # Quick test with untrained model
  python main.py --model MLP --max_games 3 --no_gui
  
  # Large-scale evaluation
  python main.py --model LightGBM --dataset large_dataset.csv --max_games 100
        """
    )

    # Model configuration
    parser.add_argument(
        "--model",
        type=str,
        choices=["MLP", "LightGBM"],
        default="MLP",
        help="ML model type to use (default: MLP)"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        help="Path to CSV dataset for training"
    )

    # Game configuration
    parser.add_argument(
        "--grid_size",
        type=int,
        default=10,
        help="Game grid size (default: 10)"
    )
    
    parser.add_argument(
        "--max_games",
        type=int,
        default=1,
        help="Maximum number of games to play (default: 1)"
    )
    
    parser.add_argument(
        "--max_steps",
        type=int,
        default=1000,
        help="Maximum steps per game (default: 1000)"
    )

    # Display options
    parser.add_argument(
        "--no_gui",
        action="store_true",
        help="Run without GUI (headless mode)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )

    return parser


def main():
    """Main entry point for supervised learning extension."""
    parser = create_parser()
    args = parser.parse_args()

    # Enhanced startup banner
    print_success("🧠 Supervised Learning Snake Game AI v0.03")
    print_info("=" * 60)
    print_info("🎯 Advanced ML Models for Snake Game Intelligence")
    print_info("📊 Training on Heuristic-Generated Datasets")
    print_info("🚀 Real-time Performance Analysis and Comparison")
    print_info("=" * 60)
    print()
    
    # Show configuration
    print_info(f"🧠 Model: {args.model}")
    print_info(f"📊 Dataset: {args.dataset if args.dataset else 'None (untrained model)'}")
    print_info(f"🎮 Max Games: {args.max_games}")
    print_info(f"📏 Grid Size: {args.grid_size}x{args.grid_size}")
    print_info(f"🖥️  GUI Mode: {'Disabled' if args.no_gui else 'Enabled'}")
    print()
    
    try:
        # Create and initialize game manager
        manager = SupervisedGameManager(args)
        manager.initialize()
        
        # Run the session
        manager.run()
        
        print()
        print_success("🎉 Supervised Learning session completed successfully!")
        print_info("📊 Check the generated logs for detailed performance analysis")
        
    except KeyboardInterrupt:
        print_info("\n⚠️  Session interrupted by user")
        print_info("💾 Session data saved where possible")
        sys.exit(0)
        
    except Exception as e:
        print_error(f"❌ Session failed: {e}")
        print_info("💡 Common solutions:")
        print_info("   - Check dataset path and format")
        print_info("   - Ensure required ML libraries are installed")
        print_info("   - Try with --verbose for detailed error information")
        
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()