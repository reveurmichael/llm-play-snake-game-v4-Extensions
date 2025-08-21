"""
Performance Monitor for Heuristics v0.04
========================================

Advanced performance monitoring and analysis for heuristic algorithms
with real-time metrics, bottleneck detection, and optimization suggestions.

Key Features:
- Real-time performance tracking
- Algorithm efficiency analysis
- Memory usage monitoring
- Bottleneck identification
- Optimization recommendations
"""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from utils.path_utils import ensure_project_root
ensure_project_root()

import time
import psutil
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from utils.print_utils import print_info, print_warning, print_success


@dataclass
class PerformanceMetrics:
    """Performance metrics for algorithm analysis."""
    algorithm_name: str
    total_games: int = 0
    total_moves: int = 0
    total_pathfinding_time: float = 0.0
    total_explanation_time: float = 0.0
    successful_paths: int = 0
    failed_paths: int = 0
    memory_usage_mb: float = 0.0
    games_per_second: float = 0.0
    moves_per_second: float = 0.0
    pathfinding_efficiency: float = 0.0
    
    # Detailed timing breakdown
    timing_breakdown: Dict[str, float] = field(default_factory=dict)
    
    def calculate_derived_metrics(self, session_duration: float):
        """Calculate derived performance metrics."""
        if session_duration > 0:
            self.games_per_second = self.total_games / session_duration
            self.moves_per_second = self.total_moves / session_duration
        
        if self.total_moves > 0:
            self.pathfinding_efficiency = self.successful_paths / self.total_moves
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        return {
            "algorithm": self.algorithm_name,
            "games": self.total_games,
            "moves": self.total_moves,
            "pathfinding_time": self.total_pathfinding_time,
            "explanation_time": self.total_explanation_time,
            "success_rate": self.successful_paths / max(1, self.total_moves),
            "memory_mb": self.memory_usage_mb,
            "games_per_sec": self.games_per_second,
            "moves_per_sec": self.moves_per_second,
            "efficiency": self.pathfinding_efficiency,
            "timing_breakdown": self.timing_breakdown
        }


class PerformanceMonitor:
    """
    Advanced performance monitor for heuristic algorithms.
    
    Tracks performance metrics, identifies bottlenecks, and provides
    optimization recommendations for algorithm implementations.
    """
    
    def __init__(self, algorithm_name: str):
        self.algorithm_name = algorithm_name
        self.metrics = PerformanceMetrics(algorithm_name)
        self.session_start_time = time.time()
        self.current_game_start = None
        self.current_move_start = None
        
        # Memory monitoring
        self.process = psutil.Process()
        self.initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        print_info(f"[PerformanceMonitor] Initialized for {algorithm_name}")
    
    def start_game(self):
        """Mark the start of a new game."""
        self.current_game_start = time.time()
        self.metrics.total_games += 1
    
    def end_game(self):
        """Mark the end of the current game."""
        if self.current_game_start:
            game_duration = time.time() - self.current_game_start
            self.metrics.timing_breakdown[f"game_{self.metrics.total_games}"] = game_duration
    
    def start_pathfinding(self):
        """Mark the start of pathfinding operation."""
        self.current_move_start = time.time()
    
    def end_pathfinding(self, success: bool):
        """Mark the end of pathfinding operation."""
        if self.current_move_start:
            pathfinding_time = time.time() - self.current_move_start
            self.metrics.total_pathfinding_time += pathfinding_time
            self.metrics.total_moves += 1
            
            if success:
                self.metrics.successful_paths += 1
            else:
                self.metrics.failed_paths += 1
    
    def record_explanation_time(self, explanation_time: float):
        """Record time spent generating explanations."""
        self.metrics.total_explanation_time += explanation_time
    
    def update_memory_usage(self):
        """Update current memory usage."""
        current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.metrics.memory_usage_mb = current_memory - self.initial_memory
    
    def get_current_performance(self) -> Dict[str, Any]:
        """Get current performance snapshot."""
        session_duration = time.time() - self.session_start_time
        self.metrics.calculate_derived_metrics(session_duration)
        self.update_memory_usage()
        
        return self.metrics.get_performance_summary()
    
    def analyze_bottlenecks(self) -> Dict[str, Any]:
        """Analyze performance bottlenecks and provide recommendations."""
        performance = self.get_current_performance()
        
        analysis = {
            "bottlenecks": [],
            "recommendations": [],
            "strengths": []
        }
        
        # Analyze pathfinding performance
        if performance["pathfinding_time"] > performance["explanation_time"] * 2:
            analysis["bottlenecks"].append("Pathfinding is the primary bottleneck")
            analysis["recommendations"].append("Consider optimizing pathfinding algorithm or using A* instead of BFS")
        
        # Analyze success rate
        if performance["success_rate"] < 0.95:
            analysis["bottlenecks"].append("Low pathfinding success rate")
            analysis["recommendations"].append("Review pathfinding logic for edge cases")
        
        # Analyze memory usage
        if performance["memory_mb"] > 100:
            analysis["bottlenecks"].append("High memory usage")
            analysis["recommendations"].append("Consider optimizing data structures or garbage collection")
        
        # Identify strengths
        if performance["success_rate"] > 0.98:
            analysis["strengths"].append("Excellent pathfinding reliability")
        
        if performance["moves_per_sec"] > 100:
            analysis["strengths"].append("High move processing throughput")
        
        if performance["memory_mb"] < 50:
            analysis["strengths"].append("Efficient memory usage")
        
        return analysis
    
    def generate_performance_report(self) -> str:
        """Generate comprehensive performance report."""
        performance = self.get_current_performance()
        analysis = self.analyze_bottlenecks()
        
        report = f"""
🏆 Performance Report: {self.algorithm_name}
{'=' * 50}

📊 Core Metrics:
   Games Played: {performance['games']:,}
   Total Moves: {performance['moves']:,}
   Success Rate: {performance['success_rate']:.1%}
   Memory Usage: {performance['memory_mb']:.1f} MB

⚡ Performance:
   Games/Second: {performance['games_per_sec']:.2f}
   Moves/Second: {performance['moves_per_sec']:.2f}
   Pathfinding Time: {performance['pathfinding_time']:.3f}s
   Explanation Time: {performance['explanation_time']:.3f}s

🎯 Analysis:
"""
        
        if analysis["strengths"]:
            report += "\n✅ Strengths:\n"
            for strength in analysis["strengths"]:
                report += f"   • {strength}\n"
        
        if analysis["bottlenecks"]:
            report += "\n⚠️ Bottlenecks:\n"
            for bottleneck in analysis["bottlenecks"]:
                report += f"   • {bottleneck}\n"
        
        if analysis["recommendations"]:
            report += "\n💡 Recommendations:\n"
            for rec in analysis["recommendations"]:
                report += f"   • {rec}\n"
        
        return report
    
    def save_performance_data(self, output_path: Path):
        """Save performance data to JSON file."""
        try:
            import json
            
            performance_data = {
                "performance_metrics": self.get_current_performance(),
                "bottleneck_analysis": self.analyze_bottlenecks(),
                "session_duration": time.time() - self.session_start_time,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            perf_file = output_path / f"{self.algorithm_name}_performance.json"
            with open(perf_file, "w", encoding="utf-8") as f:
                json.dump(performance_data, f, indent=2, ensure_ascii=False)
            
            print_success(f"📊 Performance data saved to {perf_file}")
            
        except Exception as e:
            print_warning(f"Failed to save performance data: {e}")