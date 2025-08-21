"""
Enhanced Time Statistics System
==============================

Comprehensive time tracking system for all Snake Game AI components with
elegant architecture, detailed performance analysis, and extension hooks.

Key Features:
- Universal time tracking for all task types
- Detailed performance breakdown and analysis
- Extension hooks for task-specific timing
- Beautiful reporting and visualization
- Performance optimization recommendations

Design Philosophy:
- SSOT for all time tracking across the project
- Template method pattern for extension customization
- Comprehensive performance analysis with actionable insights
- Educational value with clear timing methodology
"""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.path_utils import ensure_project_root
ensure_project_root()

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional, DefaultDict
from collections import defaultdict
from utils.print_utils import print_info, print_warning, print_success


@dataclass
class UniversalTimeStats:
    """
    Universal time statistics for all Snake Game AI tasks.
    
    Provides comprehensive timing without task-specific dependencies,
    making it perfect for heuristics, supervised learning, RL, and LLM tasks.
    """
    
    # Core timing
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    
    # Algorithm timing (universal)
    algorithm_time: float = 0.0
    decision_time: float = 0.0
    execution_time: float = 0.0
    
    # Performance breakdown
    timing_breakdown: Dict[str, float] = field(default_factory=dict)
    
    # Move-level timing
    move_times: List[float] = field(default_factory=list)
    
    def record_end_time(self) -> None:
        """Record the end time for the session."""
        self.end_time = time.time()
    
    def add_algorithm_time(self, duration: float) -> None:
        """Add time spent in algorithm computation."""
        self.algorithm_time += duration
    
    def add_decision_time(self, duration: float) -> None:
        """Add time spent in decision making."""
        self.decision_time += duration
    
    def add_execution_time(self, duration: float) -> None:
        """Add time spent in move execution."""
        self.execution_time += duration
    
    def record_move_time(self, move_duration: float) -> None:
        """Record time for individual move."""
        self.move_times.append(move_duration)
    
    def add_custom_timing(self, category: str, duration: float) -> None:
        """Add custom timing category for extension-specific tracking."""
        if category not in self.timing_breakdown:
            self.timing_breakdown[category] = 0.0
        self.timing_breakdown[category] += duration
    
    def get_total_duration(self) -> float:
        """Get total session duration."""
        end = self.end_time or time.time()
        return end - self.start_time
    
    def get_average_move_time(self) -> float:
        """Get average time per move."""
        return sum(self.move_times) / len(self.move_times) if self.move_times else 0.0
    
    def get_performance_analysis(self) -> Dict[str, Any]:
        """Get comprehensive performance analysis."""
        total_duration = self.get_total_duration()
        
        analysis = {
            "total_duration": total_duration,
            "algorithm_percentage": (self.algorithm_time / total_duration * 100) if total_duration > 0 else 0,
            "decision_percentage": (self.decision_time / total_duration * 100) if total_duration > 0 else 0,
            "execution_percentage": (self.execution_time / total_duration * 100) if total_duration > 0 else 0,
            "average_move_time": self.get_average_move_time(),
            "moves_per_second": len(self.move_times) / total_duration if total_duration > 0 else 0,
            "efficiency_score": self._calculate_efficiency_score(),
            "bottlenecks": self._identify_bottlenecks(),
            "recommendations": self._generate_recommendations()
        }
        
        return analysis
    
    def _calculate_efficiency_score(self) -> float:
        """Calculate efficiency score (0-100) based on timing distribution."""
        total_time = self.get_total_duration()
        if total_time == 0:
            return 0.0
        
        # Ideal distribution: 70% algorithm, 20% decision, 10% execution
        algorithm_ratio = self.algorithm_time / total_time
        decision_ratio = self.decision_time / total_time
        execution_ratio = self.execution_time / total_time
        
        # Score based on how close to ideal distribution
        algorithm_score = max(0, 100 - abs(algorithm_ratio - 0.7) * 100)
        decision_score = max(0, 100 - abs(decision_ratio - 0.2) * 100)
        execution_score = max(0, 100 - abs(execution_ratio - 0.1) * 100)
        
        return (algorithm_score + decision_score + execution_score) / 3
    
    def _identify_bottlenecks(self) -> List[str]:
        """Identify performance bottlenecks."""
        bottlenecks = []
        total_time = self.get_total_duration()
        
        if total_time == 0:
            return bottlenecks
        
        # Check for timing issues
        if self.algorithm_time / total_time > 0.8:
            bottlenecks.append("Algorithm computation dominates execution time")
        
        if self.decision_time / total_time > 0.3:
            bottlenecks.append("Decision making is taking too long")
        
        if self.execution_time / total_time > 0.2:
            bottlenecks.append("Move execution overhead is high")
        
        # Check move timing consistency
        if len(self.move_times) > 10:
            avg_time = self.get_average_move_time()
            max_time = max(self.move_times)
            if max_time > avg_time * 5:
                bottlenecks.append("Inconsistent move timing - some moves much slower")
        
        return bottlenecks
    
    def _generate_recommendations(self) -> List[str]:
        """Generate performance optimization recommendations."""
        recommendations = []
        bottlenecks = self._identify_bottlenecks()
        
        if "Algorithm computation dominates" in str(bottlenecks):
            recommendations.append("Consider algorithm optimization or caching")
            recommendations.append("Profile algorithm implementation for hotspots")
        
        if "Decision making is taking too long" in str(bottlenecks):
            recommendations.append("Optimize decision logic or reduce complexity")
            recommendations.append("Consider pre-computation of common decisions")
        
        if "Move execution overhead" in str(bottlenecks):
            recommendations.append("Optimize move execution pipeline")
            recommendations.append("Review GUI update frequency if using visual mode")
        
        if "Inconsistent move timing" in str(bottlenecks):
            recommendations.append("Investigate timing variability causes")
            recommendations.append("Consider move complexity normalization")
        
        # Add positive recommendations for good performance
        efficiency = self._calculate_efficiency_score()
        if efficiency > 80:
            recommendations.insert(0, "🏆 Excellent timing distribution - well optimized!")
        elif efficiency > 60:
            recommendations.insert(0, "✅ Good timing distribution - minor optimizations possible")
        
        return recommendations
    
    def asdict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        end = self.end_time or time.time()
        
        base_dict = {
            "start_time": datetime.fromtimestamp(self.start_time).strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": datetime.fromtimestamp(end).strftime("%Y-%m-%d %H:%M:%S"),
            "total_duration_seconds": end - self.start_time,
            "algorithm_time": self.algorithm_time,
            "decision_time": self.decision_time,
            "execution_time": self.execution_time,
            "average_move_time": self.get_average_move_time(),
            "total_moves": len(self.move_times),
            "moves_per_second": len(self.move_times) / (end - self.start_time) if (end - self.start_time) > 0 else 0
        }
        
        # Add custom timing breakdown
        if self.timing_breakdown:
            base_dict["timing_breakdown"] = self.timing_breakdown
        
        # Add performance analysis
        performance = self.get_performance_analysis()
        base_dict["performance_analysis"] = {
            "efficiency_score": performance["efficiency_score"],
            "bottlenecks": performance["bottlenecks"],
            "recommendations": performance["recommendations"]
        }
        
        return base_dict


@dataclass
class LLMTimeStats(UniversalTimeStats):
    """
    LLM-specific time statistics extending universal timing.
    
    Adds LLM-specific timing categories while maintaining compatibility
    with the universal timing system.
    """
    
    # LLM-specific timing
    llm_communication_time: float = 0.0
    prompt_generation_time: float = 0.0
    response_parsing_time: float = 0.0
    
    def add_llm_comm(self, duration: float) -> None:
        """Add LLM communication time."""
        self.llm_communication_time += duration
        self.add_custom_timing("llm_communication", duration)
    
    def add_prompt_time(self, duration: float) -> None:
        """Add prompt generation time."""
        self.prompt_generation_time += duration
        self.add_custom_timing("prompt_generation", duration)
    
    def add_parsing_time(self, duration: float) -> None:
        """Add response parsing time."""
        self.response_parsing_time += duration
        self.add_custom_timing("response_parsing", duration)
    
    def asdict(self) -> Dict[str, Any]:
        """Convert to dictionary with LLM-specific fields."""
        base_dict = super().asdict()
        
        # Add LLM-specific timing
        base_dict.update({
            "llm_communication_time": self.llm_communication_time,
            "prompt_generation_time": self.prompt_generation_time,
            "response_parsing_time": self.response_parsing_time
        })
        
        return base_dict


@dataclass
class HeuristicTimeStats(UniversalTimeStats):
    """
    Heuristic-specific time statistics for pathfinding algorithms.
    
    Adds pathfinding-specific timing categories while maintaining
    compatibility with universal timing system.
    """
    
    # Pathfinding-specific timing
    pathfinding_time: float = 0.0
    validation_time: float = 0.0
    explanation_time: float = 0.0
    
    def add_pathfinding_time(self, duration: float) -> None:
        """Add pathfinding computation time."""
        self.pathfinding_time += duration
        self.add_algorithm_time(duration)  # Also count as algorithm time
        self.add_custom_timing("pathfinding", duration)
    
    def add_validation_time(self, duration: float) -> None:
        """Add state validation time."""
        self.validation_time += duration
        self.add_custom_timing("validation", duration)
    
    def add_explanation_time(self, duration: float) -> None:
        """Add explanation generation time."""
        self.explanation_time += duration
        self.add_custom_timing("explanation", duration)
    
    def asdict(self) -> Dict[str, Any]:
        """Convert to dictionary with heuristic-specific fields."""
        base_dict = super().asdict()
        
        # Add heuristic-specific timing
        base_dict.update({
            "pathfinding_time": self.pathfinding_time,
            "validation_time": self.validation_time,
            "explanation_time": self.explanation_time
        })
        
        return base_dict


@dataclass
class SupervisedTimeStats(UniversalTimeStats):
    """
    Supervised learning-specific time statistics.
    
    Adds ML-specific timing categories while maintaining compatibility
    with universal timing system.
    """
    
    # ML-specific timing
    model_prediction_time: float = 0.0
    feature_extraction_time: float = 0.0
    preprocessing_time: float = 0.0
    
    # Training timing (if applicable)
    training_time: float = 0.0
    validation_time: float = 0.0
    
    def add_prediction_time(self, duration: float) -> None:
        """Add model prediction time."""
        self.model_prediction_time += duration
        self.add_algorithm_time(duration)  # Also count as algorithm time
        self.add_custom_timing("model_prediction", duration)
    
    def add_feature_extraction_time(self, duration: float) -> None:
        """Add feature extraction time."""
        self.feature_extraction_time += duration
        self.add_custom_timing("feature_extraction", duration)
    
    def add_preprocessing_time(self, duration: float) -> None:
        """Add data preprocessing time."""
        self.preprocessing_time += duration
        self.add_custom_timing("preprocessing", duration)
    
    def add_training_time(self, duration: float) -> None:
        """Add model training time."""
        self.training_time += duration
        self.add_custom_timing("training", duration)
    
    def asdict(self) -> Dict[str, Any]:
        """Convert to dictionary with ML-specific fields."""
        base_dict = super().asdict()
        
        # Add ML-specific timing
        base_dict.update({
            "model_prediction_time": self.model_prediction_time,
            "feature_extraction_time": self.feature_extraction_time,
            "preprocessing_time": self.preprocessing_time,
            "training_time": self.training_time
        })
        
        return base_dict


class TimeStatsManager:
    """
    Comprehensive time statistics manager for performance analysis.
    
    Provides centralized timing management with detailed analysis,
    optimization recommendations, and beautiful reporting.
    """
    
    def __init__(self, task_type: str = "universal"):
        self.task_type = task_type
        self.time_stats = self._create_time_stats()
        self.session_timings = []
        
    def _create_time_stats(self) -> UniversalTimeStats:
        """Create appropriate time stats based on task type."""
        if self.task_type.lower() == "llm":
            return LLMTimeStats()
        elif self.task_type.lower() == "heuristic":
            return HeuristicTimeStats()
        elif self.task_type.lower() == "supervised":
            return SupervisedTimeStats()
        else:
            return UniversalTimeStats()
    
    def start_timing(self, category: str) -> float:
        """Start timing for a specific category."""
        return time.time()
    
    def end_timing(self, category: str, start_time: float) -> float:
        """End timing and record duration."""
        duration = time.time() - start_time
        self.time_stats.add_custom_timing(category, duration)
        return duration
    
    def time_operation(self, category: str):
        """Context manager for timing operations."""
        return TimingContext(self, category)
    
    def analyze_session_performance(self) -> Dict[str, Any]:
        """Analyze overall session performance."""
        analysis = self.time_stats.get_performance_analysis()
        
        # Add session-level insights
        analysis["session_insights"] = self._generate_session_insights()
        analysis["optimization_opportunities"] = self._identify_optimization_opportunities()
        
        return analysis
    
    def _generate_session_insights(self) -> List[str]:
        """Generate session-level performance insights."""
        insights = []
        
        total_duration = self.time_stats.get_total_duration()
        avg_move_time = self.time_stats.get_average_move_time()
        
        if total_duration < 60:
            insights.append("⚡ Fast session execution - excellent performance")
        elif total_duration > 300:
            insights.append("🐌 Long session duration - consider optimization")
        
        if avg_move_time < 0.1:
            insights.append("🚀 Excellent move processing speed")
        elif avg_move_time > 1.0:
            insights.append("⏰ Slow move processing - optimization needed")
        
        return insights
    
    def _identify_optimization_opportunities(self) -> List[str]:
        """Identify specific optimization opportunities."""
        opportunities = []
        
        # Analyze timing breakdown
        breakdown = self.time_stats.timing_breakdown
        if breakdown:
            # Find dominant timing category
            max_category = max(breakdown.items(), key=lambda x: x[1])
            if max_category[1] > sum(breakdown.values()) * 0.6:
                opportunities.append(f"🎯 Optimize {max_category[0]} - dominates execution time")
        
        # Check move timing variability
        move_times = self.time_stats.move_times
        if len(move_times) > 10:
            avg_time = sum(move_times) / len(move_times)
            max_time = max(move_times)
            if max_time > avg_time * 3:
                opportunities.append("📊 High move timing variability - investigate outliers")
        
        return opportunities
    
    def generate_timing_report(self) -> str:
        """Generate comprehensive timing report."""
        analysis = self.analyze_session_performance()
        
        report = f"""
🏆 Performance Timing Report - {self.task_type.title()}
{'=' * 60}

⏱️  Session Overview:
   Total Duration: {analysis['total_duration']:.2f} seconds
   Average Move Time: {analysis['average_move_time']:.4f} seconds
   Moves per Second: {analysis['moves_per_second']:.2f}
   Efficiency Score: {analysis['efficiency_score']:.1f}/100

📊 Time Distribution:
   Algorithm: {analysis['algorithm_percentage']:.1f}%
   Decision: {analysis['decision_percentage']:.1f}%
   Execution: {analysis['execution_percentage']:.1f}%

"""
        
        # Add timing breakdown if available
        if self.time_stats.timing_breakdown:
            report += "🔍 Detailed Breakdown:\n"
            for category, duration in sorted(self.time_stats.timing_breakdown.items(), key=lambda x: x[1], reverse=True):
                percentage = (duration / analysis['total_duration'] * 100) if analysis['total_duration'] > 0 else 0
                report += f"   {category}: {duration:.3f}s ({percentage:.1f}%)\n"
        
        # Add insights
        if analysis["session_insights"]:
            report += "\n💡 Session Insights:\n"
            for insight in analysis["session_insights"]:
                report += f"   {insight}\n"
        
        # Add optimization opportunities
        if analysis["optimization_opportunities"]:
            report += "\n🚀 Optimization Opportunities:\n"
            for opportunity in analysis["optimization_opportunities"]:
                report += f"   {opportunity}\n"
        
        # Add recommendations
        if analysis["recommendations"]:
            report += "\n📈 Recommendations:\n"
            for rec in analysis["recommendations"]:
                report += f"   {rec}\n"
        
        report += "=" * 60
        
        return report
    
    def export_timing_data(self, output_path: str):
        """Export timing data to JSON file."""
        try:
            import json
            
            timing_data = {
                "task_type": self.task_type,
                "time_stats": self.time_stats.asdict(),
                "performance_analysis": self.analyze_session_performance(),
                "export_timestamp": datetime.now().isoformat()
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(timing_data, f, indent=2, ensure_ascii=False)
            
            print_success(f"📊 Timing data exported to {output_path}")
            
        except Exception as e:
            print_warning(f"Failed to export timing data: {e}")


class TimingContext:
    """Context manager for easy timing of operations."""
    
    def __init__(self, manager: TimeStatsManager, category: str):
        self.manager = manager
        self.category = category
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            self.manager.time_stats.add_custom_timing(self.category, duration)


# Factory function for creating appropriate time stats
def create_time_stats(task_type: str) -> UniversalTimeStats:
    """Create appropriate time stats for task type."""
    if task_type.lower() == "llm":
        return LLMTimeStats()
    elif task_type.lower() == "heuristic":
        return HeuristicTimeStats()
    elif task_type.lower() == "supervised":
        return SupervisedTimeStats()
    else:
        return UniversalTimeStats()


# Export all classes and utilities
__all__ = [
    "UniversalTimeStats",
    "LLMTimeStats", 
    "HeuristicTimeStats",
    "SupervisedTimeStats",
    "TimeStatsManager",
    "TimingContext",
    "create_time_stats"
]