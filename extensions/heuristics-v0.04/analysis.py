"""
Advanced Analysis Tools for Heuristics v0.04
============================================

Comprehensive analysis utilities for heuristic algorithm performance,
dataset quality, and comparative studies with elegant visualizations.

Key Features:
- Algorithm performance comparison and benchmarking
- Dataset quality analysis and validation
- Path efficiency and optimization analysis
- Statistical insights and trend analysis
- Export capabilities for research and reporting
"""

from __future__ import annotations
import sys
from pathlib import Path

# Ensure UTF-8 encoding for cross-platform compatibility (SUPREME_RULE NO.7)
import os
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from utils.path_utils import ensure_project_root
ensure_project_root()

import json
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from utils.print_utils import print_info, print_warning, print_error, print_success


class HeuristicAnalyzer:
    """
    Advanced analyzer for heuristic algorithm performance and datasets.
    
    Provides comprehensive analysis capabilities including performance
    comparison, dataset quality assessment, and optimization recommendations.
    """
    
    def __init__(self, log_directory: str):
        self.log_dir = Path(log_directory)
        self.analysis_results = {}
        
        if not self.log_dir.exists():
            raise ValueError(f"Log directory does not exist: {log_directory}")
        
        print_info(f"[HeuristicAnalyzer] Initialized for {log_directory}")
    
    def analyze_session_performance(self) -> Dict[str, Any]:
        """Analyze overall session performance."""
        try:
            # Load session summary
            summary_file = self.log_dir / "summary.json"
            if not summary_file.exists():
                raise FileNotFoundError("No summary.json found in log directory")
            
            with open(summary_file, 'r', encoding='utf-8') as f:
                summary = json.load(f)
            
            # Calculate performance metrics
            analysis = {
                "algorithm": summary.get("algorithm", "Unknown"),
                "total_games": summary.get("total_games", 0),
                "average_score": summary.get("average_score", 0),
                "total_steps": summary.get("total_steps", 0),
                "session_duration": summary.get("session_duration_seconds", 0),
                "efficiency_metrics": self._calculate_efficiency_metrics(summary),
                "performance_grade": self._calculate_performance_grade(summary),
                "recommendations": self._generate_recommendations(summary)
            }
            
            self.analysis_results["session_performance"] = analysis
            return analysis
            
        except Exception as e:
            print_error(f"Error analyzing session performance: {e}")
            return {}
    
    def analyze_dataset_quality(self) -> Dict[str, Any]:
        """Analyze quality of generated datasets."""
        analysis = {
            "csv_analysis": {},
            "jsonl_analysis": {},
            "quality_score": 0,
            "recommendations": []
        }
        
        try:
            # Analyze CSV dataset
            csv_files = list(self.log_dir.glob("*.csv"))
            if csv_files:
                csv_file = csv_files[0]  # Take first CSV file
                analysis["csv_analysis"] = self._analyze_csv_dataset(csv_file)
            
            # Analyze JSONL dataset
            jsonl_files = list(self.log_dir.glob("*.jsonl"))
            if jsonl_files:
                jsonl_file = jsonl_files[0]  # Take first JSONL file
                analysis["jsonl_analysis"] = self._analyze_jsonl_dataset(jsonl_file)
            
            # Calculate overall quality score
            analysis["quality_score"] = self._calculate_dataset_quality_score(analysis)
            analysis["recommendations"] = self._generate_dataset_recommendations(analysis)
            
            self.analysis_results["dataset_quality"] = analysis
            return analysis
            
        except Exception as e:
            print_error(f"Error analyzing dataset quality: {e}")
            return analysis
    
    def compare_algorithms(self, other_log_dirs: List[str]) -> Dict[str, Any]:
        """Compare performance with other algorithm implementations."""
        comparison = {
            "algorithms": [],
            "metrics_comparison": {},
            "rankings": {},
            "insights": []
        }
        
        try:
            # Collect data from all directories
            all_summaries = []
            
            # Add current algorithm
            current_summary = self.analyze_session_performance()
            if current_summary:
                all_summaries.append(current_summary)
            
            # Add other algorithms
            for log_dir in other_log_dirs:
                try:
                    analyzer = HeuristicAnalyzer(log_dir)
                    other_summary = analyzer.analyze_session_performance()
                    if other_summary:
                        all_summaries.append(other_summary)
                except Exception as e:
                    print_warning(f"Could not analyze {log_dir}: {e}")
            
            if len(all_summaries) < 2:
                print_warning("Need at least 2 algorithms for comparison")
                return comparison
            
            # Perform comparison analysis
            comparison["algorithms"] = [s["algorithm"] for s in all_summaries]
            comparison["metrics_comparison"] = self._compare_metrics(all_summaries)
            comparison["rankings"] = self._rank_algorithms(all_summaries)
            comparison["insights"] = self._generate_comparison_insights(all_summaries)
            
            return comparison
            
        except Exception as e:
            print_error(f"Error comparing algorithms: {e}")
            return comparison
    
    def _calculate_efficiency_metrics(self, summary: Dict[str, Any]) -> Dict[str, float]:
        """Calculate efficiency metrics from session summary."""
        metrics = {}
        
        total_games = summary.get("total_games", 0)
        total_steps = summary.get("total_steps", 0)
        total_score = summary.get("total_score", 0)
        session_duration = summary.get("session_duration_seconds", 0)
        
        if total_steps > 0:
            metrics["score_per_step"] = total_score / total_steps
        
        if session_duration > 0:
            metrics["games_per_minute"] = (total_games * 60) / session_duration
            metrics["steps_per_second"] = total_steps / session_duration
        
        if total_games > 0:
            metrics["average_steps_per_game"] = total_steps / total_games
            metrics["average_score_per_game"] = total_score / total_games
        
        return metrics
    
    def _calculate_performance_grade(self, summary: Dict[str, Any]) -> str:
        """Calculate performance grade based on metrics."""
        avg_score = summary.get("average_score", 0)
        efficiency = summary.get("score_per_step", 0)
        
        if avg_score >= 15 and efficiency >= 0.3:
            return "A+ (Excellent)"
        elif avg_score >= 10 and efficiency >= 0.2:
            return "A (Very Good)"
        elif avg_score >= 7 and efficiency >= 0.15:
            return "B (Good)"
        elif avg_score >= 5 and efficiency >= 0.1:
            return "C (Fair)"
        else:
            return "D (Needs Improvement)"
    
    def _generate_recommendations(self, summary: Dict[str, Any]) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []
        
        avg_score = summary.get("average_score", 0)
        efficiency = summary.get("score_per_step", 0)
        
        if avg_score < 10:
            recommendations.append("Consider using A* algorithm for better pathfinding")
            recommendations.append("Increase max_steps to allow longer games")
        
        if efficiency < 0.15:
            recommendations.append("Focus on path optimization to improve efficiency")
            recommendations.append("Consider safety-focused algorithms like BFS-Safe-Greedy")
        
        if summary.get("total_games", 0) < 100:
            recommendations.append("Generate more games for better statistical significance")
        
        return recommendations
    
    def _analyze_csv_dataset(self, csv_file: Path) -> Dict[str, Any]:
        """Analyze CSV dataset quality and characteristics."""
        try:
            df = pd.read_csv(csv_file)
            
            analysis = {
                "total_records": len(df),
                "features": len(df.columns) - 1,  # Exclude target
                "file_size_mb": csv_file.stat().st_size / (1024 * 1024),
                "move_distribution": {},
                "data_quality": {}
            }
            
            # Analyze move distribution
            if 'move' in df.columns:
                move_counts = df['move'].value_counts()
                analysis["move_distribution"] = move_counts.to_dict()
                analysis["data_quality"]["move_balance"] = move_counts.min() / move_counts.max()
            
            # Check for missing values
            missing_values = df.isnull().sum().sum()
            analysis["data_quality"]["missing_values"] = missing_values
            analysis["data_quality"]["completeness"] = 1 - (missing_values / (len(df) * len(df.columns)))
            
            return analysis
            
        except Exception as e:
            print_error(f"Error analyzing CSV dataset: {e}")
            return {}
    
    def _analyze_jsonl_dataset(self, jsonl_file: Path) -> Dict[str, Any]:
        """Analyze JSONL dataset quality and characteristics."""
        try:
            line_count = 0
            valid_lines = 0
            total_tokens = 0
            
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line_count += 1
                    try:
                        data = json.loads(line.strip())
                        valid_lines += 1
                        
                        # Estimate token count (rough approximation)
                        if 'prompt' in data and 'completion' in data:
                            prompt_tokens = len(data['prompt'].split())
                            completion_tokens = len(data['completion'].split())
                            total_tokens += prompt_tokens + completion_tokens
                            
                    except json.JSONDecodeError:
                        continue
            
            analysis = {
                "total_lines": line_count,
                "valid_lines": valid_lines,
                "file_size_mb": jsonl_file.stat().st_size / (1024 * 1024),
                "data_quality": {
                    "validity_rate": valid_lines / max(1, line_count),
                    "estimated_tokens": total_tokens,
                    "avg_tokens_per_record": total_tokens / max(1, valid_lines)
                }
            }
            
            return analysis
            
        except Exception as e:
            print_error(f"Error analyzing JSONL dataset: {e}")
            return {}
    
    def _calculate_dataset_quality_score(self, analysis: Dict[str, Any]) -> float:
        """Calculate overall dataset quality score (0-100)."""
        score = 0
        
        # CSV quality factors
        csv_analysis = analysis.get("csv_analysis", {})
        if csv_analysis:
            # Record count (up to 30 points)
            record_count = csv_analysis.get("total_records", 0)
            score += min(30, record_count / 1000 * 30)
            
            # Data quality (up to 20 points)
            data_quality = csv_analysis.get("data_quality", {})
            completeness = data_quality.get("completeness", 0)
            score += completeness * 20
        
        # JSONL quality factors
        jsonl_analysis = analysis.get("jsonl_analysis", {})
        if jsonl_analysis:
            # Validity rate (up to 25 points)
            validity = jsonl_analysis.get("data_quality", {}).get("validity_rate", 0)
            score += validity * 25
            
            # Token richness (up to 25 points)
            avg_tokens = jsonl_analysis.get("data_quality", {}).get("avg_tokens_per_record", 0)
            if avg_tokens > 100:
                score += min(25, (avg_tokens / 1000) * 25)
        
        return min(100, score)
    
    def _generate_dataset_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate dataset improvement recommendations."""
        recommendations = []
        
        csv_analysis = analysis.get("csv_analysis", {})
        jsonl_analysis = analysis.get("jsonl_analysis", {})
        
        # CSV recommendations
        if csv_analysis:
            if csv_analysis.get("total_records", 0) < 1000:
                recommendations.append("Generate more CSV records for robust ML training (aim for 10,000+)")
            
            data_quality = csv_analysis.get("data_quality", {})
            if data_quality.get("move_balance", 1) < 0.5:
                recommendations.append("Improve move distribution balance in CSV dataset")
        
        # JSONL recommendations
        if jsonl_analysis:
            validity = jsonl_analysis.get("data_quality", {}).get("validity_rate", 0)
            if validity < 0.95:
                recommendations.append("Improve JSONL data quality - some lines are malformed")
            
            avg_tokens = jsonl_analysis.get("data_quality", {}).get("avg_tokens_per_record", 0)
            if avg_tokens < 200:
                recommendations.append("Use higher token variants for richer explanations")
        
        return recommendations
    
    def generate_comprehensive_report(self) -> str:
        """Generate comprehensive analysis report."""
        # Perform all analyses
        session_perf = self.analyze_session_performance()
        dataset_quality = self.analyze_dataset_quality()
        
        report = f"""
🏆 Comprehensive Heuristics Analysis Report
{'=' * 60}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

📊 Session Performance:
   Algorithm: {session_perf.get('algorithm', 'Unknown')}
   Games Played: {session_perf.get('total_games', 0):,}
   Average Score: {session_perf.get('average_score', 0):.2f}
   Total Steps: {session_perf.get('total_steps', 0):,}
   Performance Grade: {session_perf.get('performance_grade', 'Unknown')}

📈 Efficiency Metrics:
"""
        
        efficiency = session_perf.get("efficiency_metrics", {})
        for metric, value in efficiency.items():
            report += f"   {metric.replace('_', ' ').title()}: {value:.4f}\n"
        
        report += f"""
📊 Dataset Quality:
   Quality Score: {dataset_quality.get('quality_score', 0):.1f}/100
"""
        
        # Add CSV analysis
        csv_analysis = dataset_quality.get("csv_analysis", {})
        if csv_analysis:
            report += f"   CSV Records: {csv_analysis.get('total_records', 0):,}\n"
            report += f"   CSV Size: {csv_analysis.get('file_size_mb', 0):.1f} MB\n"
        
        # Add JSONL analysis
        jsonl_analysis = dataset_quality.get("jsonl_analysis", {})
        if jsonl_analysis:
            report += f"   JSONL Lines: {jsonl_analysis.get('valid_lines', 0):,}\n"
            report += f"   JSONL Size: {jsonl_analysis.get('file_size_mb', 0):.1f} MB\n"
        
        # Add recommendations
        all_recommendations = []
        all_recommendations.extend(session_perf.get("recommendations", []))
        all_recommendations.extend(dataset_quality.get("recommendations", []))
        
        if all_recommendations:
            report += "\n💡 Recommendations:\n"
            for i, rec in enumerate(all_recommendations, 1):
                report += f"   {i}. {rec}\n"
        
        report += f"\n{'=' * 60}\n"
        
        return report
    
    def save_analysis_results(self, output_file: Optional[str] = None):
        """Save analysis results to JSON file."""
        try:
            if not output_file:
                output_file = self.log_dir / "analysis_report.json"
            else:
                output_file = Path(output_file)
            
            # Ensure all analyses are performed
            if "session_performance" not in self.analysis_results:
                self.analyze_session_performance()
            if "dataset_quality" not in self.analysis_results:
                self.analyze_dataset_quality()
            
            # Add metadata
            analysis_data = {
                "analysis_timestamp": datetime.now().isoformat(),
                "log_directory": str(self.log_dir),
                "analysis_results": self.analysis_results
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(analysis_data, f, indent=2, ensure_ascii=False)
            
            print_success(f"📊 Analysis results saved to {output_file}")
            
        except Exception as e:
            print_error(f"Error saving analysis results: {e}")
    
    def _compare_metrics(self, summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compare metrics across multiple algorithms."""
        comparison = {}
        
        metrics = ["average_score", "total_steps", "session_duration_seconds"]
        
        for metric in metrics:
            values = [s.get(metric, 0) for s in summaries]
            algorithms = [s.get("algorithm", "Unknown") for s in summaries]
            
            comparison[metric] = {
                "values": dict(zip(algorithms, values)),
                "best": algorithms[values.index(max(values))] if values else "None",
                "worst": algorithms[values.index(min(values))] if values else "None"
            }
        
        return comparison
    
    def _rank_algorithms(self, summaries: List[Dict[str, Any]]) -> Dict[str, int]:
        """Rank algorithms by overall performance."""
        # Simple ranking based on average score
        sorted_summaries = sorted(summaries, key=lambda x: x.get("average_score", 0), reverse=True)
        
        rankings = {}
        for i, summary in enumerate(sorted_summaries, 1):
            algorithm = summary.get("algorithm", "Unknown")
            rankings[algorithm] = i
        
        return rankings
    
    def _generate_comparison_insights(self, summaries: List[Dict[str, Any]]) -> List[str]:
        """Generate insights from algorithm comparison."""
        insights = []
        
        if len(summaries) >= 2:
            # Find best performing algorithm
            best_summary = max(summaries, key=lambda x: x.get("average_score", 0))
            worst_summary = min(summaries, key=lambda x: x.get("average_score", 0))
            
            best_alg = best_summary.get("algorithm", "Unknown")
            worst_alg = worst_summary.get("algorithm", "Unknown")
            
            best_score = best_summary.get("average_score", 0)
            worst_score = worst_summary.get("average_score", 0)
            
            if best_score > worst_score * 1.5:
                insights.append(f"{best_alg} significantly outperforms {worst_alg} ({best_score:.1f} vs {worst_score:.1f})")
            
            # Efficiency insights
            best_efficiency = best_summary.get("efficiency_metrics", {}).get("score_per_step", 0)
            if best_efficiency > 0.2:
                insights.append(f"{best_alg} shows excellent efficiency ({best_efficiency:.3f} score/step)")
        
        return insights


def analyze_heuristics_session(log_directory: str) -> None:
    """Analyze a heuristics session and print comprehensive report."""
    try:
        analyzer = HeuristicAnalyzer(log_directory)
        report = analyzer.generate_comprehensive_report()
        
        print(report)
        
        # Save analysis results
        analyzer.save_analysis_results()
        
        print_success("✅ Analysis completed successfully!")
        
    except Exception as e:
        print_error(f"❌ Analysis failed: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Heuristics Performance Analyzer")
    parser.add_argument("log_dir", help="Path to heuristics log directory")
    parser.add_argument("--compare", nargs="*", help="Additional log directories for comparison")
    
    args = parser.parse_args()
    
    try:
        analyzer = HeuristicAnalyzer(args.log_dir)
        
        if args.compare:
            print_info("🔍 Performing comparative analysis...")
            comparison = analyzer.compare_algorithms(args.compare)
            print(json.dumps(comparison, indent=2))
        else:
            print_info("📊 Performing session analysis...")
            analyze_heuristics_session(args.log_dir)
            
    except Exception as e:
        print_error(f"❌ Analysis failed: {e}")
        sys.exit(1)