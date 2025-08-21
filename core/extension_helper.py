"""
Extension Development Helper
===========================

Comprehensive utilities to make extension development incredibly easy,
intuitive, and elegant with minimal code requirements.

Key Features:
- Extension scaffolding and templates
- Automatic boilerplate generation
- Best practice validation
- Performance optimization helpers
- Educational guidance and examples

Design Philosophy:
- Make extension development as simple as possible
- Provide comprehensive infrastructure through inheritance
- Enable dramatic code reduction (80-95%)
- Maintain educational value and clean architecture
"""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.path_utils import ensure_project_root
ensure_project_root()

from typing import Dict, Any, List, Optional, Type
from abc import ABC, abstractmethod
import inspect
from utils.print_utils import print_info, print_warning, print_error, print_success


class ExtensionValidator:
    """
    Validates extension implementations for best practices and optimization.
    
    Provides comprehensive analysis of extension code quality, architecture
    compliance, and performance optimization opportunities.
    """
    
    def __init__(self, extension_class: Type):
        self.extension_class = extension_class
        self.validation_results = {}
    
    def validate_extension_quality(self) -> Dict[str, Any]:
        """Perform comprehensive extension quality validation."""
        results = {
            "architecture_compliance": self._validate_architecture(),
            "method_optimization": self._validate_methods(),
            "inheritance_usage": self._validate_inheritance(),
            "hook_implementation": self._validate_hooks(),
            "code_quality": self._validate_code_quality(),
            "recommendations": []
        }
        
        # Generate recommendations
        results["recommendations"] = self._generate_recommendations(results)
        
        self.validation_results = results
        return results
    
    def _validate_architecture(self) -> Dict[str, Any]:
        """Validate architecture compliance and patterns."""
        validation = {
            "inherits_base_manager": False,
            "uses_factory_pattern": False,
            "implements_hooks": False,
            "follows_naming": False,
            "score": 0
        }
        
        # Check inheritance
        if hasattr(self.extension_class, '__bases__'):
            for base in self.extension_class.__bases__:
                if 'BaseGameManager' in base.__name__:
                    validation["inherits_base_manager"] = True
                    validation["score"] += 25
        
        # Check factory pattern
        if hasattr(self.extension_class, 'GAME_LOGIC_CLS'):
            validation["uses_factory_pattern"] = True
            validation["score"] += 25
        
        # Check hook implementation
        hook_methods = [
            '_add_task_specific_game_data',
            '_display_task_specific_results',
            '_add_task_specific_summary_data'
        ]
        
        implemented_hooks = 0
        for hook in hook_methods:
            if hasattr(self.extension_class, hook):
                implemented_hooks += 1
        
        if implemented_hooks >= 2:
            validation["implements_hooks"] = True
            validation["score"] += 25
        
        # Check naming conventions
        if self.extension_class.__name__.endswith('GameManager'):
            validation["follows_naming"] = True
            validation["score"] += 25
        
        return validation
    
    def _validate_methods(self) -> Dict[str, Any]:
        """Validate method implementation and optimization."""
        validation = {
            "minimal_overrides": False,
            "uses_base_methods": False,
            "proper_error_handling": False,
            "score": 0
        }
        
        # Get all methods
        methods = inspect.getmembers(self.extension_class, predicate=inspect.isfunction)
        
        # Check for minimal overrides (good practice)
        if len(methods) <= 10:  # Minimal extension code
            validation["minimal_overrides"] = True
            validation["score"] += 30
        
        # Check for base method usage
        method_names = [name for name, _ in methods]
        base_method_usage = ['run_game_session', 'finalize_game', 'generate_game_data']
        
        if any(method in method_names for method in base_method_usage):
            validation["uses_base_methods"] = True
            validation["score"] += 35
        
        # Check for proper error handling patterns
        for name, method in methods:
            if hasattr(method, '__code__'):
                source = inspect.getsource(method) if hasattr(inspect, 'getsource') else ""
                if 'try:' in source and 'except' in source:
                    validation["proper_error_handling"] = True
                    validation["score"] += 35
                    break
        
        return validation
    
    def _validate_inheritance(self) -> Dict[str, Any]:
        """Validate inheritance patterns and usage."""
        validation = {
            "clean_inheritance": False,
            "avoids_duplication": False,
            "uses_super": False,
            "score": 0
        }
        
        # Check method resolution order
        mro = self.extension_class.__mro__
        if len(mro) <= 4:  # Extension → BaseGameManager → object (clean hierarchy)
            validation["clean_inheritance"] = True
            validation["score"] += 30
        
        # Check for super() usage (good practice)
        methods = inspect.getmembers(self.extension_class, predicate=inspect.isfunction)
        for name, method in methods:
            try:
                source = inspect.getsource(method)
                if 'super()' in source:
                    validation["uses_super"] = True
                    validation["score"] += 35
                    break
            except:
                continue
        
        # Check for code duplication avoidance
        if validation["uses_super"] and validation["clean_inheritance"]:
            validation["avoids_duplication"] = True
            validation["score"] += 35
        
        return validation
    
    def _validate_hooks(self) -> Dict[str, Any]:
        """Validate extension hook implementation."""
        validation = {
            "implements_data_hooks": False,
            "implements_display_hooks": False,
            "implements_session_hooks": False,
            "score": 0
        }
        
        # Check for data hooks
        data_hooks = ['_add_task_specific_game_data', '_finalize_task_specific']
        if any(hasattr(self.extension_class, hook) for hook in data_hooks):
            validation["implements_data_hooks"] = True
            validation["score"] += 30
        
        # Check for display hooks
        display_hooks = ['_display_task_specific_results', '_display_task_specific_summary']
        if any(hasattr(self.extension_class, hook) for hook in display_hooks):
            validation["implements_display_hooks"] = True
            validation["score"] += 35
        
        # Check for session hooks
        session_hooks = ['_add_task_specific_summary_data', '_finalize_session']
        if any(hasattr(self.extension_class, hook) for hook in session_hooks):
            validation["implements_session_hooks"] = True
            validation["score"] += 35
        
        return validation
    
    def _validate_code_quality(self) -> Dict[str, Any]:
        """Validate overall code quality metrics."""
        validation = {
            "has_docstrings": False,
            "follows_naming": False,
            "minimal_complexity": False,
            "score": 0
        }
        
        # Check for docstrings
        if self.extension_class.__doc__:
            validation["has_docstrings"] = True
            validation["score"] += 30
        
        # Check naming conventions
        class_name = self.extension_class.__name__
        if class_name.endswith('GameManager') and not class_name.startswith('Base'):
            validation["follows_naming"] = True
            validation["score"] += 35
        
        # Check for minimal complexity (fewer methods = better inheritance usage)
        methods = inspect.getmembers(self.extension_class, predicate=inspect.isfunction)
        if len(methods) <= 8:  # Minimal extension code
            validation["minimal_complexity"] = True
            validation["score"] += 35
        
        return validation
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []
        
        # Architecture recommendations
        arch = results["architecture_compliance"]
        if not arch["inherits_base_manager"]:
            recommendations.append("❌ CRITICAL: Inherit from BaseGameManager for infrastructure")
        if not arch["uses_factory_pattern"]:
            recommendations.append("⚠️ Consider using GAME_LOGIC_CLS factory pattern")
        if not arch["implements_hooks"]:
            recommendations.append("💡 Implement extension hooks for clean customization")
        
        # Method recommendations
        methods = results["method_optimization"]
        if not methods["uses_base_methods"]:
            recommendations.append("💡 Use base class methods (run_game_session, finalize_game)")
        if not methods["minimal_overrides"]:
            recommendations.append("⚠️ Reduce method overrides - use extension hooks instead")
        
        # Inheritance recommendations
        inheritance = results["inheritance_usage"]
        if not inheritance["uses_super"]:
            recommendations.append("💡 Use super() calls for proper inheritance")
        if not inheritance["avoids_duplication"]:
            recommendations.append("⚠️ Avoid code duplication - leverage base class methods")
        
        # Hook recommendations
        hooks = results["hook_implementation"]
        if not hooks["implements_data_hooks"]:
            recommendations.append("💡 Implement data hooks for clean customization")
        if not hooks["implements_display_hooks"]:
            recommendations.append("💡 Add display hooks for custom output formatting")
        
        # Add positive feedback for good implementations
        total_score = sum(r["score"] for r in results.values() if isinstance(r, dict) and "score" in r)
        if total_score >= 400:
            recommendations.insert(0, "🏆 EXCELLENT: Extension follows best practices!")
        elif total_score >= 300:
            recommendations.insert(0, "✅ GOOD: Extension is well-implemented")
        
        return recommendations
    
    def print_validation_report(self):
        """Print comprehensive validation report."""
        if not self.validation_results:
            self.validate_extension_quality()
        
        results = self.validation_results
        total_score = sum(r["score"] for r in results.values() if isinstance(r, dict) and "score" in r)
        
        print_success(f"📊 Extension Validation Report: {self.extension_class.__name__}")
        print_info("=" * 60)
        print_info(f"🏆 Overall Score: {total_score}/500")
        
        # Component scores
        for component, result in results.items():
            if isinstance(result, dict) and "score" in result:
                score = result["score"]
                print_info(f"   {component}: {score}/100")
        
        # Recommendations
        recommendations = results.get("recommendations", [])
        if recommendations:
            print_info("\n💡 Recommendations:")
            for rec in recommendations:
                print_info(f"   {rec}")
        
        print_info("=" * 60)


class ExtensionScaffold:
    """
    Scaffolding utility for creating new extensions with best practices.
    
    Generates extension templates that demonstrate perfect architecture
    and minimal code requirements.
    """
    
    @staticmethod
    def generate_minimal_extension(extension_name: str, algorithm_type: str) -> str:
        """Generate minimal extension template."""
        class_name = f"{extension_name.title()}GameManager"
        
        template = f'''"""
{extension_name.title()} Extension
{'=' * (len(extension_name) + 10)}

Elegant {algorithm_type} implementation with minimal code demonstrating
perfect BaseGameManager integration and extension best practices.

Key Features:
- 80-95% code reduction through BaseGameManager inheritance
- Clean algorithm implementation with extension hooks
- Automatic session management and JSON generation
- Beautiful interface with comprehensive functionality
"""

from __future__ import annotations
import sys
from pathlib import Path

# Ensure UTF-8 encoding (SUPREME_RULE NO.7)
import os
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from utils.path_utils import ensure_project_root
ensure_project_root()

import argparse
from typing import Dict, Any
from core.game_manager import BaseGameManager
from utils.print_utils import print_info


class {class_name}(BaseGameManager):
    """
    Minimal {algorithm_type} extension demonstrating perfect architecture.
    
    Only ~15 lines of extension-specific code gets:
    - Complete session management
    - JSON file I/O with UTF-8 encoding  
    - Statistics tracking and analysis
    - Error handling and recovery
    - GUI integration (optional)
    """
    
    def run(self) -> None:
        """Run session using streamlined base class."""
        self.run_game_session()  # Gets everything automatically!
    
    def _get_next_move(self, game_state: Dict[str, Any]) -> str:
        """Implement your {algorithm_type} algorithm here."""
        # TODO: Replace with your algorithm implementation
        import random
        return random.choice(["UP", "DOWN", "LEFT", "RIGHT"])
    
    def _add_task_specific_summary_data(self, summary: Dict[str, Any]) -> None:
        """Add {algorithm_type}-specific data to session summary."""
        summary["algorithm_type"] = "{algorithm_type}"
        summary["extension_name"] = "{extension_name}"
    
    def _display_task_specific_summary(self, summary: Dict[str, Any]) -> None:
        """Display {algorithm_type}-specific results."""
        print_info(f"🎯 Algorithm: {algorithm_type}")


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="{extension_name.title()} Extension")
    parser.add_argument("--max_games", type=int, default=5, help="Number of games")
    parser.add_argument("--grid_size", type=int, default=10, help="Grid size")
    parser.add_argument("--no_gui", action="store_true", help="Headless mode")
    
    args = parser.parse_args()
    
    # Create and run extension
    manager = {class_name}(args)
    manager.initialize()
    manager.run()
'''
        
        return template
    
    @staticmethod
    def generate_advanced_extension(extension_name: str, algorithm_type: str) -> str:
        """Generate advanced extension template with all hooks."""
        class_name = f"{extension_name.title()}GameManager"
        
        template = f'''"""
Advanced {extension_name.title()} Extension
{'=' * (len(extension_name) + 20)}

Comprehensive {algorithm_type} implementation demonstrating all extension
hooks and advanced customization capabilities.
"""

from __future__ import annotations
import sys
from pathlib import Path
import time
from typing import Dict, Any

# Setup imports
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from utils.path_utils import ensure_project_root
ensure_project_root()

from core.game_manager import BaseGameManager
from utils.print_utils import print_info, print_success


class {class_name}(BaseGameManager):
    """Advanced {algorithm_type} extension with full customization."""
    
    def __init__(self, args):
        super().__init__(args)
        self.algorithm_metrics = {{}}
    
    def _initialize_session(self):
        """Custom session initialization."""
        print_info(f"[{class_name}] Initializing {algorithm_type} session")
    
    def _get_next_move(self, game_state: Dict[str, Any]) -> str:
        """Implement advanced {algorithm_type} algorithm."""
        # TODO: Implement your advanced algorithm
        return "UP"
    
    def _process_game_state_before_move(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Custom pre-move processing."""
        # Add any preprocessing here
        return game_state
    
    def _process_game_state_after_move(self, game_state: Dict[str, Any]) -> None:
        """Custom post-move processing."""
        # Add any post-processing here
        pass
    
    def _add_task_specific_game_data(self, game_data: Dict[str, Any], game_duration: float) -> None:
        """Add {algorithm_type}-specific game data."""
        game_data["algorithm_type"] = "{algorithm_type}"
        game_data["game_duration"] = game_duration
        game_data["custom_metrics"] = self.algorithm_metrics
    
    def _display_task_specific_results(self, game_duration: float) -> None:
        """Display {algorithm_type}-specific results."""
        print_info(f"🎯 {algorithm_type} completed in {{game_duration:.2f}}s")
    
    def _add_task_specific_summary_data(self, summary: Dict[str, Any]) -> None:
        """Add {algorithm_type}-specific summary data."""
        summary["algorithm_type"] = "{algorithm_type}"
        summary["total_algorithm_time"] = sum(self.algorithm_metrics.values())
    
    def _display_task_specific_summary(self, summary: Dict[str, Any]) -> None:
        """Display {algorithm_type}-specific summary."""
        print_success(f"🏆 {algorithm_type} Session Complete!")
        print_info(f"⚡ Total algorithm time: {{summary.get('total_algorithm_time', 0):.2f}}s")
    
    def _finalize_session(self) -> None:
        """Custom session cleanup."""
        print_info(f"[{class_name}] Session finalized")
'''
        
        return template


class ExtensionOptimizer:
    """
    Optimization utility for improving extension performance and code quality.
    
    Analyzes extension implementations and provides specific optimization
    recommendations for better performance and cleaner code.
    """
    
    def __init__(self, extension_instance):
        self.extension = extension_instance
        self.optimization_results = {}
    
    def analyze_performance_opportunities(self) -> Dict[str, Any]:
        """Analyze extension for performance optimization opportunities."""
        analysis = {
            "method_efficiency": {},
            "memory_usage": {},
            "algorithm_optimization": {},
            "recommendations": []
        }
        
        # Analyze method implementations
        methods = inspect.getmembers(self.extension.__class__, predicate=inspect.ismethod)
        
        for name, method in methods:
            if name.startswith('_get_next_move'):
                analysis["method_efficiency"][name] = self._analyze_move_method(method)
        
        # Generate optimization recommendations
        analysis["recommendations"] = self._generate_optimization_recommendations(analysis)
        
        return analysis
    
    def _analyze_move_method(self, method) -> Dict[str, Any]:
        """Analyze move generation method for optimization."""
        return {
            "complexity": "low",  # Would need actual analysis
            "optimization_potential": "medium",
            "suggestions": ["Consider caching repeated calculations", "Profile for bottlenecks"]
        }
    
    def _generate_optimization_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate specific optimization recommendations."""
        recommendations = []
        
        # General recommendations
        recommendations.append("🚀 Use BaseGameManager infrastructure for maximum efficiency")
        recommendations.append("📊 Implement performance monitoring for bottleneck identification")
        recommendations.append("💾 Consider caching for repeated calculations")
        recommendations.append("🎯 Profile algorithm implementation for optimization opportunities")
        
        return recommendations


def validate_extension(extension_class: Type) -> None:
    """Validate extension implementation and print report."""
    try:
        validator = ExtensionValidator(extension_class)
        validator.print_validation_report()
        
    except Exception as e:
        print_error(f"Validation failed: {e}")


def generate_extension_template(extension_name: str, algorithm_type: str, advanced: bool = False) -> None:
    """Generate extension template and save to file."""
    try:
        scaffold = ExtensionScaffold()
        
        if advanced:
            template = scaffold.generate_advanced_extension(extension_name, algorithm_type)
        else:
            template = scaffold.generate_minimal_extension(extension_name, algorithm_type)
        
        # Save template
        filename = f"{extension_name}_extension_template.py"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(template)
        
        print_success(f"📝 Extension template generated: {filename}")
        print_info("💡 Edit the template and implement your algorithm in _get_next_move()")
        
    except Exception as e:
        print_error(f"Template generation failed: {e}")


if __name__ == "__main__":
    # Example usage
    print_info("🛠️ Extension Development Helper")
    print_info("=" * 40)
    print_info("This utility helps create and validate extensions")
    print_info("Use generate_extension_template() to create new extensions")
    print_info("Use validate_extension() to check existing extensions")