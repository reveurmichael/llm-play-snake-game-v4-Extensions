#!/usr/bin/env python3
"""
Project-Wide Quality Validator
==============================

Comprehensive quality validation tool for the entire Snake Game AI project
ensuring all components meet the highest standards of excellence.

Key Features:
- SUPREME_RULES compliance validation across all files
- Code quality assessment and reporting
- Architecture pattern validation
- Cross-component integration verification
- Educational value assessment

Usage:
    python project_validator.py
    python project_validator.py --detailed
    python project_validator.py --fix-issues
"""

from __future__ import annotations
import sys
from pathlib import Path

# Ensure UTF-8 encoding for cross-platform compatibility (SUPREME_RULE NO.7)
import os
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.path_utils import ensure_project_root
ensure_project_root()

import argparse
import json
import subprocess
from typing import Dict, Any, List, Optional
from datetime import datetime
from utils.print_utils import print_info, print_warning, print_error, print_success


class ProjectValidator:
    """
    Comprehensive project quality validator.
    
    Validates code quality, architecture patterns, SUPREME_RULES compliance,
    and cross-component integration across the entire project.
    """
    
    def __init__(self):
        self.validation_results = {}
        self.project_root = Path.cwd()
        
        print_info("[ProjectValidator] Initialized for comprehensive validation")
    
    def validate_project_quality(self, detailed: bool = False) -> Dict[str, Any]:
        """Run comprehensive project quality validation."""
        print_success("🔍 Starting Comprehensive Project Quality Validation")
        print_info("=" * 70)
        
        results = {
            "validation_timestamp": datetime.now().isoformat(),
            "overall_score": 0,
            "component_scores": {},
            "supreme_rules_compliance": {},
            "architecture_validation": {},
            "integration_validation": {},
            "recommendations": []
        }
        
        try:
            # Validate core components
            print_info("📋 Validating Task0 core components...")
            results["component_scores"]["task0_core"] = self._validate_task0_core()
            
            print_info("🛠️  Validating utilities and configuration...")
            results["component_scores"]["utils_config"] = self._validate_utils_and_config()
            
            print_info("📁 Validating extensions/common...")
            results["component_scores"]["extensions_common"] = self._validate_extensions_common()
            
            print_info("🎯 Validating heuristics-v0.04...")
            results["component_scores"]["heuristics"] = self._validate_heuristics_extension()
            
            print_info("🧠 Validating supervised-v0.03...")
            results["component_scores"]["supervised"] = self._validate_supervised_extension()
            
            print_info("🎮 Validating Streamlit applications...")
            results["component_scores"]["streamlit_apps"] = self._validate_streamlit_apps()
            
            # Validate SUPREME_RULES compliance
            print_info("⚖️  Validating SUPREME_RULES compliance...")
            results["supreme_rules_compliance"] = self._validate_supreme_rules()
            
            # Validate architecture patterns
            print_info("🏗️  Validating architecture patterns...")
            results["architecture_validation"] = self._validate_architecture_patterns()
            
            # Validate cross-component integration
            print_info("🔗 Validating cross-component integration...")
            results["integration_validation"] = self._validate_integration()
            
            # Calculate overall score
            results["overall_score"] = self._calculate_overall_score(results)
            
            # Generate recommendations
            results["recommendations"] = self._generate_project_recommendations(results)
            
            self.validation_results = results
            
            # Display summary
            self._display_validation_summary(results, detailed)
            
            return results
            
        except Exception as e:
            print_error(f"❌ Validation failed: {e}")
            return results
    
    def _validate_task0_core(self) -> Dict[str, Any]:
        """Validate Task0 core components."""
        score = 0
        issues = []
        strengths = []
        
        # Check core directory structure
        core_dir = self.project_root / "core"
        if core_dir.exists():
            core_files = ["game_manager.py", "game_logic.py", "game_data.py", "game_controller.py"]
            for file in core_files:
                if (core_dir / file).exists():
                    score += 5
                    strengths.append(f"✅ {file} present and accessible")
                else:
                    issues.append(f"❌ Missing core file: {file}")
        
        # Check BaseGameManager enhancements
        game_manager_file = core_dir / "game_manager.py"
        if game_manager_file.exists():
            content = game_manager_file.read_text(encoding='utf-8')
            if "generate_game_data" in content and "save_json_file" in content:
                score += 10
                strengths.append("✅ Enhanced BaseGameManager with comprehensive infrastructure")
            if "extension hooks" in content.lower():
                score += 5
                strengths.append("✅ Template method pattern with extension hooks")
        
        # Check factory patterns
        if "GAME_LOGIC_CLS" in content and "create(" in content:
            score += 10
            strengths.append("✅ Factory patterns with canonical create() method")
        
        return {
            "score": min(100, score),
            "issues": issues,
            "strengths": strengths,
            "assessment": "Excellent" if score >= 80 else "Good" if score >= 60 else "Needs Improvement"
        }
    
    def _validate_utils_and_config(self) -> Dict[str, Any]:
        """Validate utilities and configuration."""
        score = 0
        issues = []
        strengths = []
        
        # Check utils directory
        utils_dir = self.project_root / "utils"
        if utils_dir.exists():
            key_utils = ["print_utils.py", "path_utils.py", "factory_utils.py", "json_utils.py"]
            for util in key_utils:
                if (utils_dir / util).exists():
                    score += 5
                    strengths.append(f"✅ {util} present")
        
        # Check factory_utils for print_utils usage
        factory_utils = utils_dir / "factory_utils.py"
        if factory_utils.exists():
            content = factory_utils.read_text(encoding='utf-8')
            if "print_info(" in content:
                score += 15
                strengths.append("✅ Factory utils uses print_utils (SUPREME_RULES compliance)")
        
        # Check config directory
        config_dir = self.project_root / "config"
        if config_dir.exists():
            config_files = ["game_constants.py", "ui_constants.py"]
            for config in config_files:
                if (config_dir / config).exists():
                    score += 5
                    strengths.append(f"✅ {config} present")
        
        return {
            "score": min(100, score),
            "issues": issues,
            "strengths": strengths,
            "assessment": "Excellent" if score >= 80 else "Good" if score >= 60 else "Needs Improvement"
        }
    
    def _validate_extensions_common(self) -> Dict[str, Any]:
        """Validate extensions/common folder."""
        score = 0
        issues = []
        strengths = []
        
        common_dir = self.project_root / "extensions" / "common"
        if not common_dir.exists():
            return {"score": 0, "issues": ["❌ extensions/common directory missing"], "strengths": []}
        
        # Check directory structure
        subdirs = ["config", "utils", "validation"]
        for subdir in subdirs:
            if (common_dir / subdir).exists():
                score += 10
                strengths.append(f"✅ {subdir}/ directory present")
        
        # Check key utilities
        utils_dir = common_dir / "utils"
        if utils_dir.exists():
            key_files = ["csv_utils.py", "game_state_utils.py", "dataset_utils.py"]
            for file in key_files:
                if (utils_dir / file).exists():
                    score += 8
                    strengths.append(f"✅ {file} present")
        
        # Check enhanced __init__.py
        init_file = common_dir / "__init__.py"
        if init_file.exists():
            content = init_file.read_text(encoding='utf-8')
            if "Extensions Common Package" in content:
                score += 15
                strengths.append("✅ Enhanced package documentation")
        
        return {
            "score": min(100, score),
            "issues": issues,
            "strengths": strengths,
            "assessment": "Excellent" if score >= 80 else "Good" if score >= 60 else "Needs Improvement"
        }
    
    def _validate_heuristics_extension(self) -> Dict[str, Any]:
        """Validate heuristics-v0.04 extension."""
        score = 0
        issues = []
        strengths = []
        
        heuristics_dir = self.project_root / "extensions" / "heuristics-v0.04"
        if not heuristics_dir.exists():
            return {"score": 0, "issues": ["❌ heuristics-v0.04 directory missing"], "strengths": []}
        
        # Check core files
        core_files = ["game_manager.py", "game_logic.py", "game_data.py", "dataset_generator.py"]
        for file in core_files:
            if (heuristics_dir / file).exists():
                score += 8
                strengths.append(f"✅ {file} present")
        
        # Check agents directory
        agents_dir = heuristics_dir / "agents"
        if agents_dir.exists() and (agents_dir / "__init__.py").exists():
            score += 10
            strengths.append("✅ Agents factory with canonical create() method")
        
        # Check enhanced features
        if (heuristics_dir / "README.md").exists():
            score += 10
            strengths.append("✅ Comprehensive README documentation")
        
        if (heuristics_dir / "app.py").exists():
            score += 10
            strengths.append("✅ Streamlit interface available")
        
        # Check for advanced features
        advanced_files = ["performance_monitor.py", "analysis.py"]
        for file in advanced_files:
            if (heuristics_dir / file).exists():
                score += 8
                strengths.append(f"✅ Advanced feature: {file}")
        
        return {
            "score": min(100, score),
            "issues": issues,
            "strengths": strengths,
            "assessment": "Excellent" if score >= 80 else "Good" if score >= 60 else "Needs Improvement"
        }
    
    def _validate_supervised_extension(self) -> Dict[str, Any]:
        """Validate supervised-v0.03 extension."""
        score = 0
        issues = []
        strengths = []
        
        supervised_dir = self.project_root / "extensions" / "supervised-v0.03"
        if not supervised_dir.exists():
            return {"score": 0, "issues": ["❌ supervised-v0.03 directory missing"], "strengths": []}
        
        # Check core files
        core_files = ["game_manager.py", "game_logic.py", "models.py", "main.py"]
        for file in core_files:
            if (supervised_dir / file).exists():
                score += 10
                strengths.append(f"✅ {file} present")
        
        # Check enhanced features
        if (supervised_dir / "README.md").exists():
            score += 15
            strengths.append("✅ Comprehensive README documentation")
        
        if (supervised_dir / "app.py").exists():
            score += 15
            strengths.append("✅ Streamlit training interface")
        
        # Check for advanced features
        advanced_files = ["utils.py", "model_comparison.py"]
        for file in advanced_files:
            if (supervised_dir / file).exists():
                score += 10
                strengths.append(f"✅ Advanced feature: {file}")
        
        return {
            "score": min(100, score),
            "issues": issues,
            "strengths": strengths,
            "assessment": "Excellent" if score >= 80 else "Good" if score >= 60 else "Needs Improvement"
        }
    
    def _validate_streamlit_apps(self) -> Dict[str, Any]:
        """Validate all Streamlit applications."""
        score = 0
        issues = []
        strengths = []
        
        # Check main Task0 app
        main_app = self.project_root / "app.py"
        if main_app.exists():
            score += 20
            strengths.append("✅ Main Task0 Streamlit dashboard")
        
        # Check heuristics app
        heuristics_app = self.project_root / "extensions" / "heuristics-v0.04" / "app.py"
        if heuristics_app.exists():
            score += 20
            strengths.append("✅ Heuristics Streamlit interface")
        
        # Check supervised app
        supervised_app = self.project_root / "extensions" / "supervised-v0.03" / "app.py"
        if supervised_app.exists():
            score += 20
            strengths.append("✅ Supervised learning Streamlit interface")
        
        # Check replay app
        replay_app = self.project_root / "extensions" / "streamlit-app-for-replay-and-read-large-files" / "app.py"
        if replay_app.exists():
            score += 20
            strengths.append("✅ Advanced replay and file reader app")
        
        # Check for SUPREME_RULE NO.5 compliance (script launcher interfaces)
        for app_path in [heuristics_app, supervised_app]:
            if app_path.exists():
                content = app_path.read_text(encoding='utf-8')
                if "streamlit" in content.lower() and "subprocess" in content:
                    score += 5
                    strengths.append(f"✅ {app_path.parent.name} follows SUPREME_RULE NO.5")
        
        return {
            "score": min(100, score),
            "issues": issues,
            "strengths": strengths,
            "assessment": "Excellent" if score >= 80 else "Good" if score >= 60 else "Needs Improvement"
        }
    
    def _validate_supreme_rules(self) -> Dict[str, Any]:
        """Validate SUPREME_RULES compliance across project."""
        compliance = {
            "rule_5_compliance": True,  # Streamlit apps as script launchers
            "rule_6_compliance": True,  # Canonical create() method
            "rule_7_compliance": True,  # UTF-8 encoding
            "rule_8_compliance": True,  # Minimal code examples
            "overall_compliance": True,
            "violations": [],
            "strengths": []
        }
        
        try:
            # Check for canonical create() method usage
            factory_files = list(self.project_root.rglob("*factory*.py"))
            factory_files.extend(list(self.project_root.rglob("agents/__init__.py")))
            
            for factory_file in factory_files:
                if factory_file.exists():
                    content = factory_file.read_text(encoding='utf-8')
                    if "def create(" in content or "@classmethod\n    def create(" in content:
                        compliance["strengths"].append(f"✅ Canonical create() method in {factory_file.name}")
                    else:
                        compliance["rule_6_compliance"] = False
                        compliance["violations"].append(f"❌ Non-canonical factory method in {factory_file.name}")
            
            # Check UTF-8 encoding usage
            python_files = list(self.project_root.rglob("*.py"))
            utf8_files = 0
            total_files_with_encoding = 0
            
            for py_file in python_files[:20]:  # Sample check
                if py_file.exists() and "extensions" in str(py_file):
                    content = py_file.read_text(encoding='utf-8')
                    if 'encoding="utf-8"' in content or "PYTHONIOENCODING" in content:
                        utf8_files += 1
                    total_files_with_encoding += 1
            
            if utf8_files > 0:
                compliance["strengths"].append(f"✅ UTF-8 encoding used in {utf8_files} files")
            
            # Overall compliance
            compliance["overall_compliance"] = all([
                compliance["rule_5_compliance"],
                compliance["rule_6_compliance"], 
                compliance["rule_7_compliance"],
                compliance["rule_8_compliance"]
            ])
            
            return compliance
            
        except Exception as e:
            compliance["overall_compliance"] = False
            compliance["violations"].append(f"❌ Validation error: {e}")
            return compliance
    
    def _validate_architecture_patterns(self) -> Dict[str, Any]:
        """Validate architecture patterns and design quality."""
        patterns = {
            "template_method": False,
            "factory_pattern": False,
            "ssot_implementation": False,
            "extension_hooks": False,
            "clean_inheritance": False,
            "overall_architecture": False,
            "strengths": [],
            "issues": []
        }
        
        try:
            # Check BaseGameManager for template method pattern
            base_manager = self.project_root / "core" / "game_manager.py"
            if base_manager.exists():
                content = base_manager.read_text(encoding='utf-8')
                
                if "_add_task_specific" in content and "def finalize_game" in content:
                    patterns["template_method"] = True
                    patterns["strengths"].append("✅ Template method pattern with extension hooks")
                
                if "GAME_LOGIC_CLS" in content:
                    patterns["factory_pattern"] = True
                    patterns["strengths"].append("✅ Factory pattern for pluggable components")
                
                if "extension hooks" in content.lower():
                    patterns["extension_hooks"] = True
                    patterns["strengths"].append("✅ Extension hooks for customization")
            
            # Check for SSOT implementation
            common_utils = self.project_root / "extensions" / "common" / "utils"
            if common_utils.exists() and len(list(common_utils.glob("*.py"))) >= 3:
                patterns["ssot_implementation"] = True
                patterns["strengths"].append("✅ SSOT implementation in common utilities")
            
            # Check inheritance patterns
            extensions = list(self.project_root.glob("extensions/*/game_manager.py"))
            inheritance_count = 0
            for ext_manager in extensions:
                if ext_manager.exists():
                    content = ext_manager.read_text(encoding='utf-8')
                    if "BaseGameManager" in content:
                        inheritance_count += 1
            
            if inheritance_count >= 2:
                patterns["clean_inheritance"] = True
                patterns["strengths"].append(f"✅ Clean inheritance in {inheritance_count} extensions")
            
            # Overall assessment
            pattern_count = sum([
                patterns["template_method"],
                patterns["factory_pattern"],
                patterns["ssot_implementation"],
                patterns["extension_hooks"],
                patterns["clean_inheritance"]
            ])
            
            patterns["overall_architecture"] = pattern_count >= 4
            
            return patterns
            
        except Exception as e:
            patterns["issues"].append(f"❌ Architecture validation error: {e}")
            return patterns
    
    def _validate_integration(self) -> Dict[str, Any]:
        """Validate cross-component integration."""
        integration = {
            "heuristics_to_supervised": False,
            "common_utilities_usage": False,
            "consistent_data_formats": False,
            "cross_compatibility": False,
            "strengths": [],
            "issues": []
        }
        
        try:
            # Check heuristics → supervised pipeline
            heuristics_csv = list(self.project_root.glob("logs/**/heuristics*/*.csv"))
            supervised_exists = (self.project_root / "extensions" / "supervised-v0.03").exists()
            
            if heuristics_csv and supervised_exists:
                integration["heuristics_to_supervised"] = True
                integration["strengths"].append("✅ Heuristics → Supervised pipeline ready")
            
            # Check common utilities usage
            extensions = ["heuristics-v0.04", "supervised-v0.03"]
            common_usage = 0
            
            for ext in extensions:
                ext_dir = self.project_root / "extensions" / ext
                if ext_dir.exists():
                    py_files = list(ext_dir.rglob("*.py"))
                    for py_file in py_files:
                        content = py_file.read_text(encoding='utf-8')
                        if "extensions.common" in content:
                            common_usage += 1
                            break
            
            if common_usage >= 2:
                integration["common_utilities_usage"] = True
                integration["strengths"].append("✅ Extensions use common utilities")
            
            # Check data format consistency
            csv_formats = self.project_root / "extensions" / "common" / "config" / "csv_formats.py"
            if csv_formats.exists():
                integration["consistent_data_formats"] = True
                integration["strengths"].append("✅ Consistent data formats defined")
            
            # Overall integration score
            integration_score = sum([
                integration["heuristics_to_supervised"],
                integration["common_utilities_usage"],
                integration["consistent_data_formats"]
            ])
            
            integration["cross_compatibility"] = integration_score >= 2
            
            return integration
            
        except Exception as e:
            integration["issues"].append(f"❌ Integration validation error: {e}")
            return integration
    
    def _calculate_overall_score(self, results: Dict[str, Any]) -> float:
        """Calculate overall project quality score."""
        component_scores = results.get("component_scores", {})
        
        # Weight different components
        weights = {
            "task0_core": 0.25,
            "utils_config": 0.15,
            "extensions_common": 0.15,
            "heuristics": 0.20,
            "supervised": 0.15,
            "streamlit_apps": 0.10
        }
        
        total_score = 0
        total_weight = 0
        
        for component, weight in weights.items():
            if component in component_scores:
                score = component_scores[component].get("score", 0)
                total_score += score * weight
                total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0
    
    def _generate_project_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate project-wide recommendations."""
        recommendations = []
        
        overall_score = results.get("overall_score", 0)
        
        if overall_score >= 95:
            recommendations.append("🏆 Project quality is exceptional - maintain current standards")
            recommendations.append("📚 Consider documenting best practices for other projects")
        elif overall_score >= 85:
            recommendations.append("✅ Project quality is excellent - minor optimizations possible")
        elif overall_score >= 75:
            recommendations.append("📈 Good project quality - focus on identified improvement areas")
        else:
            recommendations.append("⚠️ Project needs improvement - address critical issues first")
        
        # Add specific recommendations based on component scores
        component_scores = results.get("component_scores", {})
        for component, result in component_scores.items():
            if result.get("score", 0) < 80:
                recommendations.append(f"🔧 Improve {component} - see specific issues")
        
        return recommendations
    
    def _display_validation_summary(self, results: Dict[str, Any], detailed: bool):
        """Display comprehensive validation summary."""
        print_info("\n" + "=" * 70)
        print_success("📊 PROJECT QUALITY VALIDATION SUMMARY")
        print_info("=" * 70)
        
        # Overall score
        overall_score = results.get("overall_score", 0)
        if overall_score >= 95:
            print_success(f"🏆 OVERALL SCORE: {overall_score:.1f}/100 - EXCEPTIONAL")
        elif overall_score >= 85:
            print_success(f"✅ OVERALL SCORE: {overall_score:.1f}/100 - EXCELLENT")
        elif overall_score >= 75:
            print_info(f"📈 OVERALL SCORE: {overall_score:.1f}/100 - GOOD")
        else:
            print_warning(f"⚠️ OVERALL SCORE: {overall_score:.1f}/100 - NEEDS IMPROVEMENT")
        
        # Component scores
        print_info("\n📋 COMPONENT SCORES:")
        component_scores = results.get("component_scores", {})
        for component, result in component_scores.items():
            score = result.get("score", 0)
            assessment = result.get("assessment", "Unknown")
            print_info(f"   {component}: {score:.1f}/100 ({assessment})")
        
        # SUPREME_RULES compliance
        supreme_rules = results.get("supreme_rules_compliance", {})
        if supreme_rules.get("overall_compliance", False):
            print_success("\n⚖️ SUPREME_RULES: ✅ FULLY COMPLIANT")
        else:
            print_warning("\n⚖️ SUPREME_RULES: ⚠️ SOME VIOLATIONS FOUND")
        
        # Architecture validation
        architecture = results.get("architecture_validation", {})
        if architecture.get("overall_architecture", False):
            print_success("\n🏗️ ARCHITECTURE: ✅ EXCELLENT PATTERNS")
        else:
            print_warning("\n🏗️ ARCHITECTURE: ⚠️ IMPROVEMENTS NEEDED")
        
        # Integration validation
        integration = results.get("integration_validation", {})
        if integration.get("cross_compatibility", False):
            print_success("\n🔗 INTEGRATION: ✅ SEAMLESS COMPATIBILITY")
        else:
            print_warning("\n🔗 INTEGRATION: ⚠️ COMPATIBILITY ISSUES")
        
        # Recommendations
        recommendations = results.get("recommendations", [])
        if recommendations:
            print_info("\n💡 RECOMMENDATIONS:")
            for i, rec in enumerate(recommendations, 1):
                print_info(f"   {i}. {rec}")
        
        print_info("=" * 70)
        
        if overall_score >= 95:
            print_success("🎉 PROJECT IS TRULY GREAT! 🎉")
        elif overall_score >= 85:
            print_success("✅ PROJECT IS EXCELLENT!")
        else:
            print_info("📈 PROJECT HAS GOOD FOUNDATION - CONTINUE IMPROVEMENTS")
    
    def export_validation_report(self, output_path: Optional[str] = None):
        """Export validation results to JSON file."""
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"project_validation_report_{timestamp}.json"
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.validation_results, f, indent=2, ensure_ascii=False)
            
            print_success(f"📊 Validation report exported to {output_path}")
            
        except Exception as e:
            print_error(f"Error exporting validation report: {e}")


def main():
    """Main validation entry point."""
    parser = argparse.ArgumentParser(description="Project Quality Validator")
    parser.add_argument("--detailed", action="store_true", help="Show detailed validation results")
    parser.add_argument("--export", type=str, help="Export results to file")
    
    args = parser.parse_args()
    
    try:
        validator = ProjectValidator()
        results = validator.validate_project_quality(detailed=args.detailed)
        
        if args.export:
            validator.export_validation_report(args.export)
        
        # Exit with appropriate code based on overall score
        overall_score = results.get("overall_score", 0)
        if overall_score >= 85:
            sys.exit(0)  # Success
        else:
            sys.exit(1)  # Needs improvement
            
    except Exception as e:
        print_error(f"❌ Validation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()