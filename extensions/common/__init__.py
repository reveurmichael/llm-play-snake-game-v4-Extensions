"""
Extensions Common Package
========================

Elegant shared utilities for all Snake Game AI extensions following
SUPREME_RULES with lightweight, extensible, and educational design.

Key Components:
- **config**: Ultra-lightweight configuration constants and schemas
- **utils**: Essential helper functions (dataset, CSV, game state utilities)
- **validation**: Simple validation helpers for data integrity

Design Philosophy:
- SUPREME_RULES compliance with simple logging and canonical patterns
- OOP-based utilities designed for inheritance and extension
- No tight coupling with ML/DL/RL/LLM-specific concepts
- Lightweight and focused - heavy machinery belongs in specific extensions
- Educational value with clear, reusable patterns

Perfect for: Heuristics, Supervised Learning, RL, and all future extensions
Reference: docs/extensions-guideline/final-decision.md
"""

from importlib import import_module
from types import ModuleType
from typing import List

# Re-export the three main sub-packages as attributes so users can write e.g.
#   from extensions.common import utils
# or access `extensions.common.utils.dataset_utils` directly.

config: ModuleType = import_module("extensions.common.config")
utils: ModuleType = import_module("extensions.common.utils")
validation: ModuleType = import_module("extensions.common.validation")

# Import specific constants for easier access
from .config import EXTENSIONS_LOGS_DIR, HEURISTICS_LOG_PREFIX

__all__: List[str] = [
    "config",
    "utils",
    "validation",
    "EXTENSIONS_LOGS_DIR",
    "HEURISTICS_LOG_PREFIX",
] 