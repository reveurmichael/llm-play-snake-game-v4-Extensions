"""
Heuristic Extension Configuration - Standalone Implementation
===========================================================

Configuration constants specific to heuristic pathfinding algorithms.
Implements standalone principle by defining extension-specific constants
without cross-extension dependencies.

Design Philosophy:
- SSOT: Single source of truth for heuristic configuration
- Standalone: No dependencies on other extensions
- Fail-Fast: Immediate validation of configuration values
- KISS: Simple, direct configuration without fallbacks
"""

from __future__ import annotations

# Heuristic-specific configuration constants
DEFAULT_GRID_SIZE = 10
DEFAULT_MAX_GAMES = 100
DEFAULT_MAX_STEPS = 500
DEFAULT_ALGORITHM = "BFS-512"

# Algorithm-specific constants
BFS_MAX_QUEUE_SIZE = 10000
EXPLANATION_TOKEN_LIMITS = {
    "BFS-512": 512,
    "BFS-1024": 1024, 
    "BFS-2048": 2048,
    "BFS-4096": 4096,
    "BFS-SAFE-GREEDY-4096": 4096
}

# Dataset generation constants
CSV_OUTPUT_ENABLED = True
JSONL_OUTPUT_ENABLED = True
DATASET_VALIDATION_ENABLED = True

# Performance constants
MAX_PATHFINDING_TIME_SECONDS = 30.0
MAX_EXPLANATION_GENERATION_TIME_SECONDS = 10.0

def validate_algorithm_name(algorithm: str) -> bool:
    """
    Validate algorithm name with fail-fast principle.
    
    Args:
        algorithm: Algorithm name to validate
        
    Returns:
        bool: True if valid
        
    Raises:
        ValueError: If algorithm name is invalid (fail-fast)
    """
    valid_algorithms = list(EXPLANATION_TOKEN_LIMITS.keys()) + ["BFS", "BFS-SAFE-GREEDY"]
    
    if algorithm not in valid_algorithms:
        raise ValueError(f"[SSOT] Invalid algorithm: {algorithm}. Valid: {valid_algorithms}")
    
    return True

def validate_grid_size(grid_size: int) -> bool:
    """
    Validate grid size with fail-fast principle.
    
    Args:
        grid_size: Grid size to validate
        
    Returns:
        bool: True if valid
        
    Raises:
        ValueError: If grid size is invalid (fail-fast)
    """
    if not isinstance(grid_size, int):
        raise TypeError(f"[SSOT] Grid size must be integer, got {type(grid_size)}")
    
    if grid_size < 5 or grid_size > 50:
        raise ValueError(f"[SSOT] Grid size must be 5-50, got {grid_size}")
    
    return True

def get_token_limit(algorithm: str) -> int:
    """
    Get token limit for algorithm with fail-fast validation.
    
    Args:
        algorithm: Algorithm name
        
    Returns:
        int: Token limit for explanations
        
    Raises:
        ValueError: If algorithm not found (fail-fast)
    """
    validate_algorithm_name(algorithm)
    
    return EXPLANATION_TOKEN_LIMITS.get(algorithm, 1024)  # Default fallback

# Export all configuration
__all__ = [
    "DEFAULT_GRID_SIZE",
    "DEFAULT_MAX_GAMES", 
    "DEFAULT_MAX_STEPS",
    "DEFAULT_ALGORITHM",
    "BFS_MAX_QUEUE_SIZE",
    "EXPLANATION_TOKEN_LIMITS",
    "CSV_OUTPUT_ENABLED",
    "JSONL_OUTPUT_ENABLED",
    "DATASET_VALIDATION_ENABLED",
    "MAX_PATHFINDING_TIME_SECONDS",
    "MAX_EXPLANATION_GENERATION_TIME_SECONDS",
    "validate_algorithm_name",
    "validate_grid_size", 
    "get_token_limit"
]