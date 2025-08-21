"""
Supervised Learning Agents
=========================

Factory for creating supervised learning agents that use trained ML models
for intelligent Snake game decision making.

Design Philosophy:
- Agent pattern: Each model type has its own agent implementation
- JSON output: Agents generate game data in JSON format (no JSONL needed)
- No explanations: Models provide direct move predictions without explanations
- CSV training: Models train on CSV datasets from logs folder
"""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from utils.print_utils import print_info, print_warning, print_error
from typing import Dict, Any, Optional


class SupervisedAgentFactory:
    """Factory for creating supervised learning agents."""
    
    _agents = {}
    
    @classmethod
    def register_agent(cls, name: str, agent_class):
        """Register an agent class."""
        cls._agents[name] = agent_class
    
    @classmethod
    def create_agent(cls, agent_name: str, **kwargs):
        """Create agent by name."""
        if agent_name not in cls._agents:
            available = list(cls._agents.keys())
            print_error(f"Unknown agent: {agent_name}. Available: {available}")
            return None
        
        try:
            agent_class = cls._agents[agent_name]
            return agent_class(**kwargs)
        except Exception as e:
            print_error(f"Failed to create agent {agent_name}: {e}")
            return None
    
    @classmethod
    def list_available_agents(cls) -> Dict[str, str]:
        """List available agents with descriptions."""
        agents = {}
        for name, agent_class in cls._agents.items():
            description = getattr(agent_class, 'description', 'No description available')
            agents[name] = description
        return agents


# Import and register agents
def _register_agents():
    """Register all available agents."""
    try:
        from .agent_mlp import MLPAgent
        SupervisedAgentFactory.register_agent("mlp", MLPAgent)
    except ImportError as e:
        print_warning(f"MLP agent not available: {e}")
    
    try:
        from .agent_lightgbm import LightGBMAgent
        SupervisedAgentFactory.register_agent("lightgbm", LightGBMAgent)
    except ImportError as e:
        print_warning(f"LightGBM agent not available: {e}")


# Register agents on import
_register_agents()

# Factory instance for easy access
agent_factory = SupervisedAgentFactory()

__all__ = ["SupervisedAgentFactory", "agent_factory"]