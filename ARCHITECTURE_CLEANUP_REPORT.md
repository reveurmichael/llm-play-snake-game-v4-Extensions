# 🚀 Forward-Looking Architecture Cleanup

## ✅ **Mission: Clean ROOT Folder from Extension Pollution**

Following the forward-looking architecture principles:
- **No Backward Compatibility**: Remove deprecated code entirely
- **No Code Pollution**: Remove extension-specific terminology from ROOT
- **No Over-Preparation**: Remove over-prepared code for future tasks

## 🎯 **Files Requiring Cleanup**

### **Core Files with Extension Pollution:**
- `core/game_data.py` - Contains HeuristicGameData, RLGameData classes
- `core/game_manager.py` - References heuristics, RL, supervised learning
- `core/game_summary_generator.py` - Extension-specific comments
- `core/game_loop.py` - Heuristic/RL specific logic
- `core/game_stats_manager.py` - Extension-specific examples
- `core/game_rounds.py` - Heuristic-specific round logic
- `core/game_agents.py` - Extension agent references

### **Config Files with Extension Pollution:**
- `config/game_constants.py` - Heuristic/RL/supervised references
- `config/prompt_templates.py` - Heuristic references

### **LLM Files with Extension Pollution:**
- `llm/parsing_utils.py` - Heuristic references
- `llm/agent_llm.py` - Extension agent references

## 🧹 **Cleanup Strategy**

### **1. Remove Extension-Specific Classes**
- Remove `HeuristicGameData` from `core/game_data.py`
- Remove `RLGameData` from `core/game_data.py`
- Remove any over-prepared extension classes

### **2. Clean Comments and Documentation**
- Replace "heuristics", "RL", "supervised" with generic "extensions"
- Remove specific algorithm references (BFS, DFS, A*, etc.)
- Keep only generic, future-proof terminology

### **3. Remove Over-Preparation**
- Remove unused extension hooks
- Remove over-engineered base classes
- Keep only what Task0 actually needs

### **4. Maintain Self-Contained Extensions**
- Ensure extensions don't pollute ROOT
- Extensions should be fully self-contained
- No extension terminology in ROOT folder

## 🎯 **Target Architecture**

### **Clean ROOT Structure:**
```
ROOT/
├── core/           # Generic game engine (no extension terminology)
├── config/         # Generic game configuration  
├── llm/           # Task0 LLM-specific code only
├── utils/         # Generic utilities
└── extensions/    # All extension-specific code
```

### **Generic Terminology Only:**
- "extensions" instead of "heuristics", "RL", "supervised"
- "agents" instead of specific agent types
- "algorithms" instead of BFS, DFS, A*
- "models" instead of MLP, LightGBM

## ✅ **Implementation Plan**

1. **Clean core/game_data.py**: Remove extension classes, generic comments
2. **Clean core/game_manager.py**: Remove extension references
3. **Clean core/game_loop.py**: Generic agent handling
4. **Clean core/game_stats*.py**: Generic statistics handling  
5. **Clean config files**: Remove extension-specific constants
6. **Clean llm files**: Remove extension references
7. **Validate**: Ensure no extension terminology in ROOT

## 🏆 **Expected Outcome**

- **Pure ROOT folder**: No extension-specific terminology
- **Self-contained extensions**: All extension code in extensions/
- **Future-proof**: Clean foundation for any future tasks
- **No backward compatibility**: Fresh, consistent architecture

This cleanup ensures the ROOT folder is truly generic and extension-agnostic, following forward-looking architecture principles.