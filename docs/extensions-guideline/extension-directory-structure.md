# Extension Directory Structure & Evolution Standards

## 🎯 **Current Streamlined Architecture**

Based on the enhanced BaseGameManager and successful implementations of heuristics-v0.04 and supervised-v0.03, extensions now follow a dramatically simplified structure.

## 🌟 **Minimal Extension Template (12 lines)**

### **Ultra-Simple Extensions** - Complete functionality with minimal code:

```python
class MyExtensionManager(BaseGameManager):
    def run(self):
        self.run_game_session()  # Gets everything automatically!
    
    def _get_next_move(self, game_state):
        return my_algorithm(game_state)  # Your algorithm here
    
    def _add_task_specific_summary_data(self, summary):
        summary["algorithm"] = "my_algorithm"
```

**Directory Structure:**
```
extensions/my-extension-v0.03/
├── __init__.py                    # Package exports
├── game_manager.py                # 12-50 lines total
├── main.py                        # CLI interface
└── README.md                      # Documentation
```

```python
class StandardExtensionManager(BaseGameManager):
    def run(self):
        self.run_game_session()
    
    def _get_next_move(self, game_state):
        # Your algorithm implementation
        return algorithm.predict(game_state)
    
    def _add_task_specific_game_data(self, game_data, game_duration):
        game_data["algorithm"] = self.algorithm_name
        game_data["custom_metrics"] = self.calculate_metrics()
    
    def _display_task_specific_summary(self, summary):
        print_info(f"🧠 Algorithm: {self.algorithm_name}")
        print_info(f"📊 Custom metric: {summary.get('custom_metric', 0)}")
```

**Directory Structure:**
```
extensions/standard-extension-v0.03/
├── __init__.py                    # Package exports
├── game_manager.py                # 30-50 lines with hooks
├── game_logic.py                  # Algorithm-specific logic (optional)
├── models.py                      # Algorithm implementations
├── main.py                        # CLI interface
├── app.py                         # Streamlit launcher (SUPREME_RULE NO.5)
└── README.md                      # Documentation
```

## 🏗️ **Advanced Extension Template (60-100 lines)**

### **Complex Extensions** - Full customization with all hooks:

```python
class AdvancedExtensionManager(BaseGameManager):
    def _initialize_session(self):
        # Custom session setup
        
    def _initialize_game_specific_rounds(self):
        # Custom rounds initialization
        
    def _process_game_state_before_move(self, game_state):
        # Custom pre-processing
        return processed_state
    
    def _get_next_move(self, game_state):
        # Complex algorithm
        
    def _validate_move_custom(self, move, game_state):
        # Custom validation
        
    def _process_game_state_after_move(self, game_state):
        # Custom post-processing
        
    def _finalize_task_specific(self, game_data, game_duration):
        # Custom finalization (e.g., dataset updates)
        
    def _create_extension_subdirectories(self):
        # Custom directory structure
```

**Directory Structure:**
```
extensions/advanced-extension-v0.04/
├── __init__.py                    # Package exports
├── game_manager.py                # 60-100 lines with all hooks
├── game_logic.py                  # Custom game logic
├── game_data.py                   # Custom data handling
├── agents/                        # Algorithm implementations
│   ├── __init__.py
│   ├── agent_primary.py
│   └── agent_secondary.py
├── dataset_generator.py           # Custom dataset generation
├── state_management.py            # Custom state handling
├── scripts/
│   ├── main.py                   # Enhanced CLI
│   └── utilities.py              # Helper scripts
├── app.py                         # Streamlit launcher
└── README.md                      # Comprehensive docs
```

## 📊 **Current Successful Implementations**

### **Heuristics v0.04** - Advanced Extension Example:
```
extensions/heuristics-v0.04/
├── __init__.py
├── game_manager.py                # ~400 lines (streamlined from 520)
├── game_logic.py                  # ~360 lines (clean, focused)
├── game_data.py                   # Heuristic-specific data
├── agents/                        # Multi-algorithm support
├── dataset_generator.py           # CSV/JSONL generation
├── state_management.py            # Pre/post-move validation
├── scripts/main.py                # CLI interface
├── app.py                         # Streamlit launcher
└── README.md
```

### **Supervised v0.03** - Standard Extension Example:
```
extensions/supervised-v0.03/
├── __init__.py
├── game_manager.py                # ~180 lines (ultra-streamlined)
├── game_logic.py                  # ~150 lines (ML integration)
├── models.py                      # MLP and LightGBM models
├── main.py                        # CLI interface
└── README.md                      # Comprehensive documentation
```

## 🎯 **Key Benefits of Streamlined Architecture**

### **Dramatic Code Reduction:**
- ✅ **Minimal Extensions**: 12 lines for complete functionality
- ✅ **Standard Extensions**: 30-50 lines vs previous 200+ lines
- ✅ **Advanced Extensions**: 60-100 lines vs previous 300+ lines
- ✅ **Infrastructure Inherited**: 80-95% of code provided by BaseGameManager

### **Automatic Features Included:**
- ✅ **Session Management**: Start/end tracking with timestamps
- ✅ **Statistics Collection**: Game scores, steps, rounds, performance metrics
- ✅ **JSON File I/O**: Automatic saving/loading with UTF-8 encoding
- ✅ **Directory Management**: Organized structure with custom subdirectories
- ✅ **Game Controller**: Automatic GUI/headless detection and setup
- ✅ **Rounds Management**: Step-by-step tracking with validation
- ✅ **Error Handling**: Comprehensive error recovery throughout
- ✅ **Limits Management**: Automatic game limits enforcement

### **Extension Development Experience:**
- ✅ **Focus on Algorithm**: Extensions implement only algorithm-specific logic
- ✅ **Template Method Pattern**: Consistent structure with customization hooks
- ✅ **Factory Pattern Support**: Easy algorithm/model selection
- ✅ **SOLID Compliance**: Clean inheritance and composition
- ✅ **Educational Value**: Perfect examples of software engineering principles

## 🔗 **Related Documentation**

- **[`core.md`](core.md)**: Enhanced BaseGameManager architecture details
- **[`final-decision.md`](final-decision.md)**: SUPREME_RULES governance system
- **[`extensions-v0.04.md`](extensions-v0.04.md)**: Advanced extension patterns
- **[`factory-design-pattern.md`](factory-design-pattern.md)**: Factory implementation guide

---

**The streamlined extension architecture demonstrates how proper abstraction and template method patterns can reduce extension code by 80-95% while maintaining full functionality and enhancing code quality.**
