# Unified Streamlit Architecture Guide

## 🎯 **Core Streamlit Philosophy**

Streamlit applications in v0.03+ extensions follow:
- **Object-Oriented Design** with clear class hierarchies
- **Script-Runner Pattern** - UI launches scripts via subprocess
- **Modular Dashboard Components** organized in dashboard/ folder
- **Canonical Patterns** - Demonstrates factory patterns and simple logging throughout

### **Educational Value**
- **Consistent User Experience**: Same interface patterns across all extensions
- **Modular Architecture**: Clear separation between UI and backend logic
- **Canonical Patterns**: Factory patterns ensure consistent component creation
- **Simple Logging**: Print statements provide clear operation visibility

## 🏗️ **Base Architecture Pattern**

### **Core Base Class**
```python
import streamlit as st
import subprocess
from abc import ABC, abstractmethod
from utils.factory_utils import SimpleFactory

class BaseExtensionApp(ABC):
    """
    Base class for all extension Streamlit applications
    
    Design Pattern: Template Method Pattern (Streamlit Architecture)
    Purpose: Provides common interface for all Streamlit applications
    Educational Value: Shows how canonical patterns work with
    Streamlit applications while maintaining simple logging.
    
    Reference: final-decision.md for canonical patterns
    """
    
    def __init__(self):
        print_info(f"[BaseExtensionApp] Initializing {self.__class__.__name__}")  # Simple logging - SUPREME_RULES
        self.setup_page_config()
        self.main()
    
    @abstractmethod
    def get_extension_name(self) -> str:
        """Return the human-readable extension name"""
        pass
    
    @abstractmethod
    def get_available_algorithms(self) -> list[str]:
        """Return list of available algorithms"""
        pass
    
    def setup_page_config(self) -> None:
        """Configure Streamlit page settings"""
        st.set_page_config(
            page_title=f"{self.get_extension_name()} Dashboard",
            layout="wide"
        )
        print_info(f"[BaseExtensionApp] Page config set for {self.get_extension_name()}")  # Simple logging
    
    def main(self) -> None:
        """Main application flow"""
        st.title(f"🐍 {self.get_extension_name()} Dashboard")
        self.render_sidebar()
        self.render_main_content()
        print_info(f"[BaseExtensionApp] Main content rendered")  # Simple logging
    
    def render_sidebar(self) -> None:
        """Render sidebar with common controls"""
        with st.sidebar:
            st.selectbox("Algorithm", self.get_available_algorithms())
            st.selectbox("Grid Size", [8, 10, 12, 16, 20])
        print_info(f"[BaseExtensionApp] Sidebar rendered")  # Simple logging
    
    @abstractmethod
    def render_main_content(self) -> None:
        """Render main content - implemented by subclasses"""
        pass
    
    def launch_script(self, script_name: str, **params):
        """Launch script via subprocess with simple logging"""
        print_info(f"[BaseExtensionApp] Launching script: {script_name}")  # Simple logging - SUPREME_RULES
        
        cmd = ["python", f"scripts/{script_name}"]
        for key, value in params.items():
            cmd.extend([f"--{key}", str(value)])
        
        result = subprocess.run(cmd)
        print_info(f"[BaseExtensionApp] Script completed with exit code: {result.returncode}")  # Simple logging
        return result
```

## 🔧 **Extension Implementation**

```python
# extensions/heuristics-v0.03/app.py

from extensions.common.utils.path_utils import ensure_project_root
ensure_project_root()

from utils.factory_utils import SimpleFactory
from extensions.common.app_utils import BaseExtensionApp

class HeuristicStreamlitApp(BaseExtensionApp):
    """
    Heuristic extension Streamlit application
    
    Design Pattern: Template Method Pattern (Streamlit Implementation)
    Purpose: Demonstrates canonical patterns for heuristic algorithms
    Educational Value: Shows how SUPREME_RULES apply to Streamlit applications
    while maintaining simple logging throughout.
    
    Reference: final-decision.md for canonical patterns
    """
    
    def __init__(self):
        # Use canonical factory pattern for initialization
        factory = SimpleFactory()
        factory.register("heuristic", self.__class__)
        
        print_info(f"[HeuristicStreamlitApp] Initializing heuristic dashboard")  # Simple logging - SUPREME_RULES
        super().__init__()
    
    def get_extension_name(self) -> str:
        return "Heuristic Pathfinding"
    
    def get_available_algorithms(self) -> list[str]:
        return ["BFS", "ASTAR", "DFS", "HAMILTONIAN"]
    
    def render_main_content(self) -> None:
        """Render main content with simple logging"""
        print_info(f"[HeuristicStreamlitApp] Rendering main content")  # Simple logging - SUPREME_RULES
        
        tab1, tab2 = st.tabs(["Run", "Evaluate"])
        
        with tab1:
            if st.button("Run Algorithm"):
                print_info(f"[HeuristicStreamlitApp] Run button clicked")  # Simple logging
                self.launch_script("main.py")

if __name__ == "__main__":
    HeuristicStreamlitApp()
```

## 📊 **Simple Logging Standards for Streamlit Operations**

### **Required Logging Pattern (SUPREME_RULES)**
All Streamlit operations MUST use simple print statements as established in `final-decision.md`:

```python
# ✅ CORRECT: Simple logging for Streamlit operations (SUPREME_RULES compliance)
def streamlit_operation(operation_type: str, parameters: dict):
            print_info(f"[Streamlit] Performing {operation_type}")  # Simple logging - REQUIRED
    
    # Streamlit operation logic
    result = perform_streamlit_operation(parameters)
    
    print_success(f"[Streamlit] {operation_type} completed successfully")  # Simple logging
    return result

# ❌ FORBIDDEN: Complex logging frameworks (violates SUPREME_RULES)
# import logging
# logger = logging.getLogger(__name__)

# def streamlit_operation(operation_type: str, parameters: dict):
#     logger.info(f"Performing {operation_type}")  # FORBIDDEN - complex logging
#     # This violates final-decision.md SUPREME_RULES
```

## 🎓 **Educational Applications with Canonical Patterns**

### **Streamlit Pattern Benefits**
- **Consistent User Experience**: Same interface patterns across all extensions
- **Modular Architecture**: Clear separation between UI and backend logic
- **Canonical Patterns**: Factory patterns ensure consistent component creation
- **Educational Value**: Learn Streamlit development through consistent patterns

### **Pattern Consistency**
- **Canonical Method**: All Streamlit components use consistent patterns
- **Simple Logging**: Print statements provide clear operation visibility
- **Educational Value**: Canonical patterns enable predictable learning
- **SUPREME_RULES**: Streamlit systems follow same standards as other components

## 📋 **SUPREME_RULES Implementation Checklist for Streamlit Patterns**

### **Mandatory Requirements**
- [ ] **Canonical Method**: All Streamlit components use consistent patterns (SUPREME_RULES requirement)
- [ ] **Simple Logging**: Uses utils/print_utils.py functions only for all Streamlit operations (final-decision.md compliance)
- [ ] **GOOD_RULES Reference**: References `final-decision.md` in all Streamlit documentation
- [ ] **Pattern Consistency**: Follows canonical patterns across all Streamlit implementations

### **Streamlit-Specific Standards**
- [ ] **Object-Oriented Design**: Clear class hierarchies with inheritance
- [ ] **Script-Runner Pattern**: UI launches scripts via subprocess
- [ ] **Modular Components**: Dashboard components organized in dashboard/ folder
- [ ] **Error Handling**: Simple logging for all error conditions

### **Educational Integration**
- [ ] **Clear Examples**: Simple examples using canonical patterns
- ✅ **Pattern Documentation**: Clear explanation of Streamlit pattern benefits
- ✅ **SUPREME_RULES Compliance**: All examples follow final-decision.md standards
- ✅ **Cross-Reference**: Links to related patterns and principles
- ✅ **Advanced Features**: Professional interfaces with comprehensive functionality
- ✅ **Performance Analysis**: Multi-experiment comparison and visualization
- ✅ **Large File Support**: Handle datasets up to 10GB+ efficiently

## 🎯 **Current Streamlit Implementations**

### **Task0 Dashboard (`app.py`)**
- **Comprehensive analytics**: Game session analysis and replay
- **Multi-mode support**: PyGame and Web replay options
- **Session management**: Continue interrupted sessions
- **Human gameplay**: Interactive human vs AI modes
- **Professional design**: Beautiful interface with organized tabs

### **Heuristics v0.04 Interface (`extensions/heuristics-v0.04/app.py`)**
- **Algorithm selection**: All pathfinding algorithms with descriptions
- **Performance estimates**: Real-time execution time predictions
- **Configuration management**: Intuitive parameter selection
- **Dataset generation**: Automated CSV/JSONL creation
- **SUPREME_RULE NO.5 compliance**: Pure script launcher interface

### **Supervised v0.03 Training App (`extensions/supervised-v0.03/app.py`)**
- **Model training interface**: Beautiful ML training with progress monitoring
- **Dataset validation**: Automatic dataset quality analysis
- **Model comparison**: Advanced benchmarking with statistical analysis
- **Dependency management**: Automatic PyTorch/LightGBM availability checking
- **Performance visualization**: Training metrics and model comparison charts

### **Advanced Replay & File Reader (`extensions/streamlit-app-for-replay-and-read-large-files/app.py`)**
- **Professional replay system**: PyGame and Web replay with speed control
- **Large file reader**: Handle 10GB+ files with efficient pagination
- **Multi-format support**: JSON, CSV, JSONL with format-specific rendering
- **Performance analysis**: Multi-experiment comparison with visual charts
- **Advanced filtering**: File type, size, and name-based filtering

## 🚀 **Streamlit Excellence Standards**

### **SUPREME_RULE NO.5 Compliance**
All Streamlit apps serve as **script launchers only**:
- **Parameter selection**: Intuitive interfaces for configuration
- **Subprocess execution**: Backend script execution with progress monitoring
- **No game logic**: Pure UI/launcher functionality
- **Professional design**: Beautiful, responsive, user-friendly interfaces

### **Advanced Features Implemented**
- **Real-time progress**: Live updates during long-running operations
- **Error handling**: Comprehensive error recovery with helpful messages
- **Performance monitoring**: Execution time estimates and optimization tips
- **Cross-platform compatibility**: UTF-8 encoding and path management
- **Professional polish**: Beautiful design with intuitive workflows

## 🔗 **Cross-References and Integration**

### **Related Documents**
- **`final-decision.md`**: SUPREME_RULES for canonical Streamlit patterns
- **`app.md`**: Application architecture patterns
- **`scripts.md`**: Script architecture and execution
- **`standalone.md`**: Standalone extension principles

### **Implementation Files**
- **`extensions/common/utils/factory_utils.py`**: Canonical factory utilities
- **`extensions/common/utils/path_utils.py`**: Path management with factory patterns
- **`extensions/common/utils/csv_schema.py`**: Schema utilities with factory patterns

### **Educational Resources**
- **Design Patterns**: Streamlit pattern as foundation for web applications
- **SUPREME_RULES**: Canonical patterns ensure consistency across all extensions
- **Simple Logging**: Print statements provide clear operation visibility
- **OOP Principles**: Streamlit pattern demonstrates effective inheritance 