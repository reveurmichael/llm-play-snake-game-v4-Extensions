![](./img/a.jpg)

# 🐍 Snake Game AI: Comprehensive Multi-Algorithm Framework

An advanced, educational Snake Game AI framework featuring LLM-powered gameplay, heuristic algorithms, supervised learning, and comprehensive extensibility. Designed for research, education, and AI development.

## 🎯 **Project Overview**

This project provides a complete ecosystem for Snake Game AI development with:
- **Task-0**: LLM-powered gameplay with multi-provider support
- **Extensions**: Heuristic algorithms, supervised learning, and more
- **Research Tools**: Comprehensive data generation and analysis
- **Educational Value**: Perfect for learning AI and software engineering

## 🚀 **Quick Start**

### **Task-0: LLM-Powered Snake Game**

Run the intelligent Snake game with various LLM providers:

```bash
# Single LLM (recommended for beginners)
python scripts/main.py --provider ollama --model deepseek-r1:7b

# Dual LLM (Mixture-of-Experts approach)
python scripts/main.py --provider ollama --model deepseek-r1:7b --parser-provider ollama --parser-model gemma2:9b

# Cloud providers
python scripts/main.py --provider hunyuan --model hunyuan-t1-latest
python scripts/main.py --provider deepseek --model deepseek-chat
```

### **Extensions: Complete AI Ecosystem**

```bash
# Heuristic Pathfinding Algorithms
cd extensions/heuristics-v0.04
python scripts/main.py --algorithm BFS-1024 --max-games 100  # Generate training data
streamlit run app.py  # Beautiful algorithm interface with performance estimates

# Supervised Machine Learning
cd extensions/supervised-v0.03  
python main.py --model MLP --dataset ../heuristics-v0.04/logs/.../dataset.csv --max-games 50
streamlit run app.py  # ML training interface with model comparison

# Advanced Analysis Tools
streamlit run extensions/streamlit-app-for-replay-and-read-large-files/app.py
# Professional replay system + large file reader (handles 10GB+ files)

# Task0 Comprehensive Dashboard
streamlit run app.py  # Complete analytics, replay, and session management
```

### **Research Workflow Example**
```bash
# 1. Generate high-quality training data
cd extensions/heuristics-v0.04
python scripts/main.py --algorithm BFS-2048 --max-games 1000 --grid-size 15

# 2. Train ML models on heuristic data
cd ../supervised-v0.03
python main.py --model LightGBM --dataset ../heuristics-v0.04/logs/.../BFS_dataset.csv

# 3. Compare AI approaches
streamlit run ../streamlit-app-for-replay-and-read-large-files/app.py
# Use performance comparison tools to analyze results

# 4. Advanced analysis
python ../heuristics-v0.04/analysis.py logs/heuristics_session/
python model_comparison.py  # Compare ML models with heuristics
```

## 🌟 **Key Features**

### **🧠 Advanced AI Integration**
- **Multi-Provider LLM Support**: OpenAI, Anthropic, DeepSeek, Hunyuan, Ollama
- **Mixture-of-Experts**: Dual LLM architecture for enhanced reasoning
- **Heuristic Algorithms**: BFS, A*, DFS, Hamiltonian pathfinding
- **Supervised Learning**: MLP and LightGBM models with automatic training
- **Robust Error Handling**: Graceful recovery from AI failures

### **📊 Comprehensive Data Generation**
- **Automatic Dataset Creation**: CSV for ML, JSONL for LLM fine-tuning
- **Real-time Statistics**: Performance metrics and game analytics
- **Session Management**: Complete game session tracking and replay
- **Cross-Platform Compatibility**: UTF-8 encoding and path management

### **🎮 Multiple Interfaces**
- **PyGame GUI**: Visual game interface with real-time display
- **Web Interface**: Browser-based gameplay and replay
- **Streamlit Dashboard**: Analytics, replay, and file management
- **CLI Interface**: Headless execution for automation and research

### **🔬 Research and Educational Tools**
- **Extension Framework**: Easy development of new AI algorithms
- **Replay System**: Analyze AI decision-making processes
- **Large File Reader**: Handle datasets up to 10GB+ efficiently
- **Performance Comparison**: Compare different AI approaches

## 📦 **Installation**

### **Quick Setup**
```bash
# Clone the repository
git clone <repository-url>
cd snake-game-ai

# Install dependencies
pip install -r requirements.txt

# Set up API keys (optional - for cloud LLM providers)
cp .env.example .env
# Edit .env with your API keys
```

### **API Configuration**
Set up API keys in a `.env` file:

```bash
OLLAMA_HOST=<IP_ADDRESS_OF_OLLAMA_SERVER>
HUNYUAN_API_KEY=your_hunyuan_api_key_here
DEEPSEEK_API_KEY=your_deepseek_api_key_here
MISTRAL_API_KEY=your_mistral_api_key_here
```

## 🏗️ **Advanced Architecture**

### **Perfect Foundation (Task0)**
- **Enhanced BaseGameManager**: 800+ lines of reusable infrastructure
- **Template Method Pattern**: Enables 80-95% code reduction in extensions
- **Factory Pattern Excellence**: Canonical `create()` method implementation
- **SSOT Implementation**: Single source of truth eliminating duplication
- **Multi-Interface Support**: PyGame, Web, CLI, Streamlit options

### **Extension Ecosystem**
- **Heuristics v0.04**: Complete pathfinding algorithms with dataset generation
- **Supervised v0.03**: ML models (MLP, LightGBM) with training pipeline
- **Streamlit Apps**: Professional analysis and file management tools
- **Extensions/Common**: Perfect shared utilities with SSOT compliance

### **Two-LLM Architecture (MoE Approach)**

Task0 implements a Mixture-of-Experts inspired approach where two specialized LLMs work together:

1. **Primary LLM (Game Strategy Expert)**
   - Takes the game state as input
   - Analyzes the Snake's position, apple location, and available moves
   - Generates a logical plan to navigate toward the apple
   - May or may not generate properly structured output

2. **Secondary LLM (Formatting Expert)**
   - Takes the primary LLM's output as input
   - Specializes in parsing and formatting the response
   - Ensures the final output follows the required JSON format
   - Acts as a guarantor of response quality
   - Can be disabled with `--parser-provider none` to use primary LLM output directly

## Command Line Arguments

- `--provider`: LLM provider for the primary LLM (hunyuan, ollama, deepseek, or mistral)
- `--model`: Model name for the primary LLM
- `--parser-provider`: LLM provider for the secondary LLM (defaults to primary provider if not specified). Use "none" to skip using a parser.
- `--parser-model`: Model name for the secondary LLM
- `--max-games`: Maximum number of games to play
- `--pause-between-moves`: Pause between sequential moves in seconds
- `--max-steps`: Maximum steps a snake can take in a single game (default: 400)
- `--sleep-before-launching`: Time to sleep (in minutes) before launching the program
- `--max-empty-moves`: Maximum consecutive empty moves before game over
- `--no-gui`: Run without the graphical interface (text-only mode)
- `--log-dir`: Directory to store log files

## Project Structure

The codebase is organized in a modular structure to ensure maintainability and separation of concerns:

- `/core`: Core game engine components
- `/gui`: Graphical user interface components
- `/llm`: Language model integration  
- `/utils`: Utility modules  
- `/replay`: Replay functionality
- `/web`: Web version of the Snake game
  
- Main modules:
  - `main.py`: Entry point with command-line argument parsing
  - `config.py`: Configuration constants
  - `app.py`: Streamlit dashboard for analyzing game statistics and replaying games
  - `replay.py`: Command-line interface for replaying saved games (pygame version)
  - `replay_web.py`: Command-line interface for replaying saved games (web version)
  - `human_play.py`: Human-playable version of the Snake game (pygame version)
  - `human_play_web.py`: Human-playable version of the Snake game (web version)

## Data Output

The system generates structured output for each game session:

- `game_N.json`: Contains complete data for game number N, including moves, statistics, and time metrics
- `summary.json`: Contains aggregated statistics for the entire session
- `prompts/`: Directory containing all prompts sent to the LLMs
- `responses/`: Directory containing all responses received from the LLMs

## Game Termination Conditions

The snake game will terminate under any of the following conditions:
1. Snake hits a wall (boundary of the game board)
2. Snake collides with its own body
3. Maximum steps limit is reached (default: 400 steps)
4. Three consecutive empty moves occur without ERROR
   - Empty moves are checked immediately when detected
   - An empty move occurs when the LLM returns `{"moves":[], "reasoning":"..."}`
   - If the reasoning contains "SOMETHING_IS_WRONG", the consecutive count is reset
5. A game error occurs
   - The system will catch errors, log them, and continue to the next game
   - Error information is saved in the game summary

After game termination (for any reason), the system will automatically start the next game until the maximum number of games is reached.

## How this Project Resembles a Real Research Project

- **Comprehensive Logging**: Extensive JSON file logging that tracks nearly everything, potentially useful for future analysis
- **Multiple Execution Modes**: Supports both visual and non-visual (headless) modes
- **Replay Functionality**: Enables result verification and analysis
- **Analysis Dashboard**: Includes tools for preliminary result analysis and experiment parameter adjustment
- **Rapid Prototyping**: Supports quick testing with minimal parameters:
  ```
  python main.py --provider ollama --model mistral:7b --parser-provider ollama --parser-model mistral:7b --max-games 1 --no-gui --sleep-before-launching 1 --max-steps 3 --max-consecutive-something-is-wrong-allowed 0
  ```

## What's Missing for a Complete Research Project

- **Automated Launch System**: More script-based and parallelized application launching instead of command-line arguments
- **Resource Management**: Helper scripts for GPU usage monitoring and application launch coordination
- **Logging System**: Implementation of standard logging levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- **Data Analysis Pipeline**: Comprehensive data processing and analysis tools (for reporting and paper publication)

## If You Want to Change the Code

As a fundamental rule, the schema of our `game_N.json` and `summary.json` files is now fixed and should never be modified. You can, on the contrary, modify the code to change how those values are calculated.








