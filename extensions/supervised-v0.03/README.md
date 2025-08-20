# Supervised Learning Extension v0.03

This extension demonstrates supervised learning approaches for Snake Game AI, including Multi-Layer Perceptron (MLP) and LightGBM models trained on heuristic-generated datasets.

## Features

- **Multiple ML Models**: Support for MLP (PyTorch) and LightGBM
- **Dataset Integration**: Train on heuristic-generated CSV datasets
- **Clean Architecture**: Extends BaseGameManager with minimal code
- **Performance Metrics**: Model accuracy and prediction time tracking
- **Grid-Size Agnostic**: Works with any grid size (10x10, 15x15, etc.)

## Installation

### Dependencies

```bash
# Core dependencies (already installed in main project)
pip install numpy pandas

# Optional: For MLP model
pip install torch torchvision

# Optional: For LightGBM model  
pip install lightgbm
```

## Usage

### Basic Usage

```bash
# Run with MLP model (requires PyTorch)
python main.py --model MLP --dataset /path/to/dataset.csv --max_games 5

# Run with LightGBM model (requires LightGBM)
python main.py --model LightGBM --dataset /path/to/dataset.csv --max_games 5

# Run without dataset (untrained model)
python main.py --model MLP --max_games 3
```

### Advanced Options

```bash
# Custom grid size
python main.py --model MLP --dataset data.csv --grid_size 15 --max_games 5

# Headless mode (no GUI)
python main.py --model LightGBM --dataset data.csv --no_gui --max_games 10

# Verbose training output
python main.py --model MLP --dataset data.csv --verbose --max_games 3
```

### Using Heuristics-Generated Datasets

First, generate a dataset using the heuristics extension:

```bash
# Generate dataset with heuristics
cd ../heuristics-v0.04
python main.py --algorithm BFS --max_games 100 --grid_size 10

# Use the generated dataset for supervised learning
cd ../supervised-v0.03
python main.py --model MLP --dataset ../heuristics-v0.04/logs/.../BFS_dataset.csv --max_games 5
```

## Architecture

### Design Patterns

- **Template Method**: Inherits BaseGameManager session management
- **Factory Pattern**: Model creation through `create_model()` function
- **Strategy Pattern**: Pluggable ML models (MLP, LightGBM)

### Class Hierarchy

```
BaseGameManager (core)
└── SupervisedGameManager
    └── Uses SupervisedGameLogic
        └── Uses BaseModel implementations
            ├── MLPModel (PyTorch)
            └── LightGBMModel (LightGBM)
```

### Feature Engineering

The models use 16 grid-size agnostic features:

1. **Head Position** (4 features): Normalized x, y, distances to walls
2. **Apple Position** (4 features): Normalized position, distance from head
3. **Movement Directions** (4 features): Valid moves in each direction
4. **Body Proximity** (4 features): Danger detection in each direction

## Model Details

### MLP Model (PyTorch)

- **Architecture**: 16 → 64 → 32 → 16 → 4 neurons
- **Activation**: ReLU with Dropout (0.2)
- **Optimizer**: Adam (lr=0.001)
- **Training**: 50 epochs with early stopping

### LightGBM Model

- **Type**: Gradient boosting classifier
- **Parameters**: 31 leaves, 0.1 learning rate
- **Features**: Supports feature importance analysis
- **Training**: 100 rounds with early stopping

## Output

The extension generates:

- **Game Logs**: Individual game results (`game_1.json`, etc.)
- **Session Summary**: Overall performance metrics (`summary.json`)
- **Console Output**: Real-time game progress and statistics

### Sample Output

```
🧠 Supervised Learning Snake Game AI v0.03
==================================================
[SupervisedGameManager] Training MLP model...
[SupervisedGameManager] Model trained with accuracy: 0.847
✅ 🚀 Starting supervised learning v0.03 session...
📊 Target games: 5
🧠 Model: MLP
🎯 Model accuracy: 0.847

🎮 Game 1
📊 Score: 12, Steps: 45, Duration: 0.23s
🧠 Model: MLP, Avg prediction time: 0.0023s

...

✅ ✅ Supervised learning v0.03 session completed!
🎮 Games played: 5
🏆 Total score: 45
📈 Average score: 9.0
```

## Integration with Other Extensions

### Data Flow

```
Heuristics v0.04 → CSV Dataset → Supervised v0.03 → Performance Analysis
```

### Comparison Framework

The extension outputs are compatible with the evaluation framework for comparing:
- Heuristic algorithms (BFS, A*, etc.)
- Supervised learning models (MLP, LightGBM)
- Reinforcement learning agents
- LLM-based approaches

## Development

### Adding New Models

1. Inherit from `BaseModel`
2. Implement `train()` and `predict()` methods
3. Add to `create_model()` factory function

```python
class NewModel(BaseModel):
    def __init__(self, dataset_path=None, verbose=False):
        super().__init__("NewModel")
    
    def train(self, dataset_path=None) -> float:
        # Training logic
        return accuracy
    
    def predict(self, game_state: Dict[str, Any]) -> str:
        # Prediction logic
        return move
```

### Extension Points

- **Custom Features**: Override `extract_features()` in models
- **Training Strategies**: Modify training loops in model classes
- **Evaluation Metrics**: Add custom metrics in `SupervisedGameManager`

## Educational Value

This extension demonstrates:

- **Supervised Learning**: Training models on expert demonstrations
- **Feature Engineering**: Converting game states to ML features
- **Model Comparison**: Comparing different ML approaches
- **Clean Architecture**: Extending base classes with minimal code
- **Real-world ML**: Practical machine learning pipeline implementation