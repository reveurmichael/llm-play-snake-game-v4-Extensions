import sys
import subprocess
from pathlib import Path
from typing import List

import streamlit as st


try:
    from utils.path_utils import ensure_project_root
except ModuleNotFoundError:
    # Fallback if executed in an unusual environment – resolve root manually
    project_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(project_root))
else:
    project_root = ensure_project_root()

# Ensure we run everything from project root so that relative paths work
# Note: We don't change directory here to avoid Streamlit path issues
# The path setup above ensures imports work correctly

# ---------------------------------------------------------------------------
# Import helper from unified dataset CLI to retrieve available algorithms
# ---------------------------------------------------------------------------

try:
    from extensions.common.utils.dataset_generator_cli import find_available_algorithms
except Exception:
    # Safe-fallback list in case of import issues – keeps UI usable.
    find_available_algorithms = lambda: [  # type: ignore
        "BFS",
        "BFS-SAFE-GREEDY",
    ]

# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def get_algorithm_description(algorithm: str) -> str:
    """Get user-friendly description of algorithm."""
    descriptions = {
        "BFS": "Breadth-First Search - Optimal pathfinding with guaranteed shortest path",
        "BFS-SAFE-GREEDY": "Safe Greedy BFS - BFS with safety-first approach",
        "BFS-512": "BFS with concise explanations (~512 tokens)",
        "BFS-1024": "BFS with moderate explanations (~1024 tokens)",
        "BFS-2048": "BFS with detailed explanations (~2048 tokens)",
        "BFS-4096": "BFS with comprehensive explanations (~4096 tokens)",
        "BFS-SAFE-GREEDY-4096": "Safe Greedy BFS with full explanations (~4096 tokens)",
        "ASTAR": "A* Search - Heuristic-guided optimal pathfinding",
        "DFS": "Depth-First Search - Memory-efficient exploration",
        "HAMILTONIAN": "Hamiltonian Cycle - Guaranteed survival strategy"
    }
    return descriptions.get(algorithm, "Advanced pathfinding algorithm")

st.set_page_config(page_title="Heuristics v0.04 Dataset Generator", page_icon="🐍", layout="wide")

st.title("🐍 Heuristics v0.04 Dataset Generator")
st.markdown("**Generate comprehensive datasets from heuristic pathfinding algorithms**")
st.markdown("---")

# Sidebar – parameter selection ------------------------------------------------
with st.sidebar:
    st.header("⚙️ Parameters")

    # Whether to process all algorithms or a single one ----------------------
    all_algorithms = st.checkbox("Process ALL algorithms", value=False)
    available_algorithms: List[str] = find_available_algorithms()

    if all_algorithms:
        algorithm = None  # Will be handled via CLI flag
    else:
        algorithm = st.selectbox("Algorithm", available_algorithms)

    dataset_format = st.selectbox(
        "Dataset format",
        ["both", "csv", "jsonl"],
        index=0,
        help="Choose which dataset files to generate."
    )

    grid_size: int = st.slider("Grid size", min_value=5, max_value=25, value=10)
    max_games: int = st.number_input("Max games", min_value=1, max_value=1000000000000, value=10)
    max_steps: int = st.number_input("Max steps per game", min_value=100, max_value=10000, value=500)
    verbose: bool = st.checkbox("Verbose output", value=False)
    
    # Add helpful information
    st.markdown("---")
    st.markdown("### 📊 Algorithm Info")
    if not all_algorithms and algorithm:
        st.info(f"**{algorithm}**: {get_algorithm_description(algorithm)}")
    
    st.markdown("### 📁 Output")
    st.info("Datasets will be saved to `logs/extensions/datasets/` with timestamp.")
    
    st.markdown("### 📈 Performance Estimates")
    if not all_algorithms and algorithm and max_games:
        # Estimate execution time based on algorithm and games
        time_per_game = {
            "BFS": 0.1, "BFS-SAFE-GREEDY": 0.15, "ASTAR": 0.08, 
            "DFS": 0.12, "HAMILTONIAN": 0.2
        }
        base_time = time_per_game.get(algorithm.split("-")[0], 0.1)
        estimated_time = base_time * max_games
        
        if estimated_time < 60:
            st.success(f"⚡ Estimated time: ~{estimated_time:.1f} seconds")
        elif estimated_time < 3600:
            st.warning(f"⏱️ Estimated time: ~{estimated_time/60:.1f} minutes")
        else:
            st.error(f"🕐 Estimated time: ~{estimated_time/3600:.1f} hours")
    
    st.markdown("### 💡 Tips")
    st.info("💡 Start with small max_games (10-50) to test, then increase for full datasets.")

# Main area – action -----------------------------------------------------------

# Show current configuration summary
st.markdown("### 🎯 Current Configuration")
col1, col2, col3, col4 = st.columns(4)

with col1:
    if all_algorithms:
        st.metric("Algorithms", "ALL", help="Will process all available algorithms")
    else:
        st.metric("Algorithm", algorithm or "None")

with col2:
    st.metric("Grid Size", f"{grid_size}x{grid_size}")

with col3:
    st.metric("Max Games", max_games)

with col4:
    st.metric("Format", dataset_format.upper())

st.markdown("---")

if st.button("🚀 Generate Dataset", type="primary"):
    # Build command ----------------------------------------------------------
    extension_dir = Path(__file__).parent  # heuristics-v0.04 directory
    script_path = extension_dir / "scripts" / "main.py"

    cmd: List[str] = [sys.executable, str(script_path)]

    if all_algorithms:
        cmd.append("--all-algorithms")
    else:
        if algorithm is None:
            st.error("Please select an algorithm or enable ALL algorithms option.")
            st.stop()
        cmd.extend(["--algorithm", algorithm])

    cmd.extend(["--format", dataset_format])
    cmd.extend(["--grid-size", str(grid_size)])
    cmd.extend(["--max-games", str(max_games)])
    cmd.extend(["--max-steps", str(max_steps)])

    if verbose:
        cmd.append("--verbose")

    # Display the command for transparency ----------------------------------
    st.code(" ".join(cmd), language="bash")

    # Run the command --------------------------------------------------------
    with st.spinner("Running dataset generation… this may take a while 🤖"):
        result = subprocess.run(
            cmd,
            cwd=str(extension_dir),  # Execute inside the extension folder
            capture_output=True,
            text=True,
        )

    # Output handling -------------------------------------------------------
    if result.returncode == 0:
        st.success("Dataset generation completed successfully ✨")
    else:
        st.error(f"Dataset generation failed (exit code {result.returncode})")

    # Always show stdout / stderr for transparency --------------------------
    with st.expander("📜 Script output (stdout + stderr)"):
        st.text(result.stdout + "\n" + result.stderr) 