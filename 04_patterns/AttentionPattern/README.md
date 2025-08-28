# AttentionPattern

A mini-project to generate and visualize **self-attention patterns** commonly used in Transformer models.  
This is the sequence-level counterpart to **KernelVisualizer** (which focused on spatial patterns).

## Features
- Basic attention pattern generators:
  - **Identity**: each token attends only to itself
  - **Uniform**: each token attends to all others equally
  - **Band**: each token attends only to its neighbors within ±k
- Visualization with matplotlib heatmaps (headless-friendly)
- CLI interface to generate and save attention heatmaps

## Project Structure
```bash
AttentionPattern/
├── attention_pattern/
│ ├── attention.py # pattern generators
│ └── visualize.py # heatmap visualization
├── main.py # CLI entrypoint
├── tests/ # pytest samples
└── README.md
```

## Installation
```bash
# inside your venv
pip install -r requirements.txt
pip install -r requirements-dev.txt   # optional (pytest, black, ruff)
```

## Quick Start
```bash
cd ../..   # go to 04_patterns/

# Identity pattern, save as PNG
python -m AttentionPattern.main --pattern identity --n 6 --save AttentionPattern/out/identity.png

# Uniform pattern, n=8
python -m AttentionPattern.main --pattern uniform --n 8 --save AttentionPattern/out/uniform.png

# Band pattern, n=10, k=2
python -m AttentionPattern.main --pattern band --n 10 --k 2 --save AttentionPattern/out/band.png
```

## CLI Options
```lua
--pattern identity|uniform|band   Pattern type
--n <int>                         Sequence length
--k <int>                         Band width (for band pattern)
--save <path>                     Path to save heatmap (png/jpg)
--show                            (Optional) show via matplotlib GUI
```

## Notes
- By default, heatmaps are saved to file; use --show only if a GUI is available.
- WSL2/server environments: images are saved with Agg backend (no GUI blocking).
- Useful for learning/teaching how Transformers restrict or bias attention.

