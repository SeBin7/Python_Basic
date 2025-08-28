# KernelVisualizer

A mini-project to apply canonical **2D convolution kernels (filters)** to images and visualize the results.  
Runs headless on WSL2 or server environments (no GUI required).

## Features
- Canonical kernels: `sobel_x`, `sobel_y`, `laplacian_3x3`, `sharpen_3x3`,  
  `gaussian_3x3/5x5`, `box_blur_3x3/5x5`, `emboss_3x3`, `identity_3x3`
- Pure NumPy-based 2D correlation (for educational purposes), with padding/stride and border modes
- Headless friendly: `MPLBACKEND=Agg` is automatically set, default behavior is save-only
- Visualization normalization: `--viz none|abs|minmax` (useful for edge/derivative kernels)

## Project Structure
```bash
KernelVisualizer/
├── assets/ # sample images
├── kernel_visualizer/
│ ├── kernels.py # kernel registry (immutable float32 arrays)
│ ├── apply.py # 2D correlation (H×W / RGB)
│ └── visualize.py # headless-safe plotting helper
├── out/ # example outputs
├── main.py # CLI entrypoint (headless-friendly)
└── tests/ # pytest unit tests
```

## Installation
```bash
# inside your venv
pip install -r requirements.txt
pip install -r requirements-dev.txt   # optional: pytest/ruff/black
```

## Quick Start
```bash
cd ../..   # go to 04_patterns/

# Apply Gaussian blur
python -m KernelVisualizer.main \
  --image KernelVisualizer/assets/cat.jpg \
  --kernel gaussian_5x5

# Apply Sobel X (grayscale) with abs normalization
python -m KernelVisualizer.main \
  --image KernelVisualizer/assets/cat.jpg \
  --gray --kernel sobel_x --viz abs \
  --save KernelVisualizer/out/cat_sobelx.png

# Apply Laplacian with minmax normalization
python -m KernelVisualizer.main \
  --image KernelVisualizer/assets/cat.jpg \
  --gray --kernel laplacian_3x3 --viz minmax \
  --save KernelVisualizer/out/cat_laplacian.png
```
## CLI Options
```lua
--image <path>          Input image (required)
--kernel <name>         Kernel name
--gray                  Convert to grayscale before filtering
--padding same|valid    Default = same
--stride <int>          Default = 1
--border reflect|constant|edge
--save <path>           Output path (if omitted: <input_stem>_<kernel>.png)
--show                  (Optional) show with matplotlib (GUI required)
--viz none|abs|minmax   Visualization normalization (recommended for edges)
```
## Notes
Educational implementation → can be slow on very high-res images.

For quick tests: downscale to 256–512px.

For production speed: replace with SciPy convolve2d.

## Test
```bash
cd KernelVisualizer
python -m pytest -q
```

