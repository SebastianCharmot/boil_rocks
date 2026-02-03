# Image Color Removal & Grayscale Analysis

Two tools for analyzing and processing images by removing a selected color and computing grayscale metrics.

## Files

- `process.py` — Command-line batch processor for 3 images (original, treated, boiled)
- `streamlit_app.py` — Interactive web app with color selection and visualization
- `requirements.txt` — Python dependencies

## Installation

Recommended: use a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Command-line (process.py)

Expects an `images/` directory with `original`, `treated`, and `boiled` images (supports .jpg, .jpeg, .png).

```bash
python process.py [images_dir]
```

Outputs:
- `comparison_3x2.png` — Before/after for each image
- `detailed_3x3.png` — Before, after, and grayscale metrics

### Interactive Web App (streamlit_app.py)

```bash
streamlit run streamlit_app.py
```

Features:
- Upload three images (original, treated, boiled)
- **Click on any pixel to extract that color for removal (in HSV space)**
- Optionally fine-tune with HSV sliders (Hue, Saturation, Value)
- Preview color removal results in real-time with adjustable HSV tolerance
- View computed grayscale metrics
- See mapped values (original → 0, treated → 100, boiled → interpolated)
- Download composite images

## How it Works

1. **Color Detection**: For a selected color, the app identifies pixels where that color's dominant channel is significantly higher than the others.
2. **Mask Creation**: Creates a transparency mask based on color similarity.
3. **Grayscale Analysis**: Computes average luminosity (0.299R + 0.587G + 0.114B) only on non-masked pixels.
4. **Mapping**: Maps original → 0 and treated → 100 on the grayscale scale; boiled's position is interpolated.
5. **Composites**: Generates visual summaries (3×2 and 3×3 grids).

