# Remove green pixels from an image

This small script loads `after.JPG` (or a given path), removes green pixels by making them transparent, saves outputs, and displays before/after images.

Files created:
- `process.py` — main script
- `after_no_green.png` — processed image with transparency
- `after_no_green.jpg` — processed image with green replaced by white (JPEG)

Install dependencies (recommended in a venv):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run:

```bash
python process.py [path/to/after.JPG]
```

If you want just the saved files without opening an interactive window, run in an environment without GUI or modify `process.py` to skip plotting.
