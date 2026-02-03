"""Streamlit app for batch processing images: remove selected colors and compute metrics.

Users can:
1. Upload three images (original, treated, boiled)
2. Click on each image to select a color to remove
3. View processed images with the color removed
4. See computed grayscale metrics and mapping (original -> 0, treated -> 100)
5. Download composite images (3x2 and 3x3 detailed view)
"""

import io
import os
from typing import Tuple, List, Optional

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from streamlit_image_coordinates import streamlit_image_coordinates


st.set_page_config(page_title="Image Color Removal & Analysis", layout="wide")
st.title("Image Color Removal & Grayscale Analysis")


def hsv_to_rgb(h: float, s: float, v: float) -> Tuple[int, int, int]:
    """Convert HSV (H: 0-360, S: 0-1, V: 0-1) to RGB (0-255)."""
    import colorsys
    r, g, b = colorsys.hsv_to_rgb(h / 360.0, s, v)
    return (int(r * 255), int(g * 255), int(b * 255))


def rgb_to_hsv(r: int, g: int, b: int) -> Tuple[int, int, int]:
    """Convert RGB (0-255) to HSV (H: 0-180, S: 0-255, V: 0-255)."""
    r_norm, g_norm, b_norm = r / 255.0, g / 255.0, b / 255.0
    max_val = max(r_norm, g_norm, b_norm)
    min_val = min(r_norm, g_norm, b_norm)
    delta = max_val - min_val
    
    v_hsv = max_val
    s_hsv = delta / max_val if max_val != 0 else 0
    
    if delta == 0:
        h_hsv = 0
    elif max_val == r_norm:
        h_hsv = (60 * (((g_norm - b_norm) / delta) % 6)) % 360
    elif max_val == g_norm:
        h_hsv = (60 * ((b_norm - r_norm) / delta + 2)) % 360
    else:
        h_hsv = (60 * ((r_norm - g_norm) / delta + 4)) % 360
    
    h_uint = int((h_hsv / 2) % 180)
    s_uint = int(s_hsv * 255)
    v_uint = int(v_hsv * 255)
    
    return (h_uint, s_uint, v_uint)


def compute_hsv_mask(arr: np.ndarray, target_color: Tuple[int, int, int], 
                     h_threshold: int = 15, s_threshold: int = 40, v_threshold: int = 40) -> np.ndarray:
    """Return boolean mask where True indicates a pixel matching target_color in HSV space.
    
    arr: HxWx3 or HxWx4 RGB(A) uint8 array
    target_color: (H, S, V) tuple for color to remove (HSV space, H: 0-180, S: 0-255, V: 0-255)
    """
    if arr.shape[2] == 4:
        rgb = arr[..., :3]
    else:
        rgb = arr
    
    # Convert RGB to HSV
    rgb_norm = rgb.astype(np.float32) / 255.0
    r = rgb_norm[..., 0]
    g = rgb_norm[..., 1]
    b = rgb_norm[..., 2]
    
    max_val = np.maximum(np.maximum(r, g), b)
    min_val = np.minimum(np.minimum(r, g), b)
    delta = max_val - min_val
    
    v = max_val
    # Suppress divide by zero warning - we handle it with np.where
    with np.errstate(divide='ignore', invalid='ignore'):
        s = np.where(max_val != 0, delta / max_val, 0)
    
    h = np.zeros_like(r)
    mask_r = (max_val == r) & (delta != 0)
    mask_g = (max_val == g) & (delta != 0)
    mask_b = (max_val == b) & (delta != 0)
    
    h[mask_r] = (60 * (((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6)) % 360
    h[mask_g] = (60 * ((b[mask_g] - r[mask_g]) / delta[mask_g] + 2)) % 360
    h[mask_b] = (60 * ((r[mask_b] - g[mask_b]) / delta[mask_b] + 4)) % 360
    
    h_uint = (h / 2).astype(np.uint8)
    s_uint = (s * 255).astype(np.uint8)
    v_uint = (v * 255).astype(np.uint8)
    
    target_h, target_s, target_v = target_color
    
    h_diff = np.abs(h_uint.astype(np.int16) - int(target_h))
    h_diff = np.minimum(h_diff, 180 - h_diff)
    
    s_diff = np.abs(s_uint.astype(np.int16) - int(target_s))
    v_diff = np.abs(v_uint.astype(np.int16) - int(target_v))
    
    mask = (h_diff <= h_threshold) & (s_diff <= s_threshold) & (v_diff <= v_threshold)
    
    return mask


def compute_rgb_mask(arr: np.ndarray, target_color: Tuple[int, int, int],
                     ratio: float = 1.2, diff_thresh: int = 40) -> np.ndarray:
    """Return boolean mask where True indicates a pixel matching target_color in RGB space.
    
    Uses adaptive logic: pixel matches if the dominant channel is significantly
    higher than the other two.
    
    arr: HxWx3 or HxWx4 RGB(A) uint8 array
    target_color: (R, G, B) tuple for color to remove (0-255)
    """
    if arr.shape[2] == 4:
        rgb = arr[..., :3]
    else:
        rgb = arr
    
    r = rgb[..., 0].astype(np.int16)
    g = rgb[..., 1].astype(np.int16)
    b = rgb[..., 2].astype(np.int16)
    
    target_r, target_g, target_b = int(target_color[0]), int(target_color[1]), int(target_color[2])
    
    # Determine which channel is dominant in the target color
    channels = [target_r, target_g, target_b]
    dominant_idx = np.argmax(channels)
    
    # Build a mask based on the dominant color channel
    if dominant_idx == 0:  # Red dominant
        mask = (r > (ratio * g)) & (r > (ratio * b)) & ((r - np.maximum(g, b)) > diff_thresh)
    elif dominant_idx == 1:  # Green dominant
        mask = (g > (ratio * r)) & (g > (ratio * b)) & ((g - np.maximum(r, b)) > diff_thresh)
    else:  # Blue dominant
        mask = (b > (ratio * r)) & (b > (ratio * g)) & ((b - np.maximum(r, g)) > diff_thresh)
    
    return mask


def remove_color_pixels(img: Image.Image, target_color: Tuple[int, int, int], color_space: str,
                        h_threshold: int = 15, s_threshold: int = 40, v_threshold: int = 40) -> Tuple[Image.Image, np.ndarray]:
    """Return RGBA image with target color pixels made transparent and the mask."""
    arr = np.array(img.convert("RGB"))
    
    if color_space == "HSV":
        mask = compute_hsv_mask(arr, target_color, h_threshold=h_threshold, s_threshold=s_threshold, v_threshold=v_threshold)
    else:
        # For RGB: h_threshold is ratio, s_threshold is diff_thresh
        ratio = h_threshold  
        diff_thresh = s_threshold
        mask = compute_rgb_mask(arr, target_color, ratio=ratio, diff_thresh=diff_thresh)
    
    rgba_arr = np.array(img.convert("RGBA"))
    out = rgba_arr.copy()
    out[..., 3] = np.where(mask, 0, rgba_arr[..., 3])
    
    return Image.fromarray(out), mask


def avg_grayscale_where_not_masked(arr: np.ndarray, mask: np.ndarray) -> float:
    """Compute average grayscale (luminosity) on pixels where mask is False."""
    r = arr[..., 0].astype(np.float32)
    g = arr[..., 1].astype(np.float32)
    b = arr[..., 2].astype(np.float32)
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    
    valid = ~mask
    if not np.any(valid):
        return float('nan')
    return float(lum[valid].mean())


def make_3x2_composite(orig_imgs: List[Image.Image], proc_imgs: List[Image.Image]) -> Image.Image:
    """Create a 3x2 composite: [original | processed] for each of 3 images."""
    rows = len(orig_imgs)
    if rows == 0:
        return None
    w, h = orig_imgs[0].size
    comp = Image.new('RGB', (w * 2, h * rows), (255, 255, 255))
    
    for i, (o, p) in enumerate(zip(orig_imgs, proc_imgs)):
        comp.paste(o.convert('RGB'), (0, i * h))
        white = Image.new('RGB', p.size, (255, 255, 255))
        pasted = Image.alpha_composite(white.convert('RGBA'), p).convert('RGB')
        comp.paste(pasted, (w, i * h))
    
    return comp


def make_3x3_detailed(orig_imgs: List[Image.Image], proc_imgs: List[Image.Image], 
                      mapped_vals: List[float], names: List[str]) -> Image.Image:
    """Create a 3x3 composite: [original | processed | greyscale patch with value]."""
    rows = len(orig_imgs)
    if rows == 0:
        return None
    w, h = orig_imgs[0].size
    comp = Image.new('RGB', (w * 3, h * rows), (255, 255, 255))
    
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", size=max(14, h // 15))
    except Exception:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size=max(14, h // 15))
        except Exception:
            font = ImageFont.load_default()
    
    for i, (o, p, v, name) in enumerate(zip(orig_imgs, proc_imgs, mapped_vals, names)):
        comp.paste(o.convert('RGB'), (0, i * h))
        
        white = Image.new('RGB', p.size, (255, 255, 255))
        pasted = Image.alpha_composite(white.convert('RGBA'), p).convert('RGB')
        comp.paste(pasted, (w, i * h))
        
        if np.isnan(v):
            gray_val = 0
            label = "nan"
        else:
            clipped = max(0.0, min(100.0, v))
            gray_val = int(round((clipped / 100.0) * 255))
            label = f"{clipped:.1f}"
        
        patch = Image.new('RGB', (w, h), (gray_val, gray_val, gray_val))
        draw = ImageDraw.Draw(patch)
        
        text = f"{name}\n{label}"
        text_color = (0, 0, 0) if gray_val > 128 else (255, 255, 255)
        
        try:
            bbox = draw.textbbox((0, 0), text, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
        except Exception:
            tw, th = len(text) * 8, 12
        
        draw.text(((w - tw) / 2, (h - th) / 2), text, fill=text_color, font=font)
        comp.paste(patch, (w * 2, i * h))
    
    return comp


# Sidebar for image uploads
st.sidebar.header("Upload Images")
original_file = st.sidebar.file_uploader("Upload Original Image", type=['jpg', 'jpeg', 'png'])
treated_file = st.sidebar.file_uploader("Upload Treated Image", type=['jpg', 'jpeg', 'png'])
boiled_file = st.sidebar.file_uploader("Upload Boiled Image", type=['jpg', 'jpeg', 'png'])

uploaded_images = {}
if original_file:
    uploaded_images['original'] = Image.open(original_file).convert("RGBA")
if treated_file:
    uploaded_images['treated'] = Image.open(treated_file).convert("RGBA")
if boiled_file:
    uploaded_images['boiled'] = Image.open(boiled_file).convert("RGBA")

if not uploaded_images:
    st.info("Please upload images in the sidebar to get started.")
    st.stop()

# Color space selection
color_space = st.selectbox("Choose color space for removal:", ["HSV", "RGB"], index=0)

# Normalize all images to same size (use first available image as reference)
if 'original' in uploaded_images:
    base_size = uploaded_images['original'].size
elif 'treated' in uploaded_images:
    base_size = uploaded_images['treated'].size
elif 'boiled' in uploaded_images:
    base_size = uploaded_images['boiled'].size
else:
    st.error("No images uploaded")
    st.stop()

for key in uploaded_images:
    if uploaded_images[key].size != base_size:
        uploaded_images[key] = uploaded_images[key].resize(base_size, Image.LANCZOS)

# Initialize session state for color selections
if 'selected_colors' not in st.session_state:
    st.session_state.selected_colors = {
        'original': None,
        'treated': None,
        'boiled': None
    }

# Step 1: Color Selection
st.header("Step 1: Select Color to Remove")
st.write(f"Click on any pixel in each image to extract and use that color for removal ({color_space} space).")

cols = st.columns(3)
image_names = ['original', 'treated', 'boiled']

for idx, name in enumerate(image_names):
    if name not in uploaded_images:
        continue
    
    with cols[idx]:
        st.subheader(name.capitalize())
        img = uploaded_images[name]
        
        # Display clickable image
        coords = streamlit_image_coordinates(img, key=f"img_{name}")
        
        if coords is not None:
            # Extract the color at clicked coordinates
            x, y = int(coords['x']), int(coords['y'])
            rgb_array = np.array(img.convert("RGB"))
            
            # Clamp coordinates to image bounds
            x = max(0, min(x, rgb_array.shape[1] - 1))
            y = max(0, min(y, rgb_array.shape[0] - 1))
            
            # Get RGB
            r, g, b = rgb_array[y, x, :3]
            
            if color_space == "HSV":
                # Convert RGB to HSV
                picked_color = rgb_to_hsv(r, g, b)
            else:
                # Keep as RGB
                picked_color = (r, g, b)
            
            st.session_state.selected_colors[name] = picked_color
        
        # Show current selection
        current_color = st.session_state.selected_colors.get(name)
        if current_color:
            if color_space == "HSV":
                h_val, s_val, v_val = current_color
                # Create preview in RGB for visualization
                h_preview = (h_val * 2) % 360
                preview_rgb = hsv_to_rgb(h_preview, s_val / 255.0, v_val / 255.0)
                color_preview = Image.new('RGB', (100, 50), preview_rgb)
                st.image(color_preview, width=100)
                st.write(f"Selected: HSV({h_val}, {s_val}, {v_val})")
            else:
                r_val, g_val, b_val = current_color
                color_preview = Image.new('RGB', (100, 50), (r_val, g_val, b_val))
                st.image(color_preview, width=100)
                st.write(f"Selected: RGB({r_val}, {g_val}, {b_val})")
        else:
            st.write("Click on the image above to select a color")

# Advanced settings
with st.expander("Advanced Settings"):
    if color_space == "HSV":
        st.write("Adjust HSV tolerance ranges:")
        h_threshold = st.slider("Hue tolerance", 0, 90, 34, 1)
        s_threshold = st.slider("Saturation tolerance (0-255)", 0, 100, 47, 1)
        v_threshold = st.slider("Value tolerance (0-255)", 0, 100, 84, 1)
    else:
        st.write("Adjust RGB detection parameters (ratio & threshold):")
        h_threshold = st.slider("Ratio (higher = stricter)", 1.0, 2.5, 1.05, 0.01)
        s_threshold = st.slider("Difference threshold (higher = stricter)", 0, 100, 10, 1)
        v_threshold = 0  # Not used for RGB

# Manual color adjustment (optional)
# with st.expander("Manually Adjust Colors (Optional)"):
#     if color_space == "HSV":
#         st.write("Fine-tune the selected colors with HSV sliders:")
#     else:
#         st.write("Fine-tune the selected colors with RGB sliders:")
    
#     adjust_cols = st.columns(3)
#     for idx, name in enumerate(image_names):
#         if name not in uploaded_images:
#             continue
        
#         with adjust_cols[idx]:
#             st.subheader(f"Adjust {name.capitalize()}")
#             current = st.session_state.selected_colors.get(name, (0, 0, 0))
            
#             if color_space == "HSV":
#                 st.write("DEBUG current:", current, type(current))
#                 h = st.slider(f"Hue (0-180)", 0, 180, current[0], key=f"adj_h_{name}")
#                 s = st.slider(f"Saturation (0-255)", 0, 255, current[1], key=f"adj_s_{name}")
#                 v = st.slider(f"Value (0-255)", 0, 255, current[2], key=f"adj_v_{name}")
#                 st.session_state.selected_colors[name] = (h, s, v)
#             else:
#                 r = st.slider(f"Red (0-255)", 0, 255, current[0], key=f"adj_r_{name}")
#                 g = st.slider(f"Green (0-255)", 0, 255, current[1], key=f"adj_g_{name}")
#                 b = st.slider(f"Blue (0-255)", 0, 255, current[2], key=f"adj_b_{name}")
#                 st.session_state.selected_colors[name] = (r, g, b)

# Step 2: Preview color removal
st.header("Step 2: Preview Color Removal")

processed_images = {}
masks = {}
rgb_arrays = {}

for name in image_names:
    if name not in uploaded_images:
        continue
    
    current_color = st.session_state.selected_colors.get(name)
    if current_color is None:
        st.warning(f"No color selected for {name}")
        continue
    
    color = current_color
    img = uploaded_images[name]
    proc_img, mask = remove_color_pixels(img, color, color_space, h_threshold=h_threshold, s_threshold=s_threshold, v_threshold=v_threshold)
    processed_images[name] = proc_img
    masks[name] = mask
    rgb_arrays[name] = np.array(img.convert("RGB"))

# Display preview
preview_cols = st.columns(3)
for idx, name in enumerate(image_names):
    if name not in processed_images:
        continue
    
    with preview_cols[idx]:
        st.subheader(f"{name.capitalize()} (Color Removed)")
        st.image(processed_images[name])

# Step 3: Compute metrics
st.header("Step 3: Grayscale Analysis")

if len(processed_images) == 3 and all(n in processed_images for n in image_names):
    # Compute average grayscale where not masked
    avgs = {}
    for name in image_names:
        avgs[name] = avg_grayscale_where_not_masked(rgb_arrays[name], masks[name])
    
    st.subheader("Average Grayscale (Luminosity) on Non-Removed Pixels")
    metric_cols = st.columns(3)
    for idx, name in enumerate(image_names):
        with metric_cols[idx]:
            val = avgs[name]
            if np.isnan(val):
                st.metric(name.capitalize(), "N/A")
            else:
                st.metric(name.capitalize(), f"{val:.1f}")
    
    # Mapping: original -> 0, treated -> 100
    mapped = [float('nan')] * 3
    mapped[0] = 0.0
    mapped[1] = 100.0
    
    orig_val = avgs['original']
    treated_val = avgs['treated']
    boiled_val = avgs['boiled']
    
    if np.isnan(orig_val) or np.isnan(treated_val) or np.isnan(boiled_val):
        mapped[2] = float('nan')
    elif treated_val == orig_val:
        mapped[2] = 0.0
    else:
        mapped[2] = 100.0 * (boiled_val - orig_val) / (treated_val - orig_val)
    
    st.subheader("Mapped Values (Original → 0, Treated → 100)")
    mapped_cols = st.columns(3)
    for idx, (name, val) in enumerate(zip(image_names, mapped)):
        with mapped_cols[idx]:
            if np.isnan(val):
                st.metric(name.capitalize(), "N/A")
            else:
                st.metric(name.capitalize(), f"{val:.2f}")
    
    # Step 4: Display composites
    st.header("Step 4: Composite Images")
    
    # Prepare ordered lists
    orig_list = [uploaded_images[n].convert("RGBA") for n in image_names]
    proc_list = [processed_images[n] for n in image_names]
    
    # Create 3x2 composite
    comp_3x2 = make_3x2_composite(orig_list, proc_list)
    if comp_3x2:
        st.subheader("3×2 Comparison (Original | Color Removed)")
        st.image(comp_3x2)
        
        # Allow download
        buf = io.BytesIO()
        comp_3x2.save(buf, format='PNG')
        buf.seek(0)
        st.download_button(
            label="Download 3×2 Comparison",
            data=buf,
            file_name="comparison_3x2.png",
            mime="image/png"
        )
    
    # Create 3x3 detailed composite
    comp_3x3 = make_3x3_detailed(orig_list, proc_list, mapped, image_names)
    if comp_3x3:
        st.subheader("3×3 Detailed (Original | Color Removed | Grayscale Value)")
        st.image(comp_3x3)
        
        # Allow download
        buf = io.BytesIO()
        comp_3x3.save(buf, format='PNG')
        buf.seek(0)
        st.download_button(
            label="Download 3×3 Detailed",
            data=buf,
            file_name="detailed_3x3.png",
            mime="image/png"
        )

else:
    st.info("Upload all three images and select colors to see metrics and composites.")
