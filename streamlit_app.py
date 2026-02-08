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


def avg_hsv_lightness_where_not_masked(arr: np.ndarray, mask: np.ndarray) -> float:
    """Compute average HSV lightness (V value) on pixels where mask is False."""
    r = arr[..., 0].astype(np.float32) / 255.0
    g = arr[..., 1].astype(np.float32) / 255.0
    b = arr[..., 2].astype(np.float32) / 255.0
    
    # V is the maximum of R, G, B
    v = np.maximum(np.maximum(r, g), b) * 255.0
    
    valid = ~mask
    if not np.any(valid):
        return float('nan')
    return float(v[valid].mean())


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


def apply_lighting_adjustment_to_image(img: Image.Image, offset: float) -> Image.Image:
    """Apply lighting adjustment (brightness shift) to an image."""
    arr = np.array(img.convert('RGB')).astype(np.int32)
    
    # Apply offset to all channels
    adjusted = arr + int(offset)
    
    # Clamp to valid range
    adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
    
    return Image.fromarray(adjusted, 'RGB')


def make_lighting_adjustment_viz(orig_imgs: List[Image.Image], patch_avgs: dict, 
                                  names: List[str], ref_patch_v: float) -> Image.Image:
    """Create a side-by-side visualization of original vs lighting-adjusted images."""
    rows = len(names)
    if rows == 0:
        return None
    
    w, h = orig_imgs[0].size
    
    # Create composite: [Original | Adjusted] for each image
    comp = Image.new('RGB', (w * 2, h * rows), (255, 255, 255))
    
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", size=max(10, h // 25))
    except Exception:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size=max(10, h // 25))
        except Exception:
            font = ImageFont.load_default()
    
    for i, (img, name) in enumerate(zip(orig_imgs, names)):
        patch_v = patch_avgs.get(name, float('nan'))
        
        if np.isnan(patch_v) or np.isnan(ref_patch_v):
            offset = 0
        else:
            offset = patch_v - ref_patch_v
        
        # Original image
        comp.paste(img.convert('RGB'), (0, i * h))
        
        # Adjusted image
        adjusted_img = apply_lighting_adjustment_to_image(img, -offset)
        comp.paste(adjusted_img, (w, i * h))
        
        # Add labels
        draw = ImageDraw.Draw(comp)
        
        # Original label
        label_orig = f"{name.upper()}"
        try:
            bbox = draw.textbbox((0, 0), label_orig, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
        except Exception:
            tw, th = len(label_orig) * 6, 10
        
        draw.text(((w - tw) / 2, i * h + 8), label_orig, fill=(0, 0, 0), font=font)
        
        # Adjusted label with offset info
        label_adj = f"Adjusted (offset: {-offset:+.1f})"
        try:
            bbox = draw.textbbox((0, 0), label_adj, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
        except Exception:
            tw, th = len(label_adj) * 6, 10
        
        draw.text((w + (w - tw) / 2, i * h + 8), label_adj, fill=(0, 0, 0), font=font)
    
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

# Resize very large images to prevent decompression bomb warning
# PIL warns when images exceed ~89MP; we'll cap at a reasonable size for processing
MAX_PIXELS = 50_000_000  # ~50 megapixels

for key in uploaded_images:
    w, h = uploaded_images[key].size
    total_pixels = w * h
    if total_pixels > MAX_PIXELS:
        # Calculate new size maintaining aspect ratio
        scale = (MAX_PIXELS / total_pixels) ** 0.5
        new_w = int(w * scale)
        new_h = int(h * scale)
        uploaded_images[key] = uploaded_images[key].resize((new_w, new_h), Image.LANCZOS)
        st.sidebar.info(f"📌 {key.capitalize()} image resized from {w}×{h} to {new_w}×{new_h} for processing")

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

# Initialize session state for color selections and patches
if 'selected_colors' not in st.session_state:
    st.session_state.selected_colors = {
        'original': None,
        'treated': None,
        'boiled': None
    }

if 'selected_patches' not in st.session_state:
    st.session_state.selected_patches = {
        'original': None,
        'treated': None,
        'boiled': None
    }

# Step 1: Color Selection
st.header("Step 1: Select Patch for Background Color")
st.write(f"Click and drag to select a rectangular patch in each image. The average color from the patch will be used for removal ({color_space} space).")

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
            # Store the coordinate click (streamlit-image-coordinates returns single clicks)
            # We'll use it to initialize a patch selection UI
            x, y = int(coords['x']), int(coords['y'])
            rgb_array = np.array(img.convert("RGB"))
            
            # Clamp coordinates to image bounds
            x = max(0, min(x, rgb_array.shape[1] - 1))
            y = max(0, min(y, rgb_array.shape[0] - 1))
            
            # Store the clicked point
            st.session_state.selected_patches[name] = {
                'point': (x, y),
                'clicked': True
            }
        
        # Show patch selection controls
        patch_info = st.session_state.selected_patches.get(name)
        if patch_info and patch_info.get('clicked'):
            st.info(f"✓ Point selected at ({patch_info['point'][0]}, {patch_info['point'][1]})")
            
            # Allow user to define patch size around the clicked point
            patch_size = st.slider(
                f"Patch size (pixels)", 
                5, 100, 20, 
                key=f"patch_size_{name}",
                help="Radius around clicked point for averaging"
            )
            
            x_click, y_click = patch_info['point']
            x_min = max(0, x_click - patch_size)
            x_max = min(rgb_array.shape[1], x_click + patch_size)
            y_min = max(0, y_click - patch_size)
            y_max = min(rgb_array.shape[0], y_click + patch_size)
            
            # Extract patch and compute average color
            patch = rgb_array[y_min:y_max, x_min:x_max, :3]
            avg_r = int(np.mean(patch[..., 0]))
            avg_g = int(np.mean(patch[..., 1]))
            avg_b = int(np.mean(patch[..., 2]))
            
            if color_space == "HSV":
                # Convert average RGB to HSV
                picked_color = rgb_to_hsv(avg_r, avg_g, avg_b)
            else:
                # Keep as average RGB
                picked_color = (avg_r, avg_g, avg_b)
            
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
                st.write(f"**Selected Patch Avg:** HSV({h_val}, {s_val}, {v_val})")
            else:
                r_val, g_val, b_val = current_color
                color_preview = Image.new('RGB', (100, 50), (r_val, g_val, b_val))
                st.image(color_preview, width=100)
                st.write(f"**Selected Patch Avg:** RGB({r_val}, {g_val}, {b_val})")
        else:
            st.write("👆 Click on the image to select a starting point")

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
    # Compute average HSV lightness where not masked
    avgs = {}
    for name in image_names:
        avgs[name] = avg_hsv_lightness_where_not_masked(rgb_arrays[name], masks[name])
    
    st.subheader("Average HSV Lightness (V) on Non-Removed Pixels")
    metric_cols = st.columns(3)
    for idx, name in enumerate(image_names):
        with metric_cols[idx]:
            val = avgs[name]
            if np.isnan(val):
                st.metric(name.capitalize(), "N/A")
            else:
                st.metric(name.capitalize(), f"{val:.1f}")
    
    # Compute patch HSV lightness for lighting adjustment
    st.subheader("Average HSV Lightness (V) of Selected Patches")
    patch_avgs = {}
    for name in image_names:
        patch_info = st.session_state.selected_patches.get(name)
        if patch_info and patch_info.get('clicked'):
            rgb_array = np.array(uploaded_images[name].convert("RGB"))
            x_click, y_click = patch_info['point']
            
            # Get patch size from session state or use default
            patch_key = f"patch_size_{name}"
            if patch_key in st.session_state:
                patch_size = st.session_state[patch_key]
            else:
                patch_size = 20
            
            x_min = max(0, x_click - patch_size)
            x_max = min(rgb_array.shape[1], x_click + patch_size)
            y_min = max(0, y_click - patch_size)
            y_max = min(rgb_array.shape[0], y_click + patch_size)
            
            patch = rgb_array[y_min:y_max, x_min:x_max, :3]
            patch_r = np.mean(patch[..., 0]).astype(np.float32) / 255.0
            patch_g = np.mean(patch[..., 1]).astype(np.float32) / 255.0
            patch_b = np.mean(patch[..., 2]).astype(np.float32) / 255.0
            
            # Compute HSV lightness (V) of patch
            patch_v = np.maximum(np.maximum(patch_r, patch_g), patch_b) * 255.0
            patch_avgs[name] = patch_v
        else:
            patch_avgs[name] = float('nan')
    
    patch_cols = st.columns(3)
    for idx, name in enumerate(image_names):
        with patch_cols[idx]:
            val = patch_avgs[name]
            if np.isnan(val):
                st.metric(f"{name.capitalize()} Patch", "N/A")
            else:
                st.metric(f"{name.capitalize()} Patch", f"{val:.1f}")
    
    # Lighting-adjusted HSV lightness values
    st.subheader("Average HSV Lightness (V) on Non-Removed Pixels - Lighting Adjusted")
    
    # Use original patch HSV lightness as reference
    ref_patch_v = patch_avgs.get('original', float('nan'))
    
    adjusted_avgs = {}
    if not np.isnan(ref_patch_v):
        # Adjust all values based on patch HSV lightness difference
        for name in image_names:
            if not np.isnan(avgs[name]) and not np.isnan(patch_avgs[name]):
                # Lighting offset: difference between current patch and reference patch
                offset = patch_avgs[name] - ref_patch_v
                adjusted_avgs[name] = avgs[name] - offset
            else:
                adjusted_avgs[name] = float('nan')
    else:
        adjusted_avgs = avgs  # Fallback if no valid reference
    
    adj_cols = st.columns(3)
    for idx, name in enumerate(image_names):
        with adj_cols[idx]:
            val = adjusted_avgs[name]
            if np.isnan(val):
                st.metric(name.capitalize(), "N/A")
            else:
                st.metric(name.capitalize(), f"{val:.1f}")
    
    # Visualize lighting adjustment effect
    st.subheader("Lighting Adjustment Visualization")
    orig_list = [uploaded_images[n].convert("RGB") for n in image_names]
    viz_img = make_lighting_adjustment_viz(orig_list, patch_avgs, image_names, ref_patch_v)
    if viz_img:
        st.image(viz_img)
    
    # Coat Percentage calculation using the new formula
    st.subheader("Coat Percentage")
    
    orig_val = adjusted_avgs['original']
    treated_val = adjusted_avgs['treated']
    boiled_val = adjusted_avgs['boiled']
    
    coat_percentage = float('nan')
    if not np.isnan(orig_val) and not np.isnan(treated_val) and not np.isnan(boiled_val):
        if orig_val != treated_val:
            coat_percentage = 100.0 * (1.0 - (boiled_val - treated_val) / orig_val)
    
    coat_cols = st.columns(3)
    for idx, (name, val) in enumerate(zip(['Original', 'Treated', 'Boiled'], [0.0, 100.0, coat_percentage])):
        with coat_cols[idx]:
            if np.isnan(val):
                st.metric(name, "N/A")
            else:
                st.metric(name, f"{val:.2f}")
    
    # Display the formula
    st.write("**Formula:**")
    st.latex(r"Coat\,\% = 100 \times \left(1 - \frac{V_{Boiled} - V_{Treated}}{V_{Original}}\right)")
    
    st.caption("Where V = Average HSV Lightness on Non-Removed Pixels (Lighting Adjusted)")
    
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
    # comp_3x3 = make_3x3_detailed(orig_list, proc_list, mapped, image_names)
    # if comp_3x3:
    #     st.subheader("3×3 Detailed (Original | Color Removed | Grayscale Value)")
    #     st.image(comp_3x3)
        
    #     # Allow download
    #     buf = io.BytesIO()
    #     comp_3x3.save(buf, format='PNG')
    #     buf.seek(0)
    #     st.download_button(
    #         label="Download 3×3 Detailed",
    #         data=buf,
    #         file_name="detailed_3x3.png",
    #         mime="image/png"
    #     )

else:
    st.info("Upload all three images and select colors to see metrics and composites.")

# Add process explanation at the end
st.divider()
st.header("How the Grayscale Analysis Works")

st.subheader("1. HSV Lightness (V Value)")
st.write("""
The analysis uses the **V (Value) component from the HSV color space** instead of traditional grayscale luminosity.
In HSV:
- **H (Hue):** The color itself (0-360°)
- **S (Saturation):** How pure the color is (0-100%)
- **V (Value):** The **brightness** of the color (0-100%)

The V value is calculated simply as the **maximum of the R, G, and B channels** (after normalizing to 0-1). 
This gives us a pure brightness measurement independent of color saturation.
""")

st.subheader("2. Non-Removed Pixels Analysis")
st.write("""
After removing the selected color, we measure the average V (brightness) of all remaining pixels. 
This tells us how bright the treated/boiled rocks are after the coating removal.
""")

st.subheader("3. Lighting Reference (Patch Selection)")
st.write("""
Different images may have been photographed under different lighting conditions. To account for this, 
we use the **selected patch as a lighting reference**. The patch represents the same background/coating 
across all three images, so differences in patch brightness indicate lighting differences.
""")

st.subheader("4. Lighting Adjustment")
st.write("""
The lighting offset is calculated as:
- **Offset = Patch V (current image) - Patch V (original image)**

This offset is then **subtracted from all non-removed pixel values**:
- **Adjusted V = Raw V - Offset**

This normalization ensures all three images are compared on the same lighting baseline, making the 
comparison fair and accurate regardless of lighting variations.
""")

st.subheader("5. Coat Percentage Calculation")
st.write("""
The final coat percentage tells us what fraction of the original coating remains on the boiled rock:

$$Coat\\,\\% = 100 \\times \\left(1 - \\frac{V_{Boiled} - V_{Treated}}{V_{Original}}\\right)$$

- **100%** = Boiled surface is as bright as treated (no coating removed)
- **0%** = Boiled surface is as bright as original untreated rock (all coating removed)
- **Values between** = Partial coating removal
""")

