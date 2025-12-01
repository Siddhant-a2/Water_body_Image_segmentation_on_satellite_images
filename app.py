import io

import numpy as np
import streamlit as st
import rasterio
from rasterio.io import MemoryFile
from PIL import Image

# 🔹 Import the official inference helper from the WatNet repo
# Make sure watnet_infer.py is in the same folder as this app.py,
# and model/watnet.h5 exists as in the original repo.
from infer import watnet_infer


# ---------- DATA READING ----------

def read_tif_to_array(uploaded_file):
    """
    Read an uploaded 6-band GeoTIFF and return:
    - img: (H, W, C) float32 array (raw values, no scaling)
    - profile: rasterio profile (for georeferencing & pixel size)
    """
    file_bytes = uploaded_file.read()

    with MemoryFile(file_bytes) as memfile:
        with memfile.open() as src:
            profile = src.profile
            arr = src.read()  # (C, H, W)

    if arr.shape[0] != 6:
        raise ValueError(
            f"Expected 6 bands, but found {arr.shape[0]}. "
            "Make sure your .tif is a 6-band Sentinel-2 stack (B2,B3,B4,B8,B11,B12)."
        )

    # (C, H, W) -> (H, W, C)
    img = np.transpose(arr, (1, 2, 0)).astype("float32")

    # Replace NaNs / inf just in case
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)

    return img, profile


# ---------- VISUALIZATION HELPERS ----------

def make_rgb_preview(img_6band):
    """
    Make a contrast-stretched RGB preview from a 6-band image.
    Assumes band order [B2, B3, B4, B8, B11, B12].
    RGB = (B4, B3, B2) -> indices [2, 1, 0].
    Works with any numeric scale (uint16 0–10000, float32, etc.).
    """
    try:
        rgb = img_6band[..., [2, 1, 0]]  # (H, W, 3)
    except Exception:
        return None

    rgb = rgb.astype("float32")
    out = np.zeros_like(rgb)

    # Contrast stretch each channel using 2–98 percentile
    for ch in range(3):
        band = rgb[..., ch]
        vmin = np.percentile(band, 2)
        vmax = np.percentile(band, 98)
        if vmax > vmin:
            band_stretched = (band - vmin) / (vmax - vmin)
        else:
            band_stretched = band
        out[..., ch] = np.clip(band_stretched, 0.0, 1.0)

    rgb_display = (out * 255).astype(np.uint8)
    return rgb_display


def mask_to_png_bytes(mask):
    """
    Convert a 2D uint8/0-1 mask to grayscale PNG bytes.
    0 -> black, 1 -> white.
    """
    # If mask is not 0/1, binarize: >0 -> 1
    m = (mask > 0).astype(np.uint8)
    img_uint8 = (m * 255).astype(np.uint8)
    pil_img = Image.fromarray(img_uint8, mode="L")

    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    return buf


# ---------- AREA STATS ----------

# def compute_area_stats(water_mask, profile):
#     """
#     Compute water area stats using GeoTIFF profile.

#     Assumes "water" is any pixel > 0 in water_mask.
#     Falls back to 10m x 10m pixels if transform is missing/invalid.
#     """
#     H, W = water_mask.shape
#     total_pixels = H * W
#     water_pixels = int(np.count_nonzero(water_mask > 0))

#     transform = profile.get("transform", None)
#     pixel_area_m2 = None

#     if transform is not None:
#         try:
#             pixel_width = float(transform[0])
#             pixel_height = float(transform[4])
#             pixel_area_m2 = abs(pixel_width * pixel_height)
#         except Exception:
#             pixel_area_m2 = None

#     pixel_size_assumed = False
#     if not pixel_area_m2 or pixel_area_m2 <= 0:
#         pixel_size_m = 10.0  # assume Sentinel-2 10m
#         pixel_area_m2 = pixel_size_m ** 2
#         pixel_size_assumed = True

#     water_area_m2 = water_pixels * pixel_area_m2
#     water_area_km2 = water_area_m2 / 1e6
#     coverage_percent = (water_pixels / total_pixels * 100.0) if total_pixels > 0 else 0.0

#     return {
#         "total_pixels": total_pixels,
#         "water_pixels": water_pixels,
#         "pixel_area_m2": pixel_area_m2,
#         "water_area_m2": water_area_m2,
#         "water_area_km2": water_area_km2,
#         "coverage_percent": coverage_percent,
#         "pixel_size_assumed": pixel_size_assumed,
#     }

import math
# import numpy as np

def compute_area_stats(water_map, profile):
    """
    Compute water area stats using GeoTIFF metadata.

    - Automatically figures out which pixels are "water"
      (0/1 mask or continuous probability).
    - Handles both projected CRS (meters) and geographic CRS (degrees, e.g. EPSG:4326).
    - Falls back to 10m x 10m pixels if we still can't determine pixel size.

    Parameters
    ----------
    water_map : 2D numpy array
        Output from watnet_infer (can be binary or continuous).
    profile : dict
        rasterio dataset.profile.

    Returns
    -------
    dict with:
        total_pixels, water_pixels,
        pixel_width_m, pixel_height_m, pixel_area_m2,
        water_area_m2, water_area_km2,
        coverage_percent, pixel_size_assumed
    """

    wm = np.array(water_map)

    # ----------- Decide what is "water" -----------
    unique_vals = np.unique(wm)
    # Case 1: looks like a binary mask {0, 1}
    if np.all(np.isin(unique_vals, [0, 1])):
        water_mask = (wm == 1)
    else:
        # Case 2: continuous map (probability / score)
        # Use a 0.5 * max threshold by default
        max_val = wm.max()
        if max_val <= 0:
            water_mask = np.zeros_like(wm, dtype=bool)
        else:
            thr = 0.5 * max_val
            water_mask = wm >= thr

    H, W = wm.shape
    total_pixels = H * W
    water_pixels = int(np.count_nonzero(water_mask))

    # ----------- Derive pixel size in meters -----------
    transform = profile.get("transform", None)
    crs = profile.get("crs", None)

    pixel_width_m = None
    pixel_height_m = None
    pixel_area_m2 = None
    pixel_size_assumed = False

    if transform is not None:
        # rasterio Affine: (a, b, c, d, e, f)
        a = float(transform[0])  # pixel width (x scale)
        b = float(transform[1])  # usually 0
        c = float(transform[2])  # top-left x
        d = float(transform[3])  # usually 0
        e = float(transform[4])  # pixel height (y scale, negative for north-up)
        f = float(transform[5])  # top-left y

        try:
            # Projected CRS (units in meters)
            if crs is not None and getattr(crs, "is_projected", False):
                pixel_width_m = abs(a)
                pixel_height_m = abs(e)
                pixel_area_m2 = pixel_width_m * pixel_height_m

            # Geographic CRS (units in degrees, e.g. EPSG:4326)
            elif crs is not None and getattr(crs, "is_geographic", False):
                # Approximate meters per degree at scene center latitude
                center_row = H / 2.0
                center_col = W / 2.0

                x_center = c + a * center_col + b * center_row
                y_center = f + d * center_col + e * center_row   # latitude in degrees

                lat_rad = np.deg2rad(y_center)
                meters_per_deg_lon = 111320.0 * math.cos(lat_rad)
                meters_per_deg_lat = 110540.0

                pixel_width_m = abs(a) * meters_per_deg_lon
                pixel_height_m = abs(e) * meters_per_deg_lat
                pixel_area_m2 = pixel_width_m * pixel_height_m

            # Unknown CRS, but we have a transform – just assume numbers are meters
            else:
                pixel_width_m = abs(a)
                pixel_height_m = abs(e)
                pixel_area_m2 = pixel_width_m * pixel_height_m

        except Exception:
            pixel_width_m = None
            pixel_height_m = None
            pixel_area_m2 = None

    # Fallback if everything above failed or area is non-positive
    if pixel_area_m2 is None or pixel_area_m2 <= 0:
        pixel_width_m = 10.0
        pixel_height_m = 10.0
        pixel_area_m2 = pixel_width_m * pixel_height_m
        pixel_size_assumed = True

    water_area_m2 = water_pixels * pixel_area_m2
    water_area_km2 = water_area_m2 / 1e6
    coverage_percent = (water_pixels / total_pixels * 100.0) if total_pixels > 0 else 0.0

    return {
        "total_pixels": total_pixels,
        "water_pixels": water_pixels,
        "pixel_width_m": pixel_width_m,
        "pixel_height_m": pixel_height_m,
        "pixel_area_m2": pixel_area_m2,
        "water_area_m2": water_area_m2,
        "water_area_km2": water_area_km2,
        "coverage_percent": coverage_percent,
        "pixel_size_assumed": pixel_size_assumed,
    }


# ---------- STREAMLIT APP ----------

st.set_page_config(page_title="WatNet Water Mapping", layout="wide")

st.title("🌊 WatNet – Surface Water Mapping (Streamlit Demo, using watnet_infer.py)")
st.write(
    """
This app wraps the **official WatNet inference function** (`watnet_infer`) from the GitHub repo.

Upload a **6-band Sentinel-2 GeoTIFF** (bands 2, 3, 4, 8, 11, 12 stacked), and the app will:

- Show an **RGB preview** (B4,B3,B2).
- Call `watnet_infer(rsimg)` to get the water map.
- Compute **water area** in m² and km².
"""
)

uploaded_file = st.file_uploader(
    "Upload 6-band Sentinel-2 GeoTIFF (.tif / .tiff)",
    type=["tif", "tiff"],
)

if uploaded_file is not None:
    try:
        st.info("Reading image...")
        img_6band, profile = read_tif_to_array(uploaded_file)

        H, W, C = img_6band.shape
        st.write(f"Image shape: **{H} × {W} × {C}** (H × W × bands)")

        # ---------- RGB preview ----------
        st.subheader("📌 Input Image – RGB View (B4,B3,B2)")
        rgb_display = make_rgb_preview(img_6band)
        if rgb_display is not None:
            st.image(rgb_display, caption="RGB Preview (B4,B3,B2)", use_container_width=True)
        else:
            st.warning("Could not create RGB preview from the provided image.")

        # ---------- Run WatNet ----------
        if st.button("Run Water Mapping with watnet_infer"):
            with st.spinner("Running watnet_infer(rsimg)..."):
                # NOTE: watnet_infer in the repo expects rsimg with shape (H, W, 6)
                water_map = watnet_infer(img_6band)

            # Ensure 2D mask
            water_map = np.squeeze(water_map)
            if water_map.ndim != 2:
                st.warning(f"Unexpected water_map shape from watnet_infer: {water_map.shape}")
            else:
                st.success("Inference completed!")

                # Show some debug stats
                st.write(
                    f"Mask stats – unique values: {np.unique(water_map)}, "
                    f"min: {water_map.min()}, max: {water_map.max()}"
                )

                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("🟦 Predicted Water Mask")
                    st.image(
                        (water_map > 0).astype(np.uint8) * 255,
                        caption="Water Mask (water=white)",
                        use_container_width=True,
                    )

                with col2:
                    st.subheader("🔍 Raw Water Map (normalized for display)")
                    wm = water_map.astype("float32")
                    # Normalize for display (0–1)
                    vmin = np.percentile(wm, 2)
                    vmax = np.percentile(wm, 98)
                    if vmax > vmin:
                        wm_disp = (wm - vmin) / (vmax - vmin)
                    else:
                        wm_disp = wm
                    wm_disp = np.clip(wm_disp, 0.0, 1.0)
                    st.image(
                        wm_disp,
                        caption="Water Map (scaled to 0–1 for visualization)",
                        use_container_width=True,
                        clamp=True,
                    )

                # ---------- Area statistics ----------
                st.subheader("📐 Water Area Statistics")
                stats = compute_area_stats(water_map, profile)

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Water pixels", f"{stats['water_pixels']:,}")
                with c2:
                    st.metric("Water area (m²)", f"{stats['water_area_m2']:.2f}")
                with c3:
                    st.metric("Water area (km²)", f"{stats['water_area_km2']:.4f}")

                st.write(f"Water coverage: **{stats['coverage_percent']:.4f}%** of the scene")

                st.write(
                    f"Approx. pixel size: "
                    f"{stats['pixel_width_m']:.2f} m × {stats['pixel_height_m']:.2f} m "
                    f"(area ≈ {stats['pixel_area_m2']:.2f} m² per pixel)"
                )

                if stats["pixel_size_assumed"]:
                    st.caption(
                        "⚠ Pixel size/area estimated (default 10 m × 10 m). "
                        "Consider reprojecting your GeoTIFF to a projected CRS for higher accuracy."
                    )
                else:
                    st.caption("✅ Pixel size derived from GeoTIFF transform and CRS.")


                # ---------- Download mask ----------
                st.subheader("⬇ Download Results")
                png_buf = mask_to_png_bytes(water_map)
                st.download_button(
                    label="Download Water Mask (PNG)",
                    data=png_buf,
                    file_name="water_mask.png",
                    mime="image/png",
                )

    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    st.info("Please upload a 6-band .tif to begin.")
