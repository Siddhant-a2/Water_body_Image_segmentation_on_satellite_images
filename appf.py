import io
import math

import numpy as np
import streamlit as st
from rasterio.io import MemoryFile
from PIL import Image

from infer import watnet_infer  # from your repo


# ---------- PAGE CONFIG & BASIC STYLING ----------

st.set_page_config(page_title="WatNet – Water Mapping (Past vs Present)", layout="wide")

st.markdown(
    """
    <style>
    .center-title {
        text-align: center;
        font-weight: 700;
        font-size: 2.0rem;
        margin-bottom: 0.2rem;
    }
    .subtle-caption {
        text-align: center;
        color: #808080;
        margin-bottom: 1.2rem;
        font-size: 0.95rem;
    }
    .block-card {
        padding: 1rem 1.3rem;
        border-radius: 0.8rem;
        border: 1px solid #3d3d3d33;
        background: #11111111;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="center-title">🌊 WatNet – Surface Water Mapping (Past vs Present)</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtle-caption">Upload two 6-band Sentinel-2 GeoTIFFs (B2, B3, B4, B8, B11, B12): one for the past and one for the present, to compare water bodies.</div>',
    unsafe_allow_html=True,
)


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
    m = (mask > 0).astype(np.uint8)
    img_uint8 = (m * 255).astype(np.uint8)
    pil_img = Image.fromarray(img_uint8, mode="L")

    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    return buf


# ---------- AREA STATS ----------

def compute_area_stats(water_map, profile):
    """
    Compute water area stats using GeoTIFF metadata.
    Handles binary or continuous maps and different CRS types.
    """
    wm = np.array(water_map)

    # Decide what is "water"
    unique_vals = np.unique(wm)
    if np.all(np.isin(unique_vals, [0, 1])):  # binary
        water_mask = (wm == 1)
    else:
        max_val = wm.max()
        if max_val <= 0:
            water_mask = np.zeros_like(wm, dtype=bool)
        else:
            thr = 0.5 * max_val
            water_mask = wm >= thr

    H, W = wm.shape
    total_pixels = H * W
    water_pixels = int(np.count_nonzero(water_mask))

    transform = profile.get("transform", None)
    crs = profile.get("crs", None)

    pixel_width_m = None
    pixel_height_m = None
    pixel_area_m2 = None
    pixel_size_assumed = False

    if transform is not None:
        a = float(transform[0])  # pixel width
        e = float(transform[4])  # pixel height
        c = float(transform[2])  # top-left x
        f = float(transform[5])  # top-left y
        d = float(transform[3])
        b = float(transform[1])

        try:
            # Projected CRS (meters)
            if crs is not None and getattr(crs, "is_projected", False):
                pixel_width_m = abs(a)
                pixel_height_m = abs(e)
                pixel_area_m2 = pixel_width_m * pixel_height_m

            # Geographic CRS (degrees)
            elif crs is not None and getattr(crs, "is_geographic", False):
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

            # Unknown CRS, assume transform is in meters
            else:
                pixel_width_m = abs(a)
                pixel_height_m = abs(e)
                pixel_area_m2 = pixel_width_m * pixel_height_m

        except Exception:
            pixel_width_m = pixel_height_m = pixel_area_m2 = None

    # Fallback
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


# ---------- MAIN APP LAYOUT ----------

with st.sidebar:
    st.markdown("### 1️⃣ Upload Images")
    uploaded_past = st.file_uploader(
        "Past image – 6-band Sentinel-2 GeoTIFF (.tif / .tiff)",
        type=["tif", "tiff"],
        key="past",
    )
    uploaded_present = st.file_uploader(
        "Present image – 6-band Sentinel-2 GeoTIFF (.tif / .tiff)",
        type=["tif", "tiff"],
        key="present",
    )
    run_infer = st.button("▶ Run Water Mapping (Both)", use_container_width=True)
    st.caption("Tip: Stack bands in order: B2, B3, B4, B8, B11, B12 for both images.")

if uploaded_past is not None and uploaded_present is not None:
    try:
        # Read both images
        img_past, profile_past = read_tif_to_array(uploaded_past)
        img_present, profile_present = read_tif_to_array(uploaded_present)

        H_p, W_p, C_p = img_past.shape
        H_c, W_c, C_c = img_present.shape

        st.markdown(
            f'<p style="text-align:center; margin-top:0.5rem;">'
            f"Past image: <b>{H_p} × {W_p}</b> (H × W), <b>{C_p}</b> bands  |  "
            f"Present image: <b>{H_c} × {W_c}</b> (H × W), <b>{C_c}</b> bands</p>",
            unsafe_allow_html=True,
        )

        rgb_past = make_rgb_preview(img_past)
        rgb_present = make_rgb_preview(img_present)

        water_past = None
        water_present = None

        if run_infer:
            with st.spinner("Running WatNet inference on both images..."):
                water_past = watnet_infer(img_past)
                water_present = watnet_infer(img_present)
                water_past = np.squeeze(water_past)
                water_present = np.squeeze(water_present)

        # Side-by-side layout: past vs present
        with st.container():
            st.markdown('<div class="block-card">', unsafe_allow_html=True)
            col_past, col_present = st.columns(2, vertical_alignment="top")

            # ----- PAST -----
            with col_past:
                st.subheader("🕰 Past – Input & Water Mask")
                if rgb_past is not None:
                    st.image(rgb_past, use_container_width=True, caption="Past – RGB View")
                else:
                    st.info("Could not create RGB preview for past image.")

                if water_past is not None and water_past.ndim == 2:
                    binary_mask_past = (water_past > 0).astype(np.uint8) * 255
                    st.image(binary_mask_past, use_container_width=True, caption="Past – Water Mask (white)")
                elif water_past is None:
                    st.caption("Run the model from the sidebar to see predictions for the past image.")

            # ----- PRESENT -----
            with col_present:
                st.subheader("📅 Present – Input & Water Mask")
                if rgb_present is not None:
                    st.image(rgb_present, use_container_width=True, caption="Present – RGB View")
                else:
                    st.info("Could not create RGB preview for present image.")

                if water_present is not None and water_present.ndim == 2:
                    binary_mask_present = (water_present > 0).astype(np.uint8) * 255
                    st.image(binary_mask_present, use_container_width=True, caption="Present – Water Mask (white)")
                elif water_present is None:
                    st.caption("Run the model from the sidebar to see predictions for the present image.")

            st.markdown('</div>', unsafe_allow_html=True)

        # Metrics + downloads + comparison
        if (
            water_past is not None and water_past.ndim == 2 and
            water_present is not None and water_present.ndim == 2
        ):
            stats_past = compute_area_stats(water_past, profile_past)
            stats_present = compute_area_stats(water_present, profile_present)

            st.markdown("### 📐 Water Area Summary – Past vs Present")

            # Past metrics
            st.markdown("#### 🕰 Past Image")
            mcol1, mcol2, mcol3, mcol4 = st.columns(4)
            with mcol1:
                st.metric("Water pixels", f"{stats_past['water_pixels']:,}")
            with mcol2:
                st.metric("Coverage (%)", f"{stats_past['coverage_percent']:.3f}")
            with mcol3:
                st.metric("Water area (m²)", f"{stats_past['water_area_m2']:.2f}")
            with mcol4:
                st.metric("Water area (km²)", f"{stats_past['water_area_km2']:.4f}")

            # Present metrics
            st.markdown("#### 📅 Present Image")
            ccol1, ccol2, ccol3, ccol4 = st.columns(4)
            with ccol1:
                st.metric("Water pixels", f"{stats_present['water_pixels']:,}")
            with ccol2:
                st.metric("Coverage (%)", f"{stats_present['coverage_percent']:.3f}")
            with ccol3:
                st.metric("Water area (m²)", f"{stats_present['water_area_m2']:.2f}")
            with ccol4:
                st.metric("Water area (km²)", f"{stats_present['water_area_km2']:.4f}")

            # --- Comparison (difference) ---
            st.markdown("#### 🔁 Change in Water Area (Present − Past)")
            diff_km2 = stats_present["water_area_km2"] - stats_past["water_area_km2"]
            diff_m2 = stats_present["water_area_m2"] - stats_past["water_area_m2"]
            diff_cover = stats_present["coverage_percent"] - stats_past["coverage_percent"]

            dcol1, dcol2, dcol3 = st.columns(3)
            with dcol1:
                st.metric(
                    "Water area (km²)",
                    f"{stats_present['water_area_km2']:.4f}",
                    delta=f"{diff_km2:+.4f} km²",
                )
            with dcol2:
                st.metric(
                    "Water area (m²)",
                    f"{stats_present['water_area_m2']:.2f}",
                    delta=f"{diff_m2:+.2f} m²",
                )
            with dcol3:
                st.metric(
                    "Coverage (%)",
                    f"{stats_present['coverage_percent']:.3f}",
                    delta=f"{diff_cover:+.3f} pts",
                )

            # --- Human-readable summary: increased or decreased? ---
            if diff_km2 > 0:
                change_text = f"⬆ Water body area has **increased** by **{diff_km2:.4f} km²** compared to the past image."
                st.success(change_text)
            elif diff_km2 < 0:
                change_text = f"⬇ Water body area has **decreased** by **{abs(diff_km2):.4f} km²** compared to the past image."
                st.error(change_text)
            else:
                st.info("➖ Water body area shows **no net change** between past and present images.")

            st.caption(
                "Note: Past and present images may have different pixel sizes/CRS; "
                "area values are computed independently for each image."
            )

            # Pixel info (from both)
            pixel_info_past = (
                f"{stats_past['pixel_width_m']:.2f} m × {stats_past['pixel_height_m']:.2f} m "
                f"(~{stats_past['pixel_area_m2']:.2f} m²/pixel) [Past]"
            )
            pixel_info_present = (
                f"{stats_present['pixel_width_m']:.2f} m × {stats_present['pixel_height_m']:.2f} m "
                f"(~{stats_present['pixel_area_m2']:.2f} m²/pixel) [Present]"
            )
            st.caption(f"Approx. pixel size: {pixel_info_past}")
            st.caption(f"Approx. pixel size: {pixel_info_present}")

            if stats_past["pixel_size_assumed"] or stats_present["pixel_size_assumed"]:
                st.caption("⚠ Pixel size estimated for at least one image (default 10 m × 10 m).")

            # Download masks
            st.markdown("#### ⬇ Download Water Masks")
            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                png_buf_past = mask_to_png_bytes(water_past)
                st.download_button(
                    label="Download Past Water Mask (PNG)",
                    data=png_buf_past,
                    file_name="water_mask_past.png",
                    mime="image/png",
                )
            with col_dl2:
                png_buf_present = mask_to_png_bytes(water_present)
                st.download_button(
                    label="Download Present Water Mask (PNG)",
                    data=png_buf_present,
                    file_name="water_mask_present.png",
                    mime="image/png",
                )

    except Exception as e:
        st.error(f"Error processing files: {e}")
elif uploaded_past is None and uploaded_present is None:
    st.info("Upload **both** past and present 6-band Sentinel-2 .tif files from the sidebar to get started.")
else:
    st.warning("Please upload **both** a past image and a present image before running the model.")
