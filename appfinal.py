import io
import math

import numpy as np
import streamlit as st
from rasterio.io import MemoryFile
from PIL import Image, ImageDraw
from scipy import ndimage  # for connected-components

from infer import watnet_infer  # from your repo


# ---------- PAGE CONFIG & BASIC STYLING ----------

st.set_page_config(page_title="Water Mapping (Past vs Present)", layout="wide")

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
    table.water-table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 0.5rem;
        font-size: 0.9rem;
    }
    table.water-table th, table.water-table td {
        border: 1px solid #44444455;
        padding: 0.3rem 0.5rem;
        text-align: left;
    }
    table.water-table th {
        background-color: #222222aa;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="center-title">Surface Water Mapping (Past vs Present)</div>',
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


# ---------- WATER MASK HELPER ----------

def get_water_mask(wm):
    """
    Convert a predicted map to a boolean water mask.
    Reuses the same logic for both total-area and per-object stats.
    """
    wm = np.array(wm)
    unique_vals = np.unique(wm)

    # Binary case (0/1)
    if np.all(np.isin(unique_vals, [0, 1])):
        water_mask = (wm == 1)
    else:
        # Continuous probabilities/scores
        max_val = wm.max()
        if max_val <= 0:
            water_mask = np.zeros_like(wm, dtype=bool)
        else:
            thr = 0.5 * max_val
            water_mask = wm >= thr

    return water_mask


# ---------- AREA STATS (TOTAL) ----------

def compute_area_stats(water_map, profile):
    """
    Compute water area stats using GeoTIFF metadata.
    Handles binary or continuous maps and different CRS types.
    """
    wm = np.array(water_map)
    water_mask = get_water_mask(wm)

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


# ---------- INDIVIDUAL WATER BODIES & SIZE CATEGORIES ----------

# def compute_object_area_categories(water_map, stats):
#     """
#     Categorize connected water bodies into size groups:
#       - very small: < 1 km²
#       - small:      1–10 km²
#       - medium:     10–50 km²
#       - large:      > 50 km²

#     Returns:
#         summary_dict, labeled_map, label_categories, object_stats
#         where object_stats is a list of:
#             {
#               "id": "WB-001",
#               "label": 1,
#               "area_km2": ...,
#               "size": "very_small" | "small" | "medium" | "large"
#             }
#     """
#     wm = np.array(water_map)
#     water_mask = get_water_mask(wm)

#     result = {
#         "very_small": {"count": 0, "total_area_km2": 0.0},
#         "small":      {"count": 0, "total_area_km2": 0.0},
#         "medium":     {"count": 0, "total_area_km2": 0.0},
#         "large":      {"count": 0, "total_area_km2": 0.0},
#     }

#     if not np.any(water_mask):
#         return result, None, None, []

#     labeled, num_features = ndimage.label(water_mask)
#     if num_features == 0:
#         return result, labeled, None, []

#     pixel_area_m2 = stats["pixel_area_m2"]

#     indices = np.arange(1, num_features + 1)
#     pixel_counts = ndimage.sum(
#         water_mask.astype(np.int32),
#         labels=labeled,
#         index=indices,
#     )

#     label_categories = {}
#     object_stats = []

#     for label_idx, count in zip(indices, pixel_counts):
#         area_km2 = (count * pixel_area_m2) / 1e6

#         if area_km2 <= 0:
#             continue

#         # Thresholds:
#         # < 1       -> very small
#         # 1–10      -> small
#         # 10–50     -> medium
#         # > 50      -> large
#         if area_km2 < 1.0:
#             key = "very_small"
#         elif 1.0 <= area_km2 < 10.0:
#             key = "small"
#         elif 10.0 <= area_km2 <= 50.0:
#             key = "medium"
#         else:
#             key = "large"

#         result[key]["count"] += 1
#         result[key]["total_area_km2"] += area_km2
#         label_categories[int(label_idx)] = key

#         object_stats.append(
#             {
#                 "id": f"WB-{int(label_idx):03d}",
#                 "label": int(label_idx),
#                 "area_km2": float(area_km2),
#                 "size": key,
#             }
#         )

#     # Sort objects by area (descending) for nicer table
#     object_stats.sort(key=lambda d: d["area_km2"], reverse=True)

#     return result, labeled, label_categories, object_stats


def compute_object_area_categories(water_map, stats):
    """
    Categorize connected water bodies into size groups:
      - very small: < 1 km²
      - small:      1–10 km²
      - medium:     10–50 km²
      - large:      > 50 km²

    Water bodies smaller than 0.001 km² are ignored.
    """

    MIN_THRESHOLD = 0.001  # km² cutoff

    wm = np.array(water_map)
    water_mask = get_water_mask(wm)

    # Return empty result if no water detected
    if not np.any(water_mask):
        return {
            "very_small": {"count": 0, "total_area_km2": 0.0},
            "small":      {"count": 0, "total_area_km2": 0.0},
            "medium":     {"count": 0, "total_area_km2": 0.0},
            "large":      {"count": 0, "total_area_km2": 0.0},
        }, None, None, []

    result = {
        "very_small": {"count": 0, "total_area_km2": 0.0},
        "small":      {"count": 0, "total_area_km2": 0.0},
        "medium":     {"count": 0, "total_area_km2": 0.0},
        "large":      {"count": 0, "total_area_km2": 0.0},
    }

    labeled, num_features = ndimage.label(water_mask)
    if num_features == 0:
        return result, labeled, None, []

    pixel_area_m2 = stats["pixel_area_m2"]

    indices = np.arange(1, num_features + 1)
    pixel_counts = ndimage.sum(water_mask.astype(np.int32), labels=labeled, index=indices)

    label_categories = {}
    object_stats = []

    # Iterate through detected objects
    for label_idx, count in zip(indices, pixel_counts):

        area_km2 = (count * pixel_area_m2) / 1e6  # convert to km²

        # Skip tiny objects
        if area_km2 < MIN_THRESHOLD:
            continue

        # Categorize by size
        if area_km2 < 1.0:
            key = "very_small"
        elif area_km2 < 10.0:
            key = "small"
        elif area_km2 <= 50.0:
            key = "medium"
        else:
            key = "large"

        result[key]["count"] += 1
        result[key]["total_area_km2"] += area_km2
        label_categories[int(label_idx)] = key

        object_stats.append({
            "id": f"WB-{int(label_idx):03d}",
            "label": int(label_idx),
            "area_km2": round(area_km2, 6),
            "size": key
        })

    # Sort largest first
    object_stats.sort(key=lambda x: x["area_km2"], reverse=True)

    return result, labeled, label_categories, object_stats



def make_category_color_mask(labeled, label_categories):
    """
    Create an RGB image where each water body is colored
    based on its category (very small / small / medium / large).
    """
    if labeled is None or label_categories is None:
        return None

    h, w = labeled.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    # Colors (RGB)
    color_map = {
        "very_small": (249, 212, 4),  # very small -> #f9d404
        "small":      (0, 112, 255),  # blue
        "medium":     (0, 200, 83),   # green
        "large":      (244, 67, 54),  # red
    }

    for label_idx, cat in label_categories.items():
        color = color_map.get(cat, (255, 255, 255))
        mask = (labeled == label_idx)
        rgb[mask] = color

    return rgb


def annotate_water_bodies(category_mask, labeled, object_stats):
    """
    Draw water body IDs (WB-001, WB-002, ...) on top of the
    colored category mask, at each object's centroid.
    """
    if category_mask is None or labeled is None or not object_stats:
        return category_mask

    img_pil = Image.fromarray(category_mask)
    draw = ImageDraw.Draw(img_pil)

    indices = [obj["label"] for obj in object_stats]

    if len(indices) > 0:
        centers = ndimage.center_of_mass(
            np.ones_like(labeled, dtype=np.uint8),
            labels=labeled,
            index=indices,
        )
    else:
        centers = []

    for obj, center in zip(object_stats, centers):
        if center is None:
            continue
        row, col = center
        x = int(col)
        y = int(row)

        text = obj["id"]
        # Outline
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            draw.text((x + dx, y + dy), text, fill=(0, 0, 0))
        # Main text
        draw.text((x, y), text, fill=(255, 255, 255))

    return np.array(img_pil)


def render_size_category_block(title, cat):
    """
    Render a compact summary for size categories in Streamlit.
    """
    st.markdown(f"##### {title} – Water Body Size Distribution")
    col_v, col_s, col_m, col_l = st.columns(4)

    with col_v:
        st.metric(
            "Very Small (<1 km²)",
            value=f"{cat['very_small']['count']}",
            delta=f"Total: {cat['very_small']['total_area_km2']:.3f} km²",
        )
    with col_s:
        st.metric(
            "Small (1–10 km²)",
            value=f"{cat['small']['count']}",
            delta=f"Total: {cat['small']['total_area_km2']:.3f} km²",
        )
    with col_m:
        st.metric(
            "Medium (10–50 km²)",
            value=f"{cat['medium']['count']}",
            delta=f"Total: {cat['medium']['total_area_km2']:.3f} km²",
        )
    with col_l:
        st.metric(
            "Large (>50 km²)",
            value=f"{cat['large']['count']}",
            delta=f"Total: {cat['large']['total_area_km2']:.3f} km²",
        )


def render_size_legend():
    """
    Render a color legend for very small / small / medium / large categories.
    """
    st.markdown("###### Legend – Water Body Size Categories")
    cols = st.columns(4)
    entries = [
        ("Very Small (<1 km²)", "#f9d404"),
        ("Small (1–10 km²)", "#0070ff"),
        ("Medium (10–50 km²)", "#00c853"),
        ("Large (>50 km²)", "#f44336"),
    ]
    for col, (label, color) in zip(cols, entries):
        with col:
            st.markdown(
                f"""
                <div style="display:flex;align-items:center;margin-bottom:0.3rem;">
                    <div style="width:18px;height:18px;background:{color};
                                margin-right:0.4rem;border-radius:3px;"></div>
                    <span style="font-size:0.9rem;">{label}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )


# def render_per_object_table(title, object_stats):
#     """
#     Render a table:
#         Water Body ID | Area (km²) | Size
#     using HTML (no pandas dependency).
#     """
#     st.markdown(f"#### 📋 {title} – Per-Water-Body Details")

#     if not object_stats:
#         st.info("No individual water bodies detected.")
#         return

#     size_display = {
#         "very_small": "Very Small",
#         "small": "Small",
#         "medium": "Medium",
#         "large": "Large",
#     }

#     rows_html = ""
#     for obj in object_stats:
#         wb_id = obj["id"]
#         area_str = f"{obj['area_km2']:.4f}"
#         size_str = size_display.get(obj["size"], obj["size"])
#         rows_html += f"""
#         <tr>
#             <td>{wb_id}</td>
#             <td>{area_str}</td>
#             <td>{size_str}</td>
#         </tr>
#         """

#     table_html = f"""
#     <table class="water-table">
#         <thead>
#             <tr>
#                 <th>Water Body ID</th>
#                 <th>Area (km²)</th>
#                 <th>Size</th>
#             </tr>
#         </thead>
#         <tbody>
#             {rows_html}
#         </tbody>
#     </table>
#     """

#     st.markdown(table_html, unsafe_allow_html=True)

def render_per_object_table(title, object_stats):
    """
    Render a table:
        Water Body ID | Area (km²) | Size
    using Markdown (no HTML, no pandas).
    """
    st.markdown(f"#### 📋 {title} – Per-Water-Body Details")

    if not object_stats:
        st.info("No individual water bodies detected.")
        return

    size_display = {
        "very_small": "Very Small",
        "small": "Small",
        "medium": "Medium",
        "large": "Large",
    }

    lines = []
    # Markdown header
    lines.append("| Water Body ID | Area (km²) | Size |")
    lines.append("|---------------|------------|------|")

    for obj in object_stats:
        wb_id = obj["id"]
        area_str = f"{obj['area_km2']:.4f}"
        size_str = size_display.get(obj["size"], obj["size"])
        lines.append(f"| {wb_id} | {area_str} | {size_str} |")

    table_md = "\n".join(lines)
    st.markdown(table_md)



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
                    st.image(
                        binary_mask_past,
                        use_container_width=True,
                        caption="Past – Water Mask (white)",
                    )
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
                    st.image(
                        binary_mask_present,
                        use_container_width=True,
                        caption="Present – Water Mask (white)",
                    )
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

            # ---------- PER-OBJECT SIZE CATEGORIES & COLOR MAPS ----------
            cat_past, labels_past, label_cat_past, objs_past = compute_object_area_categories(water_past, stats_past)
            cat_present, labels_present, label_cat_present, objs_present = compute_object_area_categories(water_present, stats_present)

            category_mask_past = make_category_color_mask(labels_past, label_cat_past)
            category_mask_present = make_category_color_mask(labels_present, label_cat_present)

            # Annotate IDs on top of colored masks
            annotated_past = annotate_water_bodies(category_mask_past, labels_past, objs_past)
            annotated_present = annotate_water_bodies(category_mask_present, labels_present, objs_present)

            st.markdown("### 🗺️ Water Bodies by Size Category (with IDs)")
            render_size_legend()
            col_cat_past, col_cat_present = st.columns(2)
            with col_cat_past:
                if annotated_past is not None:
                    st.image(
                        annotated_past,
                        use_container_width=True,
                        caption="Past – Water Bodies by Size Category (IDs shown)",
                    )
                else:
                    st.caption("No water bodies detected in the past image.")
            with col_cat_present:
                if annotated_present is not None:
                    st.image(
                        annotated_present,
                        use_container_width=True,
                        caption="Present – Water Bodies by Size Category (IDs shown)",
                    )
                else:
                    st.caption("No water bodies detected in the present image.")

            # ---------- SIZE CATEGORY SUMMARY ----------
            render_size_category_block("Past", cat_past)
            render_size_category_block("Present", cat_present)

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

            # ---------- PER-WATER-BODY TABLES ----------
            st.markdown("### 📊 Detailed Water Body Tables")
            col_tbl_past, col_tbl_present = st.columns(2)
            with col_tbl_past:
                render_per_object_table("Past Image", objs_past)
            with col_tbl_present:
                render_per_object_table("Present Image", objs_present)

            # Download masks
            st.markdown("#### ⬇ Download Water Masks (Binary)")
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
