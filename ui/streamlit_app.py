"""TerraOps UI — a THIN CLIENT of the serving API.

Design rule: this app NEVER loads the model or preprocesses images itself. It only
talks HTTP to the API (/predict, /predict/batch, /model-info). Consequences:
  - one serving path, one preprocessing, one model copy — no third skew surface;
  - this image needs neither torch nor the model — it knows classes only as the
    STRINGS the API returns in its JSON. The serving contract is the API, not code.

It always shows WHICH model version produced a prediction (from the API response),
so a user can trace any answer back to a governed registry version.
"""

import io
import math
import os

import folium
import requests
import streamlit as st
from streamlit_folium import st_folium

API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")

# Color per EuroSAT land-use class. Keyed by the class STRINGS the API returns;
# unknown labels fall back to gray. No model knowledge leaks in here.
CLASS_COLORS = {
    "AnnualCrop": "#e8d44d", "Forest": "#1b7837", "HerbaceousVegetation": "#a6d96a",
    "Highway": "#636363", "Industrial": "#d73027", "Pasture": "#91cf60",
    "PermanentCrop": "#fee08b", "Residential": "#fc8d59", "River": "#4575b4",
    "SeaLake": "#313695",
}
DEFAULT_COLOR = "#999999"

st.set_page_config(page_title="TerraOps — EuroSAT", page_icon="🛰️", layout="wide")


# --- API calls (all failures degrade to a friendly message) --------------------

def get_model_info():
    try:
        r = requests.get(f"{API_URL}/model-info", timeout=5)
        return r.json() if r.status_code == 200 else None
    except requests.RequestException:
        return None


def post_predict(name, data):
    r = requests.post(f"{API_URL}/predict",
                      files={"file": (name, data, "application/octet-stream")}, timeout=30)
    r.raise_for_status()
    return r.json()


def post_predict_batch(items):
    files = [("files", (n, d, "application/octet-stream")) for n, d in items]
    r = requests.post(f"{API_URL}/predict/batch", files=files, timeout=120)
    r.raise_for_status()
    return r.json()


# --- Sidebar: which model am I talking to? (traceability) ----------------------

st.sidebar.title("🛰️ TerraOps")
st.sidebar.caption(f"API: {API_URL}")
info = get_model_info()
if info:
    st.sidebar.success(f"Champion **v{info['model_version']}**")
    st.sidebar.write(f"Registry: `{info['registry_model']}@{info['alias']}`")
    st.sidebar.write(f"Device: `{info['device']}` · {info['num_classes']} classes")
else:
    st.sidebar.error("API not ready (no model loaded). "
                     "Start the stack and promote a champion, then POST /reload.")
if st.sidebar.button("🔄 Refresh model info"):
    st.rerun()

tab_single, tab_map = st.tabs(["🔍 Classify an image", "🗺️ Land-use map"])


# --- Tab 1: single-image classification ----------------------------------------

with tab_single:
    st.header("Classify one tile")
    up = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "tif", "tiff"],
                          key="single")
    if up is not None:
        data = up.read()
        col_img, col_res = st.columns([1, 2])
        col_img.image(data, caption=up.name, width=220)
        try:
            pred = post_predict(up.name, data)
        except requests.RequestException as exc:
            col_res.error(f"Prediction failed: {exc}")
        else:
            col_res.metric("Predicted class", pred["predicted_class"],
                           f"{pred['confidence'] * 100:.1f}% confidence")
            col_res.caption(f"Served by champion **v{pred['model_version']}**")
            # Probabilities, highest first — the string keys come straight from the API.
            probs = dict(sorted(pred["probabilities"].items(),
                                key=lambda kv: kv[1], reverse=True))
            col_res.bar_chart(probs)


# --- Tab 2: a grid of classified tiles, colored by land use --------------------

with tab_map:
    st.header("Classified land-use grid")
    st.caption("Upload several tiles — each becomes a colored cell on the grid, "
               "classified by the served model in one batched call.")

    center = [48.8566, 2.3522]  # arbitrary demo center (Paris); tiles have no geo-coords
    cell = 0.01                 # cell size in degrees

    ups = st.file_uploader("Upload tiles for the grid", accept_multiple_files=True,
                           type=["png", "jpg", "jpeg", "tif", "tiff"], key="grid")

    fmap = folium.Map(location=center, zoom_start=13, tiles="CartoDB positron")
    legend_classes = []

    if ups:
        items = [(u.name, u.read()) for u in ups]
        try:
            result = post_predict_batch(items)
        except requests.RequestException as exc:
            st.error(f"Batch prediction failed: {exc}")
            result = None

        if result:
            preds = result["predictions"]
            st.success(f"{len(preds)} tiles classified by champion "
                       f"v{result['model_version']}")
            cols = math.ceil(math.sqrt(len(preds)))
            for idx, (pred, (name, _)) in enumerate(zip(preds, items)):
                row, col = divmod(idx, cols)
                cls = pred["predicted_class"]
                color = CLASS_COLORS.get(cls, DEFAULT_COLOR)
                legend_classes.append(cls)
                south = center[0] - row * cell
                west = center[1] + col * cell
                folium.Rectangle(
                    bounds=[[south - cell, west], [south, west + cell]],
                    color=color, fill=True, fill_color=color, fill_opacity=0.75, weight=1,
                    tooltip=f"{name}: {cls} ({pred['confidence'] * 100:.0f}%)",
                ).add_to(fmap)

    st_folium(fmap, width=900, height=520)

    shown = [c for c in CLASS_COLORS if c in set(legend_classes)] or list(CLASS_COLORS)
    st.markdown("**Legend** &nbsp; " + " &nbsp; ".join(
        f"<span style='background:{CLASS_COLORS[c]};padding:2px 8px;border-radius:3px;"
        f"color:#fff'>{c}</span>" for c in shown), unsafe_allow_html=True)
