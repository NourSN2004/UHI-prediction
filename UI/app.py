import os
import streamlit as st
from folium.plugins import Draw
from streamlit_folium import st_folium
import folium
import numpy as np
import torch
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import mercantile
import requests
from datetime import datetime, timedelta

# ----------------------------------------------------------------------------
# 1. CONFIGURATION
# ----------------------------------------------------------------------------

st.set_page_config(page_title="NYC LST Predictor", layout="wide")

MAPBOX_TOKEN = os.getenv("MAPBOX_API_KEY", "").strip()
USE_MAPBOX = bool(MAPBOX_TOKEN)
MAPBOX_URL = (
    "https://api.mapbox.com/styles/v1/mapbox/satellite-v9/tiles/{z}/{x}/{y}"
    f"?access_token={MAPBOX_TOKEN}"
) if USE_MAPBOX else None

HEADERS = {"User-Agent": "NYC-LST-Predictor/0.1 (your_email@example.com)"}
MAX_TILES = 4
MODEL_PATH = Path("data/models/best_transunet_ftl.pth")  # Make sure this model is uploaded
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------------------------------------------------------
# 2. LST DENORMALIZATION PARAMETERS
# ----------------------------------------------------------------------------

LST_MIN_C = 0.0
LST_MAX_C = 50.0

# ----------------------------------------------------------------------------
# 3. MODEL LOADING & PREDICTION
# ----------------------------------------------------------------------------

@st.cache_resource
def load_lst_model():
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 16, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(16, 1, kernel_size=3, padding=1)
    )
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    state_dict = ckpt.get('model_state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state_dict, strict=False)
    model.to(DEVICE)
    model.eval()
    return model

def predict_lst(image_array: np.ndarray, model: torch.nn.Module) -> np.ndarray:
    tensor = torch.from_numpy(image_array.transpose(2, 0, 1)).unsqueeze(0).float().to(DEVICE)
    with torch.no_grad():
        out = model(tensor)
    return out.squeeze().cpu().numpy()

# ----------------------------------------------------------------------------
# 4. STREAMLIT UI
# ----------------------------------------------------------------------------

st.title("🌆 NYC LST Predictor")

# Add date input
current_date = datetime.today()
max_date = current_date + timedelta(days=365*2)
prediction_date = st.date_input("Select Date for LST Prediction", min_value=datetime(2022, 1, 1), max_value=max_date)
st.session_state.prediction_date = prediction_date

@st.cache_data
def create_map(center=[40.7128, -74.0060], zoom=12):
    m = folium.Map(location=center, zoom_start=zoom, tiles=None)
    tile_url = MAPBOX_URL or "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
    attr = "Mapbox Satellite" if USE_MAPBOX else "OpenStreetMap"
    folium.TileLayer(tiles=tile_url, attr=attr, name=attr).add_to(m)
    Draw().add_to(m)
    return m

m = create_map()
out = st_folium(m, width=700, height=500)

if out and out.get("all_drawings"):
    draw = out["all_drawings"][0]
    if draw["geometry"]["type"] == "Polygon":
        coords = draw["geometry"]["coordinates"][0]
        west, south = coords[0]
        east, north = coords[2]

        zoom = 18
        tiles = list(mercantile.tiles(west, south, east, north, zoom))
        xs, ys = [t.x for t in tiles], [t.y for t in tiles]
        n_x, n_y = max(xs)-min(xs)+1, max(ys)-min(ys)+1

        if n_x > MAX_TILES or n_y > MAX_TILES:
            st.warning(f"Selected region is {n_x}×{n_y} tiles (max {MAX_TILES}). Please draw smaller.")
        else:
            st.info(f"Region = {n_x}×{n_y} tiles. Click ▶ Run Model.")
            if st.button("Run Model"):
                with st.spinner(f"Fetching & resizing tiles for {prediction_date}..."):
                    mosaic = Image.new('RGB', (n_x*256, n_y*256))
                    min_x, min_y = min(xs), min(ys)
                    for t in tiles:
                        url = MAPBOX_URL.format(z=zoom, x=t.x, y=t.y) if USE_MAPBOX \
                              else f"https://tile.openstreetmap.org/{zoom}/{t.x}/{t.y}.png"
                        r = requests.get(url, headers=HEADERS); r.raise_for_status()
                        tile_img = Image.open(BytesIO(r.content)).convert('RGB')
                        mosaic.paste(tile_img, ((t.x-min_x)*256, (t.y-min_y)*256))
                    mosaic = mosaic.resize((224, 224), Image.BILINEAR)
                    img_arr = np.array(mosaic)

                model = load_lst_model()
                lst_pred = predict_lst(img_arr, model)

                # Denormalize
                lst_pred = lst_pred * (LST_MAX_C - LST_MIN_C) + LST_MIN_C

                # Re-normalize to range for color mapping
                lst_min, lst_max = lst_pred.min(), lst_pred.max()
                if lst_max > lst_min:
                    lst_pred = (lst_pred - lst_min) / (lst_max - lst_min)
                    lst_pred = lst_pred * (LST_MAX_C - LST_MIN_C) + LST_MIN_C
                else:
                    lst_pred = np.full_like(lst_pred, (LST_MAX_C + LST_MIN_C)/2)

                st.subheader(f"Predicted LST (°C) for {prediction_date}")
                fig, ax = plt.subplots(figsize=(4, 4))
                im = ax.imshow(lst_pred, cmap="jet", vmin=LST_MIN_C, vmax=LST_MAX_C)
                ax.axis('off')
                cbar = fig.colorbar(im, ax=ax, label="LST (°C)")
                cbar.set_ticks(np.linspace(LST_MIN_C, LST_MAX_C, 6))
                st.pyplot(fig)
    else:
        st.warning("Please draw a polygon.")
else:
    st.info("Draw a polygon to select a patch.")
