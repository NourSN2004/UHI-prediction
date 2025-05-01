# 🌇 Predicting NYC Land Surface Temperature

## 📖 Overview
This repository delivers a suite of deep-learning models for high-resolution (30 m, 16-day) Land Surface Temperature (LST) prediction in New York City. We leverage Landsat 8–derived indices (NDVI, NDBI, MNDWI) and hourly NOAA meteorological observations to benchmark architectures including CNNs, U-Nets, Vision Transformers, GANs, LSTMs/GRUs, and physics-informed networks.

<details>
  <summary>📂 Table of Contents</summary>

  - [📖 Overview](#📖-overview)
  - [📁 Project Structure](#📁-project-structure)
  - [🚀 Quick Start](#🚀-quick-start)
  - [🖥️ UI Application](#🖥️-ui-application)
  - [💾 Preprocessing](#💾-preprocessing)
  - [📂 Datasets](#📂-datasets)
  - [🔍 Architectures](#🔍-architectures)
  - [🧪 Testing](#🧪-testing)
  - [📊 Report & Results](#📊-report--results)
  - [🔑 Key Findings](#🔑-key-findings)
  - [👩‍💻 Authors](#👩‍💻-authors)
  - [📧 Contact](#📧-contact)
  - [📜 License](#📜-license)

</details>

---

## 📁 Project Structure
```
├── Architectures/             # Model implementations
├── Preprocessing/             # Jupyter notebook for data prep
├── testing/                   # Test scripts & evaluation notebooks
├── EECE_693_Final_Report.pdf  # Project report
├── app.py                     # Streamlit UI application
└── README.md                  # This file
```

---

## 🚀 Quick Start
1. Open the `Preprocessing.ipynb` notebook in Google Colab.
2. Mount your Google Drive containing Landsat 8 and NOAA files.
3. Update path variables (`LST_PATH`, `METEO_CSV_PATH`) in the first cell.
4. Run all preprocessing cells to generate 64×64 patches and clean data.
5. Navigate to an architecture folder (e.g., `Architectures/TransUnet`) and execute training/evaluation.

---

## 🖥️ UI Application
An interactive Streamlit interface for on-the-fly LST prediction.

Users can select any date within the available range and draw or specify any custom patch over New York City to retrieve the predicted LST for that location and time.

**Launch steps:**
```bash
cd path/to/project
# (Optional) setup venv
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\\Scripts\\activate  # Windows

# Install dependencies
pip install streamlit folium streamlit-folium numpy torch pillow mercantile requests

# Run the app
streamlit run app.py
```

Then open `http://localhost:8501` in your browser to use the NYC LST Predictor UI.

---

## 💾 Preprocessing
**Setup:**
1. **Landsat 8 Data:** Export LST bands via USGS EarthExplorer or Google Earth Engine.
2. **NOAA Data:** Download hourly meteorological CSVs from the NOAA Global Hourly portal.
3. **Configure Paths:** Set `LST_PATH` and `METEO_CSV_PATH` in `Preprocessing.ipynb`.

Run the notebook to produce cleaned datasets ready for modeling.

---

## 📂 Datasets
- **Meteorological Data:** [Drive Link](https://drive.google.com/file/d/1ss4D_ZkzQWdW9VIsAOJFZBPHo05u04sR/view)
- **LST + Meteo Patches:** [Drive Folder](https://drive.google.com/drive/folders/1nXb8mzun6akRigNKNxWN9S0lplsE6m3V)

---

## 🔍 Architectures
<details>
<summary>Click to expand model list</summary>

- **CNN_MLP:** Baseline CNN + MLP fusion.
- **Unet:** Standard U-Net variants (ResNet encoder, SEBlock) with different losses (L1, Smooth L1, Focal-Tversky).
- **Unet-Vit-hybrid:** TransUNet (MiT-B0 + U-Net decoder).
- **LSTM_vit:** ViT backbone + 6-hour LSTM for weather.
- **GAN_meteo:** Conditional GAN for meteorological data augmentation.
- **GRU:** ViT encoder + GRU for weather sequences.
- **TransUnet:** U-Net with self-attention and Focal-Tversky loss.
- **VIT_PINN_transformer:** Physics-informed ViT using Newtonian cooling priors.
- **Vit:** Vision Transformer baseline with hyperparameter tuning and last-layer unfreezing.
- **Final_Optimal:** Physics-informed ViT achieving RMSE ≈ 0.21 °C.

</details>

---

## 🧪 Testing
Test scripts and evaluation notebooks reside in `testing/`. They use held-out 2023 LST and meteorological records.

**Run tests:**
1. Mount your data in Colab or local paths.
2. Update path variables in test notebooks.
3. Execute all cells or run scripts directly.

---

## 📊 Report & Results
See **EECE_693_Final_Report.pdf** for full details.

- **Best RMSE:** 0.21 °C (Physics-informed ViT)
- **Ablation Studies:** Loss functions & fusion strategies.
- **Visuals:** Loss curves, attention maps, spatial error heatmaps.

---

## 🔑 Key Findings
- **ViT (baseline):** Strong performance with hyperparameter tuning & selective unfreezing.
- **TransUnet:** Improved boundary accuracy via self-attention & Focal-Tversky loss.
- **VIT_PINN_transformer:** Lowest RMSE (~0.21 °C) by embedding physical priors.

---

## 👩‍💻 Authors
- Ahmad Hlayhel (@ahmadhlayhel)
- John Lahoud (@lahoudjohn)
- Malak Ammar (@ammar-m-s)
- **Nour Shammaa** (@NourSN2004)

Special thanks to Dr. Mariette Awad and Dr. Hadi Mobasher.

---

## 📧 Contact
For inquiries or collaboration, email **nour.shammaa@aub.edu.lb**.

---

## 📜 License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

