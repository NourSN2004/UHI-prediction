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
**Setup Instructions:**
1. **Download Official Landsat 8 Data**
   - Use USGS EarthExplorer or Google Earth Engine to export LST bands for NYC.
2. **Download Hourly Meteorological Data**
   - Fetch CSVs from NOAA Global Hourly: https://www.ncei.noaa.gov/access/search/data-search/global-hourly
3. **Update Preprocessing Paths**
   - In the first cell of Preprocessing.ipynb, set LST_PATH to your Landsat folder and METEO_CSV_PATH to the NOAA CSV location.

Once paths are set, run the notebook cells to generate 64×64 patches and clean the data.

---

## 📂 Datasets
- **Cleaned Meteorological Data:** [Google Drive File](https://drive.google.com/file/d/1ss4D_ZkzQWdW9VIsAOJFZBPHo05u04sR/view?usp=drive_link)
- **Final cleaned LST with Meteo:** [Google Drive Folder](https://drive.google.com/drive/folders/1nXb8mzun6akRigNKNxWN9S0lplsE6m3V?usp=drive_link)

---

## 🔍 Architectures
<details>
<summary>List of implementations</summary>

- **CNN_MLP**: Baseline CNN + MLP fusion of image & weather tokens.
- **Unet**: Standard U-Net with multiple variants — loss functions include Smooth L1, L1, and Focal-Tversky; one implementation features a ResNet encoder, while another integrates the SEBlock described in the report.
- **Unet-Vit-hybrid**: TransUNet combining the Mix Transformer (MiT-B0) backbone with a U-Net decoder for spatio-temporal fusion.
- **LSTM_vit**: Vision Transformer backbone fused with a 6-hour LSTM head to incorporate sequential meteorological inputs.
- **GAN_meteo**: Conditional GAN architecture for data augmentation, generating synthetic meteorological feature maps only.
- **GRU**: Fusion model using a Vision Transformer encoder with a GRU module to process weather sequences.
- **TransUnet**: Transformer-augmented U-Net that leverages self-attention layers and employs the Focal-Tversky loss for improved boundary delineation.
- **VIT_PINN_transformer**: Physics-informed Vision Transformer integrating Newtonian cooling priors to enforce physically realistic temperature decay patterns.
- **Vit**: Vision Transformer baseline model, enhanced through hyperparameter tuning and selective unfreezing of the final layer.
- **Final_Optimal**: Hyperparameter-tuned best-performing configuration using a physics-informed Vision Transformer (achieved RMSE ≈ 0.21 °C).

</details>

---

## 🧪 Testing
We evaluated our data pipelines and model implementations using held-out LST and meteorological records from **2023**. All relevant test scripts and evaluation notebooks are located in the testing/ directory.

To run tests:
1. Mount your Google Drive in Colab or ensure local access to your data directories.
2. Update the path variables in the first cell of each testing notebook to point to your LST and meteo data.
3. Execute the notebook cells or run the test scripts directly in Colab or locally.

---

## 📊 Report & Results
The full project report is available as **EECE_693_Final_Report.pdf**. Highlights:
- **Best RMSE**: 0.21 °C (Physics-informed Transformer)
- **Ablation studies** on loss functions and data fusion strategies.
- **Visualizations**: Loss curves, attention maps, spatial error heatmaps.

---

## 🔑 Key Findings
- **Vit (Vision Transformer)** with hyperparameter tuning and selective unfreezing of the final layer achieved strong convergence and low loss metrics.
- **TransUnet** variant further improved loss performance, particularly at boundary regions, thanks to self-attention and Focal-Tversky loss.
- The **VIT_PINN_transformer** (physics-informed Vision Transformer) was our most optimal model, integrating physical priors and achieving the lowest RMSE (~0.21 °C).

---

## 👩‍💻 Authors
- Ahmad Hlayhel (@ahmadhlayhel)
- John Lahoud (@lahoudjohn)
- Malak Ammar (@ammar-m-s)
- **Nour Shammaa** (@NourSN2004)

---

Special Thanks to Dr. Mariette Awad and Dr. Hadi Mobasher for their support and guidance.

---

## 📧 Contact
For inquiries or collaboration, email **nour.shammaa@aub.edu.lb**.

---

## 📜 License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

