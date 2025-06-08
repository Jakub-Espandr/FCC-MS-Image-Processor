<p align="center">
  <a href="https://i.imghippo.com/files/pWPz7890sA.jpg">
    <img src="https://i.imghippo.com/files/pWPz7890sA.jpg" alt="FCC MS Image Processor" width="250"/>
  </a>
</p>

<h1 align="center">FlyCamCzech MultiSpectral Image Processor</h1>
<p align="center"><em>(FlyCamCzech | Jakub Ešpandr)</em></p>

---

## Overview

The **FlyCamCzech MultiSpectral Image Processor (FCC MS Image Processor)** is a powerful and user-friendly application designed for analyzing multispectral UAV imagery. Developed as part of the thesis of **Bc. Jakub Ešpandr** (*Classification of UAV Image Data within Selected Areas*), the software allows for intuitive calculation, visualization, and export of over 50 vegetation indices.

Originally developed for scientific and environmental monitoring purposes, the tool is now accessible via a modern **web interface**, usable from any device with a browser.

---

## Key Features

- 🌱 **Vegetation Index Calculation**  
  Supports more than **50 spectral indices** used in plant health and land cover analysis (NDVI, GNDVI, CIg, SAVI, VARI...).
  
- 🖼️ **Multispectral Image Import**  
  Load aligned image bands from modified camera setups or UAV sensors.

- 📊 **Visualization Tools**  
  Real-time display of processed indices using colormaps and overlays.

- 💾 **Data Export**  
  Save results as processed images or raw data for further analysis.

- 🌐 **Web Interface**  
  The entire app runs in-browser via a lightweight Python web server.

---

<details>
<summary><strong>📚 Supported Vegetation Indices (Click to expand)</strong></summary>

- BGI, BI, BNDVI, BWDRVI  
- CIg, CIred, CIVE, CVI, CVI2  
- DVI, ENDVI, ENDVI_RGN, EVI, EVI2, EVI_Mod  
- ExG, ExGR, ExR  
- GBNDVI, GCC, GLI, GLI2  
- GNDVI, GNDWI, GOSAVI, GRNDVI, GRVI, MExG  
- MSAVI, MTVI2, NDBI, NDBI-Blue  
- NGBDI, NGBVI, NDGI, NDVI, NDVI_Mod  
- NDWI, NDTI, NG, NGRDI  
- OSAVI, PRI, RGBVI, RDVI, RGRI  
- SAVI, SCI, SR, TGI, TVI  
- UI, VARI, VDVI, VEG, WDRVI

</details>

---

## Technologies Used

- **Python** – Core processing logic  
- **OpenCV** – Image reading and manipulation  
- **Matplotlib** – Plotting and visualization  
- **Flask / Tkinter / PyQt** – Web and desktop GUI  
- **Custom colormaps** – Tailored for scientific readability

---

## About

This application simplifies complex multispectral analysis workflows for both professionals and enthusiasts in:

- 🌾 **Precision Agriculture**  
- 🌍 **Environmental Monitoring**  
- 🧪 **Scientific Research**  
- 🌳 **Vegetation Health Assessment**

It is intended to reduce the technical barrier for vegetation monitoring using UAVs.

---

## License

Licensed under the **Non-Commercial Public License (NCPL v1.0)**  
© 2025 Jakub Ešpandr – FlyCamCzech.eu

---
