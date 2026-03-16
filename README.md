# Cardiac Microstructure Analysis – PhD Research Code

This repository contains analysis scripts developed during my PhD for studying cardiac microstructure in Tetralogy of Fallot using high-resolution cardiac imaging datasets, including synchrotron-based HiP-CT imaging.

The repository includes tools for:

- Conduction system right bundle branch morphology and depth analysis
- Ventricular wall thickness measurement from binary myocardial masks
- Mesh processing and geometric analysis
- Segmentation processing and ROI interpolation

These scripts were developed for processing and analysing cardiac imaging datasets obtained from high-resolution CT and synchrotron imaging pipelines.

Author: **Vaishnavi Sabarigirivasan**  
Institute: **UCL Institute of Cardiovascular Science**

---

# Repository Structure

scripts/

imagej_roi_interpolate.ijm
ImageJ macro for ROI detection and interpolation

decimate_obj_batch.py
Batch mesh decimation for OBJ meshes using PyVista

rbb_depth_analysis.py
Slice-wise depth and cross-sectional area analysis from NIfTI volumes

mask_wall_thickness.py
Interactive RV/LV wall thickness measurement from binary myocardial masks
