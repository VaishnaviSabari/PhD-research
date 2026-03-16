# ============================================================
# Script: decimate_obj_batch.py
# Author: Vaishnavi Sabarigirivasan
# Repository: conduction-tools
#
# Description:
# Batch-processing utility to reduce the size of 3D surface meshes
# stored as .OBJ files. Uses PyVista's mesh decimation method to
# reduce the number of faces while maintaining overall geometry.
#
# For each .OBJ file in the input folder:
#   1. The mesh is loaded with PyVista.
#   2. The mesh is decimated (e.g., reduce faces by 95%).
#   3. The reduced mesh is saved in the output folder with prefix
#      "REDUCED_".
#
# Typical Use Case:
# - Preparing large meshes from imaging data for visualization,
#   virtual reality (VR) environments, or faster processing.
#
# Key Parameters:
# - input_folder: path to directory containing OBJ files.
# - output_folder: path to save reduced meshes.
# - reduction: proportion of faces to remove (0–1).
#
# Requirements:
# - Python 3.10+ recommended
# - PyVista, tqdm
#
# ============================================================

import os
from pathlib import Path
import pyvista as pv
from tqdm import tqdm


def decimate_folder(input_folder: Path, output_folder: Path, reduction: float = 0.95) -> None:
    """
    Decimate all OBJ meshes in a folder and save reduced versions.
    """
    output_folder.mkdir(parents=True, exist_ok=True)
    obj_files = [p for p in input_folder.iterdir() if p.suffix.lower() == ".obj"]
    print(f"Found {len(obj_files)} .obj files to process in {input_folder}.")

    for path in tqdm(obj_files, desc="Processing Meshes", ncols=75):
        out_path = output_folder / f"REDUCED_{path.name}"
        try:
            mesh = pv.read(path)
            reduced = mesh.decimate(reduction)
            reduced.save(out_path)
        except Exception as e:
            print(f"Error processing {path.name}: {e}")


if __name__ == "__main__":
    # Example: update these paths to your own folders
    input_folder = Path(r"path/to/your/input/folder")
    output_folder = Path(r"path/to/your/output/folder")
    decimate_folder(input_folder, output_folder, reduction=0.95)
