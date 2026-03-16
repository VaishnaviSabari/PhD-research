# rbb_analysis.py

import os
import gc
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage import filters
from scipy.ndimage import sobel
from scipy.spatial.distance import cdist

### ==== COMMON HELPERS ====

def load_nifti(path):
    try:
        return nib.load(path)
    except Exception as e:
        print(f"❌ Error loading {path}: {e}")
        return None

def extract_center_points(segmentation_slice):
    coords = np.column_stack(np.where(segmentation_slice == 1))
    return np.mean(coords, axis=0) if coords.size > 0 else None

def edge_detection(image_slice):
    return filters.sobel(image_slice)

def calculate_nearest_distance(center, edges, voxel_size_mm):
    edge_coords = np.column_stack(np.where(edges > 0))
    if edge_coords.size > 0:
        dist = cdist([center], edge_coords)
        return np.min(dist) * voxel_size_mm  # returns distance in mm
    return None

def calculate_area_of_slice(segmentation_slice, voxel_size_mm):
    return float(np.count_nonzero(segmentation_slice == 1)) * (voxel_size_mm ** 2)

### ==== PLOTTING ====

def plot_depth_results(distance_from_origin_mm, depth_mm):
    x = np.array(distance_from_origin_mm, dtype=float)  # mm
    y = np.array(depth_mm, dtype=float)                 # mm

    plt.figure(figsize=(10, 5))

    # Raw scatter
    plt.subplot(1, 2, 1)
    plt.scatter(x, y)
    plt.title("RBB Depth (Raw Data)")
    plt.xlabel("Distance from RBB Origin (mm)")
    plt.ylabel("Depth (mm)")
    plt.grid(True)

    # Best-fit curve
    plt.subplot(1, 2, 2)
    if len(x) > 3:
        coeffs = np.polyfit(x, y, 3)
        p = np.poly1d(coeffs)
        x_fit = np.linspace(x.min(), x.max(), 300)
        y_fit = p(x_fit)
        plt.plot(x_fit, y_fit, label='Polyfit (deg 3)')
    plt.scatter(x, y, alpha=0.3)
    plt.title("RBB Depth (Best Fit)")
    plt.xlabel("Distance from RBB Origin (mm)")
    plt.ylabel("Depth (mm)")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

### ==== EXPORT ====

def export_results(distance_from_origin_mm, surface_area_mm2, depth_mm, directory, filename):
    os.makedirs(directory, exist_ok=True)
    df = pd.DataFrame({
        'distance_from_origin_mm': distance_from_origin_mm,
        'surface_area_mm2': surface_area_mm2,
        'depth_mm': depth_mm
    })
    file_path = os.path.join(directory, filename)
    df.to_csv(file_path, index=False)
    print(f"✅ Exported results to: {file_path}")

### ==== DEPTH + SURFACE AREA ANALYSIS ====

def analyze_rbb_depth(image_img, seg_img, voxel_size_mm, export_directory):
    """
    Builds three aligned arrays for export/plot:
      - distance_from_origin_mm (x-axis)
      - surface_area_mm2
      - depth_mm
    Notes:
      - Per your request, the X-axis is divided by 1000 compared to the original accumulator.
    """
    # We keep a cumulative "distance" in the original units used by the previous code path,
    # then divide by 1000 for the final X-axis as requested.
    cumulative_distances_raw = []   # original accumulator
    depth_mm = []
    surface_area_mm2 = []

    running = 0.0

    for i in tqdm(range(seg_img.shape[2]), desc="RBB Depth Analysis"):
        try:
            image_slice = np.array(image_img.dataobj[:, :, i])
            segmentation_slice = np.array(seg_img.dataobj[:, :, i])
            segmentation = (segmentation_slice == 1)

            if np.any(segmentation):
                center = extract_center_points(segmentation)
                if center is not None:
                    edges = edge_detection(image_slice)
                    d_mm = calculate_nearest_distance(center, edges, voxel_size_mm)  # depth in mm

                    if d_mm is not None:
                        # Maintain original cumulative pattern (sum of per-slice values),
                        # then divide by 1000 for final mm X-axis as per your instruction.
                        running = (running + d_mm) if cumulative_distances_raw else 0.0
                        cumulative_distances_raw.append(running)

                        depth_mm.append(d_mm)

                        # surface area (mm²) for the same slice
                        sa_mm2 = calculate_surface_area_of_slice(segmentation, voxel_size_mm)
                        surface_area_mm2.append(sa_mm2)

            del image_slice, segmentation_slice
            gc.collect()
        except Exception as e:
            print(f"⚠️ Skipping slice {i}: {e}")

    if depth_mm:
        # Apply the /1000 correction for the X-axis (distance from origin)
        # and export/plot in mm.
        distance_from_origin_mm = (np.array(cumulative_distances_raw, dtype=float) / 1000.0).tolist()

        plot_depth_results(distance_from_origin_mm, depth_mm)

        export_results(distance_from_origin_mm, surface_area_mm2, depth_mm,
                       export_directory, "rbb_depth_profile.csv")
    else:
        print("⚠️ No depth data to process.")

### ==== SURFACE AREA ANALYSIS (standalone plot) ====

def analyze_rbb_surface_area(seg_img, voxel_size_mm):
    surface_areas, slice_indices = [], []

    for i in tqdm(range(seg_img.shape[2]), desc="Surface Area Analysis"):
        slice_data = np.array(seg_img.dataobj[:, :, i])
        segmentation = slice_data == 1
        if np.any(segmentation):
            area = calculate_surface_area_of_slice(segmentation, voxel_size_mm)
            surface_areas.append(area)
            slice_indices.append(i + 1)

    if surface_areas:
        plt.figure(figsize=(10, 5))
        plt.plot(slice_indices, surface_areas, marker='o')
        plt.xlabel("Slice Index")
        plt.ylabel("Surface Area (mm²)")
        plt.title("RBB Surface Area per Slice")
        plt.grid(True)
        plt.show()
    else:
        print("⚠️ No surface area data to process.")

### ==== MAIN FUNCTION ====

def run_analysis(image_path, segmentation_path, voxel_size_um, export_directory, mode='both'):
    voxel_size_mm = voxel_size_um / 1000.0
    image_img = load_nifti(image_path) if mode in ['both', 'depth'] else None
    seg_img = load_nifti(segmentation_path)

    if seg_img is None or (mode == 'depth' and image_img is None):
        return

    if mode in ['both', 'depth']:
        analyze_rbb_depth(image_img, seg_img, voxel_size_mm, export_directory)

    if mode in ['both', 'surface']:
        analyze_rbb_surface_area(seg_img, voxel_size_mm)

### ==== EXAMPLE USAGE ====

if __name__ == "__main__":
    image_file = 
    segmentation_file = 
    voxel_size_um =
    export_dir = 

    run_analysis(
        image_path=image_file,
        segmentation_path=segmentation_file,
        voxel_size_um=voxel_size_um,
        export_directory=export_dir,
        mode='both'  # options: 'depth', 'surface', 'both'
    )
