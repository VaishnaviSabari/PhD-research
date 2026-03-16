#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import numpy as np
import tifffile as tiff
import imageio.v3 as iio
import matplotlib.pyplot as plt


# -----------------------------
# I/O
# -----------------------------
def read_image(filepath):
    if filepath.lower().endswith((".tif", ".tiff")):
        return tiff.imread(filepath)
    elif filepath.lower().endswith((".jp2", ".png", ".jpg", ".jpeg")):
        return iio.imread(filepath)
    else:
        raise ValueError(f"Unsupported file format: {filepath}")


def list_stack_files(stack_path):
    files = sorted(
        os.path.join(stack_path, f)
        for f in os.listdir(stack_path)
        if f.lower().endswith((".tif", ".tiff", ".jp2", ".png", ".jpg", ".jpeg"))
    )
    if not files:
        raise ValueError(f"No mask slices found in: {stack_path}")
    return files


def to_bool_mask(arr):
    return arr > 0


# -----------------------------
# Interactive point selection
# -----------------------------
def select_points_on_mask(mask2d, slice_idx):
    while True:
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(mask2d, cmap="gray", vmin=0, vmax=1)
        ax.set_title(
            f"Slice {slice_idx}\n"
            "Click 4 points:\n"
            "1–2 RV (inner→outer), 3–4 LV (inner→outer)"
        )
        plt.tight_layout()
        plt.show(block=False)

        pts = plt.ginput(4, timeout=0)
        plt.close(fig)

        pts = np.asarray(pts, dtype=float)
        if pts.shape == (4, 2):
            return pts

        print(f"[Slice {slice_idx}] {pts.shape[0]} points selected; retrying.")


# -----------------------------
# Anchor selection + interpolation
# -----------------------------
def pick_anchor_slices(start_slice, end_slice, num_mid=3):
    fracs = np.linspace(0, 1, num_mid + 2)[1:-1]
    mids = start_slice + np.rint(fracs * (end_slice - start_slice)).astype(int)
    mids = np.clip(mids, start_slice + 1, end_slice - 1)
    anchors = np.unique(np.array([start_slice, *mids, end_slice], dtype=int))
    return anchors


def interpolate_points(anchor_slices, anchor_points, all_slices):
    xs = np.interp(all_slices, anchor_slices, anchor_points[:, 0])
    ys = np.interp(all_slices, anchor_slices, anchor_points[:, 1])
    return np.stack([xs, ys], axis=1)


# -----------------------------
# Mask-based refinement
# -----------------------------
def refine_pair_on_mask(mask, p_in, p_out, extra_px=25, n_samples=500, min_run=3):
    p_in = np.asarray(p_in, dtype=float)
    p_out = np.asarray(p_out, dtype=float)

    v = p_out - p_in
    L = np.linalg.norm(v)
    if L < 1e-6:
        return None, None, False

    u = v / L
    ts = np.linspace(-extra_px, L + extra_px, n_samples)
    pts = p_in[None, :] + ts[:, None] * u[None, :]

    xs = np.rint(pts[:, 0]).astype(int)
    ys = np.rint(pts[:, 1]).astype(int)

    h, w = mask.shape
    inside = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    xs, ys = xs[inside], ys[inside]

    if xs.size == 0:
        return None, None, False

    vals = mask[ys, xs].astype(bool)
    white_idx = np.where(vals)[0]
    if white_idx.size == 0:
        return None, None, False

    splits = np.where(np.diff(white_idx) > 1)[0]
    runs = np.split(white_idx, splits + 1)
    best = max(runs, key=len)

    if best.size < min_run:
        return None, None, False

    i0, i1 = best[0], best[-1]
    return (
        np.array([xs[i0], ys[i0]], float),
        np.array([xs[i1], ys[i1]], float),
        True,
    )


def euclidean(p1, p2):
    return np.linalg.norm(np.asarray(p1) - np.asarray(p2))


# -----------------------------
# QC grid plotting
# -----------------------------
def plot_overlay_grid(overlays, ncols=10, figsize_per_cell=2.2):
    n = len(overlays)
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * figsize_per_cell, nrows * figsize_per_cell),
        squeeze=False
    )

    for ax in axes.ravel():
        ax.axis("off")

    for i, item in enumerate(overlays):
        r, c = divmod(i, ncols)
        ax = axes[r, c]

        ax.imshow(item["mask"], cmap="gray", vmin=0, vmax=1)

        def draw(pred, ref, color):
            pin, pout = pred
            ax.plot([pin[0], pout[0]], [pin[1], pout[1]], "--", lw=0.8, c=color)
            if ref is not None:
                rin, rout = ref
                ax.plot([rin[0], rout[0]], [rin[1], rout[1]], lw=2.0, c=color)

        draw(item["rv_pred"], item["rv_ref"], "tab:red")
        draw(item["lv_pred"], item["lv_ref"], "tab:blue")

        ax.set_title(f"{item['slice']}", fontsize=8)

    fig.tight_layout()
    plt.show()


# -----------------------------
# Main
# -----------------------------
def main(args):
    files = list_stack_files(args.mask_path)

    voxel_mm = args.voxel_um / 1000.0
    anchors = pick_anchor_slices(args.start, args.end, args.num_mid)
    print(f"Anchor slices: {anchors.tolist()}")

    rv_in_a, rv_out_a, lv_in_a, lv_out_a = [], [], [], []

    for s in anchors:
        mask = to_bool_mask(read_image(files[s]))
        pts = select_points_on_mask(mask, s)
        rv_in_a.append(pts[0]); rv_out_a.append(pts[1])
        lv_in_a.append(pts[2]); lv_out_a.append(pts[3])

    rv_in_a = np.asarray(rv_in_a)
    rv_out_a = np.asarray(rv_out_a)
    lv_in_a = np.asarray(lv_in_a)
    lv_out_a = np.asarray(lv_out_a)

    all_slices = np.arange(args.start, args.end + 1)
    rv_in = interpolate_points(anchors, rv_in_a, all_slices)
    rv_out = interpolate_points(anchors, rv_out_a, all_slices)
    lv_in = interpolate_points(anchors, lv_in_a, all_slices)
    lv_out = interpolate_points(anchors, lv_out_a, all_slices)

    overlays = []
    rv_vals, lv_vals = [], []

    for i, s in enumerate(all_slices):
        mask = to_bool_mask(read_image(files[s]))

        rv_rin, rv_rout, ok_rv = refine_pair_on_mask(
            mask, rv_in[i], rv_out[i], args.extra_px, args.n_samples, args.min_run
        )
        lv_rin, lv_rout, ok_lv = refine_pair_on_mask(
            mask, lv_in[i], lv_out[i], args.extra_px, args.n_samples, args.min_run
        )

        if ok_rv:
            rv_vals.append(euclidean(rv_rin, rv_rout) * voxel_mm)
        if ok_lv:
            lv_vals.append(euclidean(lv_rin, lv_rout) * voxel_mm)

        overlays.append({
            "slice": int(s),
            "mask": mask,
            "rv_pred": (rv_in[i], rv_out[i]),
            "rv_ref": (rv_rin, rv_rout) if ok_rv else None,
            "lv_pred": (lv_in[i], lv_out[i]),
            "lv_ref": (lv_rin, lv_rout) if ok_lv else None,
        })

    rv_vals = np.asarray(rv_vals)
    lv_vals = np.asarray(lv_vals)

    if rv_vals.size:
        print(f"RV thickness: {rv_vals.mean():.3f} ± {rv_vals.std():.3f} mm (n={rv_vals.size})")
    if lv_vals.size:
        print(f"LV thickness: {lv_vals.mean():.3f} ± {lv_vals.std():.3f} mm (n={lv_vals.size})")

    plot_overlay_grid(overlays, ncols=args.qc_cols)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mask-only LV/RV wall thickness measurement.")
    parser.add_argument("--mask_path", required=True, help="Folder containing binary mask slices")
    parser.add_argument("--start", type=int, required=True, help="Start slice index (inclusive)")
    parser.add_argument("--end", type=int, required=True, help="End slice index (inclusive)")
    parser.add_argument("--voxel_um", type=float, required=True, help="Voxel size in micrometers")
    parser.add_argument("--num_mid", type=int, default=3, help="Number of intermediate anchor slices")
    parser.add_argument("--extra_px", type=int, default=25)
    parser.add_argument("--n_samples", type=int, default=500)
    parser.add_argument("--min_run", type=int, default=3)
    parser.add_argument("--qc_cols", type=int, default=10)

    main(parser.parse_args())
