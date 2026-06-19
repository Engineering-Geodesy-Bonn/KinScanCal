"""DEM and GeoTIFF utilities for pointcloudlib."""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import rasterio
from rasterio.transform import from_origin

from pointcloudlib.pointcloud import PointCloud


@dataclass
class PlaneAlignment:
    normal: np.ndarray
    centroid: np.ndarray
    rotation: np.ndarray
    translation: np.ndarray
    z_offset: float


def _sample_points(xyz: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    if xyz.shape[0] <= max_points:
        return xyz

    rng = np.random.default_rng(seed)
    idx = rng.choice(xyz.shape[0], size=max_points, replace=False)
    return xyz[idx, :]


def fit_plane_pca(
    pc: PointCloud,
    max_points: int = 2_000_000,
    seed: int = 13,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a plane using PCA on a sampled subset.

    Returns the plane normal and centroid.
    """

    xyz = _sample_points(pc.xyz, max_points=max_points, seed=seed)
    centroid = np.mean(xyz, axis=0)
    xyz_centered = xyz - centroid
    _, _, vt = np.linalg.svd(xyz_centered, full_matrices=False)
    normal = vt[-1, :]

    # Ensure a stable orientation (positive Z)
    if normal[2] < 0:
        normal = -normal

    if verbose:
        sample_count = xyz.shape[0]
        print(
            f"[INFO] Plane fit sample: {sample_count:_} points, "
            f"centroid=({centroid[0]:.3f}, {centroid[1]:.3f}, {centroid[2]:.3f})"
        )
        print(
            f"[INFO] Plane normal: ({normal[0]:.6f}, {normal[1]:.6f}, {normal[2]:.6f})"
        )

    return normal, centroid


def rotation_matrix_from_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute rotation matrix that aligns vector a to vector b."""

    a_unit = a / np.linalg.norm(a)
    b_unit = b / np.linalg.norm(b)

    v = np.cross(a_unit, b_unit)
    c = np.dot(a_unit, b_unit)
    s = np.linalg.norm(v)

    if s == 0:
        # Parallel or anti-parallel
        if c > 0:
            return np.eye(3)
        # 180 deg rotation: pick an orthogonal axis
        axis = np.array([1.0, 0.0, 0.0])
        if abs(a_unit[0]) > 0.9:
            axis = np.array([0.0, 1.0, 0.0])
        v = np.cross(a_unit, axis)
        v = v / np.linalg.norm(v)
        return _rodrigues(v, np.pi)

    v_unit = v / s
    return _rodrigues(v_unit, np.arctan2(s, c))


def _rodrigues(k: np.ndarray, theta: float) -> np.ndarray:
    kx, ky, kz = k
    K = np.array(
        [
            [0.0, -kz, ky],
            [kz, 0.0, -kx],
            [-ky, kx, 0.0],
        ]
    )
    I = np.eye(3)
    return I + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def build_plane_alignment(
    normal: np.ndarray, centroid: np.ndarray, verbose: bool = False
) -> PlaneAlignment:
    """Build rotation + translation to align plane normal with +Z.

    The translation rotates around the plane centroid and shifts the plane to z=0.
    """

    target = np.array([0.0, 0.0, 1.0])
    rotation = rotation_matrix_from_vectors(normal, target)

    # Rotate around centroid: x' = R x + (c - R c)
    translation = centroid - rotation @ centroid

    # After rotation, plane passes through centroid at z=centroid_z.
    z_offset = centroid[2]

    if verbose:
        print(
            "[INFO] Alignment rotation matrix:\n"
            f"[{rotation[0, 0]: .6f} {rotation[0, 1]: .6f} {rotation[0, 2]: .6f}]\n"
            f"[{rotation[1, 0]: .6f} {rotation[1, 1]: .6f} {rotation[1, 2]: .6f}]\n"
            f"[{rotation[2, 0]: .6f} {rotation[2, 1]: .6f} {rotation[2, 2]: .6f}]"
        )
        print(
            f"[INFO] Alignment translation: ({translation[0]:.3f}, "
            f"{translation[1]:.3f}, {translation[2]:.3f})"
        )
        print(f"[INFO] Alignment z_offset: {z_offset:.3f}")

    return PlaneAlignment(
        normal=normal,
        centroid=centroid,
        rotation=rotation,
        translation=translation,
        z_offset=z_offset,
    )


def apply_alignment(pc: PointCloud, alignment: PlaneAlignment) -> None:
    """Apply alignment in-place (rotation + translation + z reference)."""

    xyz = (alignment.rotation @ pc.xyz.T).T + alignment.translation
    xyz[:, 2] = xyz[:, 2] - alignment.z_offset
    pc.xyz = xyz


def apply_z_rotation(pc: PointCloud, rotmat: np.ndarray) -> None:
    """Apply a rotation about Z in-place."""

    xyz = (rotmat @ pc.xyz.T).T
    pc.xyz = xyz


def _rotation_from_pca_xy(xy: np.ndarray) -> np.ndarray:
    """Compute Z-rotation to align major axis with X."""

    xy_centered = xy - np.mean(xy, axis=0)
    _, _, vt = np.linalg.svd(xy_centered, full_matrices=False)
    direction = vt[0, :]
    angle = np.arctan2(direction[1], direction[0])

    c = np.cos(-angle)
    s = np.sin(-angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _sample_xy_across_files(
    las_files: list[Path],
    alignment: PlaneAlignment,
    max_points: int,
    seed: int,
    apply_sor: bool = False,
    sor_neighbors: int = 50,
    sor_std_ratio: float = 1.0,
) -> np.ndarray:
    """Sample XY points across files after plane alignment."""

    if max_points <= 0:
        raise ValueError("max_points must be positive")

    per_file = max(1, max_points // max(1, len(las_files)))
    samples: list[np.ndarray] = []

    from pointcloudlib.utils import VerboseLevel

    for i, las_path in enumerate(las_files):
        pc = PointCloud()
        pc.read(str(las_path))
        if apply_sor:
            pc, _ = pc.pcdenoise(
                nb_neighbors=sor_neighbors,
                std_ratio=sor_std_ratio,
                verbose=VerboseLevel.SILENT
            )
        apply_alignment(pc, alignment)
        xyz = _sample_points(pc.xyz, max_points=per_file, seed=seed + i)
        samples.append(xyz[:, :2])

    return np.vstack(samples)


def compute_xy_intersection(bboxes: Iterable[np.ndarray]) -> np.ndarray:
    """Compute intersection bbox from input 3D bboxes."""

    bboxes = list(bboxes)
    min_x = max(b[0] for b in bboxes)
    max_x = min(b[1] for b in bboxes)
    min_y = max(b[2] for b in bboxes)
    max_y = min(b[3] for b in bboxes)
    min_z = min(b[4] for b in bboxes)
    max_z = max(b[5] for b in bboxes)

    if min_x >= max_x or min_y >= max_y:
        raise ValueError("No XY overlap found across point clouds")

    return np.array([min_x, max_x, min_y, max_y, min_z, max_z])


def compute_xy_union(bboxes: Iterable[np.ndarray]) -> np.ndarray:
    """Compute union bbox from input 3D bboxes."""

    bboxes = list(bboxes)
    min_x = min(b[0] for b in bboxes)
    max_x = max(b[1] for b in bboxes)
    min_y = min(b[2] for b in bboxes)
    max_y = max(b[3] for b in bboxes)
    min_z = min(b[4] for b in bboxes)
    max_z = max(b[5] for b in bboxes)

    return np.array([min_x, max_x, min_y, max_y, min_z, max_z])


def _grid_shape(xylimits: np.ndarray, dxy: float) -> tuple[int, int, int]:
    min_x, max_x, min_y, max_y = xylimits
    cols = int(np.ceil((max_x - min_x) / dxy))
    rows = int(np.ceil((max_y - min_y) / dxy))
    return rows, cols, rows * cols


def compute_dem_grid(
    xyz: np.ndarray,
    xylimits: np.ndarray,
    dxy: float,
    nodata: float,
    agg: str = "max",
    origin: str = "lower",
) -> tuple[np.ndarray, rasterio.Affine]:
    """Compute a DEM grid and affine transform.

    agg options: max, mean, median
    """

    min_x, max_x, min_y, max_y = xylimits

    x_edges = np.arange(min_x, max_x + dxy + 1e-9, dxy)
    y_edges = np.arange(min_y, max_y + dxy + 1e-9, dxy)

    x_idx = np.digitize(xyz[:, 0], x_edges) - 1
    y_idx = np.digitize(xyz[:, 1], y_edges) - 1

    x_idx = np.clip(x_idx, 0, len(x_edges) - 2)
    y_idx = np.clip(y_idx, 0, len(y_edges) - 2)

    if origin == "lower":
        rows = y_idx
    else:
        rows = (len(y_edges) - 2) - y_idx
    cols = x_idx

    nrows = len(y_edges) - 1
    ncols = len(x_edges) - 1
    grid = np.full((nrows, ncols), nodata, dtype=np.float32)

    if xyz.shape[0] == 0:
        transform = from_origin(min_x, max_y, dxy, dxy)
        return grid, transform

    valid = ~np.isnan(xyz[:, 2])
    rows = rows[valid]
    cols = cols[valid]
    z = xyz[valid, 2].astype(np.float32)

    flat = rows * ncols + cols

    if agg == "max":
        grid = np.full((nrows, ncols), -np.inf, dtype=np.float32)
        np.maximum.at(grid, (rows, cols), z)
        grid[~np.isfinite(grid)] = nodata
    elif agg == "mean":
        sums = np.bincount(flat, weights=z, minlength=nrows * ncols)
        counts = np.bincount(flat, minlength=nrows * ncols)
        valid_cells = counts > 0
        grid_flat = grid.ravel()
        grid_flat[valid_cells] = (sums[valid_cells] / counts[valid_cells])
        grid = grid_flat.reshape((nrows, ncols))
    elif agg == "median":
        order = np.argsort(flat)
        flat_sorted = flat[order]
        z_sorted = z[order]
        changes = np.flatnonzero(np.diff(flat_sorted)) + 1
        starts = np.concatenate(([0], changes))
        ends = np.concatenate((changes, [flat_sorted.size]))
        grid_flat = grid.ravel()
        for start, end in zip(starts, ends):
            cell = flat_sorted[start]
            grid_flat[cell] = np.median(z_sorted[start:end])
        grid = grid_flat.reshape((nrows, ncols))
    else:
        raise ValueError("agg must be one of: max, mean, median")

    transform = from_origin(min_x, max_y, dxy, dxy)
    return grid, transform


def write_dem_geotiff(
    path: Path,
    grid: np.ndarray,
    transform: rasterio.Affine,
    epsg: int,
    nodata: float,
) -> None:
    """Write DEM grid to GeoTIFF."""

    path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=grid.shape[0],
        width=grid.shape[1],
        count=1,
        dtype=grid.dtype,
        crs=f"EPSG:{epsg}",
        transform=transform,
        nodata=nodata,
        compress="deflate",
    ) as dst:
        dst.write(grid, 1)


def fill_nodata_grid(
    grid: np.ndarray,
    nodata: float,
    method: str = "zero_fill",
) -> np.ndarray:
    """Fill NoData cells using nearby values."""

    mask = grid == nodata
    if not np.any(mask):
        if method == "zero_fill":
            non_finite = ~np.isfinite(grid)
            if np.any(non_finite):
                filled = grid.copy()
                filled[non_finite] = 0
                return filled.astype(grid.dtype, copy=False)
        return grid

    if method == "zero_fill":
        filled = grid.copy()
        mask_zero = mask | ~np.isfinite(grid)
        filled[mask_zero] = 0
        return filled.astype(grid.dtype, copy=False)

    raise ValueError("fill method must be: zero_fill")


def _smart_fill_from_other_grids(
    grids: list[np.ndarray], nodata: float, mode: str = "avg"
) -> list[np.ndarray]:
    if len(grids) <= 1:
        return [grid.copy() for grid in grids]

    stacked = np.stack(grids)
    valid = stacked != nodata
    sum_all = np.where(valid, stacked, 0.0).sum(axis=0, dtype=np.float64)
    count_all = valid.sum(axis=0, dtype=np.int32)

    if mode not in {"avg", "min", "max", "median"}:
        raise ValueError("smart fill mode must be: avg, min, max, median")

    filled_grids: list[np.ndarray] = []
    for idx, grid in enumerate(grids):
        grid_valid = grid != nodata
        filled = grid.copy()
        fill_mask = grid == nodata

        if mode == "avg":
            sum_others = sum_all - np.where(grid_valid, grid, 0.0)
            count_others = count_all - grid_valid.astype(np.int32)
            has_others = count_others > 0
            mean_others = np.zeros_like(grid, dtype=np.float64)
            mean_others[has_others] = sum_others[has_others] / count_others[has_others]
            filled[fill_mask & has_others] = mean_others[fill_mask & has_others].astype(
                grid.dtype, copy=False
            )
            filled_grids.append(filled)
            continue

        if mode == "median":
            mask_other = valid.copy()
            mask_other[idx, :, :] = False
            other_vals = np.where(mask_other, stacked, np.nan)
            median_others = np.nanmedian(other_vals, axis=0)
            has_others = np.isfinite(median_others)
            filled[fill_mask & has_others] = median_others[
                fill_mask & has_others
            ].astype(grid.dtype, copy=False)
            filled_grids.append(filled)
            continue

        mask_other = valid.copy()
        mask_other[idx, :, :] = False

        if mode == "min":
            masked = np.where(mask_other, stacked, np.inf)
            min_others = masked.min(axis=0)
            has_others = np.isfinite(min_others)
            filled[fill_mask & has_others] = min_others[fill_mask & has_others].astype(
                grid.dtype, copy=False
            )
        else:
            masked = np.where(mask_other, stacked, -np.inf)
            max_others = masked.max(axis=0)
            has_others = np.isfinite(max_others)
            filled[fill_mask & has_others] = max_others[fill_mask & has_others].astype(
                grid.dtype, copy=False
            )

        filled_grids.append(filled)

    return filled_grids


def write_dem_preview_png(
    path: Path,
    grid: np.ndarray,
    nodata: float,
    cmap: str = "turbo",
    vmin: float | None = None,
    vmax: float | None = None,
    xylimits: np.ndarray | None = None,
    label: str | None = None,
) -> None:
    """Write a color preview PNG with a visible colorbar."""

    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)

    masked = np.ma.masked_where(grid == nodata, grid)
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color="#111111")
    if xylimits is not None:
        extent = (xylimits[0], xylimits[1], xylimits[2], xylimits[3])
        im = ax.imshow(
            masked,
            cmap=cmap_obj,
            vmin=vmin,
            vmax=vmax,
            extent=extent,
            origin="lower",
            aspect="equal",
        )
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.grid(True, color="white", alpha=0.2, linewidth=0.5)
    else:
        im = ax.imshow(masked, cmap=cmap_obj, vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.08, label="Height (m)")
    if label:
        ax.text(
            0.02,
            0.98,
            label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            color="white",
            bbox=dict(facecolor="black", alpha=0.4, edgecolor="none"),
        )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def write_bbox_preview_png(
    path: Path,
    bboxes: list[np.ndarray],
    xylimits: np.ndarray,
    title: str,
) -> None:
    """Write a PNG showing XY bounding boxes."""

    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    colors = plt.cm.tab10.colors

    for i, bbox in enumerate(bboxes):
        x0, x1, y0, y1 = bbox[:4]
        rect = Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            fill=False,
            linewidth=1.5,
            edgecolor=colors[i % len(colors)],
        )
        ax.add_patch(rect)
        ax.text(
            x0,
            y0,
            str(i + 1),
            color=colors[i % len(colors)],
            fontsize=8,
            verticalalignment="bottom",
        )

    ax.set_xlim(xylimits[0], xylimits[1])
    ax.set_ylim(xylimits[2], xylimits[3])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(title)
    ax.grid(True, color="#cccccc", alpha=0.4, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def write_side_view_png(
    path: Path,
    xyz: np.ndarray,
    nodata: float,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
    dxy: float,
    title: str,
    xlabel: str,
    ylabel: str,
    vmin: float | None = None,
    vmax: float | None = None,
    label: str | None = None,
) -> None:
    """Write a side-view PNG using a gridded max projection."""

    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)

    if xyz.size == 0:
        grid = np.full((1, 1), nodata, dtype=np.float32)
        extent = (x_limits[0], x_limits[1], y_limits[0], y_limits[1])
    else:
        x_edges = np.arange(x_limits[0], x_limits[1] + dxy + 1e-9, dxy)
        y_edges = np.arange(y_limits[0], y_limits[1] + dxy + 1e-9, dxy)

        x_idx = np.digitize(xyz[:, 0], x_edges) - 1
        y_idx = np.digitize(xyz[:, 1], y_edges) - 1

        x_idx = np.clip(x_idx, 0, len(x_edges) - 2)
        y_idx = np.clip(y_idx, 0, len(y_edges) - 2)

        nrows = len(y_edges) - 1
        ncols = len(x_edges) - 1
        grid = np.full((nrows, ncols), nodata, dtype=np.float32)
        flat = y_idx * ncols + x_idx
        valid = ~np.isnan(xyz[:, 2])
        flat = flat[valid]
        z = xyz[valid, 2].astype(np.float32)
        grid_flat = grid.ravel()
        np.maximum.at(grid_flat, flat, z)
        grid = grid_flat.reshape((nrows, ncols))
        grid[~np.isfinite(grid)] = nodata
        extent = (x_limits[0], x_limits[1], y_limits[0], y_limits[1])

    masked = np.ma.masked_where(grid == nodata, grid)
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    cmap_obj = plt.get_cmap("turbo").copy()
    cmap_obj.set_bad(color="#111111")
    im = ax.imshow(
        masked,
        cmap=cmap_obj,
        extent=extent,
        origin="lower",
        aspect="equal",
        vmin=vmin,
        vmax=vmax,
    )
    fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.08, label="Height (m)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, color="white", alpha=0.2, linewidth=0.5)
    if label:
        ax.text(
            0.02,
            0.98,
            label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            color="white",
            bbox=dict(facecolor="black", alpha=0.4, edgecolor="none"),
        )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)



def write_combined_view_png(
    path: Path,
    nadir_grid: np.ndarray,
    nodata: float,
    cmap: str,
    vmin: float | None,
    vmax: float | None,
    xylimits: np.ndarray,
    side_local: np.ndarray,
    x_range: float,
    y_range: float,
    z_range: float,
    dxy: float,
    left_gap: float = 0.018,
    label: str | None = None,
    point_count: int | None = None,
    z_min: float | None = None,
    z_max: float | None = None,
    date_str: str | None = None,
) -> None:
    """Write a combined PNG with Nadir, X-Z, and Y-Z views."""

    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    path.parent.mkdir(parents=True, exist_ok=True)

    masked_nadir = np.ma.masked_where(nadir_grid == nodata, nadir_grid)
    nadir_extent = (xylimits[0], xylimits[1], xylimits[2], xylimits[3])

    def make_projection_grid(
        x_values: np.ndarray,
        y_values: np.ndarray,
        z_values: np.ndarray,
        x_lim: tuple[float, float],
        y_lim: tuple[float, float],
    ) -> tuple[np.ndarray, tuple[float, float, float, float]]:
        if x_values.size == 0:
            grid = np.full((1, 1), nodata, dtype=np.float32)
            extent = (x_lim[0], x_lim[1], y_lim[0], y_lim[1])
            return grid, extent
        x_edges = np.arange(x_lim[0], x_lim[1] + dxy + 1e-9, dxy)
        y_edges = np.arange(y_lim[0], y_lim[1] + dxy + 1e-9, dxy)
        x_idx = np.digitize(x_values, x_edges) - 1
        y_idx = np.digitize(y_values, y_edges) - 1
        x_idx = np.clip(x_idx, 0, len(x_edges) - 2)
        y_idx = np.clip(y_idx, 0, len(y_edges) - 2)
        nrows = len(y_edges) - 1
        ncols = len(x_edges) - 1
        grid = np.full((nrows, ncols), nodata, dtype=np.float32)
        flat = y_idx * ncols + x_idx
        valid = ~np.isnan(z_values)
        flat = flat[valid]
        z = z_values[valid].astype(np.float32)
        grid_flat = grid.ravel()
        np.maximum.at(grid_flat, flat, z)
        grid = grid_flat.reshape((nrows, ncols))
        grid[~np.isfinite(grid)] = nodata
        extent = (x_lim[0], x_lim[1], y_lim[0], y_lim[1])
        return grid, extent

    x_values = side_local[:, 0]
    y_values = side_local[:, 1]
    z_values = side_local[:, 2]

    grid_xz, extent_xz = make_projection_grid(
        x_values,
        z_values,
        z_values,
        (0.0, x_range),
        (0.0, z_range),
    )
    masked_xz = np.ma.masked_where(grid_xz == nodata, grid_xz)

    grid_yz, extent_yz = make_projection_grid(
        y_values,
        z_values,
        z_values,
        (0.0, y_range),
        (0.0, z_range),
    )
    masked_yz = np.ma.masked_where(grid_yz == nodata, grid_yz)

    fig = plt.figure(figsize=(16, 10), dpi=300)

    left_x = 0.055
    left_w = 0.56
    right_x = 0.67
    right_w = 0.25
    plot_bottom = 0.18
    plot_h = 0.70
    left_half_h = (plot_h - left_gap) / 2.0
    cbar_bottom = 0.07
    cbar_h = 0.035

    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color="#111111")
    cmap_obj_side = plt.get_cmap("turbo").copy()
    cmap_obj_side.set_bad(color="#111111")

    ax_nadir = fig.add_axes((left_x, plot_bottom + left_half_h + left_gap, left_w, left_half_h))
    im_nadir = ax_nadir.imshow(
        masked_nadir,
        cmap=cmap_obj,
        vmin=vmin,
        vmax=vmax,
        extent=nadir_extent,
        origin="lower",
        aspect="auto",
    )
    ax_nadir.set_ylabel("Y (m)")
    ax_nadir.tick_params(axis="x", which="both", labelbottom=False)
    ax_nadir.grid(True, color="white", alpha=0.2, linewidth=0.5)

    ax_xz = fig.add_axes((left_x, plot_bottom, left_w, left_half_h))
    im_xz = ax_xz.imshow(
        masked_xz,
        cmap=cmap_obj_side,
        extent=extent_xz,
        origin="lower",
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
    )
    ax_xz.set_xlabel("X (m)")
    ax_xz.set_ylabel("Z (m)")
    ax_xz.grid(True, color="white", alpha=0.2, linewidth=0.5)

    # Y-Z plot: half height, aligned to lower left plot
    ax_yz = fig.add_axes((right_x, plot_bottom, right_w, left_half_h))
    im_yz = ax_yz.imshow(
        masked_yz,
        cmap=cmap_obj_side,
        extent=extent_yz,
        origin="lower",
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
    )
    ax_yz.set_xlabel("Y (m)")
    ax_yz.set_ylabel("Z (m)")
    ax_yz.yaxis.set_label_position("right")
    ax_yz.yaxis.tick_right()
    ax_yz.tick_params(
        axis="y",
        which="both",
        left=False,
        labelleft=False,
        right=True,
        labelright=True,
        pad=2,
    )
    ax_yz.grid(True, color="white", alpha=0.2, linewidth=0.5)

    cbar_ax = fig.add_axes((left_x, cbar_bottom, right_x + right_w - left_x, cbar_h))
    fig.colorbar(
        im_yz,
        cax=cbar_ax,
        orientation="horizontal",
        label="Height (m)",
    )

    ax_nadir.set_xlim(nadir_extent[0], nadir_extent[1])
    ax_nadir.set_ylim(nadir_extent[2], nadir_extent[3])
    ax_xz.set_xlim(extent_xz[0], extent_xz[1])
    ax_xz.set_ylim(extent_xz[2], extent_xz[3])
    ax_yz.set_xlim(extent_yz[0], extent_yz[1])
    ax_yz.set_ylim(extent_yz[2], extent_yz[3])

    # Add info box in the empty upper-right area
    if point_count is not None or z_min is not None or z_max is not None or date_str is not None:
        info_box_x = right_x
        info_box_y = plot_bottom + left_half_h + left_gap
        info_box_w = right_w
        info_box_h = left_half_h - left_gap

        # Main info box with square corners (no rounded edges, no shadow)
        info_box = FancyBboxPatch(
            (info_box_x, info_box_y),
            info_box_w,
            info_box_h,
            boxstyle="square,pad=0.01",
            transform=fig.transFigure,
            facecolor="white",
            edgecolor="white",
            linewidth=1.5,
            alpha=1.0,
            zorder=51,
        )
        fig.patches.append(info_box)

        # Format info as a left-aligned table
        table_rows = []
        if date_str is not None:
            table_rows.append(("Date", date_str))
        if point_count is not None:
            table_rows.append(("Points", f"{point_count:,}"))
        if z_max is not None:
            table_rows.append(("Z max", f"{z_max:.3f} m"))
        table_rows.append(("Resolution", f"{dxy*1000:.2f} mm"))

        # Build formatted table with two columns and light gray borders
        col1_width = max(len(row[0]) for row in table_rows)
        table_lines = []
        for attr, value in table_rows:
            table_lines.append(f"{attr:<{col1_width}}  {value}")
        
        info_text = "\n".join(table_lines)

        # Draw light gray table border lines
        from matplotlib.lines import Line2D
        border_color = (0.85, 0.85, 0.85)
        
        # Calculate table dimensions
        text_x_left = info_box_x + 0.025
        text_y_top = info_box_y + info_box_h - 0.03
        table_height = info_box_h - 0.06
        table_width = info_box_w - 0.05
        
        # Draw vertical divider line (between columns)
        col_divider_x = info_box_x + info_box_w / 2
        line_divider = Line2D(
            [col_divider_x, col_divider_x],
            [info_box_y + 0.015, info_box_y + info_box_h - 0.015],
            transform=fig.transFigure,
            color=border_color,
            linewidth=1,
            zorder=51,
        )
        fig.lines.append(line_divider)
        
        # Draw horizontal border lines (top and bottom)
        line_top = Line2D(
            [info_box_x + 0.01, info_box_x + info_box_w - 0.01],
            [info_box_y + info_box_h - 0.015, info_box_y + info_box_h - 0.015],
            transform=fig.transFigure,
            color=border_color,
            linewidth=1,
            zorder=51,
        )
        fig.lines.append(line_top)
        
        line_bottom = Line2D(
            [info_box_x + 0.01, info_box_x + info_box_w - 0.01],
            [info_box_y + 0.015, info_box_y + 0.015],
            transform=fig.transFigure,
            color=border_color,
            linewidth=1,
            zorder=51,
        )
        fig.lines.append(line_bottom)

        # Place text left-aligned in the box
        text_x = info_box_x + 0.025
        text_y = info_box_y + info_box_h / 2
        fig.text(
            text_x,
            text_y,
            info_text,
            ha="left",
            va="center",
            fontsize=11,
            color="black",
            transform=fig.transFigure,
            zorder=52,
            family="monospace",
            linespacing=1.35,
        )

    fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def _pick_first_existing(paths: list[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError(
        "No matching file found. Checked: " + ", ".join(str(path) for path in paths)
    )


def _load_kinematic_calibration(path: Path) -> np.ndarray:
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data


def plot_dem_and_kinematic_calibration(
    plot_dir: str | Path,
    output_path: str | Path | None = None,
    dem_dxy: float = 0.01,
    dem_agg: str = "max",
    dem_nodata: float = -9999.0,
    dem_max_points: int | None = 3_000_000,
) -> Path:
    """Plot DEM plus kinematic calibration time series in one vertical figure."""

    import matplotlib.pyplot as plt

    plot_dir = Path(plot_dir)
    pointcloud_dir = plot_dir / "04_pointcloud"
    calibration_dir = plot_dir / "03_calibration"

    if not pointcloud_dir.exists():
        raise FileNotFoundError(f"Point cloud directory not found: {pointcloud_dir}")
    if not calibration_dir.exists():
        raise FileNotFoundError(f"Calibration directory not found: {calibration_dir}")

    pointcloud_candidates = sorted(
        [*pointcloud_dir.glob("*.las"), *pointcloud_dir.glob("*.laz")]
    )
    if not pointcloud_candidates:
        raise FileNotFoundError(f"No LAS/LAZ files found in {pointcloud_dir}")

    pointcloud_path = next(
        (path for path in pointcloud_candidates if path.name == "pc_kinematic_calibration.las"),
        pointcloud_candidates[0],
    )
    pc = PointCloud()
    pc.read(str(pointcloud_path))
    xyz = pc.xyz
    if xyz.size == 0:
        raise ValueError(f"Point cloud is empty: {pointcloud_path}")
    if dem_max_points is not None and dem_max_points > 0 and xyz.shape[0] > dem_max_points:
        xyz = _sample_points(xyz, max_points=dem_max_points, seed=13)

    xylimits = np.array(
        [
            float(np.min(xyz[:, 0])),
            float(np.max(xyz[:, 0])),
            float(np.min(xyz[:, 1])),
            float(np.max(xyz[:, 1])),
        ]
    )
    dem_grid, _ = compute_dem_grid(
        xyz=xyz,
        xylimits=xylimits,
        dxy=dem_dxy,
        nodata=dem_nodata,
        agg=dem_agg,
    )

    left_cal_path = _pick_first_existing(
        [calibration_dir / "xint_l.txt", calibration_dir / "x_l.txt"]
    )
    right_cal_path = _pick_first_existing(
        [calibration_dir / "xint_r.txt", calibration_dir / "x_r.txt"]
    )

    left_cal = _load_kinematic_calibration(left_cal_path)
    right_cal = _load_kinematic_calibration(right_cal_path)

    def _time_axis(data: np.ndarray) -> np.ndarray:
        return data[:, 0] - data[0, 0]

    def _translation_columns(data: np.ndarray) -> np.ndarray:
        if data.shape[1] < 7:
            raise ValueError(f"Unexpected calibration file shape: {data.shape}")
        return data[:, 4:7]

    def _rotation_columns(data: np.ndarray) -> np.ndarray:
        if data.shape[1] < 7:
            raise ValueError(f"Unexpected calibration file shape: {data.shape}")
        return data[:, 1:4]

    fig, axes = plt.subplots(
        5,
        1,
        figsize=(16, 22),
        gridspec_kw={"height_ratios": [2.5, 1, 1, 1, 1], "hspace": 0.35},
    )

    masked_grid = np.ma.masked_where(dem_grid == dem_nodata, dem_grid)
    image = axes[0].imshow(
        masked_grid,
        origin="lower",
        extent=(xylimits[0], xylimits[1], xylimits[2], xylimits[3]),
        aspect="equal",
        cmap="turbo",
    )
    axes[0].set_title(f"DEM from {pointcloud_path.name}")
    axes[0].set_ylabel("Y (m)")
    axes[0].grid(True, linestyle=":", linewidth=0.5, alpha=0.25)
    fig.colorbar(image, ax=axes[0], orientation="horizontal", pad=0.08, label="Height (m)")

    time_right = _time_axis(right_cal)
    time_left = _time_axis(left_cal)

    color_map = [("x", "tab:red"), ("y", "tab:green"), ("z", "tab:blue")]

    subplot_specs = [
        (axes[1], _translation_columns(right_cal), time_right, "Scanner right translation (x, y, z)", "Translation (m)"),
        (axes[2], _translation_columns(left_cal), time_left, "Scanner left translation (x, y, z)", "Translation (m)"),
        (axes[3], _rotation_columns(right_cal), time_right, "Scanner right rotation (rx, ry, rz)", "Rotation (rad)"),
        (axes[4], _rotation_columns(left_cal), time_left, "Scanner left rotation (rx, ry, rz)", "Rotation (rad)"),
    ]

    for ax, values, times, title, ylabel in subplot_specs:
        for idx, (label, color) in enumerate(color_map):
            ax.plot(times, values[:, idx], color=color, linewidth=1.6, label=label)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
        ax.legend(loc="upper right", ncol=3, fontsize=9, frameon=False)

    axes[4].set_xlabel("Time (s)")
    for ax in axes[1:]:
        ax.set_xlim(left=0.0)

    if output_path is None:
        output_path = plot_dir / "05_dem_kinematic_calibration.png"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return output_path
    



def _build_preview_side_local(
    las_path: Path,
    alignment,
    z_rotation: np.ndarray,
    apply_sor: bool,
    sor_neighbors: int,
    sor_std_ratio: float,
    pca_rotate_xy: bool,
    z_ref_mode: str,
    global_min_z: float | None,
    xy_limits: np.ndarray,
    z_clip_min: float | None,
    bbox_union_pca: np.ndarray,
    side_z_min: float,
) -> np.ndarray:
    pc = PointCloud()
    pc.read(str(las_path))
    if apply_sor:
        from pointcloudlib.utils import VerboseLevel

        pc, _ = pc.pcdenoise(
            nb_neighbors=sor_neighbors,
            std_ratio=sor_std_ratio,
            verbose=VerboseLevel.SILENT,
        )

    apply_alignment(pc, alignment)
    if pca_rotate_xy:
        apply_z_rotation(pc, z_rotation)
    if z_ref_mode == "global-min" and global_min_z is not None:
        pc.xyz[:, 2] = pc.xyz[:, 2] - global_min_z

    bbox = np.array(
        [
            xy_limits[0],
            xy_limits[1],
            xy_limits[2],
            xy_limits[3],
            -np.inf,
            np.inf,
        ]
    )
    pc_cut, _ = pc.select_by_bbox(bbox)
    if z_clip_min is not None and pc_cut.count > 0:
        keep = np.where(pc_cut.xyz[:, 2] >= z_clip_min)[0]
        pc_cut = pc_cut.select_by_index(keep)

    side_local = pc_cut.xyz.copy()
    if side_local.size > 0:
        side_local[:, 0] -= bbox_union_pca[0]
        side_local[:, 1] -= bbox_union_pca[2]
        side_local[:, 2] -= side_z_min
    return side_local


def _discover_pointclouds(plot_root: Path) -> list[Path]:
    las_files: list[Path] = []
    for root, _, files in os.walk(plot_root):
        for name in files:
            if name.lower().endswith((".las", ".laz")):
                las_files.append(Path(root) / name)
    return sorted(las_files)


def _safe_output_name(plot_root: Path, las_path: Path) -> str:
    rel = las_path.relative_to(plot_root)
    parts = rel.with_suffix("").parts
    return "__".join(parts) + ".tif"


def _extract_date_label(path: Path) -> str:
    text = path.as_posix()
    match = re.search(r"(\d{4})[-_/]?([01]\d)[-_/]?([0-3]\d)", text)
    if match:
        year, month, day = match.groups()
        return f"{year[2:]}-{month}-{day}"

    match = re.search(r"(?<!\d)(\d{2})([01]\d)([0-3]\d)(?!\d)", text)
    if match:
        year, month, day = match.groups()
        return f"{year}-{month}-{day}"

    return "??-??-??"


def process_plot_dem(
    plot_root: str | Path,
    output_dir_name: str = "dem",
    dxy: float = 0.002,
    max_plane_points: int = 2_000_000,
    epsg: int = 32632,
    nodata: float = -9999.0,
    seed: int = 13,
    agg: str = "max",
    max_grid_cells: int | None = None,
    dem_max_points: int | None = None,
    fill_nodata: bool = True,
    fill_method: str = "zero_fill",
    write_preview: bool = True,
    write_geotiff: bool = False,
    preview_cmap: str = "turbo",
    preview_local_frame: bool = True,
    pca_rotate_xy: bool = True,
    pca_rotate_max_points: int = 5_000_000,
    bbox_mode: str = "union",
    z_ref_mode: str = "global-min",
    z_clip_min: float | None = 0.0,
    apply_sor: bool = False,
    sor_neighbors: int = 50,
    sor_std_ratio: float = 1.0,
    verbose: bool = True,
) -> None:
    """Process a plot folder and export DEM GeoTIFFs for each LAS/LAZ."""

    plot_root = Path(plot_root)
    las_files = _discover_pointclouds(plot_root)
    if not las_files:
        raise FileNotFoundError(f"No LAS/LAZ files found in {plot_root}")

    print(f"[INFO] Found {len(las_files)} point clouds")

    # Fit plane on first date and build alignment
    from pointcloudlib.utils import VerboseLevel
    pc0 = PointCloud()
    pc0.read(str(las_files[0]))
    if apply_sor:
        pc0, _ = pc0.pcdenoise(
            nb_neighbors=sor_neighbors,
            std_ratio=sor_std_ratio,
            verbose=VerboseLevel.HEADER if verbose else VerboseLevel.SILENT
        )
    normal, centroid = fit_plane_pca(
        pc0, max_points=max_plane_points, seed=seed, verbose=verbose
    )
    alignment = build_plane_alignment(normal, centroid, verbose=verbose)

    z_rotation = np.eye(3)
    if pca_rotate_xy:
        xy_samples = _sample_xy_across_files(
            las_files,
            alignment,
            max_points=pca_rotate_max_points,
            seed=seed,
            apply_sor=apply_sor,
            sor_neighbors=sor_neighbors,
            sor_std_ratio=sor_std_ratio,
        )
        z_rotation = _rotation_from_pca_xy(xy_samples)
        if verbose:
            print(
                "[INFO] Z rotation matrix (PCA XY):\n"
                f"[{z_rotation[0, 0]: .6f} {z_rotation[0, 1]: .6f} {z_rotation[0, 2]: .6f}]\n"
                f"[{z_rotation[1, 0]: .6f} {z_rotation[1, 1]: .6f} {z_rotation[1, 2]: .6f}]\n"
                f"[{z_rotation[2, 0]: .6f} {z_rotation[2, 1]: .6f} {z_rotation[2, 2]: .6f}]"
            )

    if verbose:
        z_axis = np.array([0.0, 0.0, 1.0])
        angle = np.arccos(
            np.clip(
                np.dot(normal / np.linalg.norm(normal), z_axis), -1.0, 1.0
            )
        )
        print(f"[INFO] Plane tilt angle to +Z: {np.degrees(angle):.3f} deg")

    # First pass: compute bbox in aligned frame (and PCA-rotated frame)
    aligned_bboxes_no_pca: list[np.ndarray] = []
    aligned_bboxes_pca: list[np.ndarray] = []
    t0 = time.perf_counter()
    global_min: float | None = None
    global_max: float | None = None
    preview_paths: list[Path] = []
    global_min_z: float | None = None

    for i, las_path in enumerate(las_files, start=1):
        t_file = time.perf_counter()
        pc = PointCloud()
        pc.read(str(las_path))
        if apply_sor:
            pc, _ = pc.pcdenoise(
                nb_neighbors=sor_neighbors,
                std_ratio=sor_std_ratio,
                verbose=VerboseLevel.HEADER if verbose else VerboseLevel.SILENT
            )
        apply_alignment(pc, alignment)
        aligned_bboxes_no_pca.append(pc.bbox)
        if pca_rotate_xy:
            apply_z_rotation(pc, z_rotation)
        aligned_bboxes_pca.append(pc.bbox)
        if z_ref_mode == "global-min":
            z_min = float(np.min(pc.xyz[:, 2]))
            global_min_z = z_min if global_min_z is None else min(global_min_z, z_min)
        dt = time.perf_counter() - t_file
        print(f"[INFO] Pass1 {i}/{len(las_files)}: {las_path.name} ({dt:.1f}s)")
        if verbose:
            print(
                f"[DEBUG] Pass1 bbox: "
                f"x=({pc.bbox[0]:.3f}, {pc.bbox[1]:.3f}), "
                f"y=({pc.bbox[2]:.3f}, {pc.bbox[3]:.3f}), "
                f"z=({pc.bbox[4]:.3f}, {pc.bbox[5]:.3f})"
            )

    aligned_bboxes = (
        aligned_bboxes_pca if pca_rotate_xy else aligned_bboxes_no_pca
    )

    if bbox_mode == "intersection":
        bbox_result = compute_xy_intersection(aligned_bboxes)
    elif bbox_mode == "union":
        bbox_result = compute_xy_union(aligned_bboxes)
    else:
        raise ValueError("bbox_mode must be: intersection or union")

    xy_limits = bbox_result[:4]

    if verbose:
        print(
            f"[INFO] XY {bbox_mode}: x=({xy_limits[0]:.3f}, {xy_limits[1]:.3f}), "
            f"y=({xy_limits[2]:.3f}, {xy_limits[3]:.3f})"
        )
        if z_ref_mode == "global-min" and global_min_z is not None:
            print(
                f"[INFO] Global min Z after alignment: {global_min_z:.3f}"
            )

    if preview_local_frame:
        preview_limits = np.array(
            [
                0.0,
                xy_limits[1] - xy_limits[0],
                0.0,
                xy_limits[3] - xy_limits[2],
            ]
        )
    else:
        preview_limits = xy_limits.copy()

    if verbose:
        print(
            f"[INFO] Preview limits: x=({preview_limits[0]:.3f}, {preview_limits[1]:.3f}), "
            f"y=({preview_limits[2]:.3f}, {preview_limits[3]:.3f})"
        )

    rows, cols, cells = _grid_shape(xy_limits, dxy)
    if max_grid_cells is not None and cells > max_grid_cells:
        scale = np.sqrt(cells / max_grid_cells)
        dxy = dxy * scale
        rows, cols, cells = _grid_shape(xy_limits, dxy)
        print(
            f"[WARNING] Grid too large, increasing dxy to {dxy:.4f} "
            f"(cells={cells:_})"
        )

    if verbose:
        print(
            f"[INFO] DEM grid: rows={rows:_}, cols={cols:_}, cells={cells:_}, dxy={dxy:.4f}"
        )

    dt0 = time.perf_counter() - t0
    print(f"[INFO] Pass1 complete in {dt0:.1f}s")

    # Second pass: crop, build DEM, write GeoTIFF
    out_dir = plot_root / output_dir_name
    preview_root = out_dir / "png"
    preview_nadir_dir = preview_root / "nadir"
    preview_bboxes_dir = preview_root / "bboxes"
    preview_side_xz_dir = preview_root / "side_xz"
    preview_side_yz_dir = preview_root / "side_yz"
    preview_combined_dir = preview_root / "combined"
    smart_methods = {"smart_avg", "smart_min", "smart_max", "median_fill"}
    smart_mode_map = {
        "smart_avg": "avg",
        "smart_min": "min",
        "smart_max": "max",
        "median_fill": "median",
    }
    if write_preview:
        bbox_union_no_pca = compute_xy_union(aligned_bboxes_no_pca)
        bbox_union_pca = compute_xy_union(aligned_bboxes_pca)
        write_bbox_preview_png(
            preview_bboxes_dir / "bboxes_alignment.png",
            aligned_bboxes_no_pca,
            bbox_union_no_pca[:4],
            "Bboxes after alignment",
        )
        write_bbox_preview_png(
            preview_bboxes_dir / "bboxes_pca.png",
            aligned_bboxes_pca,
            bbox_union_pca[:4],
            "Bboxes after alignment + PCA rotation",
        )
    t1 = time.perf_counter()
    grids: list[np.ndarray] = []
    transforms: list[rasterio.Affine] = []
    out_paths: list[Path] = []
    out_paths_all: list[Path] = []
    before_fill_counts: list[int] = []
    filled_grids: list[np.ndarray] = []
    side_z_min: float | None = None
    side_z_max: float | None = None
    for i, las_path in enumerate(las_files, start=1):
        t_file = time.perf_counter()
        pc = PointCloud()
        pc.read(str(las_path))
        if apply_sor:
            pc, _ = pc.pcdenoise(
                nb_neighbors=sor_neighbors,
                std_ratio=sor_std_ratio,
                verbose=VerboseLevel.HEADER if verbose else VerboseLevel.SILENT
            )
        apply_alignment(pc, alignment)
        if pca_rotate_xy:
            apply_z_rotation(pc, z_rotation)
        if z_ref_mode == "global-min" and global_min_z is not None:
            pc.xyz[:, 2] = pc.xyz[:, 2] - global_min_z

        bbox = np.array(
            [
                xy_limits[0],
                xy_limits[1],
                xy_limits[2],
                xy_limits[3],
                -np.inf,
                np.inf,
            ]
        )
        pc_cut, _ = pc.select_by_bbox(bbox)

        if verbose:
            print(f"[DEBUG] pass2 point cloud size BEFORE bbox cut: {pc.count}")
            z_min_before = float(np.nanmin(pc.xyz[:, 2])) if pc.count > 0 else float('nan')
            z_max_before = float(np.nanmax(pc.xyz[:, 2])) if pc.count > 0 else float('nan')
            print(f"[DEBUG] pass2 Z-range BEFORE bbox cut: {z_min_before:.3f} .. {z_max_before:.3f}")

            print(f"[DEBUG] pass2 bbox used for cut: x=({bbox[0]:.3f}, {bbox[1]:.3f}), y=({bbox[2]:.3f}, {bbox[3]:.3f}), z=({bbox[4]:.3f}, {bbox[5]:.3f})")
            
            print(f"[DEBUG] pass2 point cloud size AFTER bbox cut: {pc_cut.count}")
            z_min_after = float(np.nanmin(pc_cut.xyz[:, 2])) if pc_cut.count > 0 else float('nan')
            z_max_after = float(np.nanmax(pc_cut.xyz[:, 2])) if pc_cut.count > 0 else float('nan')
            print(f"[DEBUG] pass2 Z-range AFTER bbox cut: {z_min_after:.3f} .. {z_max_after:.3f}")

        if z_clip_min is not None and pc_cut.count > 0:
            keep = np.where(pc_cut.xyz[:, 2] >= z_clip_min)[0]
            pc_cut = pc_cut.select_by_index(keep)

        if pc_cut.count > 0 and write_preview:
            z_min = float(np.nanmin(pc_cut.xyz[:, 2]))
            z_max = float(np.nanmax(pc_cut.xyz[:, 2]))
            global_min = z_min if global_min is None else min(global_min, z_min)
            global_max = z_max if global_max is None else max(global_max, z_max)
            side_z_min = z_min if side_z_min is None else min(side_z_min, z_min)
            side_z_max = z_max if side_z_max is None else max(side_z_max, z_max)

        if verbose:
            print(
                f"[DEBUG] Pass2 cut points: {pc_cut.count:_} "
                f"(before cut: {pc.count:_})"
            )
            if pc_cut.count > 0:
                z_min = np.nanmin(pc_cut.xyz[:, 2])
                z_max = np.nanmax(pc_cut.xyz[:, 2])
                print(f"[DEBUG] Pass2 z-range: {z_min:.3f} .. {z_max:.3f}")

        xyz = pc_cut.xyz
        if dem_max_points is not None and dem_max_points > 0:
            xyz = _sample_points(xyz, max_points=dem_max_points, seed=seed)

        grid, transform = compute_dem_grid(
            xyz,
            xy_limits,
            dxy=dxy,
            nodata=nodata,
            agg=agg,
        )
        out_name = _safe_output_name(plot_root, las_path)
        out_path = out_dir / out_name
        out_paths_all.append(out_path)

        if fill_method in smart_methods:
            grids.append(grid)
            transforms.append(transform)
            out_paths.append(out_path)
            before_fill_counts.append(int(np.sum(grid == nodata)))
            continue

        if fill_nodata:
            if verbose:
                before = int(np.sum(grid == nodata))
            if fill_method == "zero_fill":
                grid = fill_nodata_grid(grid, nodata=nodata, method=fill_method)
            elif fill_method not in smart_methods:
                raise ValueError(
                    "fill_method must be: smart_avg, smart_min, smart_max, "
                    "median_fill, or zero_fill"
                )
            if verbose:
                after = int(np.sum(grid == nodata))
                print(
                    f"[DEBUG] Fill NoData: before={before:_}, after={after:_}"
                )

        if verbose:
            finite = np.isfinite(grid) & (grid != nodata)
            finite_count = int(np.sum(finite))
            if finite_count > 0:
                g_min = float(np.min(grid[finite]))
                g_max = float(np.max(grid[finite]))
                g_mean = float(np.mean(grid[finite]))
                print(
                    f"[DEBUG] DEM stats: count={finite_count:_}, "
                    f"min={g_min:.3f}, max={g_max:.3f}, mean={g_mean:.3f}"
                )
            else:
                print("[WARNING] DEM has no finite values; all NoData")

        if write_geotiff:
            write_dem_geotiff(out_path, grid, transform, epsg=epsg, nodata=nodata)

        if write_preview:
            preview_paths.append(out_path)
        dt = time.perf_counter() - t_file
        print(
            f"[INFO] Pass2 {i}/{len(las_files)}: {las_path.name} "
            f"-> {out_path.name} ({dt:.1f}s)"
        )

    if fill_method in smart_methods and grids:
        filled_grids = grids
        if fill_nodata:
            smart_mode = smart_mode_map[fill_method]
            filled_grids = _smart_fill_from_other_grids(
                grids, nodata=nodata, mode=smart_mode
            )

        for i, (grid, transform, out_path, las_path) in enumerate(
            zip(filled_grids, transforms, out_paths, las_files), start=1
        ):
            t_file = time.perf_counter()
            if fill_nodata:
                before = before_fill_counts[i - 1]
                if verbose:
                    after = int(np.sum(grid == nodata))
                    print(
                        f"[DEBUG] Fill NoData: before={before:_}, after={after:_}"
                    )

            if verbose:
                finite = np.isfinite(grid) & (grid != nodata)
                finite_count = int(np.sum(finite))
                if finite_count > 0:
                    g_min = float(np.min(grid[finite]))
                    g_max = float(np.max(grid[finite]))
                    g_mean = float(np.mean(grid[finite]))
                    print(
                        f"[DEBUG] DEM stats: count={finite_count:_}, "
                        f"min={g_min:.3f}, max={g_max:.3f}, mean={g_mean:.3f}"
                    )
                else:
                    print("[WARNING] DEM has no finite values; all NoData")

            if write_geotiff:
                write_dem_geotiff(
                    out_path, grid, transform, epsg=epsg, nodata=nodata
                )

            if write_preview:
                preview_paths.append(out_path)
            dt = time.perf_counter() - t_file
            print(
                f"[INFO] Pass2 {i}/{len(las_files)}: {las_path.name} "
                f"-> {out_path.name} ({dt:.1f}s)"
            )

    dt1 = time.perf_counter() - t1
    print(f"[INFO] Pass2 complete in {dt1:.1f}s")

    if write_preview and preview_paths:
        if global_min is None or global_max is None:
            print("[WARNING] Preview skipped: no finite values found")
            return
        x_range = bbox_union_pca[1] - bbox_union_pca[0]
        y_range = bbox_union_pca[3] - bbox_union_pca[2]
        if side_z_min is None or side_z_max is None:
            print("[WARNING] Side-view previews skipped: no finite Z values found")
            return
        global_range = max(global_max - global_min, 0.0)
        vmax_headroom = global_max + (0.1 * global_range)
        z_max_plot = side_z_max + (0.1 * global_max)
        z_range = z_max_plot - side_z_min
        for idx, (out_path, las_path) in enumerate(zip(preview_paths, las_files)):
            label = _extract_date_label(las_path)
            if fill_method in smart_methods:
                grid = filled_grids[idx]
                nodata_val = nodata
            elif write_geotiff:
                with rasterio.open(out_path) as src:
                    grid = src.read(1)
                    nodata_val = src.nodata if src.nodata is not None else nodata
            else:
                pc = PointCloud()
                pc.read(str(las_path))
                if apply_sor:
                    from pointcloudlib.utils import VerboseLevel
                    pc, _ = pc.pcdenoise(
                        nb_neighbors=sor_neighbors,
                        std_ratio=sor_std_ratio,
                        verbose=VerboseLevel.SILENT
                    )
                apply_alignment(pc, alignment)
                if pca_rotate_xy:
                    apply_z_rotation(pc, z_rotation)
                if z_ref_mode == "global-min" and global_min_z is not None:
                    pc.xyz[:, 2] = pc.xyz[:, 2] - global_min_z
                bbox = np.array(
                    [
                        xy_limits[0],
                        xy_limits[1],
                        xy_limits[2],
                        xy_limits[3],
                        -np.inf,
                        np.inf,
                    ]
                )
                pc_cut, _ = pc.select_by_bbox(bbox)
                if z_clip_min is not None and pc_cut.count > 0:
                    keep = np.where(pc_cut.xyz[:, 2] >= z_clip_min)[0]
                    pc_cut = pc_cut.select_by_index(keep)
                xyz = pc_cut.xyz
                if dem_max_points is not None and dem_max_points > 0:
                    xyz = _sample_points(xyz, max_points=dem_max_points, seed=seed)
                grid, _ = compute_dem_grid(
                    xyz,
                    xy_limits,
                    dxy=dxy,
                    nodata=nodata,
                    agg=agg,
                )
                if fill_nodata:
                    if fill_method == "zero_fill":
                        grid = fill_nodata_grid(grid, nodata=nodata, method=fill_method)
                nodata_val = nodata

            if fill_method in smart_methods or write_geotiff:
                side_local = _build_preview_side_local(
                    las_path=las_path,
                    alignment=alignment,
                    z_rotation=z_rotation,
                    apply_sor=apply_sor,
                    sor_neighbors=sor_neighbors,
                    sor_std_ratio=sor_std_ratio,
                    pca_rotate_xy=pca_rotate_xy,
                    z_ref_mode=z_ref_mode,
                    global_min_z=global_min_z,
                    xy_limits=xy_limits,
                    z_clip_min=z_clip_min,
                    bbox_union_pca=bbox_union_pca,
                    side_z_min=side_z_min,
                )
            else:
                side_local = pc_cut.xyz.copy()
                if side_local.size > 0:
                    side_local[:, 0] -= bbox_union_pca[0]
                    side_local[:, 1] -= bbox_union_pca[2]
                    side_local[:, 2] -= side_z_min

            preview_path = preview_nadir_dir / out_path.with_suffix(".png").name
            write_dem_preview_png(
                preview_path,
                grid,
                nodata=nodata_val,
                cmap=preview_cmap,
                vmin=global_min,
                vmax=vmax_headroom,
                xylimits=preview_limits,
                label=label,
            )

            side_xz_path = preview_side_xz_dir / f"{out_path.stem}_side_xz.png"
            side_yz_path = preview_side_yz_dir / f"{out_path.stem}_side_yz.png"
            write_side_view_png(
                side_xz_path,
                np.column_stack((side_local[:, 0], side_local[:, 2], side_local[:, 2])),
                nodata=nodata,
                x_limits=(0.0, x_range),
                y_limits=(0.0, z_range),
                dxy=dxy,
                title="Side view X-Z",
                xlabel="X (m)",
                ylabel="Z (m)",
                vmin=global_min,
                vmax=vmax_headroom,
                label=label,
            )

            write_side_view_png(
                side_yz_path,
                np.column_stack((side_local[:, 1], side_local[:, 2], side_local[:, 2])),
                nodata=nodata,
                x_limits=(0.0, y_range),
                y_limits=(0.0, z_range),
                dxy=dxy,
                title="Side view Y-Z",
                xlabel="Y (m)",
                ylabel="Z (m)",
                vmin=global_min,
                vmax=vmax_headroom,
                label=label,
            )

            combined_path = preview_combined_dir / f"{out_path.stem}_combined.png"
            write_combined_view_png(
                combined_path,
                nadir_grid=grid,
                nodata=nodata,
                cmap=preview_cmap,
                vmin=global_min,
                vmax=vmax_headroom,
                xylimits=preview_limits,
                side_local=side_local,
                x_range=x_range,
                y_range=y_range,
                z_range=z_range,
                dxy=dxy,
                label=label,
                point_count=len(side_local) if side_local.size > 0 else 0,
                z_min=float(np.min(side_local[:, 2])) if side_local.size > 0 else None,
                z_max=float(np.max(side_local[:, 2])) if side_local.size > 0 else None,
                date_str=label,
            )

