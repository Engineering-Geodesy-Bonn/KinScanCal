#!/usr/bin/env python3
"""Create a combined DEM and kinematic calibration overview plot.

The script reads a plot directory with the structure:

    PlotXX/YYMMDD/04_pointcloud
    PlotXX/YYMMDD/03_calibration

It builds a DEM from the generated point cloud(s) and plots the kinematic
calibration time series for the right and left scanner beneath it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from src.calibration.kinematiccalibration import kinematiccalibration


DEFAULT_PARENT_DIR = Path(
    "/mnt/syn180/241111_FieldPheno4D_multi_crop_multi_modal/01_cropplotdata/New_structure"
)
DEFAULT_OUTPUT_NAME = "dem_kinematic_calibration.png"


def discover_plot_dirs(parent_dir: Path) -> list[Path]:
    plot_dirs: list[Path] = []
    for plot_dir in sorted(parent_dir.glob("Plot*")):
        if not plot_dir.is_dir():
            continue
        for date_dir in sorted(plot_dir.iterdir()):
            if not date_dir.is_dir():
                continue
            if (date_dir / "03_calibration").exists() and (date_dir / "04_pointcloud").exists():
                plot_dirs.append(date_dir)
    return plot_dirs


def resolve_plot_dir(
    plot_dir: str | Path | None,
    parent_dir: str | Path,
    plot_id: str | None,
    date: str | None,
) -> Path:
    if plot_dir is not None:
        return Path(plot_dir)

    if plot_id is None or date is None:
        raise ValueError("Either --plot-dir or both --plot-id and --date must be given")

    return Path(parent_dir) / plot_id / date


def load_pointcloud_xyz(pointcloud_dir: Path, max_points: int | None = 3_000_000) -> tuple[np.ndarray, Path]:
    import numpy as np

    from src.pointcloud.pointcloud import pointcloud as PointCloud

    candidates = sorted([*pointcloud_dir.glob("*.las"), *pointcloud_dir.glob("*.laz")])
    if not candidates:
        raise FileNotFoundError(f"No LAS/LAZ files found in {pointcloud_dir}")

    preferred = next((path for path in candidates if path.name == "pc_kinematic_calibration.las"), None)
    if preferred is not None:
        candidates = [preferred]

    xyz_parts: list[np.ndarray] = []
    for path in candidates:
        pc = PointCloud()
        pc.read(str(path))
        if pc.xyz.size > 0:
            xyz_parts.append(pc.xyz)

    if not xyz_parts:
        raise ValueError(f"No points found in {pointcloud_dir}")

    xyz = np.vstack(xyz_parts)
    if max_points is not None and max_points > 0 and xyz.shape[0] > max_points:
        rng = np.random.default_rng(13)
        idx = rng.choice(xyz.shape[0], size=max_points, replace=False)
        xyz = xyz[idx]

    return xyz, candidates[0]


def load_calibration_series(calibration_dir: Path, scanner: str) -> kinematiccalibration:
    import numpy as np

    from src.calibration.kinematiccalibration import kinematiccalibration

    scanner = scanner.lower()
    if scanner not in {"l", "r"}:
        raise ValueError("scanner must be 'l' or 'r'")

    calibration = kinematiccalibration()
    raw_path = calibration_dir / f"x_{scanner}.txt"
    int_path = calibration_dir / f"xint_{scanner}.txt"

    if not raw_path.exists():
        raise FileNotFoundError(f"Missing calibration file: {raw_path}")

    if int_path.exists():
        calibration.read_calibration_from_file(str(raw_path), str(int_path))
    else:
        calibration.read_calibration_from_file(str(raw_path))
        calibration.xint = None

    if calibration.x is None or np.size(calibration.x) == 0:
        raise ValueError(f"Calibration file is empty: {raw_path}")

    if calibration.x.ndim == 1:
        calibration.x = calibration.x.reshape(1, -1)
    if calibration.xint is not None and calibration.xint.ndim == 1:
        calibration.xint = calibration.xint.reshape(1, -1)

    return calibration


def load_static_calibration(calibration_dir: Path, scanner: str):
    from src.calibration.calibration import calibration

    scanner = scanner.lower()
    if scanner not in {"l", "r"}:
        raise ValueError("scanner must be 'l' or 'r'")

    static_calibration = calibration()
    xml_path = calibration_dir / f"system_config_lmi_{scanner}.xml"
    if not xml_path.exists():
        raise FileNotFoundError(f"Missing static calibration file: {xml_path}")

    static_calibration.read_calibration_from_xml(str(xml_path))
    return static_calibration


def _time_axis(data: np.ndarray) -> np.ndarray:
    import numpy as np

    return data[:, 0] - data[0, 0]


def _plot_dem(ax, xyz: np.ndarray, dxy: float, nodata: float, agg: str) -> None:
    import numpy as np

    xylimits = np.array(
        [
            float(np.min(xyz[:, 0])),
            float(np.max(xyz[:, 0])),
            float(np.min(xyz[:, 1])),
            float(np.max(xyz[:, 1])),
        ]
    )
    grid, _ = compute_dem_grid(xyz=xyz, xylimits=xylimits, dxy=dxy, nodata=nodata, agg=agg)
    masked_grid = np.ma.masked_where(grid == nodata, grid)
    image = ax.imshow(
        masked_grid,
        origin="lower",
        extent=(xylimits[0], xylimits[1], xylimits[2], xylimits[3]),
        aspect="equal",
        cmap="turbo",
    )
    #ax.set_title("Digital Elevation Model")
    ax.set_ylabel("Y (m)", fontsize=20)
    ax.set_xlabel("X (m)", fontsize=20)
    ax.tick_params(axis='y', labelsize=20)
    ax.tick_params(axis='x', labelsize=20)
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.25)
    return image


def _plot_kinematic_series(
    ax,
    calibration: kinematiccalibration,
    static_calibration,
    value_slice: slice,
    title: str,
    ylabel: str,
    use_degrees: bool = False,
) -> None:
    import numpy as np

    colors = ["tab:red", "tab:green", "tab:blue"]
    labels = ["x", "y", "z"]
    raw = np.atleast_2d(calibration.x)
    times = _time_axis(raw)
    data = raw[:, value_slice]

    if use_degrees:
        static_values = np.array([static_calibration.rx, static_calibration.ry, static_calibration.rz])
    else:
        static_values = np.array([static_calibration.tx, static_calibration.ty, static_calibration.tz])

    for idx in range(3):
        values = data[:, idx]
        if use_degrees:
            values = np.degrees(values) - static_values[idx]
        else:
            values = (values - static_values[idx]) * 1000.0
        ax.plot(times, values, color=colors[idx], marker="x", linestyle="None", markersize=4, label=labels[idx])

    if calibration.xint is not None and np.size(calibration.xint) > 0:
        interp = np.atleast_2d(calibration.xint)
        interp_times = _time_axis(interp)
        interp_data = interp[:, value_slice]
        for idx in range(3):
            values = interp_data[:, idx]
            if use_degrees:
                values = np.degrees(values) - static_values[idx]
            else:
                values = (values - static_values[idx]) * 1000.0
            ax.plot(interp_times, values, color=colors[idx], linewidth=1.2, alpha=0.95)

    #ax.set_title(title)
    ax.set_ylabel(ylabel, fontsize=20)
    ax.set_xlabel("Time (s)", fontsize=20)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.set_xlim(0, interp_times[-1])
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)


def _print_delta_statistics(
    scanner_name: str,
    calibration: kinematiccalibration,
    static_calibration,
) -> None:
    """Print bias and variation of the kinematic deltas for one scanner."""

    series = calibration.xint if calibration.xint is not None and np.size(calibration.xint) > 0 else calibration.x
    series = np.atleast_2d(series)

    translation_mm = (series[:, 4:7] - np.array([static_calibration.tx, static_calibration.ty, static_calibration.tz])) * 1000.0
    rotation_deg = np.degrees(series[:, 1:4]) - np.array([static_calibration.rx, static_calibration.ry, static_calibration.rz])

    translation_bias = np.mean(translation_mm, axis=0)
    translation_variation = np.std(translation_mm, axis=0)
    rotation_bias = np.mean(rotation_deg, axis=0)
    rotation_variation = np.std(rotation_deg, axis=0)

    print(f"| {scanner_name} translation bias (mm):   x={translation_bias[0]: .3f}, y={translation_bias[1]: .3f}, z={translation_bias[2]: .3f}")
    print(f"| {scanner_name} translation variation (mm): x={translation_variation[0]: .3f}, y={translation_variation[1]: .3f}, z={translation_variation[2]: .3f}")
    print(f"| {scanner_name} rotation bias (deg):       x={rotation_bias[0]: .3f}, y={rotation_bias[1]: .3f}, z={rotation_bias[2]: .3f}")
    print(f"| {scanner_name} rotation variation (deg):   x={rotation_variation[0]: .3f}, y={rotation_variation[1]: .3f}, z={rotation_variation[2]: .3f}")


def _sample_points(xyz, max_points: int, seed: int):
    import numpy as np

    if xyz.shape[0] <= max_points:
        return xyz

    rng = np.random.default_rng(seed)
    idx = rng.choice(xyz.shape[0], size=max_points, replace=False)
    return xyz[idx, :]


def _rotation_matrix_from_vectors(a, b):
    import numpy as np

    a_unit = a / np.linalg.norm(a)
    b_unit = b / np.linalg.norm(b)

    v = np.cross(a_unit, b_unit)
    c = np.dot(a_unit, b_unit)
    s = np.linalg.norm(v)

    if s == 0:
        if c > 0:
            return np.eye(3)
        axis = np.array([1.0, 0.0, 0.0])
        if abs(a_unit[0]) > 0.9:
            axis = np.array([0.0, 1.0, 0.0])
        v = np.cross(a_unit, axis)
        v = v / np.linalg.norm(v)
        return _rodrigues(v, np.pi)

    v_unit = v / s
    return _rodrigues(v_unit, np.arctan2(s, c))


def _rodrigues(k, theta: float):
    import numpy as np

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


def _fit_plane_pca(xyz, max_points: int = 2_000_000, seed: int = 13):
    import numpy as np

    xyz_sample = _sample_points(xyz, max_points=max_points, seed=seed)
    centroid = np.mean(xyz_sample, axis=0)
    xyz_centered = xyz_sample - centroid
    _, _, vt = np.linalg.svd(xyz_centered, full_matrices=False)
    normal = vt[-1, :]
    if normal[2] < 0:
        normal = -normal
    return normal, centroid


def _build_plane_alignment(normal, centroid):
    import numpy as np

    target = np.array([0.0, 0.0, 1.0])
    rotation = _rotation_matrix_from_vectors(normal, target)
    translation = centroid - rotation @ centroid
    return rotation, translation, centroid[2]


def _apply_alignment(xyz, rotation, translation, z_offset):
    rotated = (rotation @ xyz.T).T + translation
    rotated[:, 2] = rotated[:, 2] - z_offset
    return rotated


def _rotation_from_pca_xy(xy):
    import numpy as np

    xy_centered = xy - np.mean(xy, axis=0)
    _, _, vt = np.linalg.svd(xy_centered, full_matrices=False)
    direction = vt[0, :]
    angle = np.arctan2(direction[1], direction[0])

    c = np.cos(-angle)
    s = np.sin(-angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _apply_z_rotation(xyz, rotmat):
    return (rotmat @ xyz.T).T


def compute_dem_grid(
    xyz,
    xylimits,
    dxy,
    nodata,
    agg: str = "max",
):
    import numpy as np

    min_x, max_x, min_y, max_y = xylimits

    x_edges = np.arange(min_x, max_x + dxy + 1e-9, dxy)
    y_edges = np.arange(min_y, max_y + dxy + 1e-9, dxy)

    x_idx = np.digitize(xyz[:, 0], x_edges) - 1
    y_idx = np.digitize(xyz[:, 1], y_edges) - 1

    x_idx = np.clip(x_idx, 0, len(x_edges) - 2)
    y_idx = np.clip(y_idx, 0, len(y_edges) - 2)

    nrows = len(y_edges) - 1
    ncols = len(x_edges) - 1
    grid = np.full((nrows, ncols), nodata, dtype=np.float32)

    if xyz.shape[0] == 0:
        return grid, None

    valid = ~np.isnan(xyz[:, 2])
    rows = y_idx[valid]
    cols = x_idx[valid]
    z = xyz[valid, 2].astype(np.float32)

    if agg == "max":
        grid = np.full((nrows, ncols), -np.inf, dtype=np.float32)
        np.maximum.at(grid, (rows, cols), z)
        grid[~np.isfinite(grid)] = nodata
    elif agg == "mean":
        flat = rows * ncols + cols
        sums = np.bincount(flat, weights=z, minlength=nrows * ncols)
        counts = np.bincount(flat, minlength=nrows * ncols)
        valid_cells = counts > 0
        grid_flat = grid.ravel()
        grid_flat[valid_cells] = sums[valid_cells] / counts[valid_cells]
        grid = grid_flat.reshape((nrows, ncols))
    elif agg == "median":
        flat = rows * ncols + cols
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

    return grid, None


def create_combined_plot(
    plot_dir: Path,
    output_path: Path | None = None,
    dxy: float = 0.01,
    agg: str = "max",
    nodata: float = -9999.0,
    max_points: int | None = 3_000_000,
) -> Path:
    import matplotlib.pyplot as plt
    import numpy as np

    pointcloud_dir = plot_dir / "04_pointcloud"
    calibration_dir = plot_dir / "03_calibration"

    if not pointcloud_dir.exists():
        raise FileNotFoundError(f"Point cloud directory not found: {pointcloud_dir}")
    if not calibration_dir.exists():
        raise FileNotFoundError(f"Calibration directory not found: {calibration_dir}")

    xyz, pointcloud_path = load_pointcloud_xyz(pointcloud_dir, max_points=max_points)
    right_cal = load_calibration_series(calibration_dir, "r")
    left_cal = load_calibration_series(calibration_dir, "l")
    right_static = load_static_calibration(Path(REPO_ROOT) / "input" / "calibration", "r")
    left_static = load_static_calibration(Path(REPO_ROOT) / "input" / "calibration", "l")

    print("| ------------------------------------------------------------------------------")
    print("| Kinematic calibration delta statistics")
    _print_delta_statistics("right scanner", right_cal, right_static)
    _print_delta_statistics("left scanner", left_cal, left_static)
    print("| ------------------------------------------------------------------------------")

    normal, centroid = _fit_plane_pca(xyz, max_points=2_000_000, seed=13)
    rotation, translation, z_offset = _build_plane_alignment(normal, centroid)
    xyz = _apply_alignment(xyz, rotation, translation, z_offset)

    z_rotation = _rotation_from_pca_xy(xyz[:, :2])
    xyz = _apply_z_rotation(xyz, z_rotation)

    xyz[:, 0] = xyz[:, 0] - np.min(xyz[:, 0])
    xyz[:, 1] = xyz[:, 1] - np.min(xyz[:, 1])
    xyz[:, 2] = xyz[:, 2] - np.min(xyz[:, 2])

    fig, axes = plt.subplots(
        6,
        1,
        figsize=(16, 22),
        gridspec_kw={"height_ratios": [2.5, 0.18, 1, 1, 1, 1], "hspace": 0.40},
    )

    image = _plot_dem(axes[0], xyz=xyz, dxy=dxy, nodata=nodata, agg=agg)
    cbar = fig.colorbar(
        image,
        cax=axes[1],
        orientation="horizontal",
    )
    #cbar.ax.set_title("Height (m)", pad=8)
    cbar.ax.xaxis.set_ticks_position("bottom")
    cbar.ax.xaxis.set_label_position("bottom")
    cbar.ax.tick_params(axis="x", labelsize=10)

    _plot_kinematic_series(
        axes[2],
        right_cal,
        right_static,
        slice(4, 7),
        "Kinematic calibration translation right (x, y, z)",
        r"$\Delta t$ (mm)",
        use_degrees=False,
    )
    _plot_kinematic_series(
        axes[3],
        left_cal,
        left_static,
        slice(4, 7),
        "Kinematic calibration translation left (x, y, z)",
        r"$\Delta t$ (mm)",
        use_degrees=False,
    )
    _plot_kinematic_series(
        axes[4],
        right_cal,
        right_static,
        slice(1, 4),
        "Kinematic calibration rotation right (rx, ry, rz)",
        r"$\Delta r$ (deg)",
        use_degrees=True,
    )
    _plot_kinematic_series(
        axes[5],
        left_cal,
        left_static,
        slice(1, 4),
        "Kinematic calibration rotation left (rx, ry, rz)",
        r"$\Delta r$ (deg)",
        use_degrees=True,
    )

    axes[5].set_xlabel("Time (s)")
    fig.suptitle(f"{plot_dir.name}: DEM and kinematic calibration overview", y=0.995, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.993))

    if output_path is None:
        output_path = plot_dir / DEFAULT_OUTPUT_NAME
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot-dir", type=str, default=None, help="Full path to a single PlotXX/YYMMDD directory")
    parser.add_argument("--parent-dir", type=str, default=str(DEFAULT_PARENT_DIR), help="Root directory containing PlotXX folders")
    parser.add_argument("--plot-id", type=str, default=None, help="Plot identifier, for example P144")
    parser.add_argument("--date", type=str, default=None, help="Measurement date, for example 230516")
    parser.add_argument("--output", type=str, default=None, help="Optional explicit output image path")
    parser.add_argument("--dxy", type=float, default=0.01, help="DEM grid spacing in meters")
    parser.add_argument("--agg", type=str, default="max", choices=("max", "mean", "median"), help="DEM aggregation mode")
    parser.add_argument("--nodata", type=float, default=-9999.0, help="NoData value for the DEM")
    parser.add_argument("--max-points", type=int, default=3_000_000, help="Max number of points to sample for the DEM")
    parser.add_argument("--all", action="store_true", help="Process all PlotXX/YYMMDD folders below --parent-dir")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if not args.all and args.plot_dir is None and args.plot_id is None and args.date is None:
        args.all = True

    if args.all:
        plot_dirs = discover_plot_dirs(Path(args.parent_dir))
        if not plot_dirs:
            raise FileNotFoundError(f"No plot directories found below {args.parent_dir}")
        outputs = [
            create_combined_plot(
                plot_dir,
                output_path=None,
                dxy=args.dxy,
                agg=args.agg,
                nodata=args.nodata,
                max_points=args.max_points,
            )
            for plot_dir in plot_dirs
        ]
        for path in outputs:
            print(path)
        return 0

    plot_dir = resolve_plot_dir(args.plot_dir, args.parent_dir, args.plot_id, args.date)
    output_path = Path(args.output) if args.output is not None else None
    result = create_combined_plot(
        plot_dir,
        output_path=output_path,
        dxy=args.dxy,
        agg=args.agg,
        nodata=args.nodata,
        max_points=args.max_points,
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
