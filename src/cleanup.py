"""Clean a scan before it goes to a model: find which way is up, remove the floor.

Why: our scans include the floor or desk under the object (22-78 % of all points in the
2026-10-18 baseline). After scaling to the unit sphere the object can end up as a small
bump on a big flat plane, and the model describes the plane instead.

Only numpy, so it runs on the laptop and in both Colab environments.
"""
import os

import numpy as np

AXIS = {"x": 0, "y": 1, "z": 2}

# Measured on our Scaniverse exports (journal 02): PLY files are Z-up, OBJ files are Y-up.
# GLB/GLTF are Y-up by definition of the format.
UP_AXIS_BY_EXT = {".ply": "z", ".obj": "y", ".glb": "y", ".gltf": "y"}


def guess_up_axis(path):
    """Which axis points up in this file? A guess from the file type (default "z")."""
    return UP_AXIS_BY_EXT.get(os.path.splitext(path)[1].lower(), "z")


def find_floor(xyz, up_axis="z", threshold=0.01, iters=300, seed=0):
    """Boolean mask of the points on the floor (or desk) plane.

    RANSAC in plain words: pick 3 random points from the lowest 30 % of the scan, they
    define a plane. If that plane is roughly horizontal, count the points within
    `threshold` metres of it. Repeat `iters` times and keep the plane with the most points.
    1 cm because at 2 cm most of a keyboard (only ~2 cm tall) was deleted along with the desk.
    Planes that aren't horizontal (walls, a chair's backrest) are never picked.
    """
    rng = np.random.default_rng(seed)
    up = np.eye(3)[AXIS[up_axis]]
    heights = xyz[:, AXIS[up_axis]]
    low = xyz[heights <= np.percentile(heights, 30)]  # the floor is in the bottom part
    best = np.zeros(len(xyz), dtype=bool)
    if len(low) < 3:
        return best
    for _ in range(iters):
        p0, p1, p2 = low[rng.choice(len(low), 3, replace=False)]
        normal = np.cross(p1 - p0, p2 - p0)
        if np.linalg.norm(normal) < 1e-9:
            continue  # the 3 points were on a line
        normal /= np.linalg.norm(normal)
        if abs(normal @ up) < 0.9:
            continue  # tilted more than ~25 degrees: not a floor
        on_plane = np.abs((xyz - p0) @ normal) < threshold
        if on_plane.sum() > best.sum():
            best = on_plane
    return best


def remove_floor(xyz, rgb=None, up_axis="z", threshold=0.01, iters=300, seed=0):
    """Drop the floor plane. Returns (xyz, rgb, number_of_points_removed)."""
    floor = find_floor(xyz, up_axis, threshold, iters, seed)
    keep = ~floor
    return xyz[keep], (rgb[keep] if rgb is not None else None), int(floor.sum())
