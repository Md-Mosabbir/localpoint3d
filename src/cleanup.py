"""Clean a scan before it goes to a model: find which way is up, remove the floor, remove stray specks.

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


def find_specks(xyz, voxel=0.10, min_share=0.01):
    """Boolean mask of stray specks: small groups of points floating away from the object.

    How: cut space into cubes of `voxel` metres and join cubes that touch (also diagonally)
    into connected pieces. The object is one big piece (or a few); specks are tiny pieces
    far away. Pieces holding less than `min_share` of all points count as specks.
    10 cm / 1 %: with 5 cm / 2 % a real, separately captured chair-leg piece (Chair_05) was
    deleted; 10 cm bridges such gaps, and real specks were < 0.3 % of the points.
    Why it matters: scaling to the unit sphere uses the farthest point, so one speck
    5 m away shrinks the real object to a dot.
    """
    cells = np.floor(xyz / voxel).astype(np.int64)
    keys, point_cell = np.unique(cells, axis=0, return_inverse=True)
    point_cell = point_cell.reshape(-1)
    count = np.bincount(point_cell, minlength=len(keys))
    index = {tuple(k): i for i, k in enumerate(keys)}
    steps = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1) if (dx, dy, dz) != (0, 0, 0)]

    piece = np.full(len(keys), -1)  # which connected piece each cube belongs to
    for start in range(len(keys)):
        if piece[start] >= 0:
            continue
        piece[start] = start
        todo = [start]
        while todo:  # flood fill: visit every cube touching this piece
            x, y, z = keys[todo.pop()]
            for dx, dy, dz in steps:
                j = index.get((x + dx, y + dy, z + dz))
                if j is not None and piece[j] < 0:
                    piece[j] = start
                    todo.append(j)

    points_per_piece = np.bincount(piece, weights=count, minlength=len(keys))
    small_piece = points_per_piece < min_share * len(xyz)
    return small_piece[piece[point_cell]]


def remove_specks(xyz, rgb=None, voxel=0.10, min_share=0.01):
    """Drop stray specks. Returns (xyz, rgb, number_of_points_removed)."""
    specks = find_specks(xyz, voxel, min_share)
    keep = ~specks
    return xyz[keep], (rgb[keep] if rgb is not None else None), int(specks.sum())
