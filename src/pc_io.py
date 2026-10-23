"""Load any 3D file we care about into a plain numpy point cloud.

Supported:
  * .ply point cloud  (x, y, z + optional red/green/blue)
  * .ply / .obj / .glb / .gltf / .off / .stl mesh  -> we sample points on its surface
    (Scaniverse "Mesh" exports, ModelNet40 .off files, ShapeNet .obj files)

Everything comes back as:
  xyz  : (N, 3) float32 array of positions
  rgb  : (N, 3) float32 array of colours in [0, 1], or None if the file had no colour
  info : small dict describing what we found (handy for printing)

Only needs numpy, plyfile and trimesh, so it works both in Colab's normal Python
and inside the Python 3.10 environment we build for PointLLM.
"""
import os

import numpy as np

MESH_SAMPLES = 100_000  # how many points to sample when the file is a mesh


def load_points(path, mesh_samples=MESH_SAMPLES, seed=0):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".ply":
        from plyfile import PlyData
        ply = PlyData.read(path)
        names = [el.name for el in ply.elements]
        has_faces = "face" in names and ply["face"].count > 0
        if not has_faces:
            return _ply_vertices(ply["vertex"])
    # anything with triangles (or a non-ply file) is treated as a mesh
    return _sample_mesh(path, mesh_samples, seed)


def _ply_vertices(v):
    props = v.data.dtype.names
    if "f_dc_0" in props:
        raise ValueError("This .ply is a Gaussian splat, not a point cloud/mesh. "
                         "In Scaniverse export the scan as Mesh (OBJ or PLY) instead.")
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)

    rgb = None
    if {"red", "green", "blue"} <= set(props):
        rgb = np.stack([v["red"], v["green"], v["blue"]], axis=1).astype(np.float32)
        if rgb.max() > 1.0:  # usually stored as 0..255 integers
            rgb /= 255.0
    return xyz, rgb, {"kind": "point cloud"}


def _sample_mesh(path, n, seed):
    import trimesh
    mesh = trimesh.load(path, force="mesh", process=False)
    if len(mesh.faces) == 0:  # e.g. an .obj that only has vertices
        xyz = np.asarray(mesh.vertices, dtype=np.float32)
        return xyz, None, {"kind": "vertices only"}

    xyz, face_idx = trimesh.sample.sample_surface(mesh, n, seed=seed)
    rgb = None
    try:
        visual = mesh.visual
        if visual.kind == "texture":
            visual = visual.to_color()  # bake the texture image into per-vertex colours
        if visual.kind in ("vertex", "face"):
            vc = np.asarray(visual.vertex_colors, dtype=np.float32)[:, :3] / 255.0
            rgb = vc[mesh.faces[face_idx]].mean(axis=1)  # average the 3 corners of each triangle
    except Exception:
        rgb = None  # colour is a nice-to-have, never crash because of it
    return xyz.astype(np.float32), rgb, {"kind": "mesh", "faces": int(len(mesh.faces))}


def describe(path):
    """One-line human summary of a file, e.g. for listing scans."""
    xyz, rgb, info = load_points(path)
    size = xyz.max(axis=0) - xyz.min(axis=0)
    return (f"{os.path.basename(path)}: {info['kind']}, {len(xyz):,} points, "
            f"colour={'yes' if rgb is not None else 'no'}, "
            f"bounding box {size[0]:.2f} x {size[1]:.2f} x {size[2]:.2f} (file units)")


METHODS = {"lidar", "photo", "detail", "area", "speed"}  # words we accept as the capture method
VERSIONS = {"cleaned", "clean", "raw"}                    # optional last word, e.g. Chair_01_cleaned.ply


def parse_scan_name(path):
    """Read object / method / take / version from a file name.

    Pattern: <Object>_<take>, <Object>_<method>_<take>, optionally ending in _cleaned.
    Chair/Chair_1.ply          -> {"object": "chair",      "method": "unknown", "take": "1", "version": "raw"}
    Chair_01_cleaned.ply       -> {"object": "chair",      "method": "unknown", "take": "1", "version": "cleaned"}
    Thela_Gari_2.obj           -> {"object": "thela_gari", "method": "unknown", "take": "2", "version": "raw"}
    rickshaw_lidar_01.ply      -> {"object": "rickshaw",   "method": "lidar",   "take": "1", "version": "raw"}
    The object name may contain "_", so the method only counts if it is one of METHODS.
    Takes are numbers, so "01" and "1" are the same take (cleaned and raw files line up).
    Files that don't follow the pattern still work, the fields are just "unknown".
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    parts = stem.split("_")
    version = "raw"
    if len(parts) >= 3 and parts[-1].lower() in VERSIONS:
        version = "cleaned" if parts[-1].lower() in ("cleaned", "clean") else "raw"
        parts = parts[:-1]
    if len(parts) < 2:
        return {"object": "_".join(parts).lower(), "method": "unknown", "take": "unknown", "version": version}
    take, rest = parts[-1], parts[:-1]
    if take.isdigit():
        take = str(int(take))  # "01" -> "1"
    method = "unknown"
    if len(rest) >= 2 and rest[-1].lower() in METHODS:
        method, rest = rest[-1].lower(), rest[:-1]
    return {"object": "_".join(rest).lower(), "method": method, "take": take, "version": version}
