# Scans: how to capture, name and store them

Scans are **not** stored in git (too big). They live in the shared Google Drive folder `bd-pointcloud/scans`. Each of us adds a shortcut to it in *My Drive*, so the notebooks find it at `/content/drive/MyDrive/bd-pointcloud/scans`.

## File names

```
<object>_<method>_<take>.<ext>
```

| part | values | example |
|---|---|---|
| object | lowercase, no spaces (use `-`) | `rickshaw`, `thelagari`, `cng`, `plastic-chair` |
| method | `lidar` or `photo` | |
| take | two digits, 01, 02, … | |

Examples: `rickshaw_lidar_01.ply`, `rickshaw_photo_01.obj`, `chair_lidar_02.ply`

The notebooks read this from the file name and put it in the results CSV, so stick to it.

## Capturing with Scaniverse

- Export as **Mesh → PLY or OBJ**, with colour. Not the Gaussian splat; our loader rejects it.
- An **OBJ** comes with a `.mtl` file and a texture image. Upload all of them to the same folder, or the colour is lost.
- **Crop** the scan in the app before exporting (remove ground, people, other objects).
- Scan each object with **both** methods (LiDAR and photo), and if possible **2 takes** of each, so we can tell "method difference" from "random take-to-take difference".
- Walk a full slow loop around the object.
- Overcast / shade is better than harsh sun.
- Shiny metal and thin rods (spokes!) are hard for both methods. Note in the journal what got lost.

## Log

Write one line per capture session in `journal/01_scaniverse_lidar_vs_photogrammetry.md`: date, who, objects, phone model, conditions.
