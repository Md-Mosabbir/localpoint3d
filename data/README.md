# Scans: how to capture, name and store them

Scans are **not** stored in git (too big). They live in the shared Google Drive folder `localpoint3d/scans`. Each of us adds a shortcut to it in *My Drive*, so the notebooks find it at `/content/drive/MyDrive/localpoint3d/scans`.

## Folders and file names

One folder per object, files named `<Object>_<take>`:

```
Models/
├── Chair/      Chair_1.ply … Chair_8.ply
├── Cup/        Cup_1.ply … Cup_4.ply
├── Keyboard/   Keyboard_1.ply … Keyboard_3.ply
└── Table/      Table_1.ply, Table_2.ply
```

When we scan the same object with both methods, add the method: `Rickshaw/Rickshaw_lidar_1.ply`, `Rickshaw/Rickshaw_photo_1.obj`.

The notebooks search all subfolders. They read object / method / take from the file name and put them in the results CSV, so stick to this pattern. A local copy lives in `data/Models/`, which git ignores.

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
