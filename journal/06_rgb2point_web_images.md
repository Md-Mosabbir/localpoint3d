# 06 · RGB2Point: web images → point clouds

Shahedul's idea: we cannot scan 100 bodnas, so collect ~100 photos per object from the web and let **RGB2Point** (a model that guesses a 3D point cloud from one photo) turn each into a point cloud. Notebook: [`notebooks/rgb2point.ipynb`](../notebooks/rgb2point.ipynb).

## 2026-11-16 · Claude · the Google Images step returned nothing; replaced with Bing + DuckDuckGo

**What the notebook says about itself (not checked by us):**
- It uses the released RGB2Point checkpoint `pc1024_three.pth` from <https://github.com/JaeLee18/RGB2point>, without training.
- That checkpoint outputs 1,024 xyz points with no colour. Its training domain is ShapeNet chairs, airplanes and cars.
- The 8,192-point files are the same 1,024 points with extra points interpolated between neighbours. That adds points, not detail.

**What we measured (laptop, 2026-11-16, query "Wooden Bench Bangladesh", 30 images asked):**

| Source | Result |
|---|---|
| Google Images via `icrawler` `GoogleImageCrawler` | **0 files**; the crawler got no result list at all |
| Bing Images via `icrawler` `BingImageCrawler` | 30 files in 12 s |
| DuckDuckGo Images via `ddgs` | 30 results in 1 s |
| Openverse API (openly licensed images only) | 1 result |
| Wikimedia Commons API | 3 results |

Google answers image searches only for real browsers, so a script gets nothing. We do not try to get around that. Openverse and Wikimedia would give clear licences but have almost no pictures of our objects.

**The fix (download cell of the notebook):**
- `IMAGE_SOURCES = ["bing", "duckduckgo"]`, tried in order. It searches "<object> Bangladesh" first and then the plain name, until `TARGET_IMAGES` is reached. A failing source is skipped, not fatal.
- Own images: `OWN_IMAGES_DIR` or `UPLOAD_OWN_IMAGES` (a ZIP or files), for pictures saved by hand from Google or taken ourselves. They are used first.
- The manifest now records `image_source`, `search_query`, `source_url` and `source_page` per image.
- `pip` output is no longer hidden.

**Tests of the new cell (laptop):**

| Query | Images kept | From | Time |
|---|---|---|---|
| Wooden Bench | 100 / 100 | Bing 74, DuckDuckGo 26 | 69 s |
| Bodna | 100 / 100 | Bing 90, DuckDuckGo 10 | 88 s |

Looking at the 100 bodna pictures by eye: about **65 show a real bodna**. The rest are noise, mostly news pictures about "bonds" and Bangladesh Bank that Bing mixes in for "Bodna", plus a few other jugs and bins. Several good ones show two or more bodnas, or have advert text on them. So the manual review step in the notebook is needed, and 100 downloads will not mean 100 usable images.

**Not tested yet:**
- The notebook on Colab. Colab machines share addresses with many other users, so a search site may rate-limit there even though it worked on the laptop.
- The RGB2Point conversion cells (they need the checkpoint; we did not change them).
- Whether RGB2Point gives a believable shape for a bodna at all. It was trained on chairs, airplanes and cars, so this is the real open question. Check the one-image smoke test before making 100 clouds.

**Keep in mind:**
- Web images have unknown licences. Do not commit them to git, and check before sharing a dataset.
- These point clouds are a model's guess from one photo, not scans. Keep them separate from our Scaniverse scans in every table.

## 2026-11-17 · Claude · pictures collected on the laptop; RGB2Point makes the same chair-like cloud for every object

**Why:** searching inside Colab was slow and unreliable, so the plan became: collect the pictures once on the laptop, put them on Drive, and let the notebook only run the model ("Part B" of `notebooks/rgb2point.ipynb`).

**Collecting (laptop):** `src/web_images.py` with the search phrases in `data/web_image_queries.csv`. Pictures go to `data/web_images/<object>/` (not in git), shrunk to 640 px, with a `manifest.csv` of source URLs. Wrong pictures are moved to `data/web_images/_rejected/` after looking at numbered overview sheets.

| Object | Downloaded | Kept after review by eye |
|---|---|---|
| bodna | 250 | 88 |
| hurricane_lamp | 250 | 79 |
| kula | 250 | 60 |
| matir_jug | 250 | 72 |
| muura | 250 | 103 |
| patil | 250 | 34 |
| piri | 250 | 19 |
| rehal | 250 | 98 |
| wooden_benched | 250 | 142 |
| **total** | 2,250 | **695** |

About **1 picture in 3 is usable** (695 of 2,250). The rest are other objects, people, adverts with text, collages, or the wrong kind of the object:
- "bodna" brings news about bonds and video thumbnails.
- "kula" brings round trays.
- "matir jug" brings plastic and steel jugs.
- "hurricane lamp" brings candle holders.
- "patil" brings aluminium sheets, factories and Western saucepans. We kept only round-bellied pots.
- "piri" brings chapati boards with rolling pins, Western stools and woven Indian stools. Plain Bangladeshi wooden piris are rare online: 19 kept, most of them painted wedding piris. Whether woven "pidha" stools count as piri is our decision to make.

English descriptive phrases ("terracotta water jug", "bamboo cane mora stool") gave far better pictures than the local names. Bing often returned nothing for a phrase; DuckDuckGo carried most of the load, about 35 results per page.

**Testing the model (laptop CPU, released checkpoint `pc1024_three.pth`, the notebook's own cells):**
- Part B ran end to end on 8 pictures. A second run skipped the finished ones. The `_rejected` folder was ignored.
- Bug fixed on the way: the setup cell failed when run twice, because it tried to patch `model.py` a second time.

**The result that matters:** every object comes out as the same fuzzy, chair-like cloud. From the side, the authors' own chair picture gives a clear seat-and-back "L", and so do many of the bodna, lamp, kula and rehal pictures; the rest are shapeless.

![Side views of RGB2Point clouds: the authors' chair picture, then two pictures each of our objects](../reports/figures/2026-11-17_rgb2point_side_views.png)

*Left: the authors' chair sample. Then two pictures per object. No web pictures are shown, only the model's output.*

We measured it with the **Chamfer distance** (for each point, the distance to the nearest point of the other cloud, averaged; 0 = identical clouds). 20 pictures per object, 8 objects; clouds are about 1.02 across. Patil, piri and rehal used unreviewed pictures. Table: `results/rgb2point/chamfer_within_vs_between_2026-11-17.csv`.

| | Chamfer distance |
|---|---|
| two pictures of the **same** object | 0.023 – 0.035 |
| pictures of **different** objects | 0.028 – 0.032 |
| our objects vs the authors' chair cloud | 0.024 – 0.031 |

The three ranges are the same. If the clouds held the object's shape, the first row would be clearly smaller than the second. It is not: for piri it is even larger.

**What it means:** this checkpoint knows chairs, airplanes and cars (as its authors say). For anything else it falls back to its nearest known shape. Its clouds for our objects are not 3D data of those objects. Feeding them to PointLLM would test "does PointLLM see a chair", not our objects. Interpolating to 8,192 points changes nothing about this.

**What the pictures are still good for:** they are real photos of the objects, so they fit the image side of the project: CLIP / MetaCLIP zero-shot on photos, or fine-tuning with captions.

**If we still want 3D from photos:** use a single-image-to-3D model trained on many kinds of objects (for example TripoSR, which was trained on Objaverse), not this three-class checkpoint. Not tried yet. Its shapes would still be a model's guess, not scans.

**State on 2026-11-17, end of first pass:** all 9 objects downloaded and reviewed once. Only muura and wooden bench have 100 or more good pictures. Extra search phrases for the others are already in `data/web_image_queries.csv`; running `collect` again with a higher `--target` tops them up without bringing rejected pictures back. Not done yet, because the RGB2Point result above makes it unclear what the pictures will be used for.
