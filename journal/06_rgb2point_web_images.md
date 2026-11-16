# 06 · RGB2Point: web images → point clouds

Shahedul's idea: we cannot scan 100 bodnas, so collect ~100 photos per object from the web and let **RGB2Point** (a model that guesses a 3D point cloud from one photo) turn each into a point cloud. Notebook: [`notebooks/{UPDATED}RGB2Point_Google_Review_1024_8192.ipynb`](../notebooks/%7BUPDATED%7DRGB2Point_Google_Review_1024_8192.ipynb).

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
