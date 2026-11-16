"""Collect photos of our objects from web image searches: one folder per object.

Runs on the laptop (no GPU needed). The RGB2Point notebook then only runs the model on these folders.

    pip install icrawler ddgs pillow numpy requests
    python src/web_images.py collect data/web_image_queries.csv data/web_images --target 250
    python src/web_images.py sheets data/web_images               # numbered overview pictures in _sheets/
    python src/web_images.py reject data/web_images bodna 3 8 21  # move wrong pictures to _rejected/

The CSV has two columns, `object,query`, with several search phrases per object. For every object
we go through its phrases, ask Bing and DuckDuckGo, and keep every picture that
  - really is an image and is at least MIN_SIDE pixels on its short side,
  - is not a copy of one we already have (same file, same URL, or a near-identical picture).
Each object folder gets a `manifest.csv` saying where every picture came from.

Google Images is not used: it answers real browsers only (journal 06).
Web pictures have unknown licences: the output folder is in .gitignore, never commit it.
"""
import argparse
import csv
import hashlib
import logging
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests
from PIL import Image, ImageOps

MIN_SIDE = 128           # smaller pictures are thrown away
MAX_SIDE = 640           # bigger pictures are shrunk to this (the model only looks at 224 x 224)
MAX_MB = 15
TIMEOUT = 20             # seconds to wait for one download
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; RGB2PointDatasetResearch/1.0)"}  # an honest name
SOURCES = ["bing", "duckduckgo"]
MANIFEST_COLUMNS = ["file", "object", "query", "image_source", "source_url", "source_page",
                    "original_width", "original_height", "sha256", "license_status"]


# ---------------------------------------------------------------- searching

def search_bing(query, max_num, raw_dir):
    """Download up to max_num Bing results into raw_dir. Returns [{file, image_url, source_page}]."""
    from icrawler import ImageDownloader
    from icrawler.builtin import BingImageCrawler

    class RememberUrlDownloader(ImageDownloader):
        """icrawler's downloader, plus a note of which URL each saved file came from."""

        def process_meta(self, task):
            if task.get("success"):
                self.url_of_file[task["filename"]] = task["file_url"]

    crawler = BingImageCrawler(downloader_cls=RememberUrlDownloader, storage={"root_dir": str(raw_dir)},
                               downloader_threads=4, log_level=logging.CRITICAL)
    crawler.downloader.url_of_file = {}
    crawler.crawl(keyword=query, filters={"type": "photo"}, max_num=min(max_num, 1000),
                  min_size=(MIN_SIDE, MIN_SIDE))
    return [{"file": raw_dir / name, "image_url": url, "source_page": ""}
            for name, url in crawler.downloader.url_of_file.items()]


def download_one(job):
    """Download one image URL to a file. Returns True if it worked."""
    url, path = job
    try:
        response = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        response.raise_for_status()
        if not 2_000 < len(response.content) < MAX_MB * 1024 * 1024:
            return False
        path.write_bytes(response.content)
        return True
    except Exception:
        return False


def search_duckduckgo(query, max_num, raw_dir):
    """Download up to max_num DuckDuckGo results into raw_dir. Returns [{file, image_url, source_page}]."""
    from ddgs import DDGS

    # One call gives about 35 results, so we ask for page 1, 2, 3, ... until we have enough
    # or a page brings nothing new.
    results, seen = [], set()
    for page in range(1, 8):
        new = [r for r in DDGS().images(query, safesearch="moderate", max_results=100, page=page)
               if r.get("image") and r["image"] not in seen]
        if not new:
            break
        results += new
        seen.update(r["image"] for r in new)
        if len(results) >= max_num:
            break
        time.sleep(1)
    results = results[:max_num]
    jobs =[(r["image"], raw_dir / f"{i:06d}.img") for i, r in enumerate(results, start=1) if r.get("image")]
    with ThreadPoolExecutor(max_workers=8) as pool:      # 8 downloads at the same time
        worked = list(pool.map(download_one, jobs))
    page_of = {r.get("image"): r.get("url", "") for r in results}
    return [{"file": path, "image_url": url, "source_page": page_of.get(url, "")}
            for (url, path), ok in zip(jobs, worked) if ok]


SEARCHERS = {"bing": search_bing, "duckduckgo": search_duckduckgo}


# ---------------------------------------------------------------- checking one picture

def dhash_64(pil_image):
    """A 64-bit 'fingerprint' of a picture: two pictures that look the same get (almost) the same bits."""
    small = np.asarray(pil_image.convert("L").resize((9, 8), Image.Resampling.LANCZOS))
    bits = small[:, 1:] > small[:, :-1]
    value = 0
    for bit in bits.flatten():
        value = (value << 1) | int(bit)
    return value


def different_bits(a, b):
    return (int(a) ^ int(b)).bit_count()


def save_as_jpeg(source_path, output_path):
    """Check the file is a usable picture and save it as an RGB JPEG (shrunk to MAX_SIDE).
    Returns its details, or raises ValueError / an image error if it is not usable."""
    with Image.open(source_path) as check:
        check.verify()
    with Image.open(source_path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        width, height = im.size
        if min(width, height) < MIN_SIDE:
            raise ValueError(f"too small ({width}x{height})")
        fingerprint = dhash_64(im)
        im.thumbnail((MAX_SIDE, MAX_SIDE))               # shrinks only if bigger, keeps the proportions
        im.save(output_path, format="JPEG", quality=90, optimize=True)
    return {"original_width": width, "original_height": height, "dhash": fingerprint,
            "sha256": hashlib.sha256(Path(output_path).read_bytes()).hexdigest()}


# ---------------------------------------------------------------- one object

def collect_object(name, queries, out_dir, target, sources=SOURCES, per_search=80):
    """Fill out_dir/name/ until it holds `target` pictures. Returns how many it holds at the end.

    Safe to run again after a review: rejected pictures do not come back (their URL and their
    fingerprint are remembered), search phrases that were already used are skipped, and the
    numbering continues after the highest number so far. So to top up an object, add new
    phrases for it to the CSV and run `collect` again.
    """
    folder = Path(out_dir) / name
    rejected_folder = Path(out_dir) / "_rejected" / name
    raw_root = Path(out_dir) / "_raw" / name
    folder.mkdir(parents=True, exist_ok=True)

    rows = read_manifest(folder)                          # every picture ever collected, also rejected ones
    seen_urls = {row["source_url"] for row in rows}
    seen_sha = {row["sha256"] for row in rows}
    used_queries = {row["query"] for row in rows}
    old_files = [path for _, path in numbered_pictures(folder) + numbered_pictures(rejected_folder)]
    seen_hashes = []
    for path in old_files:
        with Image.open(path) as im:
            seen_hashes.append(dhash_64(im))
    next_number = max([int(Path(row["file"]).stem.rsplit("_", 1)[1]) for row in rows], default=0) + 1
    in_folder = len(numbered_pictures(folder))

    for query in queries:
        if query in used_queries:
            continue
        for source in sources:
            if in_folder >= target:
                break
            raw_dir = raw_root / f"{source}_{next_number:04d}"
            raw_dir.mkdir(parents=True, exist_ok=True)
            try:
                candidates = SEARCHERS[source](query, per_search, raw_dir)
            except Exception as exc:   # one search failing must not stop the whole run
                print(f"  [{name}] {source} {query!r}: failed ({exc!r})", flush=True)
                continue
            before = in_folder
            for candidate in candidates:
                if in_folder >= target:
                    break
                if candidate["image_url"] in seen_urls:
                    continue
                file_name = f"{name}_{next_number:04d}.jpg"
                try:
                    details = save_as_jpeg(candidate["file"], folder / file_name)
                except Exception:
                    (folder / file_name).unlink(missing_ok=True)
                    continue
                is_copy = details["sha256"] in seen_sha or any(
                    different_bits(details["dhash"], old) <= 2 for old in seen_hashes)
                if is_copy:
                    (folder / file_name).unlink()
                    continue
                seen_sha.add(details["sha256"])
                seen_hashes.append(details["dhash"])
                seen_urls.add(candidate["image_url"])
                rows.append({"file": file_name, "object": name, "query": query, "image_source": source,
                             "source_url": candidate["image_url"], "source_page": candidate["source_page"],
                             "original_width": details["original_width"],
                             "original_height": details["original_height"], "sha256": details["sha256"],
                             "license_status": "NOT_VERIFIED_CHECK_BEFORE_TRAINING"})
                next_number += 1
                in_folder += 1
            print(f"  [{name}] {source:10s} {query!r}: {len(candidates)} downloaded, "
                  f"{in_folder - before} new, now {in_folder}", flush=True)
            time.sleep(2)              # a short pause between searches, to be polite to the search site

    with open(folder / "manifest.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    shutil.rmtree(raw_root, ignore_errors=True)
    return in_folder


def read_manifest(folder):
    """The rows of folder/manifest.csv, or [] if this object was never collected."""
    path = Path(folder) / "manifest.csv"
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def read_queries(csv_path):
    """{object: [query, query, ...]} in the order of the file."""
    queries = {}
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            queries.setdefault(row["object"].strip(), []).append(row["query"].strip())
    return queries


# ---------------------------------------------------------------- reviewing by eye

def numbered_pictures(folder):
    """[(number, file)] for the pictures in one object folder; bodna_0042.jpg has number 42."""
    return [(int(p.stem.rsplit("_", 1)[1]), p) for p in sorted(Path(folder).glob("*.jpg"))]


def make_sheets(out_dir, only=(), first_number=1, per_sheet=100, columns=10):
    """Write numbered overview pictures to out_dir/_sheets/, so wrong pictures are quick to spot.
    `only` = object names (empty = all); `first_number` skips pictures that were reviewed before."""
    from PIL import ImageDraw

    sheets_dir = Path(out_dir) / "_sheets"
    sheets_dir.mkdir(parents=True, exist_ok=True)
    cell_w, cell_h, label_h = 170, 150, 18
    for folder in sorted(p for p in Path(out_dir).iterdir() if p.is_dir() and not p.name.startswith("_")):
        if only and folder.name not in only:
            continue
        pictures = [(n, path) for n, path in numbered_pictures(folder) if n >= first_number]
        for old in sheets_dir.glob(f"{folder.name}_*.jpg"):
            old.unlink()
        for start in range(0, len(pictures), per_sheet):
            page = pictures[start:start + per_sheet]
            rows = -(-len(page) // columns)              # divide and round up
            sheet = Image.new("RGB", (columns * cell_w, rows * (cell_h + label_h)), "white")
            draw = ImageDraw.Draw(sheet)
            for i, (number, path) in enumerate(page):
                with Image.open(path) as im:
                    im = im.convert("RGB")
                    im.thumbnail((cell_w - 8, cell_h - 8))
                    x, y = (i % columns) * cell_w, (i // columns) * (cell_h + label_h)
                    sheet.paste(im, (x + (cell_w - im.width) // 2, y + (cell_h - im.height) // 2))
                    draw.text((x + 6, y + cell_h + 2), f"#{number}", fill="black")
            sheet_path = sheets_dir / f"{folder.name}_{start // per_sheet + 1:02d}.jpg"
            sheet.save(sheet_path, quality=88)
            print(sheet_path, f"({len(page)} pictures)")


def reject(out_dir, name, numbers):
    """Move the pictures with these numbers from out_dir/name/ to out_dir/_rejected/name/.
    Nothing is deleted, so a wrong rejection can be undone by moving the file back."""
    folder, rejected = Path(out_dir) / name, Path(out_dir) / "_rejected" / name
    rejected.mkdir(parents=True, exist_ok=True)
    path_of = dict(numbered_pictures(folder))
    moved = 0
    for number in numbers:
        if number in path_of:
            shutil.move(str(path_of[number]), str(rejected / path_of[number].name))
            moved += 1
    print(f"{name}: {moved} rejected, {len(numbered_pictures(folder))} kept")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    c = commands.add_parser("collect", help="search and download pictures for every object in the CSV")
    c.add_argument("queries_csv")
    c.add_argument("out_dir")
    c.add_argument("--target", type=int, default=250, help="fill every object folder up to this many pictures")
    c.add_argument("--only", nargs="*", default=[], help="collect only these objects")
    s = commands.add_parser("sheets", help="make numbered overview pictures in out_dir/_sheets/")
    s.add_argument("out_dir")
    s.add_argument("--only", nargs="*", default=[], help="only these objects")
    s.add_argument("--first", type=int, default=1, help="skip pictures with a smaller number")
    r = commands.add_parser("reject", help="move wrong pictures to out_dir/_rejected/")
    r.add_argument("out_dir")
    r.add_argument("object")
    r.add_argument("numbers", nargs="+", type=int)
    args = parser.parse_args()

    if args.command == "sheets":
        make_sheets(args.out_dir, args.only, args.first)
    elif args.command == "reject":
        reject(args.out_dir, args.object, args.numbers)
    else:
        for name, queries in read_queries(args.queries_csv).items():
            if args.only and name not in args.only:
                continue
            have = len(numbered_pictures(Path(args.out_dir) / name))
            if have >= args.target:
                print(f"{name}: already has {have} pictures", flush=True)
                continue
            print(f"{name}: has {have} pictures, collecting up to {args.target}", flush=True)
            have = collect_object(name, queries, args.out_dir, args.target)
            print(f"{name}: {have} pictures", flush=True)
        shutil.rmtree(Path(args.out_dir) / "_raw", ignore_errors=True)


if __name__ == "__main__":
    main()
