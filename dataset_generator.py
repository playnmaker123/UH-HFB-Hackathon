"""Generate four specialized synthetic YOLO datasets for SmartPack.

Expected input layout (relative to this script):

    dataset-items/
      boxes/
      hands/<batch>/<frame>.png
      hands/mask/mask/<batch>/<frame>.png
      items/<food-category>/*

Output layout:

    dataset/<food-category>/
      train|val|test/images/*.jpg
      train|val|test/labels/*.txt
      data.yaml

Each generated scene contains one box, one hand/arm, and normally one target
food item. Every category is a separate one-class YOLO dataset whose only class
is ``item``. Boxes and hands are context and are not labeled.

Requirements:
    python -m pip install pillow numpy

Example:
    python dataset_generator.py --count-per-model 3000
"""

from __future__ import annotations

import argparse
import hashlib
import os
import random
import shutil
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


PROJECT_DIR = Path(__file__).resolve().parent


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
SPLITS = (("train", 0.70), ("val", 0.20), ("test", 0.10))
RESAMPLE = Image.Resampling.LANCZOS
ASSET_CACHE_VERSION = "v1"


@dataclass(frozen=True)
class Asset:
    image: Image.Image
    label: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate specialized synthetic YOLO datasets.")
    parser.add_argument(
        "--assets-dir", type=Path, default=PROJECT_DIR / "dataset-items"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_DIR / "dataset"
    )
    parser.add_argument(
        "--count-per-model",
        type=int,
        default=3_000,
        help="Total images generated for each specialized model.",
    )
    parser.add_argument("--count", type=int, default=None, help="Alias for --count-per-model.")
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, (os.cpu_count() or 2) - 1)),
        help="Parallel image-generation workers.",
    )
    parser.add_argument(
        "--hand-cache",
        type=int,
        default=100,
        help="Number of CHOC hand/mask pairs prepared in RAM.",
    )
    parser.add_argument(
        "--hand-probability",
        type=float,
        default=1.0,
        help="Deprecated compatibility flag; one hand is always used.",
    )
    parser.add_argument(
        "--positive-probability",
        type=float,
        default=1.0,
        help="Probability that the target item appears; negatives contain an empty hand.",
    )
    parser.add_argument(
        "--track-padding",
        type=float,
        default=0.15,
        help="Context padding around the hand/item tracking crop.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Deprecated compatibility flag.")
    parser.add_argument("--no-asset-cache", action="store_true")
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Generate 30 images per category for visual review.",
    )
    return parser.parse_args()


def list_images(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def white_background_alpha(image: Image.Image) -> Image.Image:
    """Remove only nearly-white background pixels from an opaque asset."""

    rgba = image.convert("RGBA")
    existing_alpha = np.asarray(rgba.getchannel("A"), dtype=np.uint8)
    if existing_alpha.min() < 250:
        return rgba

    rgb = np.asarray(rgba.convert("RGB"), dtype=np.uint8)
    channel_range = rgb.max(axis=2).astype(np.int16) - rgb.min(axis=2).astype(np.int16)
    near_white = (rgb.min(axis=2) >= 242) & (channel_range <= 18)
    alpha = np.where(near_white, 0, 255).astype(np.uint8)
    alpha_image = Image.fromarray(alpha, mode="L").filter(ImageFilter.GaussianBlur(0.6))
    rgba.putalpha(alpha_image)
    return rgba


def trim_transparent_margins(image: Image.Image, padding: int = 3) -> Image.Image:
    alpha_box = image.getchannel("A").getbbox()
    if alpha_box is None:
        raise ValueError("asset has no visible pixels after background processing")
    left, top, right, bottom = alpha_box
    return image.crop(
        (
            max(0, left - padding),
            max(0, top - padding),
            min(image.width, right + padding),
            min(image.height, bottom + padding),
        )
    )


def largest_component_python(foreground: np.ndarray) -> np.ndarray:
    height, width = foreground.shape
    visited = np.zeros_like(foreground, dtype=bool)
    largest: list[tuple[int, int]] = []
    for y in range(height):
        for x in range(width):
            if not foreground[y, x] or visited[y, x]:
                continue
            component: list[tuple[int, int]] = []
            queue: deque[tuple[int, int]] = deque([(x, y)])
            visited[y, x] = True
            while queue:
                cx, cy = queue.popleft()
                component.append((cx, cy))
                for ny in range(max(0, cy - 1), min(height, cy + 2)):
                    for nx in range(max(0, cx - 1), min(width, cx + 2)):
                        if foreground[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            queue.append((nx, ny))
            if len(component) > len(largest):
                largest = component
    output = np.zeros_like(foreground, dtype=np.uint8)
    for x, y in largest:
        output[y, x] = 255
    return output


def keep_largest_foreground_component(image: Image.Image) -> Image.Image:
    """Remove disconnected specks left by imperfect background removal."""

    alpha = np.asarray(image.getchannel("A"), dtype=np.uint8)
    height, width = alpha.shape
    if not alpha.any():
        return image

    scale = min(1.0, 128 / max(height, width))
    probe_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    probe = Image.fromarray(alpha, mode="L").resize(probe_size, Image.Resampling.BILINEAR)
    foreground = np.asarray(probe, dtype=np.uint8) > 24

    try:
        import cv2
    except ImportError:
        cv2 = None

    if cv2 is not None:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            foreground.astype(np.uint8), connectivity=8
        )
        if count <= 1:
            return image
        largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        keep_probe = np.where(labels == largest_label, 255, 0).astype(np.uint8)
    else:
        keep_probe = largest_component_python(foreground)

    keep = Image.fromarray(keep_probe, mode="L").resize((width, height), Image.Resampling.NEAREST)
    cleaned_alpha = np.minimum(alpha, np.asarray(keep, dtype=np.uint8))
    cleaned = Image.fromarray(cleaned_alpha, mode="L")
    cleaned = cleaned.filter(ImageFilter.MinFilter(3)).filter(ImageFilter.MaxFilter(3))
    result = image.copy()
    result.putalpha(cleaned)
    return result


def cache_path_for(source_path: Path, cache_dir: Path) -> Path:
    key = hashlib.sha1(
        f"{ASSET_CACHE_VERSION}|{source_path.resolve().as_posix()}".encode("utf-8")
    ).hexdigest()
    return cache_dir / f"{key}.png"


def load_photo_asset(path: Path, cache_dir: Path | None = None) -> Asset:
    prepared_path = cache_path_for(path, cache_dir) if cache_dir else None
    if prepared_path and prepared_path.is_file():
        try:
            if prepared_path.stat().st_mtime_ns >= path.stat().st_mtime_ns:
                with Image.open(prepared_path) as source:
                    cached = source.convert("RGBA")
                    cached.load()
                return Asset(cached)
        except OSError:
            pass

    with Image.open(path) as source:
        if source.format == "JPEG":
            source.draft("RGB", (1024, 1024))
        image = ImageOps.exif_transpose(source).convert("RGBA")
    if max(image.size) > 1024:
        image.thumbnail((1024, 1024), RESAMPLE)
    image = white_background_alpha(image)
    image = keep_largest_foreground_component(image)
    image = trim_transparent_margins(image)

    if prepared_path:
        prepared_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = prepared_path.with_suffix(".tmp.png")
        image.save(temporary, format="PNG", compress_level=1)
        temporary.replace(prepared_path)
    return Asset(image)


def load_hand_asset(image_path: Path, mask_path: Path) -> Asset:
    with Image.open(image_path) as source, Image.open(mask_path) as mask_source:
        image = ImageOps.exif_transpose(source).convert("RGBA")
        mask = ImageOps.exif_transpose(mask_source).convert("L")
    if mask.size != image.size:
        mask = mask.resize(image.size, Image.Resampling.NEAREST)

    mask_array = np.asarray(mask, dtype=np.uint8)
    values, counts = np.unique(mask_array[mask_array > 0], return_counts=True)
    if len(values) == 0:
        raise ValueError("hand mask contains no foreground instance")
    selected = int(values[int(np.argmax(counts))])
    alpha = Image.fromarray(
        np.where(mask_array == selected, 255, 0).astype(np.uint8), mode="L"
    )
    image.putalpha(alpha)
    return Asset(trim_transparent_margins(image, padding=1))


def resize_max_dimension(image: Image.Image, target: int) -> Image.Image:
    scale = target / max(image.width, image.height)
    width = max(1, round(image.width * scale))
    height = max(1, round(image.height * scale))
    return image.resize((width, height), RESAMPLE)


def transform_asset(
    image: Image.Image,
    rng: random.Random,
    max_dimension: int,
    rotation_degrees: float,
    horizontal_flip_probability: float,
    vertical_flip_probability: float,
) -> Image.Image:
    image = resize_max_dimension(image, max_dimension)
    if rng.random() < horizontal_flip_probability:
        image = ImageOps.mirror(image)
    if rng.random() < vertical_flip_probability:
        image = ImageOps.flip(image)
    image = image.rotate(
        rng.uniform(-rotation_degrees, rotation_degrees),
        resample=Image.Resampling.BICUBIC,
        expand=True,
    )
    image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.75, 1.25))
    return ImageEnhance.Contrast(image).enhance(rng.uniform(0.75, 1.25))


def alpha_bbox(image: Image.Image) -> tuple[int, int, int, int] | None:
    return image.getchannel("A").getbbox()


def placed_bbox(image: Image.Image, x: int, y: int) -> tuple[int, int, int, int] | None:
    box = alpha_bbox(image)
    if box is None:
        return None
    return x + box[0], y + box[1], x + box[2], y + box[3]


def overlap_fraction(
    reference: tuple[int, int, int, int], overlay: tuple[int, int, int, int]
) -> float:
    left, top = max(reference[0], overlay[0]), max(reference[1], overlay[1])
    right, bottom = min(reference[2], overlay[2]), min(reference[3], overlay[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    reference_area = max(1, (reference[2] - reference[0]) * (reference[3] - reference[1]))
    return intersection / reference_area


def paste_asset(
    canvas: Image.Image, image: Image.Image, x: int, y: int
) -> tuple[int, int, int, int] | None:
    canvas.alpha_composite(image, (x, y))
    box = alpha_bbox(image)
    if box is None:
        return None
    left, top = max(0, x + box[0]), max(0, y + box[1])
    right, bottom = min(canvas.width, x + box[2]), min(canvas.height, y + box[3])
    return (left, top, right, bottom) if right > left and bottom > top else None


def yolo_line(class_id: int, box: tuple[int, int, int, int], size: int) -> str:
    left, top, right, bottom = box
    return (
        f"{class_id} {((left + right) / 2) / size:.6f} "
        f"{((top + bottom) / 2) / size:.6f} "
        f"{(right - left) / size:.6f} {(bottom - top) / size:.6f}"
    )


def crop_tracking_context(
    canvas: Image.Image,
    hand_box: tuple[int, int, int, int] | None,
    item_box: tuple[int, int, int, int] | None,
    output_size: int,
    padding_fraction: float,
) -> tuple[Image.Image, tuple[int, int, int, int] | None]:
    tracked = [box for box in (hand_box, item_box) if box is not None]
    if not tracked:
        return canvas, item_box

    left = min(box[0] for box in tracked)
    top = min(box[1] for box in tracked)
    right = max(box[2] for box in tracked)
    bottom = max(box[3] for box in tracked)
    tracked_width, tracked_height = max(1, right - left), max(1, bottom - top)
    side = min(
        canvas.width,
        canvas.height,
        max(2, round(max(tracked_width, tracked_height) * (1 + 2 * padding_fraction))),
    )
    center_x, center_y = (left + right) / 2, (top + bottom) / 2
    crop_left = max(0, min(canvas.width - side, round(center_x - side / 2)))
    crop_top = max(0, min(canvas.height - side, round(center_y - side / 2)))
    cropped = canvas.crop((crop_left, crop_top, crop_left + side, crop_top + side))
    cropped = cropped.resize((output_size, output_size), RESAMPLE)
    if item_box is None:
        return cropped, None

    scale = output_size / side
    transformed = (
        round(max(0, item_box[0] - crop_left) * scale),
        round(max(0, item_box[1] - crop_top) * scale),
        round(min(side, item_box[2] - crop_left) * scale),
        round(min(side, item_box[3] - crop_top) * scale),
    )
    return cropped, transformed


def make_background(size: int, rng: random.Random) -> Image.Image:
    colors = (
        (126, 119, 105),
        (143, 134, 115),
        (109, 113, 110),
        (151, 143, 128),
        (105, 125, 132),
        (119, 132, 112),
        (137, 115, 126),
        (101, 118, 140),
    )
    base = np.array(rng.choice(colors), dtype=np.int16)
    noise = np.random.default_rng(rng.randrange(2**32)).normal(0, 4, (size, size, 1))
    pixels = np.clip(base + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(pixels, mode="RGB").convert("RGBA")


def final_camera_augmentation(image: Image.Image, rng: random.Random) -> Image.Image:
    image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.75, 1.25))
    image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.75, 1.25))
    image = ImageEnhance.Color(image).enhance(rng.uniform(0.75, 1.25))
    if rng.random() < 0.55:
        image = image.filter(ImageFilter.GaussianBlur(rng.uniform(0.30, 2.0)))
    return image


def get_food_assets(items_dir: Path, cache_dir: Path | None) -> dict[str, list[Asset]]:
    if not items_dir.is_dir():
        raise FileNotFoundError(f"Missing items folder: {items_dir}")
    result: dict[str, list[Asset]] = {}
    for category in sorted(path for path in items_dir.iterdir() if path.is_dir()):
        assets: list[Asset] = []
        for path in list_images(category):
            try:
                assets.append(Asset(load_photo_asset(path, cache_dir).image, category.name))
            except (OSError, ValueError) as exc:
                print(f"WARN  Skipping {path}: {exc}")
        if assets:
            result[category.name] = assets
    if not result:
        raise RuntimeError(f"No usable food assets found in {items_dir}")
    return result


def get_hand_assets(hands_dir: Path, cache_size: int, rng: random.Random) -> list[Asset]:
    mask_root = hands_dir / "mask" / "mask"
    if not hands_dir.is_dir() or not mask_root.is_dir() or cache_size <= 0:
        return []
    frames = [
        path
        for batch in hands_dir.iterdir()
        if batch.is_dir() and batch.name != "mask"
        for path in batch.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    candidates = [path for path in frames if (mask_root / path.relative_to(hands_dir)).is_file()]
    assets: list[Asset] = []
    for path in rng.sample(candidates, k=min(cache_size, len(candidates))):
        try:
            assets.append(load_hand_asset(path, mask_root / path.relative_to(hands_dir)))
        except (OSError, ValueError) as exc:
            print(f"WARN  Skipping hand {path}: {exc}")
    return assets


def ensure_output(output_dir: Path, categories: Iterable[str]) -> None:
    old_layout = [output_dir / split for split, _ in SPLITS]
    generated = [
        output_dir / category / split / part
        for category in categories
        for split, _ in SPLITS
        for part in ("images", "labels")
    ]
    removed = 0
    for directory in old_layout + generated:
        if directory.exists():
            removed += sum(1 for _ in directory.rglob("*"))
            shutil.rmtree(directory)
    if removed:
        print(f"Removed {removed} files/folders from previous generated output.")
    for directory in generated:
        directory.mkdir(parents=True, exist_ok=True)


def write_data_yaml(category_output: Path) -> None:
    text = (
        f"path: {category_output.resolve().as_posix()}\n"
        "train: train/images\n"
        "val: val/images\n"
        "test: test/images\n\n"
        "nc: 1\n"
        "names: {0: item}\n"
    )
    (category_output / "data.yaml").write_text(text, encoding="utf-8")


def split_counts(total: int) -> dict[str, int]:
    train = round(total * 0.70)
    validation = round(total * 0.20)
    return {"train": train, "val": validation, "test": total - train - validation}


def build_scene(
    *,
    size: int,
    boxes: list[Asset],
    food_assets: dict[str, list[Asset]],
    hand_assets: list[Asset],
    target_category: str,
    positive_probability: float,
    track_padding: float,
    rng: random.Random,
) -> tuple[Image.Image, list[str]]:
    canvas = make_background(size, rng)
    labels: list[str] = []

    box = transform_asset(
        rng.choice(boxes).image,
        rng,
        rng.randint(int(size * 0.76), int(size * 0.94)),
        6,
        0.50,
        0.50,
    )
    box_x = (size - box.width) // 2 + rng.randint(-int(size * 0.05), int(size * 0.05))
    box_y = (size - box.height) // 2 + rng.randint(-int(size * 0.05), int(size * 0.05))
    paste_asset(canvas, box, box_x, box_y)

    if rng.random() >= positive_probability:
        hand = transform_asset(
            rng.choice(hand_assets).image,
            rng,
            rng.randint(int(size * 0.20), int(size * 0.42)),
            20,
            0.50,
            0.0,
        )
        hand_x = (size - hand.width) // 2 + rng.randint(-int(size * 0.15), int(size * 0.15))
        hand_y = (size - hand.height) // 2 + rng.randint(-int(size * 0.15), int(size * 0.15))
        hand_box = paste_asset(canvas, hand, hand_x, hand_y)
        cropped, _ = crop_tracking_context(canvas, hand_box, None, size, track_padding)
        return final_camera_augmentation(cropped, rng).convert("RGB"), labels

    item = transform_asset(
        rng.choice(food_assets[target_category]).image,
        rng,
        rng.randint(int(size * 0.20), int(size * 0.45)),
        25,
        0.50,
        0.50,
    )
    item_scale = rng.uniform(0.35, 0.70)
    item = item.resize(
        (max(1, round(item.width * item_scale)), max(1, round(item.height * item_scale))),
        RESAMPLE,
    )
    item_x = int(size * rng.uniform(0.30, 0.70) - item.width / 2)
    item_y = int(size * rng.uniform(0.33, 0.67) - item.height / 2)
    item_x = max(0, min(size - item.width, item_x))
    item_y = max(0, min(size - item.height, item_y))

    hand = transform_asset(
        rng.choice(hand_assets).image,
        rng,
        rng.randint(max(int(item.width * 1.1), 80), max(int(item.width * 2.0), 130)),
        20,
        0.50,
        0.0,
    )
    item_reference = placed_bbox(item, item_x, item_y)
    if item_reference is None:
        return final_camera_augmentation(canvas, rng).convert("RGB"), labels

    best_position: tuple[int, int] | None = None
    best_score = float("inf")
    desired_overlap = rng.uniform(0.05, 0.35)
    for _ in range(40):
        x = item_x + rng.randint(-int(hand.width * 0.55), int(item.width * 0.30))
        y = item_y + rng.randint(-int(hand.height * 0.45), int(item.width * 0.30))
        x = max(-hand.width // 3, min(size - hand.width * 2 // 3, x))
        y = max(-hand.height // 3, min(size - hand.height * 2 // 3, y))
        candidate = placed_bbox(hand, x, y)
        if candidate is None:
            continue
        amount = overlap_fraction(item_reference, candidate)
        if amount <= 0.40:
            best_position = (x, y)
            break
        score = abs(amount - desired_overlap)
        if score < best_score:
            best_score, best_position = score, (x, y)
    hand_x, hand_y = best_position or (item_x, item_y)

    if rng.random() < 0.50:
        hand_box = paste_asset(canvas, hand, hand_x, hand_y)
        item_box = paste_asset(canvas, item, item_x, item_y)
    else:
        item_box = paste_asset(canvas, item, item_x, item_y)
        hand_box = paste_asset(canvas, hand, hand_x, hand_y)

    cropped, transformed_item = crop_tracking_context(
        canvas, hand_box, item_box, size, track_padding
    )
    if transformed_item:
        labels.append(yolo_line(0, transformed_item, size))
    return final_camera_augmentation(cropped, rng).convert("RGB"), labels


def generate_scene_file(
    *,
    seed: int,
    index: int,
    split: str,
    category: str,
    category_output: Path,
    size: int,
    boxes: list[Asset],
    food_assets: dict[str, list[Asset]],
    hand_assets: list[Asset],
    positive_probability: float,
    track_padding: float,
) -> int:
    rng = random.Random(seed)
    image, labels = build_scene(
        size=size,
        boxes=boxes,
        food_assets=food_assets,
        hand_assets=hand_assets,
        target_category=category,
        positive_probability=positive_probability,
        track_padding=track_padding,
        rng=rng,
    )
    stem = f"{category}_{index:06d}"
    image.save(
        category_output / split / "images" / f"{stem}.jpg",
        quality=rng.randint(84, 95),
        subsampling=0,
    )
    (category_output / split / "labels" / f"{stem}.txt").write_text(
        "\n".join(labels) + ("\n" if labels else ""), encoding="utf-8"
    )
    return index


def main() -> int:
    started = time.perf_counter()
    args = parse_args()
    count_per_model = args.count if args.count is not None else args.count_per_model
    if count_per_model <= 0 or args.image_size < 128:
        raise ValueError("count must be positive and image size must be at least 128")
    if not 0 <= args.positive_probability <= 1:
        raise ValueError("--positive-probability must be between 0 and 1")
    if not 0 <= args.track_padding <= 1:
        raise ValueError("--track-padding must be between 0 and 1")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")

    rng = random.Random(args.seed)
    assets_dir = args.assets_dir.resolve()
    output_dir = args.output_dir.resolve()
    cache_dir = None if args.no_asset_cache else assets_dir / ".prepared-cache"

    boxes = [load_photo_asset(path, cache_dir) for path in list_images(assets_dir / "boxes")]
    if not boxes:
        raise FileNotFoundError(f"No usable box assets found in {assets_dir / 'boxes'}")
    food_assets = get_food_assets(assets_dir / "items", cache_dir)
    hands = get_hand_assets(assets_dir / "hands", args.hand_cache, rng)
    if not hands:
        raise FileNotFoundError(f"No usable CHOC hand/mask pairs found in {assets_dir / 'hands'}")

    categories = sorted(food_assets)
    total = 30 if args.preview else count_per_model
    counts = split_counts(total)
    asset_elapsed = time.perf_counter() - started
    ensure_output(output_dir, categories)

    print(
        f"Prepared {len(boxes)} boxes, {len(hands)} hands, and "
        f"{sum(len(group) for group in food_assets.values())} food assets in RAM."
    )
    print(f"Asset preparation time: {asset_elapsed:.1f} seconds")
    if cache_dir:
        print(f"Prepared-asset cache: {cache_dir}")
    print(f"Parallel generation workers: {args.workers}")
    print(f"Tracking-crop context padding: {args.track_padding:.0%}")

    generation_started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for category in categories:
            category_output = output_dir / category
            write_data_yaml(category_output)
            print(f"Creating {total} images for {category}: {counts}")
            for split, _ in SPLITS:
                print(f"Writing pictures to: {category_output / split / 'images'}")

            futures = []
            index = 0
            for split, _ in SPLITS:
                for _ in range(counts[split]):
                    futures.append(
                        executor.submit(
                            generate_scene_file,
                            seed=rng.getrandbits(64),
                            index=index,
                            split=split,
                            category=category,
                            category_output=category_output,
                            size=args.image_size,
                            boxes=boxes,
                            food_assets=food_assets,
                            hand_assets=hands,
                            positive_probability=args.positive_probability,
                            track_padding=args.track_padding,
                        )
                    )
                    index += 1

            completed = 0
            for future in as_completed(futures):
                future.result()
                completed += 1
                if completed % 100 == 0 or completed == total:
                    print(f"{category}: created {completed}/{total}")

    generation_elapsed = time.perf_counter() - generation_started
    elapsed = time.perf_counter() - started
    hours, remainder = divmod(int(round(elapsed)), 3600)
    minutes, seconds = divmod(remainder, 60)
    print(
        f"Done. Generated {len(categories) * total:,} images in "
        f"{hours:02d}:{minutes:02d}:{seconds:02d}."
    )
    print(f"Generation/write phase: {generation_elapsed:.1f} seconds")
    print(
        "Example training command: "
        f"yolo detect train data={output_dir / categories[0] / 'data.yaml'} model=yolo11n.pt"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
