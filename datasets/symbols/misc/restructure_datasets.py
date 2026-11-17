from __future__ import annotations

import argparse
import hashlib
import shutil
import tempfile
from collections import Counter
from pathlib import Path


DatasetEntries = dict[str, list[tuple[str, Path, int]]]
SYMBOLS_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = SYMBOLS_DIR / "dataset_new"
DATASET_SOURCES = {
    "dataset1": ("dataset1",),
    "dataset2": ("dataset2",),
    "dataset3": ("dataset3",),
    "dataset4": ("dataset4", "dataset5", "dataset6", "dataset7"),
    "dataset8": ("dataset8",),
    "dataset9": ("dataset9",),
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
CLASS_LABELS = {0, 1, 2, 3}


def image_digest(image_path: Path) -> str:
    return hashlib.sha256(image_path.read_bytes()).hexdigest()


def remove_duplicate_images(
    datasets: DatasetEntries,
) -> tuple[DatasetEntries, dict[str, tuple[int, int]]]:
    earlier_images: dict[str, tuple[str, int]] = {}
    deduplicated: DatasetEntries = {}
    removed_counts: dict[str, tuple[int, int]] = {}

    for dataset_name, entries in datasets.items():
        unique_entries = []
        removed = 0
        label_conflicts = 0
        for source_name, image_path, label in entries:
            digest = image_digest(image_path)
            earlier = earlier_images.get(digest)
            if earlier is not None:
                removed += 1
                if earlier[1] != label:
                    label_conflicts += 1
                continue
            earlier_images[digest] = (dataset_name, label)
            unique_entries.append((source_name, image_path, label))
        deduplicated[dataset_name] = unique_entries
        removed_counts[dataset_name] = (removed, label_conflicts)

    assert_no_duplicate_images(deduplicated)
    return deduplicated, removed_counts


def assert_no_duplicate_images(datasets: DatasetEntries) -> None:
    seen_images: dict[str, str] = {}
    for dataset_name, entries in datasets.items():
        for _, image_path, _ in entries:
            digest = image_digest(image_path)
            earlier_dataset = seen_images.get(digest)
            if earlier_dataset is not None:
                raise AssertionError(
                    f"Duplicate image in {dataset_name}; it already appears in "
                    f"{earlier_dataset}: {image_path}"
                )
            seen_images[digest] = dataset_name


def read_source_images(dataset_name: str) -> list[tuple[Path, int]]:
    dataset_dir = SYMBOLS_DIR / dataset_name
    data_dir = dataset_dir / "data"
    labels_dir = dataset_dir / "gt"
    if not data_dir.is_dir() or not labels_dir.is_dir():
        raise FileNotFoundError(
            f"{dataset_name} must contain both 'data' and 'gt' directories"
        )

    image_paths = sorted(path for path in data_dir.iterdir() if path.is_file())
    unsupported_images = [
        path.name for path in image_paths if path.suffix.lower() not in IMAGE_EXTENSIONS
    ]
    if unsupported_images:
        raise ValueError(
            f"Unsupported files in {data_dir}: {', '.join(unsupported_images)}"
        )

    images_by_stem: dict[str, Path] = {}
    for image_path in image_paths:
        if image_path.stem in images_by_stem:
            raise ValueError(f"Duplicate image filename stem in {data_dir}: {image_path.stem}")
        images_by_stem[image_path.stem] = image_path

    label_paths = sorted(path for path in labels_dir.iterdir() if path.is_file())
    unsupported_labels = [path.name for path in label_paths if path.suffix.lower() != ".txt"]
    if unsupported_labels:
        raise ValueError(
            f"Unsupported files in {labels_dir}: {', '.join(unsupported_labels)}"
        )

    labels_by_stem: dict[str, Path] = {}
    for label_path in label_paths:
        if label_path.stem in labels_by_stem:
            raise ValueError(f"Duplicate label filename stem in {labels_dir}: {label_path.stem}")
        labels_by_stem[label_path.stem] = label_path

    missing_labels = sorted(images_by_stem.keys() - labels_by_stem.keys())
    missing_images = sorted(labels_by_stem.keys() - images_by_stem.keys())
    if missing_labels or missing_images:
        problems = []
        if missing_labels:
            problems.append(f"images without labels: {', '.join(missing_labels)}")
        if missing_images:
            problems.append(f"labels without images: {', '.join(missing_images)}")
        raise ValueError(f"Unpaired files in {dataset_name}: {'; '.join(problems)}")

    source_images = []
    for stem, image_path in images_by_stem.items():
        label_path = labels_by_stem[stem]
        try:
            label = int(label_path.read_text(encoding="utf-8").strip())
        except ValueError as error:
            raise ValueError(f"Invalid class label in {label_path}") from error
        if label not in CLASS_LABELS:
            raise ValueError(f"Class label must be between 0 and 3 in {label_path}")
        source_images.append((image_path, label))
    return source_images


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Restructure symbol datasets into class folders without changing sources."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the input datasets and show planned class counts without copying.",
    )
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help="Replace the existing dataset_new output after successfully staging the new one.",
    )
    args = parser.parse_args()

    datasets: DatasetEntries = {}
    for output_name, source_names in DATASET_SOURCES.items():
        entries = []
        for source_name in source_names:
            entries.extend(
                (source_name, image_path, label)
                for image_path, label in read_source_images(source_name)
            )
        datasets[output_name] = entries

    if not args.dry_run and OUTPUT_DIR.exists() and not args.replace_existing:
        raise FileExistsError(
            f"{OUTPUT_DIR} already exists. Use --replace-existing to replace it."
        )

    datasets, removed_counts = remove_duplicate_images(datasets)
    for output_name, entries in datasets.items():
        counts = Counter(label for _, _, label in entries)
        duplicates_removed, label_conflicts = removed_counts[output_name]
        print(
            f"{output_name}: {len(entries)} images "
            f"(class 0: {counts[0]}, class 1: {counts[1]}, "
            f"class 2: {counts[2]}, class 3: {counts[3]}); "
            f"removed {duplicates_removed} duplicates"
        )
        if label_conflicts:
            print(
                f"  {label_conflicts} duplicate(s) had a different class label; "
                "kept the earlier dataset's label."
            )

    if args.dry_run:
        print("Dry run complete; no files were copied.")
        return

    with tempfile.TemporaryDirectory(prefix=".dataset_new-", dir=SYMBOLS_DIR) as temporary_dir:
        staging_dir = Path(temporary_dir) / "dataset_new"
        for dataset_name, entries in datasets.items():
            is_merged = len(DATASET_SOURCES[dataset_name]) > 1
            for label in CLASS_LABELS:
                (staging_dir / dataset_name / str(label)).mkdir(parents=True, exist_ok=True)
            for source_name, image_path, label in entries:
                class_dir = staging_dir / dataset_name / str(label)
                filename = (
                    f"{source_name}_{image_path.name}" if is_merged else image_path.name
                )
                shutil.copy2(image_path, class_dir / filename)
        if OUTPUT_DIR.exists():
            backup_dir = Path(temporary_dir) / "previous_dataset_new"
            OUTPUT_DIR.rename(backup_dir)
            try:
                staging_dir.rename(OUTPUT_DIR)
            except OSError:
                backup_dir.rename(OUTPUT_DIR)
                raise
        else:
            staging_dir.rename(OUTPUT_DIR)

    print(f"Created {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
