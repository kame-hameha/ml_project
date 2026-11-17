import tempfile
import unittest
from pathlib import Path

from restructure_datasets import assert_no_duplicate_images, remove_duplicate_images


class RemoveDuplicateImagesTests(unittest.TestCase):
    def test_removes_images_seen_in_earlier_datasets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            first_image = root / "first.png"
            duplicate_image = root / "duplicate.png"
            unique_image = root / "unique.png"
            first_image.write_bytes(b"same image")
            duplicate_image.write_bytes(b"same image")
            unique_image.write_bytes(b"different image")

            datasets = {
                "dataset1": [("dataset1", first_image, 1)],
                "dataset2": [
                    ("dataset2", duplicate_image, 2),
                    ("dataset2", unique_image, 3),
                ],
            }

            result, removed_counts = remove_duplicate_images(datasets)

            self.assertEqual(len(result["dataset1"]), 1)
            self.assertEqual(result["dataset2"], [("dataset2", unique_image, 3)])
            self.assertEqual(removed_counts["dataset2"], (1, 1))
            assert_no_duplicate_images(result)

    def test_duplicate_assertion_rejects_repeated_image_content(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            first_image = root / "first.png"
            second_image = root / "second.png"
            first_image.write_bytes(b"same image")
            second_image.write_bytes(b"same image")

            with self.assertRaisesRegex(AssertionError, "already appears in dataset1"):
                assert_no_duplicate_images(
                    {
                        "dataset1": [("dataset1", first_image, 1)],
                        "dataset2": [("dataset2", second_image, 1)],
                    }
                )


if __name__ == "__main__":
    unittest.main()
