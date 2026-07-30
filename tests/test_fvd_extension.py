import tempfile
import unittest
from pathlib import Path

from experiments.rebuttal.decode_lmdb_references import (
    initialize_or_validate_intent,
    select_extension_indices,
    validate_existing_manifests,
)
from experiments.rebuttal.evaluate_fvd import list_videos


class FvdReferenceExtensionTests(unittest.TestCase):
    def test_extension_is_deterministic_and_disjoint(self):
        existing = [1, 4, 8, 12]
        first = select_extension_indices(32, existing, 8, seed=7)
        second = select_extension_indices(32, existing, 8, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(len(first), len(set(first)))
        self.assertTrue(set(first).isdisjoint(existing))

    def test_existing_manifests_are_paired_by_order_and_prompt(self):
        references = [
            {
                "order": index,
                "dataset_index": index + 100,
                "output_name": f"real_{index:04d}.mp4",
                "prompt": f"prompt {index}",
            }
            for index in range(3)
        ]
        generations = [
            {
                "prompt_index": index,
                "sample_index": 0,
                "seed": index,
                "output_name": f"fake_{index:04d}.mp4",
                "prompt": f"prompt {index}",
            }
            for index in range(3)
        ]
        self.assertEqual(
            validate_existing_manifests(references, generations),
            [100, 101, 102],
        )
        generations[1]["prompt"] = "wrong"
        with self.assertRaisesRegex(ValueError, "disagree at order 1"):
            validate_existing_manifests(references, generations)

    def test_combined_directories_sort_globally_and_reject_duplicate_names(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = root / "old"
            new = root / "new"
            old.mkdir()
            new.mkdir()
            (old / "real_0000.mp4").write_bytes(b"")
            (old / "real_0001.mp4").write_bytes(b"")
            (new / "real_0002.mp4").write_bytes(b"")
            paths = list_videos([old, new], limit=-1)
            self.assertEqual(
                [path.name for path in paths],
                ["real_0000.mp4", "real_0001.mp4", "real_0002.mp4"],
            )
            (new / "real_0001.mp4").write_bytes(b"")
            with self.assertRaisesRegex(ValueError, "Duplicate video filename"):
                list_videos([old, new], limit=-1)

    def test_reference_intent_makes_resume_exact(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            intent = {"selected_dataset_indices": [3, 7], "seed": 0}
            initialize_or_validate_intent(output, intent)
            initialize_or_validate_intent(output, intent)
            with self.assertRaisesRegex(ValueError, "different reference export"):
                initialize_or_validate_intent(
                    output,
                    {"selected_dataset_indices": [4, 8], "seed": 0},
                )

if __name__ == "__main__":
    unittest.main()
