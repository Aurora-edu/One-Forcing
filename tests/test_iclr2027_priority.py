import subprocess
import sys
import unittest
from pathlib import Path

from experiments.rebuttal.run_iclr2027_priority import (
    CELLS,
    parse_checkpoint_assignments,
    selected_cells,
)


RUNNER = Path(__file__).resolve().parents[1] / "experiments/rebuttal/run_iclr2027_priority.py"


class ICLR2027PriorityTest(unittest.TestCase):
    def test_main_precedes_appendix_and_reuses_step_200(self):
        self.assertEqual([cell.name for cell in selected_cells("main")], [
            "full200_ffe", "dmd200_ffe",
        ])
        self.assertEqual([cell.name for cell in selected_cells("appendix")], [
            "full400_ffe", "full600_ffe",
        ])
        self.assertEqual(len(CELLS), 4)
        self.assertEqual({cell.schedule for cell in CELLS}, {"ffe"})

    def test_checkpoint_assignments_reject_unknown_or_duplicate_keys(self):
        self.assertEqual(set(parse_checkpoint_assignments(["full200=/tmp/model.pt"])), {"full200"})
        with self.assertRaisesRegex(ValueError, "Invalid or duplicate"):
            parse_checkpoint_assignments(["published=/tmp/model.pt"])
        with self.assertRaisesRegex(ValueError, "Invalid or duplicate"):
            parse_checkpoint_assignments(["curved300=/tmp/model.pt"])
        with self.assertRaisesRegex(ValueError, "Invalid or duplicate"):
            parse_checkpoint_assignments(["full200=/tmp/a", "full200=/tmp/b"])

    def test_plan_is_cpu_only_and_missing_inputs_fail_closed(self):
        planned = subprocess.run(
            [sys.executable, str(RUNNER), "--phase", "plan"],
            capture_output=True, text=True, check=True,
        )
        self.assertIn("Already aligned, omitted", planned.stdout)
        self.assertIn("User-excluded: trajectory rectification", planned.stdout)
        self.assertNotIn("curved300_all4", planned.stdout)
        self.assertLess(planned.stdout.index("full200_ffe"), planned.stdout.index("full400_ffe"))
        missing = subprocess.run(
            [sys.executable, str(RUNNER), "--phase", "main"],
            capture_output=True, text=True,
        )
        self.assertNotEqual(missing.returncode, 0)
        self.assertIn("Missing --checkpoint", missing.stderr)
        combined = subprocess.run(
            [sys.executable, str(RUNNER), "--phase", "all"],
            capture_output=True, text=True,
        )
        self.assertNotEqual(combined.returncode, 0)
        self.assertIn("invalid choice", combined.stderr)


if __name__ == "__main__":
    unittest.main()
