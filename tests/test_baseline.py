"""Regression check against the committed photo and Cellpose centroids."""
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


class BaselineTest(unittest.TestCase):
    def test_classical_and_committed_cellpose_agreement(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "analysis/agreement.py")],
            cwd=ROOT,
            env={**os.environ, "MPLBACKEND": "Agg"},
            text=True,
            capture_output=True,
            check=True,
            timeout=30,
        )
        self.assertIn("classical (t=175, min_distance=14): 506", result.stdout)
        self.assertIn("matched within 15px: 498", result.stdout)
        self.assertIn("16 individual disagreements", result.stdout)
