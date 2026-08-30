"""Shared setup so each analysis script runs from the repo root."""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
IMAGE = os.path.join(ROOT, "marshmallows.png")
TRUTH = 506   # careful hand count; see README caveat about anchoring
