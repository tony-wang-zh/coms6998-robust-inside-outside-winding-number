#!/usr/bin/env python3
"""
run_star.py  –  Full pipeline for the star-shaped input
========================================================
Runs all 3 steps in sequence for input_star.json

Usage:
    python run_star.py
"""

import subprocess
import sys
import os

def run(cmd):
    print(f"\n{'='*60}")
    print(f"  Running: {' '.join(cmd)}")
    print('='*60)
    result = subprocess.run([sys.executable] + cmd, check=True)
    return result

def main():
    base = os.path.dirname(os.path.abspath(__file__))
    os.chdir(base)

    # ── Step 1: Constrained Delaunay Triangulation ──
    run(["step1_cdt.py",
         "input_star.json",
         "star_step1_cdt.json"])

    # ── Step 2: Winding Number ──
    run(["step2_winding.py",
         "star_step1_cdt.json",
         "star_step2_winding.json"])

    # ── Step 3: Graph Cut Segmentation ──
    run(["step3_graphcut.py",
         "star_step2_winding.json",
         "star_step3_result.json",
         "5.0",
         "0.25"])

    print("\n" + "="*60)
    print("  Pipeline complete for STAR input!")
    print("  Output files:")
    for f in [
        "star_step1_cdt.json",        "star_step1_cdt_viz.png",
        "star_step2_winding.json",    "star_step2_winding_viz.png",
        "star_step3_result.json",     "star_step3_result_viz.png",
        "star_step3_result_comparison.png"
    ]:
        exists = "✓" if os.path.exists(f) else "✗ MISSING"
        print(f"    {exists}  {f}")
    print("="*60)

if __name__ == "__main__":
    main()
