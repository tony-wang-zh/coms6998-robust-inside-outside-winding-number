#!/usr/bin/env python3
"""
run_cshape.py  –  Full pipeline for the C-shape (horseshoe) input
==================================================================
Runs all 3 steps in sequence for input_cshape.json

Usage:
    python run_cshape.py
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
         "input_cshape.json",
         "cshape_step1_cdt.json"])

    # ── Step 2: Winding Number ──
    run(["step2_winding.py",
         "cshape_step1_cdt.json",
         "cshape_step2_winding.json"])

    # ── Step 3: Graph Cut Segmentation ──
    #   Use a higher gamma for a more aggressive smoothness penalty on this shape
    run(["step3_graphcut.py",
         "cshape_step2_winding.json",
         "cshape_step3_result.json",
         "8.0",
         "0.3"])

    print("\n" + "="*60)
    print("  Pipeline complete for C-SHAPE input!")
    print("  Output files:")
    for f in [
        "cshape_step1_cdt.json",        "cshape_step1_cdt_viz.png",
        "cshape_step2_winding.json",    "cshape_step2_winding_viz.png",
        "cshape_step3_result.json",     "cshape_step3_result_viz.png",
        "cshape_step3_result_comparison.png"
    ]:
        exists = "✓" if os.path.exists(f) else "✗ MISSING"
        print(f"    {exists}  {f}")
    print("="*60)

if __name__ == "__main__":
    main()
