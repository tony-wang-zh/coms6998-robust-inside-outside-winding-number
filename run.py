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

OUTPUT_DIR = "output"

def run(cmd):
    print(f"\n{'='*60}")
    print(f"  Running: {' '.join(cmd)}")
    print('='*60)
    result = subprocess.run([sys.executable] + cmd, check=True)
    return result

def main(input_path):
    filename = input_path.split("/")[-1]
    name = filename.split(".")[0]

    base = os.path.dirname(os.path.abspath(__file__))
    os.chdir(base)

    # ── Step 1: Constrained Delaunay Triangulation ──
    run(["step1_cdt.py",
         input_path,
         os.path.join(OUTPUT_DIR, f"{name}_step1_cdt.json")])

    # ── Step 2: Winding Number ──
    run(["step2_winding.py",
         os.path.join(OUTPUT_DIR, f"{name}_step1_cdt.json"),
         os.path.join(OUTPUT_DIR, f"{name}_step2_winding.json")])

    # ── Step 3: Graph Cut Segmentation ──
    run(["step3_graphcut.py",
         os.path.join(OUTPUT_DIR, f"{name}_step2_winding.json"),
         os.path.join(OUTPUT_DIR, f"{name}_step3_result.json"),
         "2.0",
         "0.5"])



if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run.py <input.json>")
        sys.exit(1)
    main(sys.argv[1])
