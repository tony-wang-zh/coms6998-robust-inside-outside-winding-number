"""
Step 1 (Alternative): Constrained Delaunay Triangulation via Shapely
=====================================================================
Drop-in replacement for step1_cdt.py using Shapely's built-in
`constrained_delaunay_triangles` function instead of the manual
Steiner-point insertion loop.

Install Shapely (requires GEOS >= 3.10.0):
    pip install shapely>=2.1.0

Key difference from step1_cdt.py:
  - step1_cdt.py:  manually detects crossing edges, inserts Steiner
                   points one at a time, and re-runs scipy Delaunay
                   from scratch on each iteration.
  - This script:   hands the constraint segments directly to Shapely
                   as a Polygon / MultiPolygon and lets GEOS handle
                   the full CDT in a single call.  The result is
                   guaranteed to have every constraint segment present
                   as a triangle edge — no manual iteration needed.

Input / Output / Visualization format: identical to step1_cdt.py,
so downstream step2_winding.py and step3_graphcut.py work unchanged.

Usage:
    python step1_cdt_alternative.py <input.json> <output.json>
"""

import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import shapely
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
import shapely as shp


# ── segment soup → Shapely geometry ──────────────────────────────────────────

def build_rings(vertices, segments):
    """
    Reconstruct ordered rings from an unordered list of segments.

    segments is a list of [i, j] index pairs.  We follow the chain of
    connectivity to produce one or more closed loops (rings), each as an
    ordered list of vertex indices.

    Returns: list of rings, where each ring is a list of (x, y) tuples
             in order (the first and last point are the same, closing the ring).
    """
    # Build adjacency for each vertex
    adj = {}
    for cs, ct in segments:
        adj.setdefault(cs, []).append(ct)
        adj.setdefault(ct, []).append(cs)

    visited_edges = set()
    rings = []

    for start in list(adj.keys()):
        # Try to start a new ring from this vertex if there are unvisited edges
        for nxt in adj[start]:
            edge = frozenset([start, nxt])
            if edge in visited_edges:
                continue

            # Walk the ring
            ring_indices = [start]
            prev, curr = start, nxt
            while True:
                visited_edges.add(frozenset([prev, curr]))
                ring_indices.append(curr)
                if curr == start:
                    break
                # Pick the next unvisited neighbour that isn't where we came from
                candidates = [v for v in adj[curr]
                              if frozenset([curr, v]) not in visited_edges]
                if not candidates:
                    break
                prev, curr = curr, candidates[0]

            if len(ring_indices) >= 4 and ring_indices[-1] == ring_indices[0]:
                coords = [tuple(vertices[i]) for i in ring_indices]
                rings.append(coords)

    return rings


def rings_to_shapely(rings):
    """
    Convert a list of rings into a Shapely Polygon or MultiPolygon.

    Shapely's constrained_delaunay_triangles requires polygonal input.
    When there are multiple rings, the first (largest-area) ring becomes
    the exterior shell and the remaining rings become holes, OR if they
    are disjoint they become separate polygons in a MultiPolygon.
    """
    if not rings:
        raise ValueError("No closed rings found in segment list.")

    # Compute signed area to detect orientation
    def signed_area(coords):
        n = len(coords) - 1  # last == first
        return sum(
            (coords[i][0] * coords[(i+1) % n][1] -
             coords[(i+1) % n][0] * coords[i][1])
            for i in range(n)
        ) / 2.0

    areas = [abs(signed_area(r)) for r in rings]

    if len(rings) == 1:
        return Polygon(rings[0])

    # Sort largest first
    sorted_rings = [r for _, r in sorted(zip(areas, rings), reverse=True)]

    # Build Shapely polygons per ring and test containment
    from shapely.geometry import Polygon as SPoly
    polys = [SPoly(r) for r in sorted_rings]

    # Try to form a single polygon with holes (if inner rings are inside outer)
    exterior = sorted_rings[0]
    holes = []
    extra_polys = []
    for i, (ring, poly) in enumerate(zip(sorted_rings[1:], polys[1:]), 1):
        if polys[0].contains(poly):
            holes.append(ring)
        else:
            extra_polys.append(ring)

    if not extra_polys:
        # All inner rings are holes of the outer ring
        return Polygon(exterior, holes)
    else:
        # Disjoint components → MultiPolygon
        all_polys = [Polygon(exterior, holes)] + [Polygon(r) for r in extra_polys]
        return MultiPolygon(all_polys)


# ── triangulation result → index form ────────────────────────────────────────

def shapely_result_to_indexed(geom_collection, tol=1e-10):
    """
    Convert a GeometryCollection of triangle Polygons into:
      - pts:       list of [x, y]  (deduplicated vertex list)
      - triangles: list of [i, j, k] index triples into pts

    Shapely returns triangle coordinates as floats; we snap near-identical
    points together (within `tol`) to build a clean shared vertex list.
    """
    pts = []        # list of [x, y]
    pt_index = {}   # tuple(rounded) -> index
    triangles = []

    def get_or_add(x, y):
        key = (round(x / tol) * tol, round(y / tol) * tol)
        if key not in pt_index:
            pt_index[key] = len(pts)
            pts.append([x, y])
        return pt_index[key]

    for geom in geom_collection.geoms:
        coords = list(geom.exterior.coords)
        # coords has 4 entries: v0, v1, v2, v0 (closed ring)
        idx = [get_or_add(x, y) for x, y in coords[:3]]
        triangles.append(idx)

    return pts, triangles


# ── main ─────────────────────────────────────────────────────────────────────

def main(input_path, output_path):
    # --- Load input ---
    with open(input_path) as f:
        data = json.load(f)

    raw_verts = data["vertices"]   # list of [x, y]
    raw_segs  = data["segments"]   # list of [i, j]
    
    desc = data.get("description", input_path)

    print(f"Input : {len(raw_verts)} vertices, {len(raw_segs)} constraint segments")
    print(f"Shapely version: {shapely.__version__}")

    # --- Build rings and convert to Shapely geometry ---
    rings = build_rings(raw_verts, raw_segs)
    print(f"Detected {len(rings)} closed ring(s)")
    if not rings:
        print("ERROR: No closed rings found. Check that your segments form closed loops.")
        sys.exit(1)

    shapely_geom = rings_to_shapely(rings)
    print(f"Shapely geometry type: {shapely_geom.geom_type}")

    # --- Run Shapely CDT ---
    result = shp.constrained_delaunay_triangles(shapely_geom)
    print(f"CDT result: {len(result.geoms)} triangles")

    # --- Convert result back to indexed form ---
    all_pts, triangles = shapely_result_to_indexed(result)

    # Count Steiner points: vertices in the CDT that weren't in the input
    input_pt_set = {(round(x, 8), round(y, 8)) for x, y in raw_verts}
    n_steiner = sum(
        1 for p in all_pts
        if (round(p[0], 8), round(p[1], 8)) not in input_pt_set
    )
    print(f"Vertices in CDT: {len(all_pts)} ({n_steiner} Steiner points added by Shapely)")

    # --- Save output (same format as step1_cdt.py) ---
    output = {
        "description": desc,
        "original_vertices": raw_verts,
        "vertices": all_pts,
        "triangles": triangles,
        "segments": raw_segs,
        "n_original_vertices": len(raw_verts),
        "n_steiner": n_steiner,
        "cdt_method": f"shapely {shapely.__version__} constrained_delaunay_triangles"
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved CDT to {output_path}")

    # --- Visualize (same layout as step1_cdt.py) ---
    # Left panel: input segments only
    # Right panel: CDT result with constraint segments highlighted
    from scipy.spatial import Delaunay as ScipyDelaunay

    raw_np  = np.array(raw_verts)
    all_np  = np.array(all_pts)
    tris_np = np.array(triangles)

    # Recompute plain Delaunay for left panel comparison
    plain_tri = ScipyDelaunay(raw_np)
    plain_tris = plain_tri.simplices

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f"Step 1 (Alternative – Shapely CDT): Constrained Delaunay Triangulation\n{desc}",
        fontsize=11
    )

    panel_data = [
        (raw_np,  plain_tris, "Unconstrained Delaunay\n(scipy, for comparison)"),
        (all_np,  tris_np,    f"Shapely constrained_delaunay_triangles\n({len(triangles)} triangles, {n_steiner} Steiner pts)"),
    ]

    for ax, (pts_show, tris_show, title) in zip(axes, panel_data):
        for tri in tris_show:
            poly = plt.Polygon(pts_show[tri], facecolor="#e8f4f8",
                               edgecolor="#5599bb", linewidth=0.7, alpha=0.8)
            ax.add_patch(poly)

        # Constraint segments (always drawn over original vertex indices)
        for cs, ct in raw_segs:
            ax.plot([raw_verts[cs][0], raw_verts[ct][0]],
                    [raw_verts[cs][1], raw_verts[ct][1]],
                    'r-', linewidth=2.5, zorder=5)

        # Original vertices
        ax.scatter(raw_np[:, 0], raw_np[:, 1], c='navy', s=40, zorder=6,
                   label="Input vertices")

        # Steiner points (right panel only)
        if n_steiner > 0 and pts_show is all_np:
            steiner = all_np[len(raw_verts):]
            ax.scatter(steiner[:, 0], steiner[:, 1],
                       c='orange', s=35, marker='^', zorder=7,
                       label=f"Steiner pts ({n_steiner})")

        patches = [
            mpatches.Patch(facecolor="#e8f4f8", edgecolor="#5599bb", label="Triangle"),
            mpatches.Patch(facecolor="red", label="Constraint segment"),
        ]
        ax.legend(handles=patches, fontsize=7, loc="upper right")
        ax.set_title(title, fontsize=9)
        ax.set_aspect("equal")
        ax.autoscale()
        ax.margins(0.1)

    plt.tight_layout()
    img_path = output_path.replace(".json", "_viz.png")
    plt.savefig(img_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Visualization saved to {img_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python step1_cdt_alternative.py <input.json> <output.json>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
