"""
Step 1: Constrained Delaunay Triangulation (CDT)
=================================================
2D equivalent of the paper's CDT step.

Input : JSON file with 'vertices' and 'segments'
Output: JSON file with triangulation + visualization PNG

The approach:
  1. Load vertices and constraint segments.
  2. Compute the Delaunay triangulation of ALL vertices (convex-hull tessellation).
  3. Enforce constraints: any Delaunay edge that crosses a constraint segment is
     removed by inserting the intersection point (Steiner point) and re-triangulating
     locally, until all constraint segments appear as edges in the triangulation.
  4. Save the full triangulation (we segment inside/outside in step 3).
  5. Visualize.

Usage:
    python step1_cdt.py <input_json> <output_json>
"""

import sys
import json
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.tri import Triangulation as MplTriangulation
from scipy.spatial import Delaunay


# ── geometry helpers ──────────────────────────────────────────────────────────

def seg_intersect(p1, p2, p3, p4, tol=1e-10):
    """
    check if two line segments, p1p2 and p3p4 intersect 
    Return the parameter t in [0,1] along p1->p2 where it crosses p3->p4,
    or None if no proper interior intersection exists.
    """
    d1 = p2 - p1
    d2 = p4 - p3
    cross = d1[0]*d2[1] - d1[1]*d2[0]
    if abs(cross) < tol:
        return None
    diff = p3 - p1
    t = (diff[0]*d2[1] - diff[1]*d2[0]) / cross
    u = (diff[0]*d1[1] - diff[1]*d1[0]) / cross
    if tol < t < 1 - tol and tol < u < 1 - tol:
        return t
    return None


def segments_of_triangle(tri_idx, triangles):
    """Return the 3 directed edges of a triangle as frozensets."""
    a, b, c = triangles[tri_idx]
    return [frozenset([a, b]), frozenset([b, c]), frozenset([a, c])]


def find_crossing_edges(pts, triangles, cs, ct):
    """
    Find all Delaunay edges that cross constraint segment (cs->ct).
    Returns list of frozenset edge pairs.
    """
    p_s, p_t = pts[cs], pts[ct]
    crossings = []
    seen = set()
    for tri in triangles:
        for i in range(3):
            a, b = tri[i], tri[(i+1)%3]
            key = frozenset([a, b])
            if key in seen:
                continue
            seen.add(key)
            if a in (cs, ct) or b in (cs, ct):
                continue
            t = seg_intersect(p_s, p_t, pts[a], pts[b])
            if t is not None:
                crossings.append(key)
    return crossings


def enforce_constraints(pts_list, triangles, segments):
    """
    Iteratively insert Steiner points at intersections of constraint segments
    with Delaunay edges until all constraints are present in the triangulation.
    Uses a simple re-Delaunay after each insertion.
    """
    pts = np.array(pts_list, dtype=float)
    max_iter = 200
    for iteration in range(max_iter):
        changed = False
        for cs, ct in segments:
            crossings = find_crossing_edges(pts, triangles, cs, ct)
            if not crossings:
                continue
            # Pick the first crossing edge and insert a Steiner point
            edge = next(iter(crossings.pop()))  # get one vertex of the edge
            # Find the actual edge
            for cr in [crossings[0]] if crossings else []:
                a, b = list(cr)
                break
            else:
                # Use the edge we just popped
                a, b = list(find_crossing_edges(pts, triangles, cs, ct)[0])

            # Intersection point
            p_s, p_t = pts[cs], pts[ct]
            t = seg_intersect(p_s, p_t, pts[a], pts[b])
            if t is None:
                continue
            new_pt = p_s + t * (p_t - p_s)
            # Snap to existing point if very close
            dists = np.linalg.norm(pts - new_pt, axis=1)
            closest = np.argmin(dists)
            if dists[closest] < 1e-9:
                new_idx = closest
            else:
                new_idx = len(pts)
                pts = np.vstack([pts, new_pt])

            # Re-triangulate
            tri_obj = Delaunay(pts)
            triangles = tri_obj.simplices.tolist()
            changed = True
            break  # restart loop after insertion

        if not changed:
            break

    return pts.tolist(), triangles


def constraint_edge_present(triangles, cs, ct):
    """Check if the directed edge (cs,ct) or (ct,cs) appears in the triangulation."""
    for tri in triangles:
        tri_set = set(tri)
        if cs in tri_set and ct in tri_set:
            # check it's actually an edge (shared by two vertices = edge)
            edges = [(tri[0],tri[1]),(tri[1],tri[2]),(tri[0],tri[2])]
            for a, b in edges:
                if (a==cs and b==ct) or (a==ct and b==cs):
                    return True
    return False


# ── main ──────────────────────────────────────────────────────────────────────

def main(input_path, output_path):
    # --- Load input ---
    with open(input_path) as f:
        data = json.load(f)

    raw_verts = data["vertices"]
    raw_segs  = data["segments"]
    desc = data.get("description", input_path)

    pts = np.array(raw_verts, dtype=float)
    segments = raw_segs  # list of [i, j]

    print(f"Input: {len(pts)} vertices, {len(segments)} constraint segments")

    # --- Step 1a: Initial Delaunay of all vertices ---
    tri_obj = Delaunay(pts)
    triangles = tri_obj.simplices.tolist()
    print(f"Initial Delaunay: {len(triangles)} triangles")

    # --- Step 1b: Enforce constraints ---
    pts_list, triangles = enforce_constraints(pts.tolist(), triangles, segments)
    pts = np.array(pts_list)
    n_steiner = len(pts) - len(raw_verts)
    print(f"After constraint enforcement: {len(triangles)} triangles, {n_steiner} Steiner points added")

    # --- Save output ---
    output = {
        "description": desc,
        "original_vertices": raw_verts,
        "vertices": pts.tolist(),
        "triangles": triangles,
        "segments": segments,
        "n_original_vertices": len(raw_verts),
        "n_steiner": n_steiner
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved CDT to {output_path}")

    # --- Visualize ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Step 1: Constrained Delaunay Triangulation\n{desc}", fontsize=11)

    for ax_idx, (ax, tris_to_show, title) in enumerate(zip(
        axes,
        [Delaunay(np.array(raw_verts)).simplices.tolist(), triangles],
        ["Initial Delaunay\n(all vertices, no constraints)", "After Constraint Enforcement"]
    )):
        pts_show = np.array(raw_verts) if ax_idx == 0 else pts
        tris_arr = np.array(tris_to_show)

        # Draw triangles
        for tri in tris_arr:
            poly = plt.Polygon(pts_show[tri], fill=True, facecolor="#e8f4f8",
                               edgecolor="#5599bb", linewidth=0.7, alpha=0.8)
            ax.add_patch(poly)

        # Draw constraint segments
        for cs, ct in segments:
            ax.plot([raw_verts[cs][0], raw_verts[ct][0]],
                    [raw_verts[cs][1], raw_verts[ct][1]],
                    'r-', linewidth=2.5, zorder=5)

        # Plot original vertices
        orig_pts = np.array(raw_verts)
        ax.scatter(orig_pts[:, 0], orig_pts[:, 1], c='navy', s=40, zorder=6)

        if ax_idx == 1 and n_steiner > 0:
            steiner_pts = pts[len(raw_verts):]
            ax.scatter(steiner_pts[:, 0], steiner_pts[:, 1],
                       c='orange', s=30, marker='^', zorder=6, label=f"Steiner pts ({n_steiner})")
            ax.legend(fontsize=8)

        ax.set_title(f"{title}\n({len(tris_to_show)} triangles)", fontsize=9)
        ax.set_aspect("equal")
        ax.autoscale()
        ax.margins(0.1)

        # Legend patches
        patches = [
            mpatches.Patch(facecolor="#e8f4f8", edgecolor="#5599bb", label="Triangle"),
            mpatches.Patch(facecolor="red", label="Constraint segment"),
        ]
        ax.legend(handles=patches, fontsize=7, loc="upper right")

    plt.tight_layout()
    img_path = output_path.replace(".json", "_viz.png")
    plt.savefig(img_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Visualization saved to {img_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python step1_cdt.py <input.json> <output.json>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
