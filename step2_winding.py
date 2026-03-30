"""
Step 2: Generalized Winding Number
===================================
2D equivalent of the paper's winding number formulation.

For each triangle in the CDT, we evaluate the generalized winding number
at the triangle's centroid with respect to the input constraint segments.

In 2D the winding number of a point p w.r.t. a (possibly open) polygon is:

    w(p) = (1 / 2π) * Σ_i  θ_i

where θ_i is the SIGNED angle subtended by segment i at p (using atan2).

For a closed, consistently oriented polygon:
  - w(p) = 1  →  p is inside
  - w(p) = 0  →  p is outside
  - w(p) = 2, 3, …  →  p is in an overlapping region

For an open or imperfect boundary, w(p) is a smooth (harmonic) real value
that serves as a confidence measure.

Input : JSON from step 1
Output: JSON with per-triangle winding numbers + visualization

Usage:
    python step2_winding.py <step1_output.json> <output.json>
"""

import sys
import json
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import TwoSlopeNorm


# ── winding number ─────────────────────────────────────────────────────────────

def signed_angle(a, b):
    """
    Signed angle from vector a to vector b, using the 2D atan2 formula.
    This corresponds to the θ_i term in eq. (2)/(3) of the paper.
    """
    cross = a[0]*b[1] - a[1]*b[0]   # sin component
    dot   = a[0]*b[0] + a[1]*b[1]   # cos component
    return math.atan2(cross, dot)


def winding_number_2d(p, vertices, segments):
    """
    Generalized 2D winding number of point p w.r.t. a (possibly open) polyline
    defined by `segments` over `vertices`.

    Implements eq. (2) of the paper:
        w(p) = (1/2π) * Σ θ_i
    where θ_i is the signed angle ∠(v_i - p, v_{i+1} - p).
    """
    total_angle = 0.0
    px, py = p
    for cs, ct in segments: 
        a = np.array(vertices[cs]) - np.array(p)
        b = np.array(vertices[ct]) - np.array(p)
        # Skip degenerate (point on segment endpoint)
        if np.linalg.norm(a) < 1e-12 or np.linalg.norm(b) < 1e-12:
            continue
        total_angle += signed_angle(a, b) # uses assumption that segments are oriented
    return total_angle / (2.0 * math.pi) # radians 


def triangle_centroid(pts, tri):
    """
    find center point of triangle 
    winding number of this point represents triangle 
    """
    a, b, c = pts[tri[0]], pts[tri[1]], pts[tri[2]]
    return [(a[0]+b[0]+c[0])/3, (a[1]+b[1]+c[1])/3]


# ── main ──────────────────────────────────────────────────────────────────────

def main(input_path, output_path):
    with open(input_path) as f:
        data = json.load(f)

    pts       = data["vertices"]          # all vertices (incl. Steiner)
    orig_verts= data["original_vertices"] # only original input vertices
    triangles = data["triangles"]
    segments  = data["segments"]          # constraint segments (original indices)
    desc      = data.get("description", input_path)

    print(f"Loaded: {len(pts)} vertices, {len(triangles)} triangles, {len(segments)} constraint segs")

    # --- Compute winding number at each triangle centroid ---
    winding = []
    centroids = []
    for tri in triangles:
        c = triangle_centroid(pts, tri)
        centroids.append(c)
        w = winding_number_2d(c, orig_verts, segments)
        winding.append(w)

    winding = np.array(winding)
    centroids = np.array(centroids)

    abs_w = np.abs(winding)
    print(f"Winding number range: [{winding.min():.3f}, {winding.max():.3f}]")
    print(f"  Triangles with |w| > 0.5: {(abs_w > 0.5).sum()} (likely inside)")
    print(f"  Triangles with |w| < 0.5: {(abs_w < 0.5).sum()} (likely outside)")

    # --- Save ---
    output = {**data, "winding_numbers": winding.tolist(), "centroids": centroids.tolist()}
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved winding numbers to {output_path}")

    # --- Visualize ---
    pts_np = np.array(pts)
    orig_np = np.array(orig_verts)
    tris_np = np.array(triangles)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f"Step 2: Generalized Winding Number\n{desc}", fontsize=11)

    # Left: scalar field as a heatmap (color each triangle by winding number)
    ax = axes[0]
    ax.set_title("Winding Number Field\n(per triangle centroid)", fontsize=9)

    w_min, w_max = winding.min(), winding.max()
    # Color map: blue=outside (0), white=boundary(0.5), red=inside (1+)
    norm = TwoSlopeNorm(vmin=min(w_min, -0.1), vcenter=0.5, vmax=max(w_max, 1.1))
    cmap = plt.cm.RdBu_r

    for i, tri in enumerate(tris_np):
        poly_pts = pts_np[tri]
        color = cmap(norm(winding[i]))
        patch = plt.Polygon(poly_pts, facecolor=color, edgecolor="#aaaaaa",
                            linewidth=0.4, alpha=0.9)
        ax.add_patch(patch)

    # Constraint segments
    for cs, ct in segments:
        ax.plot([orig_np[cs,0], orig_np[ct,0]],
                [orig_np[cs,1], orig_np[ct,1]], 'k-', linewidth=2.0, zorder=5)

    # Centroids colored by winding
    sc = ax.scatter(centroids[:, 0], centroids[:, 1],
                    c=winding, cmap=cmap, norm=norm, s=10, zorder=6)
    plt.colorbar(sc, ax=ax, label="w(p)", shrink=0.8)
    ax.set_aspect("equal"); ax.autoscale(); ax.margins(0.1)
    ax.set_xlabel("x"); ax.set_ylabel("y")

    # Right: binary threshold preview (w > 0.5 = inside)
    ax = axes[1]
    ax.set_title("Winding Number Threshold Preview\n(w > 0.5 = inside, before graphcut)", fontsize=9)

    inside_mask = np.abs(winding) > 0.5
    for i, tri in enumerate(tris_np):
        poly_pts = pts_np[tri]
        fc = "#e63946" if inside_mask[i] else "#a8dadc"
        patch = plt.Polygon(poly_pts, facecolor=fc, edgecolor="#555555",
                            linewidth=0.4, alpha=0.85)
        ax.add_patch(patch)

    for cs, ct in segments:
        ax.plot([orig_np[cs,0], orig_np[ct,0]],
                [orig_np[cs,1], orig_np[ct,1]], 'k-', linewidth=2.2, zorder=5)

    import matplotlib.patches as mpatches
    patches = [
        mpatches.Patch(facecolor="#e63946", label=f"Inside (w>0.5): {inside_mask.sum()}"),
        mpatches.Patch(facecolor="#a8dadc", label=f"Outside (w≤0.5): {(~inside_mask).sum()}"),
    ]
    ax.legend(handles=patches, fontsize=8, loc="upper right")
    ax.set_aspect("equal"); ax.autoscale(); ax.margins(0.1)
    ax.set_xlabel("x"); ax.set_ylabel("y")

    plt.tight_layout()
    img_path = output_path.replace(".json", "_viz.png")
    plt.savefig(img_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Visualization saved to {img_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python step2_winding.py <step1_output.json> <output.json>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
