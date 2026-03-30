"""
Step 3: Inside-Outside Segmentation via Graph Cut
==================================================
2D equivalent of the paper's energy-minimization segmentation (Section 5.1).

The winding number alone (thresholded) can produce noisy results.
We formulate an MRF energy over the CDT triangles:

    E = Σ_i u(x_i)  +  γ · Σ_{(i,j) neighbors} v(x_i, x_j)

Data term (eq. 9):
    u_i(outside) = max(w_i - 0, 0)   = max(w_i, 0)   (penalize outside when w is high)
    u_i(inside)  = max(1 - w_i, 0)               (penalize inside when w is low)

Smoothness term (eq. 10):
    v(x_i, x_j) = 0              if x_i == x_j
                = len(shared_edge) · exp(-(w_i - w_j)² / (2σ²))   otherwise

Two adjacent triangles with very different winding numbers should NOT be
smoothed across (they likely straddle a boundary), while similar-winding
neighbors are encouraged to agree.

We implement this via s-t min-cut using NetworkX max_flow (exact, for small
CDTs). For large meshes a faster solver would be used, but this matches the
paper's conceptual algorithm exactly.

Input : JSON from step 2
Output: JSON with final inside/outside labels + visualization

Usage:
    python step3_graphcut.py <step2_output.json> <output.json> [gamma] [sigma]
"""

import sys
import json
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx


# ── adjacency ─────────────────────────────────────────────────────────────────

def build_adjacency(triangles, pts):
    """
    Build triangle adjacency: two triangles are neighbors if they share an edge.
    Returns dict: tri_idx -> list of (neighbor_tri_idx, shared_edge_length)
    """
    edge_to_tris = {}
    pts_np = np.array(pts)

    for i, tri in enumerate(triangles):
        for k in range(3):
            a, b = tri[k], tri[(k+1)%3]
            edge = frozenset([a, b])
            edge_to_tris.setdefault(edge, []).append(i)

    adjacency = {i: [] for i in range(len(triangles))}
    for edge, tri_list in edge_to_tris.items():
        if len(tri_list) == 2:
            i, j = tri_list
            a, b = list(edge)
            edge_len = np.linalg.norm(pts_np[a] - pts_np[b])
            adjacency[i].append((j, edge_len))
            adjacency[j].append((i, edge_len))

    return adjacency


# ── graph cut ─────────────────────────────────────────────────────────────────

def graphcut_segment(winding, adjacency, gamma=5.0, sigma=0.25):
    """
    s-t min-cut on the CDT dual graph.

    Node 's' (source) = INSIDE
    Node 't' (sink)   = OUTSIDE

    Terminal weights:
        s -> i  (cost of labeling i as OUTSIDE) = data term u_i(outside)
        i -> t  (cost of labeling i as INSIDE)  = data term u_i(inside)

    Pairwise weights (eq. 10):
        i <-> j = gamma * edge_len * exp(-(w_i - w_j)^2 / (2*sigma^2))
    """
    n = len(winding)
    G = nx.DiGraph()
    G.add_node("s")
    G.add_node("t")

    INF = 1e9

    for i, w in enumerate(winding):
        # Use abs(w) so both CW and CCW orientations work correctly
        aw = abs(w)
        # Data term
        cost_outside = max(aw, 0.0)       # penalty if we call this OUTSIDE (high w -> inside)
        cost_inside  = max(1.0 - aw, 0.0) # penalty if we call this INSIDE (low w -> outside)

        # s -> i  with capacity = penalty for labeling OUTSIDE
        G.add_edge("s", i, capacity=cost_outside)
        # i -> t  with capacity = penalty for labeling INSIDE
        G.add_edge(i, "t", capacity=cost_inside)

    for i, neighbors in adjacency.items():
        for j, edge_len in neighbors:
            if i >= j:
                continue
            w_diff = winding[i] - winding[j]
            smooth = gamma * edge_len * math.exp(-(w_diff**2) / (2 * sigma**2))
            # Add both directions for undirected smoothness
            G.add_edge(i, j, capacity=smooth)
            G.add_edge(j, i, capacity=smooth)

    # Min-cut
    cut_value, (reachable, non_reachable) = nx.minimum_cut(G, "s", "t")
    print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"  Min-cut value: {cut_value:.4f}")

    # Nodes reachable from s = INSIDE
    labels = np.zeros(n, dtype=int)
    for node in reachable:
        if isinstance(node, int):
            labels[node] = 1  # inside

    return labels


# ── main ──────────────────────────────────────────────────────────────────────

def main(input_path, output_path, gamma=5.0, sigma=0.25):
    with open(input_path) as f:
        data = json.load(f)

    pts         = data["vertices"]
    orig_verts  = data["original_vertices"]
    triangles   = data["triangles"]
    segments    = data["segments"]
    winding     = np.array(data["winding_numbers"])
    desc        = data.get("description", input_path)

    print(f"Loaded: {len(pts)} vertices, {len(triangles)} triangles")
    print(f"Parameters: gamma={gamma}, sigma={sigma}")

    # --- Build adjacency graph ---
    adjacency = build_adjacency(triangles, pts)

    # --- Run graph cut ---
    print("Running graph cut segmentation...")
    labels = graphcut_segment(winding, adjacency, gamma=gamma, sigma=sigma)

    n_inside  = labels.sum()
    n_outside = len(labels) - n_inside
    print(f"Result: {n_inside} inside, {n_outside} outside triangles")

    # --- Save ---
    inside_tris = [triangles[i] for i in range(len(triangles)) if labels[i] == 1]
    output = {
        **data,
        "labels": labels.tolist(),
        "inside_triangles": inside_tris,
        "gamma": gamma,
        "sigma": sigma
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved segmentation to {output_path}")

    # --- Visualize (3 panels) ---
    pts_np   = np.array(pts)
    orig_np  = np.array(orig_verts)
    tris_np  = np.array(triangles)

    threshold_labels = (winding > 0.5).astype(int)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Step 3: Graph Cut Inside-Outside Segmentation\n{desc}", fontsize=11)

    panel_data = [
        (threshold_labels,  "Winding Number Threshold\n(w > 0.5, no smoothness)", "#e63946", "#a8dadc"),
        (labels,            f"Graph Cut Result\n(γ={gamma}, σ={sigma})",          "#e63946", "#a8dadc"),
        (labels,            "Final Interior Triangulation\n(inside only)",         "#457b9d", None),
    ]

    for ax, (lbl, title, in_color, out_color) in zip(axes, panel_data):
        for i, tri in enumerate(tris_np):
            poly_pts = pts_np[tri]
            if lbl[i] == 1:
                fc = in_color
                ec = "#333333"
                lw = 0.5
                alpha = 0.85
            else:
                if out_color is None:
                    continue  # skip outside in panel 3
                fc = out_color
                ec = "#bbbbbb"
                lw = 0.3
                alpha = 0.5
            patch = plt.Polygon(poly_pts, facecolor=fc, edgecolor=ec,
                                linewidth=lw, alpha=alpha)
            ax.add_patch(patch)

        # Constraint segments
        for cs, ct in segments:
            ax.plot([orig_np[cs,0], orig_np[ct,0]],
                    [orig_np[cs,1], orig_np[ct,1]], 'k-', linewidth=2.2, zorder=5)

        ax.set_title(title, fontsize=9)
        ax.set_aspect("equal"); ax.autoscale(); ax.margins(0.1)
        ax.set_xlabel("x"); ax.set_ylabel("y")

        n_in  = int(lbl.sum())
        n_out = int(len(lbl) - n_in)
        patches = [mpatches.Patch(facecolor=in_color, label=f"Inside ({n_in})")]
        if out_color:
            patches.append(mpatches.Patch(facecolor=out_color, label=f"Outside ({n_out})"))
        ax.legend(handles=patches, fontsize=8, loc="upper right")

    plt.tight_layout()
    img_path = output_path.replace(".json", "_viz.png")
    plt.savefig(img_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Visualization saved to {img_path}")

    # --- Extra: side-by-side comparison of threshold vs graphcut ---
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
    fig2.suptitle(f"Threshold vs. Graph Cut Comparison\n{desc}", fontsize=11)

    for ax, (lbl, title) in zip(axes2, [
        (threshold_labels, "Simple Threshold (w > 0.5)"),
        (labels, f"Graph Cut (γ={gamma}, σ={sigma})")
    ]):
        for i, tri in enumerate(tris_np):
            poly_pts = pts_np[tri]
            fc = "#e63946" if lbl[i] == 1 else "#a8dadc"
            ec = "#444444" if lbl[i] == 1 else "#aaaaaa"
            patch = plt.Polygon(poly_pts, facecolor=fc, edgecolor=ec,
                                linewidth=0.4, alpha=0.85)
            ax.add_patch(patch)

        for cs, ct in segments:
            ax.plot([orig_np[cs,0], orig_np[ct,0]],
                    [orig_np[cs,1], orig_np[ct,1]], 'k-', linewidth=2.2, zorder=5)

        ax.set_title(title, fontsize=10)
        ax.set_aspect("equal"); ax.autoscale(); ax.margins(0.1)

        n_in = int(lbl.sum())
        patches = [
            mpatches.Patch(facecolor="#e63946", label=f"Inside ({n_in})"),
            mpatches.Patch(facecolor="#a8dadc", label=f"Outside ({len(lbl)-n_in})"),
        ]
        ax.legend(handles=patches, fontsize=8)

    plt.tight_layout()
    cmp_path = output_path.replace(".json", "_comparison.png")
    plt.savefig(cmp_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Comparison visualization saved to {cmp_path}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python step3_graphcut.py <step2_output.json> <output.json> [gamma] [sigma]")
        sys.exit(1)

    gamma = float(sys.argv[3]) if len(sys.argv) > 3 else 5.0
    sigma = float(sys.argv[4]) if len(sys.argv) > 4 else 0.25
    main(sys.argv[1], sys.argv[2], gamma=gamma, sigma=sigma)
