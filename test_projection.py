"""
Test script for project_to_nonoverlap function
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from optimizer import find_all_overlaps, project_to_nonoverlap
from polygon_utils import (
    translate_polygon,
    get_convex_hull_vertices,
    separating_distance,
)


def plot_scenario(ax, x, movables, fixed_obstacles, title):
    """Plot the polygons at given positions."""
    pts = x.reshape(-1, 2)

    # Plot movables
    colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12"]
    for i, movable in enumerate(movables):
        # Translate polygon to current position
        poly_verts = translate_polygon(
            movable["verts"], pts[i], movable["RotationAngle"]
        )
        # Get convex hull for plotting
        hull_verts = get_convex_hull_vertices(poly_verts, closed=True)

        poly = MplPolygon(
            hull_verts,
            facecolor=colors[i % len(colors)],
            edgecolor="black",
            linewidth=2,
            alpha=0.6,
            label=f"Movable {i + 1}",
        )
        ax.add_patch(poly)

        # Plot center point
        ax.plot(pts[i, 0], pts[i, 1], "ko", markersize=8, zorder=5)

    # Plot fixed obstacles
    for i, obs in enumerate(fixed_obstacles):
        if obs.get("center") is None:
            continue
        poly_verts = translate_polygon(
            obs["verts"], obs["center"], obs["RotationAngle"]
        )
        hull_verts = get_convex_hull_vertices(poly_verts, closed=True)

        poly = MplPolygon(
            hull_verts,
            facecolor="gray",
            edgecolor="black",
            linewidth=2,
            alpha=0.3,
            hatch="///",
            label=f"Fixed Obstacle {i + 1}",
        )
        ax.add_patch(poly)

        # Plot center point
        ax.plot(obs["center"][0], obs["center"][1], "ks", markersize=8, zorder=5)

    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="upper right")

    # Set axis limits based on all objects
    all_x = [pts[i, 0] for i in range(len(movables))]
    all_y = [pts[i, 1] for i in range(len(movables))]
    for obs in fixed_obstacles:
        if obs.get("center") is not None:
            all_x.append(obs["center"][0])
            all_y.append(obs["center"][1])

    margin = 2.0
    ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
    ax.set_xlabel("X", fontsize=12)
    ax.set_ylabel("Y", fontsize=12)


# Create two overlapping square movables
movable1 = {
    "verts": np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]]),
    "RotationAngle": np.pi / 4,
}

movable2 = {
    "verts": np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]]),
    "RotationAngle": np.pi / 4,
}

# Create a fixed obstacle
fixed_obstacle = {
    "verts": np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]),
    "center": np.array([0.0, 0.0]),
    "RotationAngle": 0.0,
}

movables = [movable1, movable2]
fixed_obstacles = [fixed_obstacle]

# Create overlapping initial positions
# Position movable1 at (0, 0) and movable2 at (0.5, 0) - they will overlap
x_initial = np.array([0.0, 0.0, 0.0, 0.0])

print("=" * 60)
print("Testing project_to_nonoverlap function")
print("=" * 60)
print()

print("Initial positions:")
print(f"  Movable 1: {x_initial[0:2]}")
print(f"  Movable 2: {x_initial[2:4]}")
print()

# Find overlaps before projection
overlaps_before = find_all_overlaps(x_initial, movables, fixed_obstacles)
print(f"Overlaps before projection: {len(overlaps_before)}")
for i, (idx_i, idx_j, penetration, normal, avg_size) in enumerate(overlaps_before):
    if idx_j is None:
        print(f"  Overlap {i + 1}: Movable {idx_i} vs Fixed obstacle")
    else:
        print(f"  Overlap {i + 1}: Movable {idx_i} vs Movable {idx_j}")
    print(f"    Penetration: {penetration:.4f}")
    print(f"    Normal: [{normal[0]:.4f}, {normal[1]:.4f}]")
    print(f"    Avg Size: {avg_size:.4f}")
print()

# Apply projection with minimum separation
min_sep = 0.2  # 0.2 units gap between objects
x_projected = project_to_nonoverlap(
    x_initial, movables, fixed_obstacles, min_separation=min_sep
)

print(f"Applied projection with min_separation = {min_sep}")

print("Projected positions:")
print(f"  Movable 1: {x_projected[0:2]}")
print(f"  Movable 2: {x_projected[2:4]}")
print()

# Find overlaps after projection
overlaps_after = find_all_overlaps(
    x_projected, movables, fixed_obstacles, min_separation=min_sep
)
print(f"Overlaps after projection: {len(overlaps_after)}")
if len(overlaps_after) == 0:
    print("  ✓ All overlaps successfully resolved!")
else:
    print("  ✗ Some overlaps remain:")
    for i, (idx_i, idx_j, penetration, normal, avg_size) in enumerate(overlaps_after):
        if idx_j is None:
            print(f"    Overlap {i + 1}: Movable {idx_i} vs Fixed obstacle")
        else:
            print(f"    Overlap {i + 1}: Movable {idx_i} vs Movable {idx_j}")
        print(f"      Penetration: {penetration:.4f}")

# Verify minimum separation is maintained
print()
print("Verifying separation distances:")

pts_final = x_projected.reshape(-1, 2)

# Check movable-movable separations
for i in range(len(movables)):
    for j in range(i + 1, len(movables)):
        A_translated = translate_polygon(
            movables[i]["verts"], pts_final[i], movables[i]["RotationAngle"]
        )
        B_translated = translate_polygon(
            movables[j]["verts"], pts_final[j], movables[j]["RotationAngle"]
        )
        sep = separating_distance(A_translated, B_translated)
        status = "✓" if sep >= min_sep else "✗"
        print(
            f"  {status} Movable {i} ↔ Movable {j}: {sep:.4f} (required: {min_sep:.4f})"
        )

# Check movable-fixed separations
for i in range(len(movables)):
    for obs_idx, obs in enumerate(fixed_obstacles):
        if obs.get("center") is None:
            continue
        A_translated = translate_polygon(
            movables[i]["verts"], pts_final[i], movables[i]["RotationAngle"]
        )
        B_translated = translate_polygon(
            obs["verts"], obs["center"], obs["RotationAngle"]
        )
        sep = separating_distance(A_translated, B_translated)
        status = "✓" if sep >= min_sep else "✗"
        print(
            f"  {status} Movable {i} ↔ Fixed {obs_idx}: {sep:.4f} (required: {min_sep:.4f})"
        )

print()
print("=" * 60)
print("Test complete")
print("=" * 60)
print()

# Create visualization
print("Creating visualization...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot before projection
plot_scenario(
    ax1,
    x_initial,
    movables,
    fixed_obstacles,
    f"Before Projection\n({len(overlaps_before)} overlap{'s' if len(overlaps_before) != 1 else ''})",
)

# Plot after projection
plot_scenario(
    ax2,
    x_projected,
    movables,
    fixed_obstacles,
    f"After Projection\n({len(overlaps_after)} overlap{'s' if len(overlaps_after) != 1 else ''})",
)

plt.suptitle(
    "Project to Non-Overlap Visualization", fontsize=16, fontweight="bold", y=0.98
)
plt.tight_layout()

# Save the figure
output_file = "/workspaces/smart-annotation/projection_test_result.png"
plt.savefig(output_file, dpi=150, bbox_inches="tight")
print(f"✓ Visualization saved to: {output_file}")

# Display the plot
plt.show()
print("Done!")
