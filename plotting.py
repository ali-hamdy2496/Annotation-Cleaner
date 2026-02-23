"""
Plotting functions for visualizing optimization results.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from polygon_utils import translate_polygon, get_convex_hull_vertices, unpack_xy


def plot_regions(regions_info, movables_per_region, fixed_per_region, placement_bounds, extended_lines=None):
    """
    Plot the region splitting visualization with clear region boundaries and colored movables.
    
    Args:
        regions_info: List of region info dictionaries
        movables_per_region: List of movable lists per region
        fixed_per_region: List of fixed obstacle lists per region
        placement_bounds: Tuple of ((xmin, xmax), (ymin, ymax))
        extended_lines: Optional list of extended pipe lines
    """
    n_regions = len(regions_info)
    
    # Use distinct colors for regions
    region_colors = [
        '#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', 
        '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabed4',
        '#469990', '#dcbeff', '#9A6324', '#fffac8', '#800000',
        '#aaffc3', '#808000', '#ffd8b1', '#000075', '#a9a9a9'
    ]
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    # ============ LEFT PLOT: Regions Overview ============
    ax1 = axes[0]
    ax1.set_aspect("equal")
    ax1.set_xlim(placement_bounds[0])
    ax1.set_ylim(placement_bounds[1])
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f"Region Overview ({n_regions} regions)", fontsize=14, fontweight="bold")
    
    # Plot each region with distinct color
    for i, region in enumerate(regions_info):
        boundary = region.get("boundary", [])
        if len(boundary) < 3:
            continue
        
        color = region_colors[i % len(region_colors)]
        
        # Region polygon with strong border
        poly_patch = Polygon(
            boundary,
            closed=True,
            facecolor=color,
            alpha=0.25,
            edgecolor=color,
            linewidth=3,
        )
        ax1.add_patch(poly_patch)
        
        # Region label at centroid
        centroid = np.mean(boundary, axis=0)
        n_mov = len(movables_per_region[i]) if i < len(movables_per_region) else 0
        n_fix = len(fixed_per_region[i]) if i < len(fixed_per_region) else 0
        n_pipes = sum(1 for obs in fixed_per_region[i] if obs.get("ElementType") == "Pipe") if i < len(fixed_per_region) else 0
        
        ax1.text(
            centroid[0], centroid[1],
            f"R{i}\n{n_mov} mov\n{n_fix} fix\n({n_pipes} pipes)",
            ha="center", va="center",
            fontsize=9, fontweight="bold",
            color="black",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor=color, linewidth=2)
        )
    
    # Plot extended pipe lines (dashed red)
    if extended_lines is not None:
        for idx, line in enumerate(extended_lines):
            ax1.plot(line[:, 0], line[:, 1], "r--", linewidth=2.0, alpha=0.8, 
                    label="Extended pipes" if idx == 0 else "")
    
    # Plot actual pipes (solid orange)
    pipe_plotted = False
    for fixed_list in fixed_per_region:
        for obs in fixed_list:
            if obs.get("ElementType") != "Pipe" or obs.get("center") is None:
                continue
            verts_world = translate_polygon(obs["verts"], obs["center"], obs["RotationAngle"])
            ax1.fill(verts_world[:, 0], verts_world[:, 1], color="orange", alpha=0.8, 
                    edgecolor="darkorange", linewidth=1.5,
                    label="Actual pipes" if not pipe_plotted else "")
            pipe_plotted = True
    
    ax1.legend(loc="upper right", fontsize=10)
    
    # ============ RIGHT PLOT: Movables by Region ============
    ax2 = axes[1]
    ax2.set_aspect("equal")
    ax2.set_xlim(placement_bounds[0])
    ax2.set_ylim(placement_bounds[1])
    ax2.grid(True, alpha=0.3)
    ax2.set_title("Movables Colored by Region", fontsize=14, fontweight="bold")
    
    # Plot region boundaries (light)
    for i, region in enumerate(regions_info):
        boundary = region.get("boundary", [])
        if len(boundary) < 3:
            continue
        color = region_colors[i % len(region_colors)]
        boundary = np.array(boundary)
        ax2.plot(boundary[:, 0], boundary[:, 1], color=color, linewidth=2, alpha=0.5)
    
    # Plot fixed obstacles (gray)
    all_fixed_ids = set()
    for fixed_list in fixed_per_region:
        for obs in fixed_list:
            if obs["ElementId"] in all_fixed_ids or obs.get("center") is None:
                continue
            all_fixed_ids.add(obs["ElementId"])
            
            verts_world = translate_polygon(obs["verts"], obs["center"], obs["RotationAngle"])
            
            if obs.get("ElementType") == "Pipe":
                ax2.fill(verts_world[:, 0], verts_world[:, 1], color="orange", alpha=0.7,
                        edgecolor="darkorange", linewidth=1)
            else:
                ax2.fill(verts_world[:, 0], verts_world[:, 1], color="gray", alpha=0.4,
                        edgecolor="darkgray", linewidth=0.5)
    
    # Plot movables colored by their region
    for i, movables in enumerate(movables_per_region):
        color = region_colors[i % len(region_colors)]
        for j, mov in enumerate(movables):
            verts_world = translate_polygon(mov["verts"], mov["target"], mov["RotationAngle"])
            hull = get_convex_hull_vertices(verts_world, closed=True)
            
            ax2.fill(hull[:-1, 0], hull[:-1, 1], color=color, alpha=0.6,
                    edgecolor="black", linewidth=0.5,
                    label=f"Region {i}" if j == 0 else "")
            
            # Small dot at center
            ax2.plot(mov["target"][0], mov["target"][1], ".", color="black", markersize=2)
    
    # Create legend with region counts
    legend_elements = []
    for i in range(n_regions):
        color = region_colors[i % len(region_colors)]
        n_mov = len(movables_per_region[i])
        legend_elements.append(plt.Line2D([0], [0], marker='s', color='w', 
                                         markerfacecolor=color, markersize=12,
                                         label=f'Region {i}: {n_mov} movables'))
    ax2.legend(handles=legend_elements, loc="upper right", fontsize=9)
    
    plt.tight_layout()
    plt.savefig("regions.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved regions.png")
    
    # ============ ADDITIONAL: One plot per region ============
    if n_regions <= 8:  # Only create individual plots if not too many regions
        cols = min(4, n_regions)
        rows = (n_regions + cols - 1) // cols
        fig2, axes2 = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
        if n_regions == 1:
            axes2 = np.array([[axes2]])
        elif rows == 1:
            axes2 = axes2.reshape(1, -1)
        
        for i, region in enumerate(regions_info):
            row, col = i // cols, i % cols
            ax = axes2[row, col]
            
            boundary = np.array(region.get("boundary", []))
            if len(boundary) < 3:
                continue
            
            color = region_colors[i % len(region_colors)]
            
            # Set axis limits to region bounds with padding
            min_x, min_y = boundary.min(axis=0)
            max_x, max_y = boundary.max(axis=0)
            pad_x = (max_x - min_x) * 0.1
            pad_y = (max_y - min_y) * 0.1
            ax.set_xlim(min_x - pad_x, max_x + pad_x)
            ax.set_ylim(min_y - pad_y, max_y + pad_y)
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)
            
            n_mov = len(movables_per_region[i])
            n_fix = len(fixed_per_region[i])
            ax.set_title(f"Region {i}: {n_mov} movables, {n_fix} fixed", fontsize=11, fontweight="bold", color=color)
            
            # Region boundary
            ax.fill(boundary[:, 0], boundary[:, 1], color=color, alpha=0.15,
                   edgecolor=color, linewidth=2)
            
            # Fixed obstacles in this region
            for obs in fixed_per_region[i]:
                if obs.get("center") is None:
                    continue
                verts_world = translate_polygon(obs["verts"], obs["center"], obs["RotationAngle"])
                if obs.get("ElementType") == "Pipe":
                    ax.fill(verts_world[:, 0], verts_world[:, 1], color="orange", alpha=0.8,
                           edgecolor="darkorange", linewidth=1)
                else:
                    ax.fill(verts_world[:, 0], verts_world[:, 1], color="red", alpha=0.4,
                           edgecolor="darkred", linewidth=0.5)
            
            # Movables in this region
            for mov in movables_per_region[i]:
                verts_world = translate_polygon(mov["verts"], mov["target"], mov["RotationAngle"])
                hull = get_convex_hull_vertices(verts_world, closed=True)
                ax.fill(hull[:-1, 0], hull[:-1, 1], color="blue", alpha=0.6,
                       edgecolor="darkblue", linewidth=0.5)
                ax.plot(mov["target"][0], mov["target"][1], "k.", markersize=3)
        
        # Hide empty subplots
        for i in range(n_regions, rows * cols):
            row, col = i // cols, i % cols
            axes2[row, col].axis("off")
        
        plt.tight_layout()
        plt.savefig("regions_detail.png", dpi=300, bbox_inches="tight")
        plt.close(fig2)
        print("Saved regions_detail.png")


def plot_result(
    xvec, xvec_initial, movables, fixed_obstacles_before, fixed_obstacles_after, placement_bounds, search_radius=None
):
    """
    Plot the optimization results showing before and after states in separate files.

    Args:
        xvec: Optimized positions vector
        xvec_initial: Initial positions vector (original targets)
        movables: List of movable object dictionaries
        fixed_obstacles_before: List of fixed obstacle dictionaries
        fixed_obstacles_after: List of fixed obstacle dictionaries
        placement_bounds: Tuple of ((xmin, xmax), (ymin, ymax))
        search_radius: KD-tree search radius for visualization (optional)
    """
    pts = unpack_xy(xvec)

    # Get initial positions
    if xvec_initial is not None:
        pts_initial = unpack_xy(xvec_initial)
    else:
        pts_initial = pts  # fallback if no initial provided

    # =============== BEFORE PLOT ===============
    fig1, ax1 = plt.subplots(1, 1, figsize=(10, 8))
    ax1.set_aspect("equal")
    ax1.set_xlim(placement_bounds[0])
    ax1.set_ylim(placement_bounds[1])
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Before Optimization", fontsize=14, fontweight="bold")

    # plot fixed obstacles
    for idx, obs in enumerate(fixed_obstacles_before):
        if obs.get("center") is None:
            continue
        poly = translate_polygon(obs["verts"], obs["center"], obs["RotationAngle"])
        poly_hull = get_convex_hull_vertices(poly, closed=True)
        polygon_patch = Polygon(
            poly_hull[:-1],
            closed=True,
            facecolor="red",
            alpha=0.3,
            edgecolor="darkred",
            linewidth=0.5,
            label="Fixed Obstacles" if idx == 0 else "",
        )
        ax1.add_patch(polygon_patch)
        if obs.get("ElementType") == "Pipe":
            ax1.plot(poly_hull[:, 0], poly_hull[:, 1], "yellow", linewidth=0.25)
        else:
            ax1.plot(poly_hull[:, 0], poly_hull[:, 1], "darkred", linewidth=0.25)

    # plot movables at INITIAL positions
    for i, p in enumerate(pts_initial):
        poly = translate_polygon(movables[i]["verts"], p, movables[i]["RotationAngle"])
        poly_hull = get_convex_hull_vertices(poly, closed=True)
        polygon_patch = Polygon(
            poly_hull[:-1],
            closed=True,
            facecolor="blue",
            alpha=0.4,
            edgecolor="darkblue",
            linewidth=0.5,
            label="Movable Objects" if i == 0 else "",
        )
        ax1.add_patch(polygon_patch)
        ax1.plot(poly_hull[:, 0], poly_hull[:, 1], "darkblue", linewidth=0.25)
        
        # Show ORIGINAL target point (from pts_initial, NOT movables[i]["target"] which is modified)
        # In the "before" state, objects are at their original target positions
        t = pts_initial[i]  # FIXED: Use original position, not modified target
        hull_verts_calc = get_convex_hull_vertices(poly, closed=False)
        if len(hull_verts_calc) > 0:
            centroid = np.mean(hull_verts_calc, axis=0)
        else:
            centroid = p
        ax1.plot(
            t[0], t[1], "g*", markersize=1, label="Target Points" if i == 0 else ""
        )
        ax1.plot(
            [centroid[0], t[0]], [centroid[1], t[1]], "g--", alpha=0.5, linewidth=0.25
        )

    ax1.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig("before.png", dpi=600, bbox_inches="tight")
    plt.close(fig1)
    print("Saved before.png")

    # =============== AFTER PLOT ===============
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 8))
    ax2.set_aspect("equal")
    ax2.set_xlim(placement_bounds[0])
    ax2.set_ylim(placement_bounds[1])
    ax2.grid(True, alpha=0.3)
    ax2.set_title("After Optimization", fontsize=14, fontweight="bold")

    # plot fixed obstacles (same as before)
    for idx, obs in enumerate(fixed_obstacles_after):
        if obs.get("center") is None:
            continue
        poly = translate_polygon(obs["verts"], obs["center"], obs["RotationAngle"])
        poly_hull = get_convex_hull_vertices(poly, closed=True)
        polygon_patch = Polygon(
            poly_hull[:-1],
            closed=True,
            facecolor="red",
            alpha=0.3,
            edgecolor="darkred",
            linewidth=0.5,
            label="Fixed Obstacles" if idx == 0 else "",
        )
        ax2.add_patch(polygon_patch)
        if obs.get("ElementType") == "Pipe":
            ax2.plot(poly_hull[:, 0], poly_hull[:, 1], "yellow", linewidth=0.25)
        else:
            ax2.plot(poly_hull[:, 0], poly_hull[:, 1], "darkred", linewidth=0.25)

    # plot movables at OPTIMIZED positions
    for i, p in enumerate(pts):
        poly = translate_polygon(movables[i]["verts"], p, movables[i]["RotationAngle"])
        poly_hull = get_convex_hull_vertices(poly, closed=True)
        polygon_patch = Polygon(
            poly_hull[:-1],
            closed=True,
            facecolor="blue",
            alpha=0.4,
            edgecolor="darkblue",
            linewidth=0.5,
            label="Movable Objects" if i == 0 else "",
        )
        ax2.add_patch(polygon_patch)
        ax2.plot(poly_hull[:, 0], poly_hull[:, 1], "darkblue", linewidth=0.25)
        
        # Show ORIGINAL target point (where the object wanted to be)
        # This shows the displacement from original target to final position
        t = pts_initial[i]  # Original target position
        hull_verts_calc = get_convex_hull_vertices(poly, closed=False)
        if len(hull_verts_calc) > 0:
            centroid = np.mean(hull_verts_calc, axis=0)
        else:
            centroid = p
        ax2.plot(
            t[0], t[1], "g*", markersize=1, label="Original Targets" if i == 0 else ""
        )
        ax2.plot(
            [centroid[0], t[0]], [centroid[1], t[1]], "g--", alpha=0.5, linewidth=0.25
        )

    # Visualize KD-tree search radius for a sample of movables
    if search_radius is not None and search_radius > 0:
        # Show radius for up to 5 evenly spaced movables
        num_samples = min(5, len(movables))
        sample_indices = np.linspace(0, len(movables) - 1, num_samples, dtype=int)

        for idx, i in enumerate(sample_indices):
            t = pts_initial[i]
            circle = plt.Circle(
                t,
                search_radius,
                fill=False,
                edgecolor="orange",
                linestyle="--",
                linewidth=0.5,
                alpha=0.5,
                label="KD-tree Search Radius" if idx == 0 else "",
            )
            ax2.add_patch(circle)

    ax2.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig("after.png", dpi=600, bbox_inches="tight")
    plt.close(fig2)
    print("Saved after.png")