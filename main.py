"""
Main execution script for object placement optimization with region splitting.

This script:
1. Reads problem data (movable objects and fixed obstacles)
2. Splits the space into regions using pipe SUBTRACTION (region_simple)
3. Runs GREEDY optimization for each region (guarantees 0 overlaps)
4. Combines results and visualizes
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon, Patch
from shapely.geometry import Polygon as ShapelyPolygon

from json_helper import load_problem_data, save_optimized_output
from plotting import plot_result
from region_simple import split_into_regions
from greedy_optimizer import (
    greedy_optimize_with_regions,
    pull_movables_into_region,

)
from utils import (
    calculate_displacement_metric,
    find_all_overlaps,
)
from polygon_utils import translate_polygon, get_convex_hull_vertices


def plot_regions(regions_info, movables_per_region, fixed_per_region, placement_bounds, 
                 filename="regions_plot.png"):
    """
    Plot regions with two panels:
    - Left: Region overview with colors and labels
    - Right: Movables colored by region
    """
    # Define distinct colors for regions
    region_colors = [
        '#FF6B6B',  # Red
        '#4ECDC4',  # Teal
        '#FFE66D',  # Yellow
        '#95E1D3',  # Mint
        '#DDA0DD',  # Plum
        '#87CEEB',  # Sky blue
        '#98D8C8',  # Sea green
        '#F7DC6F',  # Gold
        '#BB8FCE',  # Purple
        '#85C1E9',  # Light blue
    ]
    
    (xmin, xmax), (ymin, ymax) = placement_bounds
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # ===== LEFT PANEL: Region Overview =====
    ax1.set_aspect('equal')
    ax1.set_xlim(xmin - 5, xmax + 5)
    ax1.set_ylim(ymin - 5, ymax + 5)
    ax1.set_title(f"Region Splitting - Overview ({len(regions_info)} regions)", fontsize=14, fontweight='bold')
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.grid(True, alpha=0.3)
    
    # Fill background with light pink (areas outside regions)
    background = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, 
                                 facecolor='#FFB6C1', alpha=0.3, zorder=0)
    ax1.add_patch(background)
    
    # Draw each region
    for i, region in enumerate(regions_info):
        color = region_colors[i % len(region_colors)]
        
        # Get region boundary
        if "shapely_polygon" in region:
            poly = region["shapely_polygon"]
            coords = list(poly.exterior.coords)
        else:
            coords = region["boundary"]
        
        # Draw region polygon
        polygon = MplPolygon(coords, alpha=0.4, facecolor=color, 
                            edgecolor='orange', linewidth=2, zorder=1)
        ax1.add_patch(polygon)
        
        # Count pipes in this region
        n_pipes = sum(1 for obs in fixed_per_region[i] if obs.get("ElementType") == "Pipe")
        
        # Add label at centroid
        if "shapely_polygon" in region:
            centroid = region["shapely_polygon"].centroid
            cx, cy = centroid.x, centroid.y
        else:
            coords_arr = np.array(coords[:-1]) if len(coords) > 0 and coords[0] == coords[-1] else np.array(coords)
            if len(coords_arr) > 0:
                cx, cy = np.mean(coords_arr, axis=0)
            else:
                cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
        
        n_mov = len(movables_per_region[i])
        n_fix = len(fixed_per_region[i])
        
        # Draw label box
        label_text = f"R{i}\n{n_mov} mov\n{n_fix} fix\n({n_pipes} pipes)"
        ax1.annotate(label_text, (cx, cy), fontsize=9, fontweight='bold',
                    ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=color, 
                             edgecolor='black', alpha=0.9))
    
    # Collect all unique pipes and fixed obstacles
    all_pipes_drawn = set()
    all_fixed_drawn = set()
    
    for fixed_list in fixed_per_region:
        for obs in fixed_list:
            if obs.get("center") is None:
                continue
            if obs.get("ElementType") == "Pipe":
                if obs["ElementId"] not in all_pipes_drawn:
                    all_pipes_drawn.add(obs["ElementId"])
                    verts = translate_polygon(obs["verts"], obs["center"], obs.get("RotationAngle", 0.0))
                    hull = get_convex_hull_vertices(verts, closed=True)
                    ax1.fill(hull[:, 0], hull[:, 1], color='orange', alpha=0.8, zorder=2)
                    ax1.plot(hull[:, 0], hull[:, 1], color='darkorange', linewidth=1, zorder=3)
            else:
                if obs["ElementId"] not in all_fixed_drawn:
                    all_fixed_drawn.add(obs["ElementId"])
                    verts = translate_polygon(obs["verts"], obs["center"], obs.get("RotationAngle", 0.0))
                    hull = get_convex_hull_vertices(verts, closed=True)
                    ax1.fill(hull[:, 0], hull[:, 1], color='gray', alpha=0.5, zorder=2)
    
    # ===== RIGHT PANEL: Movables Colored by Region =====
    ax2.set_aspect('equal')
    ax2.set_xlim(xmin - 5, xmax + 5)
    ax2.set_ylim(ymin - 5, ymax + 5)
    ax2.set_title("Movables Colored by Region", fontsize=14, fontweight='bold')
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.grid(True, alpha=0.3)
    
    # Draw pipes again on right panel
    for fixed_list in fixed_per_region:
        for obs in fixed_list:
            if obs.get("center") is None:
                continue
            if obs.get("ElementType") == "Pipe":
                verts = translate_polygon(obs["verts"], obs["center"], obs.get("RotationAngle", 0.0))
                hull = get_convex_hull_vertices(verts, closed=True)
                ax2.fill(hull[:, 0], hull[:, 1], color='orange', alpha=0.8, zorder=1)
                ax2.plot(hull[:, 0], hull[:, 1], color='darkorange', linewidth=1, zorder=2)
            else:
                verts = translate_polygon(obs["verts"], obs["center"], obs.get("RotationAngle", 0.0))
                hull = get_convex_hull_vertices(verts, closed=True)
                ax2.fill(hull[:, 0], hull[:, 1], color='gray', alpha=0.5, zorder=1)
    
    # Draw movables colored by region
    legend_handles = []
    for i, movables in enumerate(movables_per_region):
        color = region_colors[i % len(region_colors)]
        n_mov = len(movables)
        
        for mov in movables:
            target = mov["target"]
            verts = translate_polygon(mov["verts"], target, mov.get("RotationAngle", 0.0))
            hull = get_convex_hull_vertices(verts, closed=True)
            ax2.fill(hull[:, 0], hull[:, 1], color=color, alpha=0.7, 
                    edgecolor='black', linewidth=0.5, zorder=3)
        
        # Add to legend
        legend_handles.append(Patch(facecolor=color, edgecolor='black', 
                                   label=f'Region {i}: {n_mov} movables'))
    
    ax2.legend(handles=legend_handles, loc='upper right', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {filename}")


def plot_regions_after_optimization(regions_info, movables_per_region, fixed_per_region, 
                                    all_results, placement_bounds, filename="regions_after.png"):
    """
    Plot regions after optimization with updated movable positions.
    """
    # Save original targets and update with results
    saved_targets = []
    for movables, result in zip(movables_per_region, all_results):
        region_saved = []
        if len(movables) == 0 or len(result) == 0:
            saved_targets.append(region_saved)
            continue
        for i, mov in enumerate(movables):
            region_saved.append(np.array(mov["target"]).copy())
            mov["target"] = np.array(result[i * 2 : i * 2 + 2]).copy()
        saved_targets.append(region_saved)
    
    # Plot with updated positions
    plot_regions(regions_info, movables_per_region, fixed_per_region, 
                placement_bounds, filename=filename)
    
    # Restore original targets
    for movables, region_saved in zip(movables_per_region, saved_targets):
        for i, mov in enumerate(movables):
            if i < len(region_saved):
                mov["target"] = region_saved[i]


def save_original_positions(movables_per_region):
    """Save original target positions before any modifications."""
    original_positions = []
    for movables in movables_per_region:
        region_positions = []
        for mov in movables:
            region_positions.append(np.array(mov["target"]).copy())
        original_positions.append(region_positions)
    return original_positions


def restore_positions(movables_per_region, saved_positions):
    """Restore target positions from saved values."""
    for movables, region_positions in zip(movables_per_region, saved_positions):
        for mov, pos in zip(movables, region_positions):
            mov["target"] = pos.copy()


def plot_regions_with_positions(regions_info, movables_per_region, fixed_per_region, 
                                 positions, placement_bounds, filename="regions_plot.png"):
    """
    Plot regions with specific positions (not current movable targets).
    
    Args:
        positions: List of lists - positions[region_idx][mov_idx] = (x, y)
    """
    # Temporarily set positions
    saved = []
    for movables, region_positions in zip(movables_per_region, positions):
        region_saved = []
        for mov, pos in zip(movables, region_positions):
            region_saved.append(np.array(mov["target"]).copy())
            mov["target"] = np.array(pos).copy()
        saved.append(region_saved)
    
    # Plot
    plot_regions(regions_info, movables_per_region, fixed_per_region, 
                placement_bounds, filename=filename)
    
    # Restore
    for movables, region_saved in zip(movables_per_region, saved):
        for mov, pos in zip(movables, region_saved):
            mov["target"] = pos


def combine_region_results(all_results, all_x0s, movables_per_region, original_movables):
    """
    Combine optimization results from all regions back into a single result array.
    """
    n_total = len(original_movables)
    combined_result = np.zeros(n_total * 2)
    combined_x0 = np.zeros(n_total * 2)

    # Use (ElementId, SegmentIndex) as composite key to handle elements with same ElementId
    element_to_idx = {(m["ElementId"], m["SegmentIndex"]): i for i, m in enumerate(original_movables)}

    for region_idx, (result, x0, movables) in enumerate(zip(all_results, all_x0s, movables_per_region)):
        if len(movables) == 0:
            continue

        for local_idx, mov in enumerate(movables):
            original_idx = element_to_idx[(mov["ElementId"], mov["SegmentIndex"])]
            combined_result[original_idx * 2] = result[local_idx * 2]
            combined_result[original_idx * 2 + 1] = result[local_idx * 2 + 1]
            combined_x0[original_idx * 2] = x0[local_idx * 2]
            combined_x0[original_idx * 2 + 1] = x0[local_idx * 2 + 1]

    return combined_result, combined_x0


def verify_overlaps_shapely(result, movables, fixed_obstacles, min_separation=0.0):
    """
    Verify overlaps using Shapely (ground truth verification).
    """
    n = len(movables)
    pts = result.reshape(-1, 2)
    overlaps = []
    
    # Get movable polygons
    mov_polys = []
    for i, mov in enumerate(movables):
        verts = translate_polygon(mov["verts"], pts[i], mov.get("RotationAngle", 0.0))
        hull = get_convex_hull_vertices(verts, closed=False)
        try:
            poly = ShapelyPolygon(hull)
            if not poly.is_valid:
                poly = poly.buffer(0)
            mov_polys.append(poly)
        except:
            mov_polys.append(None)
    
    # Get fixed obstacle polygons
    fixed_polys = []
    for obs in fixed_obstacles:
        if obs.get("center") is None:
            continue
        verts = translate_polygon(obs["verts"], obs["center"], obs.get("RotationAngle", 0.0))
        hull = get_convex_hull_vertices(verts, closed=False)
        try:
            poly = ShapelyPolygon(hull)
            if not poly.is_valid:
                poly = poly.buffer(0)
            fixed_polys.append((obs, poly))
        except:
            continue
    
    # Check movable-movable overlaps
    for i in range(n):
        if mov_polys[i] is None:
            continue
        for j in range(i + 1, n):
            if mov_polys[j] is None:
                continue
            
            if mov_polys[i].intersects(mov_polys[j]):
                intersection = mov_polys[i].intersection(mov_polys[j])
                if hasattr(intersection, 'area') and intersection.area > 1e-6:
                    overlaps.append(("mov-mov", i, j, intersection.area))
    
    # Check movable-fixed overlaps
    for i in range(n):
        if mov_polys[i] is None:
            continue
        for obs, obs_poly in fixed_polys:
            if mov_polys[i].intersects(obs_poly):
                intersection = mov_polys[i].intersection(obs_poly)
                if hasattr(intersection, 'area') and intersection.area > 1e-6:
                    overlaps.append(("mov-fix", i, obs.get("ElementId", "?"), intersection.area))
    
    return overlaps


def main():
    np.random.seed(0)

    # ========== LOAD DATA ==========
    json_path = "AnnotationCleaner_CurveLoops.json"
    print(f"Loading data from {json_path}...")
    movables, fixed_obstacles, placement_bounds = load_problem_data(json_path)

    print(f"Loaded {len(movables)} movables and {len(fixed_obstacles)} fixed obstacles")
    print(f"Placement bounds: X={placement_bounds[0]}, Y={placement_bounds[1]}")
    
    # Count pipes
    n_pipes = sum(1 for obs in fixed_obstacles if obs.get("ElementType") == "Pipe")
    print(f"Pipes: {n_pipes}")
    
    fixed_obstacles_original = fixed_obstacles.copy()
    total_start_time = time.time()

    # ========== SPLIT INTO REGIONS (using pipe subtraction) ==========
    print("\n" + "=" * 80)
    print("STAGE 1: Splitting space into regions (pipe subtraction method)")
    print("=" * 80)
    
    start_time = time.time()
    
    # Check if region splitting makes sense
    use_regions = n_pipes > 0 and len(movables) > 15
    
    if use_regions:
        # Split into regions using pipe subtraction
        # pipe_buffer controls how much pipes are expanded to detect connections
        # min_region_area filters out tiny regions
        fixed_per_region, movables_per_region, regions_info = split_into_regions(
            movables,
            fixed_obstacles,
            placement_bounds,
            pipe_buffer=0.5,  # Large buffer for clean region separation
            min_region_area=10.0,  # Filter regions smaller than this
            min_separation=0.001,  # Objects can get this close to pipes
        )
        
        # If only 1 region created, fall back to single-region mode
        if len(regions_info) <= 1:
            print("Only 1 region created, falling back to single-region mode")
            use_regions = False
    else:
        print("Skipping region splitting (no pipes or too few movables)")
    
    if not use_regions:
        # Single region mode
        all_fixed_single = [obs for obs in fixed_obstacles if obs.get("center") is not None]
        
        (xmin, xmax), (ymin, ymax) = placement_bounds
        single_boundary = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax), (xmin, ymin)]
        single_poly = ShapelyPolygon(single_boundary)
        
        regions_info = [{
            "index": 0,
            "boundary": single_boundary,
            "area": single_poly.area,
            "shapely_polygon": single_poly,
        }]
        movables_per_region = [movables]
        fixed_per_region = [all_fixed_single]
    
    split_time = time.time() - start_time
    print(f"\nRegion splitting completed in {split_time:.2f} seconds")
    print(f"Created {len(regions_info)} regions:")
    
    total_movables = 0
    total_fixed = 0
    for i, r in enumerate(regions_info):
        n_mov = len(movables_per_region[i])
        n_fix = len(fixed_per_region[i])
        n_pipes_region = sum(1 for obs in fixed_per_region[i] if obs.get("ElementType") == "Pipe")
        total_movables += n_mov
        total_fixed += n_fix
        print(f"  Region {i}: area={r['area']:.2f}, movables={n_mov}, fixed={n_fix}, pipes={n_pipes_region}")
    
    print(f"Total: {total_movables} movables, {total_fixed} fixed obstacle assignments")

    # ========== SAVE ORIGINAL POSITIONS ==========
    # Save positions before any modifications (for "before" plot)
    original_positions = save_original_positions(movables_per_region)

    # ========== VISUALIZE REGIONS (BEFORE) ==========
    print("\nGenerating region visualization (before optimization)...")
    plot_regions_with_positions(
        regions_info,
        movables_per_region,
        fixed_per_region,
        original_positions,
        placement_bounds,
        filename="regions_before.png"
    )

    # ========== PRE-PROCESSING: PULL MOVABLES INSIDE THEIR REGIONS ==========
    print("\n" + "=" * 80)
    print("STAGE 1.5: Pull movables fully inside their assigned regions")
    print("=" * 80)
    
    total_adjusted = 0
    for region_idx, (movs, region_info) in enumerate(zip(movables_per_region, regions_info)):
        if len(movs) == 0:
            continue
        
        region_poly = region_info.get("shapely_polygon")
        if region_poly is None:
            continue
        
        adjusted = pull_movables_into_region(movs, region_poly, min_margin=0.1)
        if adjusted > 0:
            print(f"  Region {region_idx}: Pulled {adjusted} movables inside")
            total_adjusted += adjusted
    
    if total_adjusted > 0:
        print(f"  Total: {total_adjusted} movables adjusted")
    else:
        print("  All movables already inside their regions")

    # ========== GREEDY OPTIMIZE EACH REGION ==========
    print("\n" + "=" * 80)
    print("STAGE 2: Greedy placement optimization (guarantees 0 overlaps)")
    print("=" * 80)
    
    start_time = time.time()
    
    all_results, all_x0s, region_indices = greedy_optimize_with_regions(
        movables_per_region,
        fixed_per_region,
        regions_info,
        min_separation=0.01,
        search_step=0.3,
        max_search_radius=50.0,
    )
    
    opt_time = time.time() - start_time
    print(f"\nGreedy optimization completed in {opt_time:.2f} seconds ({opt_time/60:.2f} minutes)")

    # ========== COMBINE RESULTS ==========
    print("\n" + "=" * 80)
    print("STAGE 3: Combining and verifying results")
    print("=" * 80)
    
    combined_result, combined_x0 = combine_region_results(
        all_results,
        all_x0s,
        movables_per_region,
        movables,
    )
    
    # Update movables with results
    # Use (ElementId, SegmentIndex) as composite key to handle elements with same ElementId
    element_to_idx = {(m["ElementId"], m["SegmentIndex"]): i for i, m in enumerate(movables)}
    for region_movables, result in zip(movables_per_region, all_results):
        for local_idx, mov in enumerate(region_movables):
            original_idx = element_to_idx[(mov["ElementId"], mov["SegmentIndex"])]
            movables[original_idx]["target"] = result[local_idx * 2 : local_idx * 2 + 2].copy()
            combined_result[original_idx * 2] = result[local_idx * 2]
            combined_result[original_idx * 2 + 1] = result[local_idx * 2 + 1]
    
    # Get ALL fixed obstacles INCLUDING pipes for overlap checks
    all_fixed = [obs for obs in fixed_obstacles if obs.get("center") is not None]

    # ========== VERIFICATION ==========
    print("\nVerifying results...")
    
    # SAT-based check
    sat_overlaps = find_all_overlaps(combined_result, movables, all_fixed, min_separation=0.0)
    print(f"SAT-based overlap check: {len(sat_overlaps)} overlaps")
    
    # Shapely-based check (ground truth)
    shapely_overlaps = verify_overlaps_shapely(combined_result, movables, all_fixed, min_separation=0.0)
    print(f"Shapely-based overlap check: {len(shapely_overlaps)} overlaps")
    
    if len(shapely_overlaps) > 0:
        print("\nOverlap details (first 10):")
        for i, ov in enumerate(shapely_overlaps[:10]):
            ov_type, idx1, idx2, area = ov
            print(f"  [{i}] {ov_type}: {idx1} <-> {idx2}, area={area:.6f}")
        if len(shapely_overlaps) > 10:
            print(f"  ... and {len(shapely_overlaps) - 10} more")
    
    # Per-region overlap check
    print("\nPer-region overlap check:")
    for i, (movs, fixed, result) in enumerate(zip(movables_per_region, fixed_per_region, all_results)):
        if len(movs) == 0:
            continue
        region_result = []
        for mov in movs:
            orig_idx = element_to_idx[(mov["ElementId"], mov["SegmentIndex"])]
            region_result.extend(combined_result[orig_idx * 2 : orig_idx * 2 + 2])
        region_result = np.array(region_result)
        overlaps = find_all_overlaps(region_result, movs, fixed, min_separation=0.0)
        status = "✓" if len(overlaps) == 0 else "✗"
        print(f"  Region {i}: {len(overlaps)} overlaps {status}")

    # ========== METRICS ==========
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    
    displacement_metric = calculate_displacement_metric(combined_x0, combined_result)
    print(f"Average displacement: {displacement_metric:.4f}")
    print(f"Final overlaps (Shapely): {len(shapely_overlaps)}")

    # ========== SAVE OUTPUT ==========
    print("\nSaving output...")
    save_optimized_output(combined_result, movables, output_path="output.json")
    print("Saved output.json")

    # ========== VISUALIZE FINAL RESULT ==========
    print("\nGenerating final visualization...")
    
    # Convert all_results to positions format for plotting
    final_positions = []
    for result, movables_list in zip(all_results, movables_per_region):
        region_positions = []
        for i in range(len(movables_list)):
            if len(result) > i * 2 + 1:
                region_positions.append(np.array([result[i * 2], result[i * 2 + 1]]))
            else:
                region_positions.append(np.array(movables_list[i]["target"]))
        final_positions.append(region_positions)
    
    # Plot regions after optimization
    plot_regions_with_positions(
        regions_info,
        movables_per_region,
        fixed_per_region,
        final_positions,
        placement_bounds,
        filename="regions_after.png"
    )
    
    # Convert original_positions to flat array for plot_result
    combined_original = np.zeros(len(movables) * 2)
    for region_idx, (region_movables, region_positions) in enumerate(zip(movables_per_region, original_positions)):
        for local_idx, (mov, pos) in enumerate(zip(region_movables, region_positions)):
            original_idx = element_to_idx[(mov["ElementId"], mov["SegmentIndex"])]
            combined_original[original_idx * 2] = pos[0]
            combined_original[original_idx * 2 + 1] = pos[1]
    
    plot_result(
        combined_result,
        combined_original,  # Use TRUE original positions, not post-pre-processing
        movables,
        fixed_obstacles_original,
        fixed_obstacles_original,
        placement_bounds,
    )
    print("Visualization complete!")

    # ========== SUMMARY ==========
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total regions: {len(regions_info)}")
    print(f"Total movables: {len(movables)}")
    print(f"Region split time: {split_time:.2f}s")
    print(f"Optimization time: {opt_time:.2f}s")
    print(f"Total time: {time.time() - total_start_time:.2f}s")
    print(f"Average displacement: {displacement_metric:.4f}")
    print(f"Final overlaps: {len(shapely_overlaps)}")
    
    if len(shapely_overlaps) == 0:
        print("\n" + "=" * 80)
        print("SUCCESS: All overlaps resolved!")
        print("=" * 80)
    else:
        print("\n" + "!" * 80)
        print(f"WARNING: {len(shapely_overlaps)} overlaps remain!")
        print("!" * 80)
    
    return len(shapely_overlaps) == 0


if __name__ == "__main__":
    success = main()