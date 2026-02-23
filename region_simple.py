"""
Region splitting module - Simple approach using pipe subtraction.

Instead of extending pipes, we:
1. Take the placement boundary as a polygon
2. Buffer pipes slightly to detect connections (corners, T-shapes)
3. Subtract buffered pipes from the boundary
4. The result naturally forms closed regions
"""

import numpy as np
from shapely.ops import unary_union
from shapely.geometry import Polygon as ShapelyPolygon, Point, box, MultiPolygon
from shapely.prepared import prep
from polygon_utils import translate_polygon


def get_pipe_polygons(fixed_obstacles, buffer_margin=0.5):
    """
    Extract pipe obstacles as Shapely polygons with optional buffer.
    
    Args:
        fixed_obstacles: List of fixed obstacle dictionaries
        buffer_margin: Buffer to add around pipes for connection detection
        
    Returns:
        List of (pipe_obs, buffered_polygon) tuples
    """
    pipe_polygons = []
    
    for obs in fixed_obstacles:
        if obs.get("ElementType") != "Pipe":
            continue
        if obs.get("center") is None:
            continue
            
        verts_local = obs["verts"]
        center = obs["center"]
        rotation = obs.get("RotationAngle", 0.0)
        verts_world = translate_polygon(verts_local, center, rotation)
        
        try:
            poly = ShapelyPolygon(verts_world)
            if not poly.is_valid:
                poly = poly.buffer(0)
            
            # Buffer the pipe slightly for connection detection
            if buffer_margin > 0:
                buffered = poly.buffer(buffer_margin, join_style=2)  # mitre join
            else:
                buffered = poly
                
            pipe_polygons.append((obs, poly, buffered))
        except Exception as e:
            print(f"Warning: Could not create polygon for pipe {obs.get('ElementId')}: {e}")
            continue
    
    return pipe_polygons


def split_into_regions(movables, fixed_obstacles, placement_bounds,
                       pipe_buffer=0.5, min_region_area=1.0, min_separation=0.3):
    """
    Split the placement area into regions using pipes as natural barriers.

    Args:
        movables: List of movable object dictionaries
        fixed_obstacles: List of fixed obstacle dictionaries
        placement_bounds: Tuple of ((xmin, xmax), (ymin, ymax))
        pipe_buffer: Buffer around pipes for region splitting (default: 0.5)
        min_region_area: Minimum area for a region to be kept (default: 1.0)
        min_separation: Minimum separation for objects from obstacles (default: 0.3)
                       Used to expand regions back toward pipes after splitting

    Returns:
        tuple: (
            fixed_obstacles_per_region,  # List of lists - fixed obstacles per region
            movables_per_region,         # List of lists - movables per region
            regions_info                 # List of region dictionaries
        )
    """
    (xmin, xmax), (ymin, ymax) = placement_bounds
    
    # 1. Create placement boundary polygon
    boundary_poly = box(xmin, ymin, xmax, ymax)
    total_area = boundary_poly.area
    
    print(f"  Placement bounds: x=[{xmin}, {xmax}], y=[{ymin}, {ymax}], area={total_area:.1f}")
    
    # 2. Get pipe polygons (buffered for connection detection)
    pipe_data = get_pipe_polygons(fixed_obstacles, buffer_margin=pipe_buffer)
    pipe_obstacles = [p[0] for p in pipe_data]
    pipe_polygons_original = [p[1] for p in pipe_data]
    pipe_polygons_buffered = [p[2] for p in pipe_data]
    
    print(f"  Found {len(pipe_data)} pipes")
    
    # 3. Union all buffered pipes
    if len(pipe_polygons_buffered) > 0:
        pipes_union = unary_union(pipe_polygons_buffered)
    else:
        pipes_union = ShapelyPolygon()  # Empty polygon
    
    # 4. Subtract pipes from boundary to get free regions
    free_space = boundary_poly.difference(pipes_union)

    # 5. Extract individual regions
    if free_space.is_empty:
        print("  Warning: No free space after subtracting pipes!")
        regions_polys = [boundary_poly]
    elif isinstance(free_space, MultiPolygon):
        regions_polys = list(free_space.geoms)
    else:
        regions_polys = [free_space]

    # Filter out tiny regions before expansion
    regions_polys = [r for r in regions_polys if r.area >= min_region_area]

    # Keep original regions for movable assignment
    original_regions = regions_polys.copy()

    # 6. Expand regions back toward pipes to allow closer placement
    # Three separate buffer uses:
    #   1. pipe_buffer: for splitting regions (creates clean separation)
    #   2. min_separation/2: for placing movables around obstacles
    #   3. Original regions (no expansion): for assigning movables to regions
    expansion_amount = pipe_buffer - (min_separation / 2)
    if expansion_amount > 0:
        print(f"  Expanding regions by {expansion_amount:.2f} to allow closer placement to pipes")
        expanded_regions = []
        for poly in regions_polys:
            # Buffer outward to expand toward pipes
            expanded = poly.buffer(expansion_amount, join_style=2)  # mitre join
            # Clip to original boundary
            expanded = expanded.intersection(boundary_poly)
            if not expanded.is_empty:
                expanded_regions.append(expanded)
        regions_polys = expanded_regions
    
    print(f"  Created {len(regions_polys)} regions from pipe barriers")
    
    # 6. Get non-pipe fixed obstacles
    non_pipe_obstacles = [obs for obs in fixed_obstacles if obs.get("ElementType") != "Pipe"]
    
    # Create Shapely polygons for non-pipe obstacles
    fixed_obstacle_shapes = []
    for obs in non_pipe_obstacles:
        if obs.get("center") is None:
            continue
        verts_world = translate_polygon(obs["verts"], obs["center"], obs.get("RotationAngle", 0.0))
        try:
            shape = ShapelyPolygon(verts_world)
            if not shape.is_valid:
                shape = shape.buffer(0)
            fixed_obstacle_shapes.append((obs, shape))
        except:
            continue
    
    # 7. Prepare original regions for movable assignment (no expansion buffer)
    prepared_original_regions = [prep(poly) for poly in original_regions]

    # 8. Assign movables to regions using ORIGINAL (unexpanded) boundaries
    movables_per_region = [[] for _ in range(len(prepared_original_regions))]

    for mov in movables:
        center_pt = Point(mov["target"])
        assigned = False

        for i, prep_poly in enumerate(prepared_original_regions):
            if prep_poly.contains(center_pt):
                movables_per_region[i].append(mov)
                assigned = True
                break

        if not assigned:
            # Find closest region (using original regions)
            dists = [poly.distance(center_pt) for poly in original_regions]
            closest_idx = np.argmin(dists)
            movables_per_region[closest_idx].append(mov)
    
    # 9. Assign non-pipe fixed obstacles to regions
    fixed_per_region = [[] for _ in range(len(regions_polys))]
    
    for obs, obs_shape in fixed_obstacle_shapes:
        for i, region_poly in enumerate(regions_polys):
            if region_poly.intersects(obs_shape):
                intersection = region_poly.intersection(obs_shape)
                if hasattr(intersection, 'area') and intersection.area > 1e-9:
                    fixed_per_region[i].append(obs)
    
    # 10. Assign pipes to regions they border
    # Use original (non-buffered) pipe polygons with min_separation tolerance
    # This accounts for the expanded region boundaries
    for pipe_obs, pipe_poly_orig, pipe_poly_buffered in pipe_data:
        for i, region_poly in enumerate(regions_polys):
            # Check if pipe is within min_separation distance of region
            # (Region was expanded by pipe_buffer - min_separation/2, so pipes
            #  should be within min_separation distance of region boundary)
            if region_poly.distance(pipe_poly_orig) < min_separation:
                fixed_per_region[i].append(pipe_obs)
    
    # 11. Build regions_info
    regions_info = []
    final_movables_per_region = []
    final_fixed_per_region = []
    
    for i, poly in enumerate(regions_polys):
        movs = movables_per_region[i]
        
        # Skip empty regions
        if len(movs) == 0:
            continue
        
        coords = list(poly.exterior.coords)
        
        region_dict = {
            "index": len(regions_info),
            "boundary": coords,
            "area": poly.area,
            "shapely_polygon": poly,
        }
        
        regions_info.append(region_dict)
        final_movables_per_region.append(movs)
        final_fixed_per_region.append(fixed_per_region[i])
    
    # Print summary
    print(f"  Final: {len(regions_info)} regions with movables")
    for i, (movs, fixed) in enumerate(zip(final_movables_per_region, final_fixed_per_region)):
        n_pipes = sum(1 for obs in fixed if obs.get("ElementType") == "Pipe")
        print(f"    Region {i}: {len(movs)} movables, {len(fixed)} fixed ({n_pipes} pipes)")
    
    return final_fixed_per_region, final_movables_per_region, regions_info


def plot_regions(regions_info, movables_per_region, fixed_per_region, placement_bounds, 
                 pipe_obstacles=None, title="Region Splitting"):
    """
    Plot the region splitting visualization.
    
    Args:
        regions_info: List of region info dictionaries
        movables_per_region: List of movable lists per region
        fixed_per_region: List of fixed obstacle lists per region
        placement_bounds: Tuple of ((xmin, xmax), (ymin, ymax))
        pipe_obstacles: Optional list of pipe obstacles for visualization
        title: Plot title
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon
    
    n_regions = len(regions_info)
    
    # Distinct colors for regions
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
    ax1.set_title(f"{title} - Overview ({n_regions} regions)", fontsize=14, fontweight="bold")
    
    # Plot each region
    for i, region in enumerate(regions_info):
        boundary = region.get("boundary", [])
        if len(boundary) < 3:
            continue
        
        color = region_colors[i % len(region_colors)]
        
        poly_patch = MplPolygon(
            boundary,
            closed=True,
            facecolor=color,
            alpha=0.3,
            edgecolor=color,
            linewidth=3,
        )
        ax1.add_patch(poly_patch)
        
        # Label
        centroid = np.mean(boundary, axis=0)
        n_mov = len(movables_per_region[i]) if i < len(movables_per_region) else 0
        n_fix = len(fixed_per_region[i]) if i < len(fixed_per_region) else 0
        n_pipes = sum(1 for obs in fixed_per_region[i] if obs.get("ElementType") == "Pipe") if i < len(fixed_per_region) else 0
        
        ax1.text(
            centroid[0], centroid[1],
            f"R{i}\n{n_mov} mov\n{n_fix} fix\n({n_pipes} pipes)",
            ha="center", va="center",
            fontsize=9, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor=color, linewidth=2)
        )
    
    # Plot pipes
    all_fixed_ids = set()
    for fixed_list in fixed_per_region:
        for obs in fixed_list:
            if obs["ElementId"] in all_fixed_ids or obs.get("center") is None:
                continue
            all_fixed_ids.add(obs["ElementId"])
            
            verts_world = translate_polygon(obs["verts"], obs["center"], obs.get("RotationAngle", 0.0))
            
            if obs.get("ElementType") == "Pipe":
                ax1.fill(verts_world[:, 0], verts_world[:, 1], color="orange", alpha=0.8,
                        edgecolor="darkorange", linewidth=1.5)
            else:
                ax1.fill(verts_world[:, 0], verts_world[:, 1], color="gray", alpha=0.5,
                        edgecolor="darkgray", linewidth=1)
    
    # ============ RIGHT PLOT: Movables by Region ============
    ax2 = axes[1]
    ax2.set_aspect("equal")
    ax2.set_xlim(placement_bounds[0])
    ax2.set_ylim(placement_bounds[1])
    ax2.grid(True, alpha=0.3)
    ax2.set_title("Movables Colored by Region", fontsize=14, fontweight="bold")
    
    # Plot region boundaries
    for i, region in enumerate(regions_info):
        boundary = region.get("boundary", [])
        if len(boundary) < 3:
            continue
        color = region_colors[i % len(region_colors)]
        boundary = np.array(boundary)
        ax2.plot(boundary[:, 0], boundary[:, 1], color=color, linewidth=2, alpha=0.5)
    
    # Plot fixed obstacles
    all_fixed_ids = set()
    for fixed_list in fixed_per_region:
        for obs in fixed_list:
            if obs["ElementId"] in all_fixed_ids or obs.get("center") is None:
                continue
            all_fixed_ids.add(obs["ElementId"])
            
            verts_world = translate_polygon(obs["verts"], obs["center"], obs.get("RotationAngle", 0.0))
            
            if obs.get("ElementType") == "Pipe":
                ax2.fill(verts_world[:, 0], verts_world[:, 1], color="orange", alpha=0.7,
                        edgecolor="darkorange", linewidth=1)
            else:
                ax2.fill(verts_world[:, 0], verts_world[:, 1], color="gray", alpha=0.4,
                        edgecolor="darkgray", linewidth=0.5)
    
    # Plot movables colored by region
    for i, movables in enumerate(movables_per_region):
        color = region_colors[i % len(region_colors)]
        for j, mov in enumerate(movables):
            verts_world = translate_polygon(mov["verts"], mov["target"], mov.get("RotationAngle", 0.0))
            ax2.fill(verts_world[:, 0], verts_world[:, 1], color=color, alpha=0.6,
                    edgecolor="black", linewidth=0.5,
                    label=f"Region {i}" if j == 0 else "")
            ax2.plot(mov["target"][0], mov["target"][1], ".", color="black", markersize=2)
    
    # Legend
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