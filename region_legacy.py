"""
Region splitting module for dividing the optimization space using pipe extensions.

This module:
1. Extracts pipe centerlines from rectangular pipe obstacles
2. Extends pipes until they hit boundaries or other pipes
3. Creates closed regions from these extended lines
4. Assigns movables and fixed obstacles to their respective regions
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from shapely.ops import polygonize, unary_union
from shapely.geometry import LineString, Point, Polygon as ShapelyPolygon, box
from shapely.prepared import prep
from polygon_utils import translate_polygon


def split_into_regions(movables, fixed_obstacles, placement_bounds, 
                       min_movables_per_region=0, min_area_ratio=0.02, enable_merging=True):
    """
    Splits the space into regions using extended pipes and assigns movables and fixed obstacles.
    Merges small regions with neighbors to avoid over-fragmentation.
    Includes pipe obstacles in each region they border.

    Args:
        movables: List of movable object dictionaries
        fixed_obstacles: List of fixed obstacle dictionaries
        placement_bounds: Tuple of ((xmin, xmax), (ymin, ymax))
        min_movables_per_region: Minimum movables to keep a region separate (default: 5)
        min_area_ratio: Minimum area ratio (region_area / total_area) to keep separate (default: 0.02)

    Returns:
        tuple: (
            fixed_obstacles_per_region,  # List of lists - fixed obstacles (including bordering pipes) per region
            movables_per_region,         # List of lists - movables per region
            regions_info                 # List of region dictionaries with boundary, area, index
        )
    """
    # 1. Get pipe obstacles and their geometry
    pipe_obstacles = [obs for obs in fixed_obstacles if obs.get("ElementType") == "Pipe"]
    non_pipe_obstacles = [obs for obs in fixed_obstacles if obs.get("ElementType") != "Pipe"]
    
    # Create Shapely polygons for pipe obstacles
    pipe_shapes = []
    for obs in pipe_obstacles:
        verts_local = obs["verts"]
        center = obs["center"]
        rotation = obs["RotationAngle"]
        verts_world = translate_polygon(verts_local, center, rotation)
        
        try:
            shape = ShapelyPolygon(verts_world)
            if not shape.is_valid:
                shape = shape.buffer(0)
            pipe_shapes.append((obs, shape))
        except Exception:
            continue
    
    # 2. Get extended pipes for region creation
    pipe_segments = get_pipe_segments(fixed_obstacles)
    extended_lines = extend_pipes(pipe_segments, placement_bounds)

    # 3. Create boundary lines
    (xmin, xmax), (ymin, ymax) = placement_bounds
    boundary_lines = [
        np.array([[xmin, ymin], [xmax, ymin]]),  # bottom
        np.array([[xmax, ymin], [xmax, ymax]]),  # right
        np.array([[xmax, ymax], [xmin, ymax]]),  # top
        np.array([[xmin, ymax], [xmin, ymin]]),  # left
    ]

    all_lines_np = extended_lines + boundary_lines

    # 4. Convert to Shapely LineStrings and polygonize
    shapely_lines = [LineString(line) for line in all_lines_np]
    noded_lines = unary_union(shapely_lines)
    regions_polys = list(polygonize(noded_lines))

    # Filter degenerate regions (extremely small areas)
    min_area_threshold = 1e-6
    # regions_polys = [r for r in regions_polys if r.area > min_area_threshold]

    # Calculate total area for ratio comparison
    total_area = sum(r.area for r in regions_polys)

    # 5. Create Shapely polygons for non-pipe fixed obstacles
    fixed_obstacle_shapes = []
    for obs in non_pipe_obstacles:
        verts_local = obs["verts"]
        center = obs["center"]
        rotation = obs["RotationAngle"]
        verts_world = translate_polygon(verts_local, center, rotation)
        
        try:
            shape = ShapelyPolygon(verts_world)
            if not shape.is_valid:
                shape = shape.buffer(0)
            fixed_obstacle_shapes.append((obs, shape))
        except Exception:
            continue

    # 6. Prepare regions for faster containment/intersection checks
    prepared_regions = [prep(poly) for poly in regions_polys]

    # 7. Assign movables to regions (by center point containment)
    movables_per_region_temp = [[] for _ in range(len(regions_polys))]

    for mov in movables:
        center_pt = Point(mov["target"])
        assigned = False
        
        for i, prep_poly in enumerate(prepared_regions):
            if prep_poly.contains(center_pt):
                movables_per_region_temp[i].append(mov)
                assigned = True
                break

        if not assigned:
            dists = [poly.distance(center_pt) for poly in regions_polys]
            closest_idx = np.argmin(dists)
            movables_per_region_temp[closest_idx].append(mov)

    # 8. Assign non-pipe fixed obstacles to regions
    fixed_per_region_temp = [[] for _ in range(len(regions_polys))]

    for obs, obs_shape in fixed_obstacle_shapes:
        for i, region_poly in enumerate(regions_polys):
            if region_poly.intersects(obs_shape):
                intersection = region_poly.intersection(obs_shape)
                if intersection.area > 1e-9:
                    fixed_per_region_temp[i].append(obs)

    # 9. Assign PIPE obstacles to regions they border/intersect
    # A pipe borders a region if the pipe polygon touches or intersects the region
    for pipe_obs, pipe_shape in pipe_shapes:
        for i, region_poly in enumerate(regions_polys):
            # Check if pipe touches or intersects region
            if region_poly.intersects(pipe_shape) or region_poly.touches(pipe_shape):
                # Check if there's actual overlap or shared boundary
                intersection = region_poly.intersection(pipe_shape)
                # Include if there's area overlap OR significant boundary contact
                has_overlap = False
                if hasattr(intersection, 'area') and intersection.area > 1e-9:
                    has_overlap = True
                elif hasattr(intersection, 'length') and intersection.length > 1e-6:
                    has_overlap = True
                elif not intersection.is_empty:
                    has_overlap = True
                
                if has_overlap:
                    # Add pipe to this region's fixed obstacles
                    fixed_per_region_temp[i].append(pipe_obs)

    # 10. MERGE SMALL REGIONS - ONLY across extended pipe lines (not points, not actual pipes)
    n_regions = len(regions_polys)
    active_regions = set(range(n_regions))  # Start with all regions active
    
    if enable_merging:
        # Create LineStrings for extended pipes (for checking if boundary lies on extended pipe)
        extended_pipe_lines = [LineString(line) for line in extended_lines]
        
        # mergeable_pairs[key] = True means regions can be merged
        mergeable_pairs = {}
        adjacency = [set() for _ in range(n_regions)]
        
        for i in range(n_regions):
            for j in range(i + 1, n_regions):
                if not (regions_polys[i].touches(regions_polys[j]) or regions_polys[i].intersects(regions_polys[j])):
                    continue
                
                intersection = regions_polys[i].intersection(regions_polys[j])
                
                # Must share more than a point (need a line segment)
                if not hasattr(intersection, 'length') or intersection.length < 1e-6:
                    continue  # Only share a point - not adjacent for merging
                
                adjacency[i].add(j)
                adjacency[j].add(i)
                key = (min(i, j), max(i, j))
                
                # Check if an actual pipe lies on this shared boundary
                has_actual_pipe = False
                for pipe_obs, pipe_shape in pipe_shapes:
                    if pipe_shape.intersects(intersection) or pipe_shape.distance(intersection) < 0.1:
                        has_actual_pipe = True
                        break
                
                if has_actual_pipe:
                    # Cannot merge - actual pipe separates them
                    mergeable_pairs[key] = False
                    continue
                
                # Check if shared boundary lies along an extended pipe line
                is_on_extended_pipe = False
                for ext_line in extended_pipe_lines:
                    if intersection.distance(ext_line) < 0.1:
                        is_on_extended_pipe = True
                        break
                
                if is_on_extended_pipe:
                    # Can merge - separated only by extended pipe line (virtual divider)
                    mergeable_pairs[key] = True
                else:
                    # Cannot merge - they share a boundary that's not an extended pipe
                    mergeable_pairs[key] = False

        # Identify regions to merge (only merge across extended pipe lines)
        merge_target = {}
        
        region_sizes = [(i, len(movables_per_region_temp[i]), regions_polys[i].area) 
                        for i in range(n_regions)]
        region_sizes.sort(key=lambda x: (x[1], x[2]))
        
        for region_idx, n_movables, area in region_sizes:
            if region_idx not in active_regions:
                continue
                
            area_ratio = area / total_area if total_area > 0 else 0
            should_merge = (n_movables < min_movables_per_region and 
                           n_movables > 0 and 
                           area_ratio < min_area_ratio)
            
            if n_movables == 0:
                should_merge = True
            
            if should_merge:
                best_neighbor = None
                best_score = -1
                
                for neighbor_idx in adjacency[region_idx]:
                    if neighbor_idx not in active_regions:
                        while neighbor_idx in merge_target:
                            neighbor_idx = merge_target[neighbor_idx]
                    
                    if neighbor_idx not in active_regions:
                        continue
                    
                    # ONLY MERGE if they share an extended pipe line boundary
                    key = (min(region_idx, neighbor_idx), max(region_idx, neighbor_idx))
                    if not mergeable_pairs.get(key, False):
                        continue  # Cannot merge - not separated by extended pipe
                    
                    neighbor_movables = len(movables_per_region_temp[neighbor_idx])
                    shared_boundary = regions_polys[region_idx].intersection(regions_polys[neighbor_idx])
                    boundary_bonus = shared_boundary.length if hasattr(shared_boundary, 'length') else 0
                    
                    score = neighbor_movables * 10 + boundary_bonus
                    
                    if score > best_score:
                        best_score = score
                        best_neighbor = neighbor_idx
                
                if best_neighbor is not None:
                    merge_target[region_idx] = best_neighbor
                    active_regions.discard(region_idx)
                    
                    # Transfer movables
                    movables_per_region_temp[best_neighbor].extend(movables_per_region_temp[region_idx])
                    movables_per_region_temp[region_idx] = []
                    
                    # Transfer fixed obstacles (avoid duplicates by ElementId)
                    existing_ids = {obs["ElementId"] for obs in fixed_per_region_temp[best_neighbor]}
                    for obs in fixed_per_region_temp[region_idx]:
                        if obs["ElementId"] not in existing_ids:
                            fixed_per_region_temp[best_neighbor].append(obs)
                            existing_ids.add(obs["ElementId"])
                    fixed_per_region_temp[region_idx] = []
                    
                    # Merge polygons
                    merged_poly = unary_union([regions_polys[best_neighbor], regions_polys[region_idx]])
                    if hasattr(merged_poly, 'geoms'):
                        merged_poly = max(merged_poly.geoms, key=lambda p: p.area)
                    regions_polys[best_neighbor] = merged_poly
                    
                    # Update adjacency and mergeable_pairs for the merged region
                    for adj in adjacency[region_idx]:
                        if adj != best_neighbor and adj in active_regions:
                            adjacency[best_neighbor].add(adj)
                            adjacency[adj].add(best_neighbor)
                            adjacency[adj].discard(region_idx)
                            
                            # Inherit mergeability status from the old region
                            old_key = (min(region_idx, adj), max(region_idx, adj))
                            new_key = (min(best_neighbor, adj), max(best_neighbor, adj))
                            old_mergeable = mergeable_pairs.get(old_key, False)
                            existing_mergeable = mergeable_pairs.get(new_key, False)
                            mergeable_pairs[new_key] = old_mergeable and existing_mergeable
        
        print(f"  Region merging: {n_regions} initial -> {len(active_regions)} after merge")
    else:
        print(f"  Region merging: DISABLED ({n_regions} regions)")

    # 11. Build final output
    final_regions_info = []
    final_movables_per_region = []
    final_fixed_per_region = []

    for i in sorted(active_regions):
        movs = movables_per_region_temp[i]

        if len(movs) == 0:
            continue

        poly = regions_polys[i]
        coords = list(poly.exterior.coords)
        
        boundary_segments = extract_boundary_segments(
            poly, placement_bounds, extended_lines
        )

        region_dict = {
            "index": len(final_regions_info),
            "boundary": coords,
            "boundary_without_pipes": boundary_segments,
            "area": poly.area,
            "shapely_polygon": poly,
        }
        
        final_regions_info.append(region_dict)
        final_movables_per_region.append(movs)
        final_fixed_per_region.append(fixed_per_region_temp[i])

    # Print summary
    total_pipes_assigned = sum(
        sum(1 for obs in fixed_list if obs.get("ElementType") == "Pipe")
        for fixed_list in final_fixed_per_region
    )
    print(f"  Region merging: {n_regions} initial -> {len(final_regions_info)} final regions")
    print(f"  (Only merged regions sharing an extended pipe line, not actual pipes or points)")
    print(f"  Pipe obstacles assigned to regions: {total_pipes_assigned} assignments ({len(pipe_obstacles)} unique pipes)")

    return final_fixed_per_region, final_movables_per_region, final_regions_info


def extract_boundary_segments(region_poly, placement_bounds, extended_pipe_lines):
    """
    Extract the boundary segments of a region that lie on the placement bounds
    (excluding pipe-derived segments).
    
    Args:
        region_poly: Shapely Polygon for the region
        placement_bounds: ((xmin, xmax), (ymin, ymax))
        extended_pipe_lines: List of numpy arrays representing extended pipe lines
        
    Returns:
        List of coordinate pairs representing boundary segments on placement bounds
    """
    (xmin, xmax), (ymin, ymax) = placement_bounds
    tolerance = 1e-6
    
    boundary_on_bounds = []
    coords = list(region_poly.exterior.coords)
    
    for i in range(len(coords) - 1):
        p1 = coords[i]
        p2 = coords[i + 1]
        
        # Check if this segment lies on one of the placement bounds
        on_bound = False
        
        # Check bottom edge (y = ymin)
        if abs(p1[1] - ymin) < tolerance and abs(p2[1] - ymin) < tolerance:
            on_bound = True
        # Check top edge (y = ymax)
        elif abs(p1[1] - ymax) < tolerance and abs(p2[1] - ymax) < tolerance:
            on_bound = True
        # Check left edge (x = xmin)
        elif abs(p1[0] - xmin) < tolerance and abs(p2[0] - xmin) < tolerance:
            on_bound = True
        # Check right edge (x = xmax)
        elif abs(p1[0] - xmax) < tolerance and abs(p2[0] - xmax) < tolerance:
            on_bound = True
            
        if on_bound:
            boundary_on_bounds.append([list(p1), list(p2)])
    
    return boundary_on_bounds


def get_pipe_segments(fixed_obstacles):
    """
    Extracts the centerline segments for each pipe.
    
    Returns:
        List of dictionaries: {'line': np.array([p1, p2]), 'width': float, 'length': float}
    """
    segments = []

    for obs in fixed_obstacles:
        if obs.get("ElementType") != "Pipe":
            continue

        # Get absolute vertices
        verts_local = obs["verts"]
        center = obs["center"]
        rotation = obs["RotationAngle"]
        verts_world = translate_polygon(verts_local, center, rotation)

        if len(verts_world) != 4:
            continue

        # Calculate edge lengths
        num_verts = len(verts_world)
        dists = []
        for i in range(num_verts):
            p1 = verts_world[i]
            p2 = verts_world[(i + 1) % num_verts]
            d = np.linalg.norm(p2 - p1)
            dists.append((d, i, (i + 1) % num_verts))

        dists.sort(key=lambda x: x[0])

        # Short edges are the width, long edges are the length
        width = (dists[0][0] + dists[1][0]) / 2.0
        length = (dists[2][0] + dists[3][0]) / 2.0

        # Check aspect ratio - must be elongated (length/width > 10)
        if width > 0 and length / width < 20.0:
            continue

        # Get midpoints of the short edges (these define the centerline)
        idx1_a, idx1_b = dists[0][1], dists[0][2]
        mid1 = (verts_world[idx1_a] + verts_world[idx1_b]) / 2.0

        idx2_a, idx2_b = dists[1][1], dists[1][2]
        mid2 = (verts_world[idx2_a] + verts_world[idx2_b]) / 2.0

        segments.append({
            "line": np.array([mid1, mid2]),
            "width": width,
            "length": length
        })

    return segments


def ray_box_intersection(ray_origin, ray_dir, bounds):
    """
    Calculate intersection of a ray with an axis-aligned bounding box.
    
    Returns:
        float or None: Parameter t along ray where intersection occurs
    """
    (xmin, xmax), (ymin, ymax) = bounds
    t_min = -float("inf")
    t_max = float("inf")

    # X-axis slab
    if abs(ray_dir[0]) < 1e-9:
        if ray_origin[0] < xmin or ray_origin[0] > xmax:
            return None
    else:
        t1 = (xmin - ray_origin[0]) / ray_dir[0]
        t2 = (xmax - ray_origin[0]) / ray_dir[0]
        t_min = max(t_min, min(t1, t2))
        t_max = min(t_max, max(t1, t2))

    # Y-axis slab
    if abs(ray_dir[1]) < 1e-9:
        if ray_origin[1] < ymin or ray_origin[1] > ymax:
            return None
    else:
        t1 = (ymin - ray_origin[1]) / ray_dir[1]
        t2 = (ymax - ray_origin[1]) / ray_dir[1]
        t_min = max(t_min, min(t1, t2))
        t_max = min(t_max, max(t1, t2))

    if t_max >= t_min and t_max > 0:
        if t_min >= 0:
            return t_min
        return t_max

    return None


def ray_virtual_segment_intersection(ray_origin, ray_dir, p3, p4, search_radius):
    """
    Finds intersection of ray with segment p3-p4, treating p3-p4 as extended
    by search_radius on both ends.
    
    Args:
        ray_origin: Starting point of ray
        ray_dir: Direction vector of ray
        p3, p4: Endpoints of target segment
        search_radius: How far to extend the segment virtually
        
    Returns:
        float or None: Parameter t along ray if intersection valid
    """
    def cross(a, b):
        return a[0] * b[1] - a[1] * b[0]

    denom = cross(ray_dir, p4 - p3)
    if abs(denom) < 1e-9:
        return None  # Parallel lines

    # t along ray, u along segment (normalized)
    t = cross(p3 - ray_origin, p4 - p3) / denom
    u = cross(p3 - ray_origin, ray_dir) / denom

    # Calculate segment length to map u to distance
    seg_vec = p4 - p3
    seg_len = np.linalg.norm(seg_vec)
    if seg_len < 1e-9:
        return None

    # Convert u (normalized 0-1) to actual distance along segment
    dist_along_seg = u * seg_len

    # Check if intersection is valid (within extended segment range)
    if t >= -1e-5 and -search_radius <= dist_along_seg <= (seg_len + search_radius):
        return t

    return None


def extend_pipes(segments, placement_bounds):
    """
    Extends pipe segments until they hit boundaries or other pipes.
    Uses 'virtual extension' search relative to segment dimensions.
    
    Args:
        segments: List of pipe segment dictionaries
        placement_bounds: ((xmin, xmax), (ymin, ymax))
        
    Returns:
        List of numpy arrays representing extended line segments
    """
    if not segments:
        return []

    # Work on deep copy of 'line' arrays
    final_lines = [s["line"].copy() for s in segments]

    # Process each segment's ends
    for i in range(len(final_lines)):
        for end_idx in [0, 1]:
            # Current state of segment
            p_start = final_lines[i][end_idx]
            p_other = final_lines[i][1 - end_idx]

            # Direction outwards from the segment
            direction = p_start - p_other
            length = np.linalg.norm(direction)
            if length < 1e-9:
                continue
            unit_dir = direction / length

            # Find closest hit
            min_dist = float("inf")

            # 1. Bounds check (always valid fallback)
            t_box = ray_box_intersection(p_start, unit_dir, placement_bounds)
            if t_box is not None:
                min_dist = t_box

            # 2. Check against other pipes (with virtual extension tolerance)
            for j, other in enumerate(segments):
                if i == j:
                    continue
                    
                # Use original geometry for target
                p3, p4 = other["line"]

                # Search radius based on the OTHER segment's dimensions
                other_radius = min(other["width"], other["length"])

                t = ray_virtual_segment_intersection(
                    p_start, unit_dir, p3, p4, other_radius
                )
                if t is not None and t < min_dist:
                    min_dist = t

            # Extend to the closest intersection
            if min_dist < float("inf") and min_dist > 0:
                final_lines[i][end_idx] = p_start + unit_dir * min_dist

    return final_lines


def visualize_extended_pipes(
    original_segments, extended_segments_lines, bounds, filename="extended_pipes.png"
):
    """
    Visualize original and extended pipe segments.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect("equal")
    (xmin, xmax), (ymin, ymax) = bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    rect = plt.Rectangle(
        (xmin, ymin),
        xmax - xmin,
        ymax - ymin,
        fill=False,
        edgecolor="black",
        linestyle="--",
    )
    ax.add_patch(rect)

    for p1, p2 in extended_segments_lines:
        ax.plot(
            [p1[0], p2[0]],
            [p1[1], p2[1]],
            "r-",
            linewidth=2,
            alpha=0.6,
            label="Extended"
            if "Extended" not in [l.get_label() for l in ax.lines]
            else "",
        )

    for seg in original_segments:
        p1, p2 = seg["line"]
        ax.plot(
            [p1[0], p2[0]],
            [p1[1], p2[1]],
            "b-",
            linewidth=1.5,
            label="Original"
            if "Original" not in [l.get_label() for l in ax.lines]
            else "",
        )

    ax.legend()
    plt.title("Pipe Extension Visualization")
    plt.savefig(filename)
    plt.close()


def visualize_regions(
    regions_info,
    movables_per_region,
    fixed_per_region,
    placement_bounds,
    pipe_segments=None,
    extended_lines=None,
    filename="regions_visualization.png",
):
    """
    Visualize the split regions with movables and fixed obstacles.
    """
    import matplotlib.cm as cm
    
    fig, ax = plt.subplots(figsize=(14, 14))
    ax.set_aspect("equal")
    
    (xmin, xmax), (ymin, ymax) = placement_bounds
    ax.set_xlim(xmin - 0.5, xmax + 0.5)
    ax.set_ylim(ymin - 0.5, ymax + 0.5)

    # Draw placement boundary
    rect = plt.Rectangle(
        (xmin, ymin),
        xmax - xmin,
        ymax - ymin,
        fill=False,
        edgecolor="black",
        linestyle="--",
        linewidth=2,
    )
    ax.add_patch(rect)

    # Color map for regions
    n_regions = len(regions_info)
    colors = cm.tab10(np.linspace(0, 1, max(n_regions, 1)))

    # Draw extended pipe lines
    if extended_lines is not None:
        for line in extended_lines:
            p1, p2 = line
            ax.plot(
                [p1[0], p2[0]],
                [p1[1], p2[1]],
                "r-",
                linewidth=2,
                alpha=0.8,
            )

    # Draw original pipe segments
    if pipe_segments is not None:
        for seg in pipe_segments:
            p1, p2 = seg["line"]
            ax.plot(
                [p1[0], p2[0]],
                [p1[1], p2[1]],
                "b-",
                linewidth=3,
            )

    # Draw regions
    for i, region in enumerate(regions_info):
        poly_coords = region["boundary"]
        polygon = MplPolygon(
            poly_coords,
            alpha=0.2,
            facecolor=colors[i % len(colors)],
            edgecolor=colors[i % len(colors)],
            linewidth=2,
        )
        ax.add_patch(polygon)
        
        # Label region
        centroid = region["shapely_polygon"].centroid
        ax.annotate(
            f"R{region['index']}",
            (centroid.x, centroid.y),
            fontsize=12,
            fontweight="bold",
            ha="center",
            va="center",
        )

    # Draw movables
    for i, movs in enumerate(movables_per_region):
        for mov in movs:
            center = mov["target"]
            ax.plot(
                center[0], center[1],
                "o",
                color=colors[i % len(colors)],
                markersize=8,
                markeredgecolor="black",
            )

    # Draw fixed obstacles
    for i, fixed_list in enumerate(fixed_per_region):
        for obs in fixed_list:
            center = obs["center"]
            ax.plot(
                center[0], center[1],
                "s",
                color=colors[i % len(colors)],
                markersize=6,
                markeredgecolor="darkgray",
                alpha=0.7,
            )

    plt.title(f"Region Split Visualization ({n_regions} regions)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True, alpha=0.3)
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    
    return filename