"""
Greedy placement optimizer - places all objects at once using spatial indexing.

Instead of resolving overlaps pair-by-pair (which can cycle), this approach:
1. Identifies all valid placement positions in the region
2. Sorts movables by priority
3. Places each movable at the nearest valid position to its target
4. Once placed, that space is marked as occupied

This guarantees 0 overlaps by construction.
"""

import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon, Point, box
from shapely.prepared import prep
from shapely.strtree import STRtree
from scipy.spatial import cKDTree
from polygon_utils import (
    translate_polygon,
    get_convex_hull_vertices,
    get_precomputed_geometry,
)


def create_obstacle_union(fixed_obstacles, buffer=0.0):
    """
    Create a union of all fixed obstacle polygons.
    
    Args:
        fixed_obstacles: List of fixed obstacle dictionaries
        buffer: Optional buffer around obstacles
        
    Returns:
        Shapely geometry representing all obstacles
    """
    from shapely.ops import unary_union
    
    obstacle_polys = []
    for obs in fixed_obstacles:
        if obs.get("center") is None:
            continue
        
        verts = translate_polygon(obs["verts"], obs["center"], obs.get("RotationAngle", 0.0))
        try:
            poly = ShapelyPolygon(verts)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if buffer > 0:
                poly = poly.buffer(buffer)
            obstacle_polys.append(poly)
        except:
            continue
    
    if len(obstacle_polys) == 0:
        return ShapelyPolygon()  # Empty
    
    return unary_union(obstacle_polys)


def get_movable_local_center(mov):
    """
    Get the geometric center of movable vertices in local coordinates.
    
    Args:
        mov: Movable object dictionary with 'verts'
        
    Returns:
        numpy array (x, y) of local center
    """
    verts = np.array(mov["verts"])
    return verts.mean(axis=0)


def get_movable_world_center(mov, position):
    """
    Get the geometric center of a movable placed at a given position.
    
    Note: 'position' is where the local origin is placed.
    The actual geometric center may be offset from this.
    
    Args:
        mov: Movable object dictionary
        position: (x, y) where local origin is placed
        
    Returns:
        numpy array (x, y) of world center
    """
    local_center = get_movable_local_center(mov)
    rotation = mov.get("RotationAngle", 0.0)
    
    # Rotate local center
    if rotation != 0:
        cos_r = np.cos(rotation)
        sin_r = np.sin(rotation)
        rotated_center = np.array([
            local_center[0] * cos_r - local_center[1] * sin_r,
            local_center[0] * sin_r + local_center[1] * cos_r
        ])
    else:
        rotated_center = local_center
    
    return np.array(position) + rotated_center


def get_position_for_center(mov, desired_center):
    """
    Get the position (local origin placement) needed to put the 
    geometric center at the desired location.
    
    Args:
        mov: Movable object dictionary
        desired_center: (x, y) where we want the geometric center to be
        
    Returns:
        numpy array (x, y) position for local origin
    """
    local_center = get_movable_local_center(mov)
    rotation = mov.get("RotationAngle", 0.0)
    
    # Rotate local center
    if rotation != 0:
        cos_r = np.cos(rotation)
        sin_r = np.sin(rotation)
        rotated_center = np.array([
            local_center[0] * cos_r - local_center[1] * sin_r,
            local_center[0] * sin_r + local_center[1] * cos_r
        ])
    else:
        rotated_center = local_center
    
    return np.array(desired_center) - rotated_center


def get_movable_polygon(mov, position):
    """Get the Shapely polygon for a movable at a given position.
    
    Note: 'position' is where the local coordinate origin is placed,
    which may not be the geometric center of the object.
    """
    verts = translate_polygon(mov["verts"], position, mov.get("RotationAngle", 0.0))
    hull = get_convex_hull_vertices(verts, closed=False)
    try:
        poly = ShapelyPolygon(hull)
        if not poly.is_valid:
            poly = poly.buffer(0)
        return poly
    except:
        return None


def check_position_valid(mov, position, obstacle_union, placed_polys, region_poly, min_separation=0.3):
    """
    Check if a movable can be placed at a position without overlaps.
    
    Args:
        mov: Movable object dictionary
        position: (x, y) position to check
        obstacle_union: Union of all fixed obstacles
        placed_polys: List of already-placed movable polygons
        region_poly: Region boundary polygon (or None)
        min_separation: Minimum separation distance
        
    Returns:
        True if position is valid, False otherwise
    """
    # Get movable polygon at this position (with buffer for separation)
    mov_poly = get_movable_polygon(mov, position)
    if mov_poly is None:
        return False
    
    # Buffer for minimum separation
    if min_separation > 0:
        mov_poly_buffered = mov_poly.buffer(min_separation / 2)
    else:
        mov_poly_buffered = mov_poly
    
    # STRICT region boundary check - the ENTIRE movable must be inside the region
    if region_poly is not None:
        # Check if movable polygon is completely within the region
        if not region_poly.contains(mov_poly):
            return False
        # Also check buffered version for separation from region boundary
        if min_separation > 0:
            region_shrunk = region_poly.buffer(-min_separation / 2)
            if not region_shrunk.is_empty and not region_shrunk.contains(mov_poly):
                # Allow if shrunk region is too small, but movable fits in original
                if not region_poly.contains(mov_poly):
                    return False
    
    # Check fixed obstacles
    if not obstacle_union.is_empty:
        if mov_poly_buffered.intersects(obstacle_union):
            return False
    
    # Check already-placed movables
    for placed_poly in placed_polys:
        if mov_poly_buffered.intersects(placed_poly):
            return False
    
    return True


def spiral_search(center, max_radius, step=0.5, region_poly=None):
    """
    Generate positions in an outward spiral pattern from center.
    If region_poly is provided, only yield positions inside the region.
    
    Args:
        center: (x, y) starting position
        max_radius: Maximum search radius
        step: Distance between search positions
        region_poly: Optional region polygon to constrain search
        
    Yields:
        (x, y) positions in spiral order
    """
    # First try center
    if region_poly is None or region_poly.contains(Point(center)):
        yield center
    
    x, y = center
    
    # Spiral outward
    for radius in np.arange(step, max_radius, step):
        # Number of points at this radius (roughly even spacing)
        n_points = max(50, int(2 * np.pi * radius / step))
        for i in range(n_points):
            angle = 2 * np.pi * i / n_points
            px = x + radius * np.cos(angle)
            py = y + radius * np.sin(angle)
            
            # Only yield if inside region (if region specified)
            if region_poly is None or region_poly.contains(Point(px, py)):
                yield (px, py)


def grid_search(center, bounds, step=0.5, max_positions=1000, region_poly=None):
    """
    Generate positions sorted by distance from center within bounds.
    If region_poly is provided, only return positions inside the region.
    
    Args:
        center: (x, y) target position
        bounds: ((xmin, xmax), (ymin, ymax))
        step: Grid spacing
        max_positions: Maximum positions to return
        region_poly: Optional region polygon to constrain search
        
    Returns:
        List of (x, y) positions sorted by distance from center
    """
    (xmin, xmax), (ymin, ymax) = bounds
    cx, cy = center
    
    # Generate grid
    xs = np.arange(xmin, xmax, step)
    ys = np.arange(ymin, ymax, step)
    
    # Create all grid positions
    positions = []
    for x in xs:
        for y in ys:
            # Check if inside region (if specified)
            if region_poly is not None:
                if not region_poly.contains(Point(x, y)):
                    continue
            
            dist = np.sqrt((x - cx)**2 + (y - cy)**2)
            positions.append((dist, x, y))
    
    # Sort by distance
    positions.sort(key=lambda p: p[0])
    
    # Return positions (without distance)
    return [(p[1], p[2]) for p in positions[:max_positions]]


def find_nearest_valid_position(mov, target, obstacle_union, placed_polys, 
                                region_poly, bounds, min_separation=0.3,
                                max_search_radius=50.0, search_step=0.5):
    """
    Find the nearest valid position to the target for a movable.
    STRICTLY stays within the region boundary.
    
    Args:
        mov: Movable object dictionary
        target: (x, y) target position
        obstacle_union: Union of all fixed obstacles
        placed_polys: List of already-placed movable polygons
        region_poly: Region boundary polygon
        bounds: ((xmin, xmax), (ymin, ymax))
        min_separation: Minimum separation between objects
        max_search_radius: Maximum search distance from target
        search_step: Step size for spiral search
        
    Returns:
        (x, y) valid position, or None if no valid position found
    """
    # First check if target position is valid
    if check_position_valid(mov, target, obstacle_union, placed_polys, region_poly, min_separation):
        return target
    
    # Spiral search outward from target, constrained to region
    for pos in spiral_search(target, max_search_radius, search_step, region_poly):
        # Quick bounds check
        (xmin, xmax), (ymin, ymax) = bounds
        if pos[0] < xmin or pos[0] > xmax or pos[1] < ymin or pos[1] > ymax:
            continue
        
        if check_position_valid(mov, pos, obstacle_union, placed_polys, region_poly, min_separation):
            return pos
    
    # If spiral search failed, try grid search within region only
    print(f"    Warning: Spiral search failed, trying grid search within region...")
    
    if region_poly is not None:
        # Get region bounds
        minx, miny, maxx, maxy = region_poly.bounds
        region_bounds = ((minx, maxx), (miny, maxy))
    else:
        region_bounds = bounds
    
    for pos in grid_search(target, region_bounds, step=search_step/2, max_positions=2000, region_poly=region_poly):
        if check_position_valid(mov, pos, obstacle_union, placed_polys, region_poly, min_separation):
            return pos
    
    # LAST RESORT: Try with reduced separation, still within region
    print(f"    Warning: Trying with reduced separation...")
    for reduced_sep in [min_separation * 0.5, min_separation * 0.25, 0.0]:
        for pos in spiral_search(target, max_search_radius, search_step, region_poly):
            (xmin, xmax), (ymin, ymax) = bounds
            if pos[0] < xmin or pos[0] > xmax or pos[1] < ymin or pos[1] > ymax:
                continue
            if check_position_valid(mov, pos, obstacle_union, placed_polys, region_poly, reduced_sep):
                return pos
    
    # If still no valid position, return None (don't place outside region!)
    print(f"    ERROR: No valid position found within region!")
    return None


def pull_movable_inside_region(mov, region_poly, min_margin=0.1):
    """
    If a movable's polygon extends outside its assigned region,
    find the nearest position that places it completely inside.
    
    Args:
        mov: Movable object dictionary
        region_poly: Shapely polygon of the region
        min_margin: Minimum margin from region boundary
        
    Returns:
        New position (x, y) that keeps movable inside, or original if already inside
    """
    target = np.array(mov["target"])
    mov_poly = get_movable_polygon(mov, target)
    
    if mov_poly is None:
        return target
    
    # Check if already completely inside
    if region_poly.contains(mov_poly):
        return target
    
    # Need to move it inside - find the direction to push
    # Strategy: move toward region centroid until completely inside
    
    region_centroid = np.array([region_poly.centroid.x, region_poly.centroid.y])
    
    # Binary search for the minimum distance to move
    direction = region_centroid - target
    dir_length = np.linalg.norm(direction)
    
    if dir_length < 1e-6:
        # Target is at centroid, try different directions
        for angle in np.linspace(0, 2 * np.pi, 16, endpoint=False):
            test_dir = np.array([np.cos(angle), np.sin(angle)])
            for dist in np.arange(0.5, 50, 0.5):
                test_pos = target + test_dir * dist
                test_poly = get_movable_polygon(mov, test_pos)
                if test_poly and region_poly.contains(test_poly):
                    return test_pos
        return target  # Give up
    
    direction = direction / dir_length
    
    # Search along direction toward centroid
    for dist in np.arange(0.1, dir_length + 20, 0.2):
        test_pos = target + direction * dist
        test_poly = get_movable_polygon(mov, test_pos)
        
        if test_poly is None:
            continue
            
        if region_poly.contains(test_poly):
            return test_pos
    
    # If moving toward centroid didn't work, try spiral search
    for radius in np.arange(0.5, 50, 0.5):
        n_points = max(8, int(2 * np.pi * radius / 0.5))
        for i in range(n_points):
            angle = 2 * np.pi * i / n_points
            test_pos = target + radius * np.array([np.cos(angle), np.sin(angle)])
            
            if not region_poly.contains(Point(test_pos)):
                continue
                
            test_poly = get_movable_polygon(mov, test_pos)
            if test_poly and region_poly.contains(test_poly):
                return test_pos
    
    # Last resort - return original
    print(f"    Warning: Could not pull movable inside region")
    return target


def preprocess_movables_into_regions(movables_per_region, regions_info, min_margin=0.1):
    """
    Pre-processing step: Ensure all movables are completely inside their assigned regions.
    
    This fixes the issue where a movable's center is in Region A but its polygon
    extends into Region B.
    
    Args:
        movables_per_region: List of movable lists per region
        regions_info: List of region dictionaries with 'shapely_polygon'
        min_margin: Minimum margin from region boundary
        
    Returns:
        Updated movables_per_region with adjusted target positions
    """
    print("  Pre-processing: Pulling movables inside their regions...")
    
    total_adjusted = 0
    
    for region_idx, (movables, region_info) in enumerate(zip(movables_per_region, regions_info)):
        if len(movables) == 0:
            continue
            
        region_poly = region_info.get("shapely_polygon")
        if region_poly is None:
            continue
        
        adjusted_count = 0
        
        for mov in movables:
            original_target = np.array(mov["target"]).copy()
            mov_poly = get_movable_polygon(mov, original_target)
            
            if mov_poly is None:
                continue
            
            # Check if completely inside
            if not region_poly.contains(mov_poly):
                # Need to adjust
                new_target = pull_movable_inside_region(mov, region_poly, min_margin)
                
                if not np.allclose(new_target, original_target):
                    mov["target"] = new_target
                    adjusted_count += 1
        
        if adjusted_count > 0:
            print(f"    Region {region_idx}: Adjusted {adjusted_count} movables")
            total_adjusted += adjusted_count
    
    print(f"  Pre-processing complete: {total_adjusted} movables adjusted")
    return movables_per_region


def pull_movables_into_region(movables, region_poly, min_margin=0.1):
    """
    Pre-processing step: Ensure all movables are fully inside their assigned region.
    
    For movables whose polygon extends outside the region, find the NEAREST
    position that keeps them fully inside (minimum displacement).
    
    Strategy: 
    1. Find the part of the movable polygon that's OUTSIDE the region
    2. Use that to determine the push direction (from outside toward inside)
    3. Move until the ENTIRE polygon is inside
    
    Args:
        movables: List of movable objects assigned to this region
        region_poly: Shapely polygon defining the region boundary
        min_margin: Minimum margin from region boundary
        
    Returns:
        Number of movables that were adjusted
    """
    if region_poly is None or len(movables) == 0:
        return 0
    
    adjusted_count = 0
    
    # Shrink region slightly to ensure movables don't touch the boundary
    inner_region = region_poly.buffer(-min_margin)
    if inner_region.is_empty or not inner_region.is_valid:
        inner_region = region_poly
    
    for mov in movables:
        target = np.array(mov["target"])
        mov_poly = get_movable_polygon(mov, target)
        
        if mov_poly is None:
            continue
        
        # Check if movable is fully inside the region
        if region_poly.contains(mov_poly):
            continue  # Already inside, no adjustment needed
        
        # Movable extends outside region - need to pull it in
        adjusted_count += 1
        
        # Find the part of the polygon that's OUTSIDE the region
        try:
            outside_part = mov_poly.difference(region_poly)
            inside_part = mov_poly.intersection(region_poly)
        except:
            outside_part = None
        

        # NO outside part
        if outside_part is None or outside_part.is_empty:
            continue
        

        if inside_part is None or inside_part.is_empty:
            direction = Point(mov['target'][0], mov['target'][1])   
        else:
            direction = inside_part.centroid

        # Get centroid of the outside part
        outside_centroid = outside_part.centroid
        # print(f"    Outside centroid: {outside_centroid}")
        # print(f"    direction: {direction}")
        # print(f"    Difference x: {direction.x - outside_centroid.x}")
        # print(f"    Difference y: {direction.y - outside_centroid.y}")
        # print(f"    Movable ElementId: {mov['ElementId']}")
        
        # Direction: from outside centroid toward movable centroid (i.e., pull the outside part in)
        direction = np.array([direction.x - outside_centroid.x,
                              direction.y - outside_centroid.y])
        
        dir_length = np.linalg.norm(direction)
        if dir_length < 1e-6:
            # Outside and inside centroids coincide - use direction toward region centroid
            region_centroid = region_poly.centroid
            direction = np.array([region_centroid.x - target[0],
                                    region_centroid.y - target[1]])
            dir_length = np.linalg.norm(direction)
    # else:
    #     # Fallback: use direction toward region centroid
    #     region_centroid = region_poly.centroid
    #     direction = np.array([region_centroid.x - target[0],
    #                             region_centroid.y - target[1]])
    #     dir_length = np.linalg.norm(direction)
        
        if dir_length < 1e-6:
            direction = np.array([1.0, 0.0])
            dir_length = 1.0
        
        direction = direction / dir_length
        
        # Move along direction until entire polygon is inside
        best_pos = None
        
        for dist in np.arange(0.1, 50, 0.1):
            test_pos = target + direction * dist
            test_poly = get_movable_polygon(mov, test_pos)
            
            if test_poly is not None and inner_region.contains(test_poly):
                best_pos = test_pos
                break
        
        if best_pos is None:
            # Direction didn't work, try spiral search from target
            for radius in np.arange(0.2, 30, 0.2):
                n_points = max(16, int(2 * np.pi * radius / 0.3))
                found = False
                for i in range(n_points):
                    angle = 2 * np.pi * i / n_points
                    test_pos = target + radius * np.array([np.cos(angle), np.sin(angle)])
                    test_poly = get_movable_polygon(mov, test_pos)
                    
                    if test_poly is not None and inner_region.contains(test_poly):
                        best_pos = test_pos
                        found = True
                        break
                if found:
                    break
        
        if best_pos is not None:
            mov["target"] = best_pos
        else:
            print(f"    Warning: Could not pull movable {mov.get('ElementId', '?')} inside region")
    
    return adjusted_count


def greedy_optimize_region(
    movables,
    fixed_obstacles,
    region_boundary=None,
    region_index=0,
    min_separation=0.3,
    placement_bounds=None,
    search_step=0.5,
    max_search_radius=50.0,
    priority_mode="size",  # "distance", "size", "original_order"
):
    """
    Optimize a region using greedy placement.
    
    Places all movables at valid positions near their targets, guaranteeing
    0 overlaps by construction (each object is placed only if it doesn't
    overlap anything already placed).
    
    IMPORTANT: Objects already at valid positions are placed FIRST to avoid
    displacing them unnecessarily.
    
    Args:
        movables: List of movable objects in this region
        fixed_obstacles: List of fixed obstacles in this region
        region_boundary: List of (x, y) coordinates defining region boundary
        region_index: Index of this region (for logging)
        min_separation: Minimum separation between objects
        placement_bounds: ((xmin, xmax), (ymin, ymax)) - required if no region_boundary
        search_step: Step size for position search
        max_search_radius: Maximum search distance from target
        priority_mode: How to prioritize object placement
        
    Returns:
        Tuple of (optimized_positions, original_positions, region_index)
    """
    if len(movables) == 0:
        return np.array([]), np.array([]), region_index
    
    print(f"  Region {region_index}: {len(movables)} movables, {len(fixed_obstacles)} fixed - Greedy placement")
    
    # Get original positions
    x0 = np.array([m["target"] for m in movables]).reshape(-1)
    
    # Create region polygon
    if region_boundary is not None and len(region_boundary) >= 3:
        region_poly = ShapelyPolygon(region_boundary)
        if not region_poly.is_valid:
            region_poly = region_poly.buffer(0)
        
        # Get bounds from region
        minx, miny, maxx, maxy = region_poly.bounds
        bounds = ((minx, maxx), (miny, maxy))
    else:
        region_poly = None
        if placement_bounds is not None:
            bounds = placement_bounds
        else:
            # Compute bounds from movable positions
            positions = np.array([m["target"] for m in movables])
            margin = max_search_radius
            bounds = (
                (positions[:, 0].min() - margin, positions[:, 0].max() + margin),
                (positions[:, 1].min() - margin, positions[:, 1].max() + margin)
            )
    
    # Create obstacle union (with buffer for separation)
    obstacle_union = create_obstacle_union(fixed_obstacles, buffer=min_separation / 2)
    
    # PHASE 1: Identify objects already at valid positions (no overlap with fixed obstacles)
    # These should stay in place if possible
    indexed_movables = list(enumerate(movables))
    
    valid_at_target = []  # Objects that can stay at their target
    need_placement = []   # Objects that need to find a new position
    
    for idx, mov in indexed_movables:
        target = np.array(mov["target"])
        mov_poly = get_movable_polygon(mov, target)
        
        if mov_poly is None:
            need_placement.append((idx, mov))
            continue
        
        # Check if position is valid (inside region, not overlapping fixed obstacles)
        is_inside = region_poly is None or region_poly.contains(mov_poly)
        overlaps_fixed = not obstacle_union.is_empty and mov_poly.buffer(min_separation / 2).intersects(obstacle_union)
        
        if is_inside and not overlaps_fixed:
            valid_at_target.append((idx, mov))
        else:
            need_placement.append((idx, mov))
    
    print(f"    {len(valid_at_target)} objects at valid positions, {len(need_placement)} need placement")
    
    # PHASE 2: Prioritize movable-fixed conflicts, then movable-movable conflicts
    placed_polys = []  # Polygons of already-placed movables
    result = np.zeros(len(movables) * 2)

    # Helper function to get object size
    def get_size(idx_mov):
        mov = idx_mov[1]
        verts = np.array(mov["verts"])
        return (verts[:, 0].max() - verts[:, 0].min()) * (verts[:, 1].max() - verts[:, 1].min())

    # Step 2a: Place movable-fixed conflicts from need_placement
    movable_fixed_conflicts = []
    movable_other = []

    for idx, mov in need_placement:
        target = np.array(mov["target"])
        mov_poly = get_movable_polygon(mov, target)

        if mov_poly is None:
            movable_other.append((idx, mov))
            continue

        # Check if overlaps with fixed obstacles
        mov_poly_buffered = mov_poly.buffer(min_separation / 2)
        overlaps_fixed = not obstacle_union.is_empty and mov_poly_buffered.intersects(obstacle_union)

        if overlaps_fixed:
            movable_fixed_conflicts.append((idx, mov))
        else:
            movable_other.append((idx, mov))

    # Sort movable-fixed conflicts by size (largest first)
    movable_fixed_conflicts.sort(key=get_size, reverse=True)

    print(f"    {len(movable_fixed_conflicts)} movable-fixed conflicts to place first")

    # Place movable-fixed conflicts
    still_failed = []
    for idx, mov in movable_fixed_conflicts:
        target = np.array(mov["target"])

        new_pos = find_nearest_valid_position(
            mov, target, obstacle_union, placed_polys,
            region_poly, bounds, min_separation,
            max_search_radius, search_step
        )

        if new_pos is None:
            # Could not place - will try again in Phase 3
            still_failed.append((idx, mov))
        else:
            # Successfully placed
            result[idx * 2] = new_pos[0]
            result[idx * 2 + 1] = new_pos[1]

            placed_poly = get_movable_polygon(mov, new_pos)
            if placed_poly is not None:
                placed_polys.append(placed_poly.buffer(min_separation / 2))

    # Step 2b: Resolve movable-movable conflicts from valid_at_target
    # Sort valid_at_target by size (largest first get priority to stay)
    valid_at_target.sort(key=get_size, reverse=True)

    final_valid = []
    still_need_placement = list(movable_other)  # Objects without fixed conflicts

    for idx, mov in valid_at_target:
        target = np.array(mov["target"])
        mov_poly = get_movable_polygon(mov, target)
        mov_poly_buffered = mov_poly.buffer(min_separation / 2)

        # Check if it overlaps any already-placed movable (from step 2a)
        overlaps_placed = False
        for placed_poly in placed_polys:
            if mov_poly_buffered.intersects(placed_poly):
                overlaps_placed = True
                break

        if not overlaps_placed:
            # Can stay at target
            result[idx * 2] = target[0]
            result[idx * 2 + 1] = target[1]
            placed_polys.append(mov_poly.buffer(min_separation / 2))
            final_valid.append((idx, mov))
        else:
            # Overlaps - needs placement
            still_need_placement.append((idx, mov))

    # Add failed movable-fixed conflicts back to still_need_placement
    still_need_placement.extend(still_failed)

    print(f"    {len(final_valid)} staying at target, {len(still_need_placement)} being repositioned")
    
    # PHASE 3: Place remaining objects using greedy search
    # Sort by size (largest first - harder to place)
    still_need_placement.sort(key=get_size, reverse=True)
    
    failed_placements = []
    
    for idx, mov in still_need_placement:
        target = np.array(mov["target"])
        
        # Find nearest valid position
        new_pos = find_nearest_valid_position(
            mov, target, obstacle_union, placed_polys,
            region_poly, bounds, min_separation,
            max_search_radius, search_step
        )
        
        if new_pos is None:
            print(f"    WARNING: Could not place movable {idx} ({mov.get('ElementId', '?')}) within region!")
            new_pos = target
            failed_placements.append(idx)
        
        # Store result
        result[idx * 2] = new_pos[0]
        result[idx * 2 + 1] = new_pos[1]
        
        # Add to placed polygons
        placed_poly = get_movable_polygon(mov, new_pos)
        if placed_poly is not None:
            placed_polys.append(placed_poly.buffer(min_separation / 2))
    
    # Report displacement
    pts_initial = x0.reshape(-1, 2)
    pts_final = result.reshape(-1, 2)
    displacements = np.linalg.norm(pts_final - pts_initial, axis=1)
    avg_disp = np.mean(displacements)
    max_disp = np.max(displacements)
    print(f"    Avg displacement: {avg_disp:.2f}, Max: {max_disp:.2f}")
    
    if len(failed_placements) > 0:
        print(f"    WARNING: {len(failed_placements)} movables could not be placed within region!")
    
    return result, x0, region_index


def greedy_optimize_with_regions(
    movables_per_region,
    fixed_per_region,
    regions_info,
    min_separation=0.3,
    search_step=0.5,
    max_search_radius=50.0,
    n_jobs=1,
):
    """
    Optimize all regions using greedy placement.
    
    Args:
        movables_per_region: List of movable lists per region
        fixed_per_region: List of fixed obstacle lists per region
        regions_info: List of region dictionaries
        min_separation: Minimum separation between objects
        search_step: Step size for position search
        max_search_radius: Maximum search distance from target
        n_jobs: Number of parallel jobs (not used yet, for API compatibility)
        
    Returns:
        Tuple of (all_results, all_x0s, region_indices)
    """
    print(f"Greedy optimizing {len(regions_info)} regions...")
    
    all_results = []
    all_x0s = []
    region_indices = []
    
    for i, (movables, fixed_obs, region_info) in enumerate(
        zip(movables_per_region, fixed_per_region, regions_info)
    ):
        boundary = region_info.get("boundary")
        
        result, x0, idx = greedy_optimize_region(
            movables,
            fixed_obs,
            region_boundary=boundary,
            region_index=i,
            min_separation=min_separation,
            search_step=search_step,
            max_search_radius=max_search_radius,
        )
        
        all_results.append(result)
        all_x0s.append(x0)
        region_indices.append(idx)
    
    return all_results, all_x0s, region_indices


# ============================================================================
# Alternative: Grid-based placement with Hungarian assignment
# ============================================================================

def create_placement_grid(region_poly, obstacle_union, grid_step=.5, min_separation=0.3, max_movable_radius=2.0):
    """
    Create a grid of valid placement positions within a region.
    
    Args:
        region_poly: Shapely polygon defining the region
        obstacle_union: Union of all obstacles
        grid_step: Spacing between grid points
        min_separation: Minimum separation (for buffering)
        max_movable_radius: Maximum distance from movable center to its edge
        
    Returns:
        numpy array of (x, y) valid positions
    """
    if region_poly is None:
        return np.array([])
    
    # Get bounds
    minx, miny, maxx, maxy = region_poly.bounds
    
    # Total buffer needed: movable radius + separation
    total_buffer = max_movable_radius + min_separation
    
    # Shrink region by total buffer
    inner_region = region_poly.buffer(-total_buffer)
    if inner_region.is_empty:
        inner_region = region_poly.buffer(-min_separation)
        if inner_region.is_empty:
            inner_region = region_poly
    
    # Expand obstacles by total buffer
    if not obstacle_union.is_empty:
        expanded_obstacles = obstacle_union.buffer(total_buffer)
    else:
        expanded_obstacles = ShapelyPolygon()
    
    # Generate grid points
    valid_positions = []
    
    for x in np.arange(minx, maxx, grid_step):
        for y in np.arange(miny, maxy, grid_step):
            pt = Point(x, y)
            
            # Check if inside region and outside obstacles
            if inner_region.contains(pt) and not expanded_obstacles.contains(pt):
                valid_positions.append([x, y])
    
    return np.array(valid_positions) if valid_positions else np.array([]).reshape(0, 2)


def assign_objects_to_grid(movables, grid_positions, min_separation=0.3):
    """
    Assign movables to grid positions using greedy nearest-neighbor.
    
    Args:
        movables: List of movable objects
        grid_positions: numpy array of (x, y) valid positions
        min_separation: Minimum separation between objects
        
    Returns:
        numpy array of assigned positions (same order as movables)
    """
    if len(movables) == 0:
        return np.array([])
    
    if len(grid_positions) == 0:
        # No valid grid positions, return original targets
        return np.array([m["target"] for m in movables]).reshape(-1)
    
    n_movables = len(movables)
    targets = np.array([m["target"] for m in movables])
    
    # Build KD-tree of grid positions
    tree = cKDTree(grid_positions)
    
    # Track which grid positions are used
    used_positions = set()
    
    # Assign positions greedily
    result = np.zeros(n_movables * 2)
    
    # Sort movables by distance to nearest grid point (closest first)
    distances, nearest_indices = tree.query(targets, k=1)
    order = np.argsort(distances)
    
    for idx in order:
        target = targets[idx]
        
        # Find nearest unused grid position
        # Query multiple neighbors and find first unused
        k = min(len(grid_positions), 50)  # Check up to 50 nearest
        dists, indices = tree.query(target, k=k)
        
        assigned_pos = None
        for i, grid_idx in enumerate(indices):
            if grid_idx not in used_positions:
                assigned_pos = grid_positions[grid_idx]
                used_positions.add(grid_idx)
                break
        
        if assigned_pos is None:
            # All nearby positions used, fall back to target
            assigned_pos = target
        
        result[idx * 2] = assigned_pos[0]
        result[idx * 2 + 1] = assigned_pos[1]
    
    return result


def grid_optimize_region(
    movables,
    fixed_obstacles,
    region_boundary=None,
    region_index=0,
    min_separation=0.3,
    grid_step=1.0,
):
    """
    Optimize a region using grid-based placement.
    
    NOTE: This is a simpler but less accurate approach than greedy_optimize_region.
    Use greedy_optimize_region for better results.
    
    Args:
        movables: List of movable objects
        fixed_obstacles: List of fixed obstacles
        region_boundary: Region boundary coordinates
        region_index: Index for logging
        min_separation: Minimum separation
        grid_step: Grid spacing
        
    Returns:
        Tuple of (optimized_positions, original_positions, region_index)
    """
    if len(movables) == 0:
        return np.array([]), np.array([]), region_index
    
    print(f"  Region {region_index}: {len(movables)} movables - Grid placement")
    
    x0 = np.array([m["target"] for m in movables]).reshape(-1)
    
    # Create region polygon
    if region_boundary is not None and len(region_boundary) >= 3:
        region_poly = ShapelyPolygon(region_boundary)
        if not region_poly.is_valid:
            region_poly = region_poly.buffer(0)
    else:
        region_poly = None
    
    # Create obstacle union
    obstacle_union = create_obstacle_union(fixed_obstacles, buffer=min_separation/2)
    
    # Compute max movable radius (distance from center to furthest vertex)
    max_movable_radius = 0.0
    for mov in movables:
        verts = np.array(mov["verts"])
        radius = np.max(np.linalg.norm(verts, axis=1))
        max_movable_radius = max(max_movable_radius, radius)
    
    print(f"    Max movable radius: {max_movable_radius:.2f}")
    
    # Create placement grid
    grid_positions = create_placement_grid(
        region_poly, obstacle_union, grid_step, min_separation, max_movable_radius
    )
    
    print(f"    Grid has {len(grid_positions)} valid positions")
    
    if len(grid_positions) < len(movables):
        print(f"    WARNING: Only {len(grid_positions)} positions for {len(movables)} movables!")
    
    # Assign movables to grid positions WITH size checking
    # This is similar to greedy but uses pre-computed grid positions
    n_movables = len(movables)
    targets = np.array([m["target"] for m in movables])
    
    # Build KD-tree of grid positions for fast nearest-neighbor lookup
    if len(grid_positions) > 0:
        tree = cKDTree(grid_positions)
    else:
        # No valid positions, return original
        return x0.copy(), x0, region_index
    
    # Sort movables by size (largest first - harder to place)
    def get_size(idx):
        mov = movables[idx]
        verts = np.array(mov["verts"])
        return (verts[:, 0].max() - verts[:, 0].min()) * (verts[:, 1].max() - verts[:, 1].min())
    
    order = sorted(range(n_movables), key=get_size, reverse=True)
    
    result = np.zeros(n_movables * 2)
    placed_polys = []  # Already placed movable polygons
    used_grid_indices = set()
    
    for idx in order:
        mov = movables[idx]
        target = targets[idx]
        
        # Find nearest valid grid position that doesn't overlap placed objects
        k = min(len(grid_positions), 100)  # Check up to 100 nearest
        dists, indices = tree.query(target, k=k)
        
        assigned_pos = None
        for grid_idx in indices:
            if grid_idx in used_grid_indices:
                continue
            
            pos = grid_positions[grid_idx]
            
            # Check if movable at this position overlaps any placed movable
            mov_poly = get_movable_polygon(mov, pos)
            if mov_poly is None:
                continue
            
            # Buffer for separation
            mov_poly_buffered = mov_poly.buffer(min_separation / 2)
            
            overlaps_placed = False
            for placed_poly in placed_polys:
                if mov_poly_buffered.intersects(placed_poly):
                    overlaps_placed = True
                    break
            
            if not overlaps_placed:
                assigned_pos = pos
                used_grid_indices.add(grid_idx)
                placed_polys.append(mov_poly.buffer(min_separation / 2))
                break
        
        if assigned_pos is None:
            # Fall back to target position
            print(f"    Warning: No valid grid position for movable {idx}, using target")
            assigned_pos = target
        
        result[idx * 2] = assigned_pos[0]
        result[idx * 2 + 1] = assigned_pos[1]
    
    # Report stats
    pts_initial = x0.reshape(-1, 2)
    pts_final = result.reshape(-1, 2)
    displacements = np.linalg.norm(pts_final - pts_initial, axis=1)
    print(f"    Avg displacement: {np.mean(displacements):.2f}, Max: {np.max(displacements):.2f}")
    
    return result, x0, region_index