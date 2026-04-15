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
from concurrent.futures import ThreadPoolExecutor, as_completed
import shapely as _shp
from shapely.geometry import Polygon as ShapelyPolygon, Point, box
from shapely.strtree import STRtree
from scipy.spatial import cKDTree
from polygon_utils import (
    translate_polygon,
    get_convex_hull_vertices,
    get_precomputed_geometry,
)


def _vec_contains(geom, xs, ys):
    """
    Vectorized point-in-polygon using Shapely 2.0's GEOS-backed contains_xy.
    Handles Polygon, MultiPolygon, and holes natively at C speed.

    Args:
        geom: Any Shapely geometry
        xs, ys: 1-D numpy arrays of coordinates

    Returns:
        Boolean numpy array, True where point is inside geom
    """
    return _shp.contains_xy(geom, xs, ys)


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


def _precompute_hull(mov):
    """Pre-compute the local convex hull for a movable (rotation applied, no translation).
    Returns rotated hull vertices relative to local origin, or None."""
    verts = np.array(mov["verts"], dtype=float)
    rotation = mov.get("RotationAngle", 0.0)
    if rotation != 0:
        cos_r = np.cos(rotation)
        sin_r = np.sin(rotation)
        rot = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
        verts = verts @ rot.T
    hull = get_convex_hull_vertices(verts, closed=False)
    if len(hull) < 3:
        return None
    return hull


def get_movable_polygon(mov, position):
    """Get the Shapely polygon for a movable at a given position.

    Note: 'position' is where the local coordinate origin is placed,
    which may not be the geometric center of the object.
    """
    # Use cached hull if available
    local_hull = mov.get("_cached_hull")
    if local_hull is None:
        local_hull = _precompute_hull(mov)
        mov["_cached_hull"] = local_hull
    if local_hull is None:
        return None
    # Fast translate: just add position offset
    world_hull = local_hull + np.asarray(position)
    try:
        poly = ShapelyPolygon(world_hull)
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
    
    # Check already-placed movables (vectorized)
    if placed_polys and np.any(_shp.intersects(mov_poly_buffered, np.array(placed_polys))):
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
    x, y = center

    # First try center
    if region_poly is None or region_poly.contains(Point(center)):
        yield center

    # Early radius cutoff: a ring beyond the farthest bounding-box corner
    # cannot contain any region point, so stop there
    if region_poly is not None:
        minx, miny, maxx, maxy = region_poly.bounds
        corners_x = np.array([minx, minx, maxx, maxx])
        corners_y = np.array([miny, maxy, miny, maxy])
        max_useful_radius = float(np.sqrt((corners_x - x) ** 2 + (corners_y - y) ** 2).max())
    else:
        max_useful_radius = max_radius

    # Spiral outward
    for radius in np.arange(step, max_radius, step):
        # Early cutoff: ring is entirely beyond the region's bounding box
        if region_poly is not None and radius > max_useful_radius:
            break

        # Vectorized angle generation for this ring
        n_points = max(50, int(2 * np.pi * radius / step))
        angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
        ring_x = x + radius * np.cos(angles)
        ring_y = y + radius * np.sin(angles)

        # Batch region containment using vectorized Path check
        if region_poly is not None:
            mask = _vec_contains(region_poly, ring_x, ring_y)
            ring_x = ring_x[mask]
            ring_y = ring_y[mask]

        for px, py in zip(ring_x, ring_y):
            yield (float(px), float(py))


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

    # Vectorized grid generation
    xs = np.arange(xmin, xmax, step)
    ys = np.arange(ymin, ymax, step)
    xx, yy = np.meshgrid(xs, ys)
    pts_x = xx.ravel()
    pts_y = yy.ravel()

    if region_poly is not None and len(pts_x) > 0:
        minx, miny, maxx, maxy = region_poly.bounds
        bbox = (pts_x >= minx) & (pts_x <= maxx) & (pts_y >= miny) & (pts_y <= maxy)
        pts_x, pts_y = pts_x[bbox], pts_y[bbox]
        if len(pts_x) > 0:
            mask = _vec_contains(region_poly, pts_x, pts_y)
            pts_x, pts_y = pts_x[mask], pts_y[mask]

    if len(pts_x) == 0:
        return []

    # Distance-sorted order using cKDTree
    pts = np.column_stack([pts_x, pts_y])
    k = min(max_positions, len(pts))
    _, indices = cKDTree(pts).query(np.array([cx, cy]), k=k)
    indices = np.atleast_1d(indices)

    return [(float(pts[i, 0]), float(pts[i, 1])) for i in indices]


def build_valid_slots(obstacle_union, region_poly, bounds, step, min_separation, max_object_radius=0.0):
    """
    Pre-generate all grid slot centers valid for any object (center-based filter).

    Filters by:
    - Region containment (slot center inside region)
    - Fixed obstacle clearance (slot center outside obstacle buffered by
      max_object_radius + min_separation, so that any object whose bounding
      radius <= max_object_radius cannot overlap the obstacle when centred here)

    Returns:
        (slots, tree): numpy (N, 2) array and cKDTree, or (empty array, None)
    """
    (xmin, xmax), (ymin, ymax) = bounds
    xs = np.arange(xmin, xmax, step)
    ys = np.arange(ymin, ymax, step)
    xx, yy = np.meshgrid(xs, ys)
    pts_x = xx.ravel()
    pts_y = yy.ravel()

    if region_poly is not None and len(pts_x) > 0:
        minx, miny, maxx, maxy = region_poly.bounds
        bbox = (pts_x >= minx) & (pts_x <= maxx) & (pts_y >= miny) & (pts_y <= maxy)
        pts_x, pts_y = pts_x[bbox], pts_y[bbox]
        if len(pts_x) > 0:
            mask = _vec_contains(region_poly, pts_x, pts_y)
            pts_x, pts_y = pts_x[mask], pts_y[mask]

    if not obstacle_union.is_empty and len(pts_x) > 0:
        # Buffer = max object radius + separation: guarantees no object centred
        # at a surviving slot can overlap the obstacle (for convex objects).
        obs_clear = max_object_radius + min_separation
        obs_buffered = obstacle_union.buffer(obs_clear)
        obs_minx, obs_miny, obs_maxx, obs_maxy = obs_buffered.bounds
        obs_bbox = (pts_x >= obs_minx) & (pts_x <= obs_maxx) & (pts_y >= obs_miny) & (pts_y <= obs_maxy)
        if np.any(obs_bbox):
            in_obs = np.zeros(len(pts_x), dtype=bool)
            in_obs[obs_bbox] = _vec_contains(obs_buffered, pts_x[obs_bbox], pts_y[obs_bbox])
            pts_x, pts_y = pts_x[~in_obs], pts_y[~in_obs]

    if len(pts_x) == 0:
        return np.empty((0, 2)), None

    slots = np.column_stack([pts_x, pts_y])
    return slots, cKDTree(slots)


def find_nearest_valid_position(mov, target, obstacle_union, placed_polys,
                                placed_positions, placed_max_radii,
                                region_poly, valid_slots, slot_tree,
                                min_separation=0.3, max_search_radius=None,
                                search_step=0.5):
    """
    Find the nearest pre-validated grid slot for a movable.

    Uses pre-built valid_slots (filtered against fixed obstacles and region)
    and a vectorized center-distance prefilter to avoid expensive polygon
    intersection checks against placed objects that are clearly far away.

    Args:
        mov: Movable object dictionary
        target: (x, y) target position
        obstacle_union: Union of all fixed obstacles (for target check only)
        placed_polys: List of already-placed movable polygons (buffered)
        placed_positions: List of [x, y] placement origins for placed objects
        placed_max_radii: List of max-radii from origin for placed objects
        region_poly: Region boundary polygon
        valid_slots: numpy (N, 2) array of pre-validated grid positions
        slot_tree: cKDTree built from valid_slots
        min_separation: Minimum separation between objects
        max_search_radius: Max distance from target to search (None = unlimited)

    Returns:
        (x, y) valid position, or None if no valid position found
    """
    # Check target position first (may already be valid)
    if check_position_valid(mov, target, obstacle_union, placed_polys, region_poly, min_separation):
        return target

    target_arr = np.array(target, dtype=float)
    search_r = max_search_radius if (max_search_radius and max_search_radius > 0) else 50.0

    # --- Pre-compute data shared by both spiral and grid search ---
    verts = np.array(mov["verts"])
    obj_max_radius = float(np.sqrt(verts[:, 0] ** 2 + verts[:, 1] ** 2).max())
    has_placed = len(placed_polys) > 0
    placed_pos_arr = np.array(placed_positions) if has_placed else None
    placed_rad_arr = np.array(placed_max_radii) if has_placed else None
    placed_polys_arr = np.array(placed_polys, dtype=object) if has_placed else None

    # Pre-shrink region by object radius: if the CENTER of the object is inside
    # this shrunk region, the full polygon (max extent = obj_max_radius) is
    # guaranteed to be inside the original region.  Avoids expensive
    # region_poly.contains(mov_poly) for the vast majority of positions.
    if region_poly is not None:
        region_inner = region_poly.buffer(-obj_max_radius)
        if region_inner.is_empty or not region_inner.is_valid:
            region_inner = None
        else:
            _shp.prepare(region_inner)
    else:
        region_inner = None

    half_sep = min_separation / 2

    # PRIMARY: Spiral search — ring-by-ring with batch distance pre-filter.
    # Positions are snapped to grid multiples of search_step so the final
    # layout has horizontal/vertical alignment.  A set tracks checked grid
    # cells to avoid redundant work across rings.
    cx, cy = float(target_arr[0]), float(target_arr[1])
    checked_grid = set()  # (snapped_x, snapped_y) already tested

    if region_poly is not None:
        rb = region_poly.bounds
        _cx = np.array([rb[0], rb[0], rb[2], rb[2]])
        _cy = np.array([rb[1], rb[3], rb[1], rb[3]])
        max_useful_r = float(np.sqrt((_cx - cx) ** 2 + (_cy - cy) ** 2).max())
    else:
        max_useful_r = search_r

    for radius in np.arange(search_step, min(search_r, max_useful_r + search_step), search_step):
        n_pts = max(50, int(2 * np.pi * radius / search_step))
        angles = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
        ring_x = cx + radius * np.cos(angles)
        ring_y = cy + radius * np.sin(angles)

        # Snap to grid for horizontal/vertical alignment
        ring_x = np.round(ring_x / search_step) * search_step
        ring_y = np.round(ring_y / search_step) * search_step

        # Vectorized region point-containment filter
        if region_poly is not None:
            mask = _vec_contains(region_poly, ring_x, ring_y)
            ring_x, ring_y = ring_x[mask], ring_y[mask]
        if len(ring_x) == 0:
            continue

        # Batch center-distance to all placed objects for the whole ring.
        # dist_sq[i, j] = squared distance from ring position i to placed object j.
        if has_placed:
            ring_pts = np.column_stack([ring_x, ring_y])          # (N, 2)
            diffs = ring_pts[:, None, :] - placed_pos_arr[None, :, :]  # (N, P, 2)
            dist_sq = np.sum(diffs * diffs, axis=2)                # (N, P)
            clearance_sq = (obj_max_radius + placed_rad_arr + min_separation) ** 2  # (P,)

        # Fast-path region inner containment for whole ring (vectorized)
        if region_inner is not None:
            center_inside = _vec_contains(region_inner, ring_x, ring_y)
        else:
            center_inside = np.zeros(len(ring_x), dtype=bool)

        for i in range(len(ring_x)):
            px, py = float(ring_x[i]), float(ring_y[i])
            grid_key = (px, py)
            if grid_key in checked_grid:
                continue
            checked_grid.add(grid_key)
            pos = (px, py)
            mov_poly = None
            mov_buffered = None

            # --- Region containment ---
            if region_poly is not None:
                if not center_inside[i]:
                    mov_poly = get_movable_polygon(mov, pos)
                    if mov_poly is None or not region_poly.contains(mov_poly):
                        continue

            # --- Fixed obstacles ---
            if not obstacle_union.is_empty:
                if mov_poly is None:
                    mov_poly = get_movable_polygon(mov, pos)
                if mov_poly is None:
                    continue
                mov_buffered = mov_poly.buffer(half_sep)
                if mov_buffered.intersects(obstacle_union):
                    continue

            # --- Placed objects (only nearby) ---
            if has_placed:
                nearby = np.where(dist_sq[i] < clearance_sq)[0]
                if len(nearby) > 0:
                    if mov_poly is None:
                        mov_poly = get_movable_polygon(mov, pos)
                    if mov_poly is None:
                        continue
                    if mov_buffered is None:
                        mov_buffered = mov_poly.buffer(half_sep)
                    if np.any(_shp.intersects(mov_buffered, placed_polys_arr[nearby])):
                        continue

            # All checks passed
            return pos

    # FALLBACK: Grid search — pre-built valid_slots with fast KD-tree lookup.
    # May find positions that the spiral missed (different alignment / spacing).
    if valid_slots is not None and len(valid_slots) > 0:
        # Limit candidate slots to within max_search_radius.
        if max_search_radius is not None and max_search_radius > 0:
            n = len(slot_tree.query_ball_point(target_arr, r=max_search_radius))
        else:
            n = len(valid_slots)

        if n > 0:
            def _try_batch(indices):
                """Check a batch of slot indices; return the first valid (x, y) or None."""
                for slot_idx in indices:
                    pos = valid_slots[slot_idx]
                    mov_poly = None
                    mov_buffered = None

                    if has_placed:
                        dists = np.linalg.norm(placed_pos_arr - pos, axis=1)
                        nearby = np.where(dists < obj_max_radius + placed_rad_arr + min_separation)[0]
                        if len(nearby) > 0:
                            mov_poly = get_movable_polygon(mov, pos)
                            if mov_poly is None:
                                continue
                            mov_buffered = mov_poly.buffer(half_sep)
                            if np.any(_shp.intersects(mov_buffered, placed_polys_arr[nearby])):
                                continue

                    if region_poly is not None:
                        if mov_poly is None:
                            mov_poly = get_movable_polygon(mov, pos)
                        if mov_poly is None or not region_poly.contains(mov_poly):
                            continue

                    if not obstacle_union.is_empty:
                        if mov_poly is None:
                            mov_poly = get_movable_polygon(mov, pos)
                        if mov_poly is None:
                            continue
                        if mov_buffered is None:
                            mov_buffered = mov_poly.buffer(half_sep)
                        if mov_buffered.intersects(obstacle_union):
                            continue

                    return (float(pos[0]), float(pos[1]))
                return None

            prev_k = 0
            for batch_k in [min(500, n), min(5000, n), n]:
                if batch_k <= prev_k:
                    continue
                _, indices = slot_tree.query(target_arr, k=batch_k)
                result = _try_batch(np.atleast_1d(indices)[prev_k:])
                if result is not None:
                    return result
                prev_k = batch_k

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

    # Precompute region dimensions for feasibility check
    minx, miny, maxx, maxy = region_poly.bounds
    region_w = maxx - minx
    region_h = maxy - miny

    # Shrink region slightly to ensure movables don't touch the boundary
    inner_region = region_poly.buffer(-min_margin)
    if inner_region.is_empty or not inner_region.is_valid:
        inner_region = region_poly

    for mov in movables:
        target = np.array(mov["target"])
        mov_poly = get_movable_polygon(mov, target)

        if mov_poly is None:
            continue

        # Skip objects that are physically too large to ever fit in this region
        verts = np.array(mov["verts"])
        obj_w = verts[:, 0].max() - verts[:, 0].min()
        obj_h = verts[:, 1].max() - verts[:, 1].min()
        if obj_w > region_w or obj_h > region_h:
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
        # Use binary-search-like approach: first find the range, then refine
        best_pos = None
        dists = np.arange(0.1, 50, 0.1)
        if len(dists) > 0:
            # Batch: generate all test positions along direction
            test_positions = target + np.outer(dists, direction)  # (N, 2)
            # Quick filter: center must be inside region
            center_mask = _vec_contains(inner_region, test_positions[:, 0], test_positions[:, 1])
            for idx in np.where(center_mask)[0]:
                test_pos = test_positions[idx]
                test_poly = get_movable_polygon(mov, test_pos)
                if test_poly is not None and inner_region.contains(test_poly):
                    best_pos = test_pos
                    break

        if best_pos is None:
            # Direction didn't work, try spiral search from target
            for radius in np.arange(0.2, 30, 0.2):
                n_points = max(16, int(2 * np.pi * radius / 0.3))
                angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
                ring_x = target[0] + radius * np.cos(angles)
                ring_y = target[1] + radius * np.sin(angles)
                # Batch filter: center inside region
                mask = _vec_contains(inner_region, ring_x, ring_y)
                found = False
                for i in np.where(mask)[0]:
                    test_pos = np.array([ring_x[i], ring_y[i]])
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

    # Prepare geometries for fast repeated GEOS operations (Shapely 2.0)
    if region_poly is not None:
        _shp.prepare(region_poly)
    _shp.prepare(obstacle_union)

    # Compute per-region max object radius (bounding circle from local origin).
    # Used by build_valid_slots to guarantee no object overlaps fixed obstacles.
    # IMPORTANT: exclude cant_fit objects — they are larger than the region and
    # their inflated radius would cause obstacle_union.buffer(radius) to cover
    # the entire region, leaving zero valid slots for all other objects.
    max_object_radius = 0.0
    if region_poly is not None:
        _rb = region_poly.bounds  # (minx, miny, maxx, maxy)
        _reg_w, _reg_h = _rb[2] - _rb[0], _rb[3] - _rb[1]
    else:
        _reg_w = _reg_h = None
    for mov in movables:
        verts = np.array(mov["verts"])
        if _reg_w is not None:
            obj_w = verts[:, 0].max() - verts[:, 0].min()
            obj_h = verts[:, 1].max() - verts[:, 1].min()
            if obj_w > _reg_w or obj_h > _reg_h:
                continue  # cant_fit object — skip, it doesn't need grid slots
        r = float(np.sqrt(verts[:, 0] ** 2 + verts[:, 1] ** 2).max())
        if r > max_object_radius:
            max_object_radius = r

    # Pre-generate all valid grid slots for this region (once, shared across all objects)
    valid_slots, slot_tree = build_valid_slots(
        obstacle_union, region_poly, bounds, search_step, min_separation, max_object_radius
    )
    print(f"    {len(valid_slots)} valid grid slots pre-generated (step={search_step}, obj_r={max_object_radius:.2f})")

    # PHASE 1: Identify objects already at valid positions (no overlap with fixed obstacles)
    # These should stay in place if possible — classified in parallel since each
    # object's check is independent.

    # Compute region dimensions for feasibility check
    if region_poly is not None:
        minx, miny, maxx, maxy = region_poly.bounds
        region_w = maxx - minx
        region_h = maxy - miny
    else:
        region_w = region_h = None

    def _classify_object(args):
        idx, mov = args
        if region_w is not None:
            verts = np.array(mov["verts"])
            if (verts[:, 0].max() - verts[:, 0].min() > region_w or
                    verts[:, 1].max() - verts[:, 1].min() > region_h):
                return idx, "cant_fit"

        target = np.array(mov["target"])
        mov_poly = get_movable_polygon(mov, target)
        if mov_poly is None:
            return idx, "need"

        is_inside = region_poly is None or region_poly.contains(mov_poly)
        overlaps_fixed = (not obstacle_union.is_empty and
                          mov_poly.buffer(min_separation / 2).intersects(obstacle_union))
        return idx, "valid" if (is_inside and not overlaps_fixed) else "need"

    valid_at_target = []
    need_placement = []
    cant_fit = []

    with ThreadPoolExecutor() as executor:
        for idx, label in executor.map(_classify_object, enumerate(movables)):
            mov = movables[idx]
            if label == "cant_fit":
                cant_fit.append((idx, mov))
            elif label == "valid":
                valid_at_target.append((idx, mov))
            else:
                need_placement.append((idx, mov))

    if cant_fit:
        print(f"    {len(cant_fit)} objects too large to fit in region — kept as-is")
    print(f"    {len(valid_at_target)} objects at valid positions, {len(need_placement)} need placement")

    placed_polys = []       # buffered polygons of already-placed movables
    placed_positions = []   # placement origin [x, y] for each placed object
    placed_max_radii = []   # max distance from origin to any vertex, per placed object
    result = np.zeros(len(movables) * 2)

    def _record_placed(mov, pos, mov_poly):
        placed_polys.append(mov_poly.buffer(min_separation / 2))
        placed_positions.append([pos[0], pos[1]])
        verts = np.array(mov["verts"])
        placed_max_radii.append(float(np.sqrt(verts[:, 0] ** 2 + verts[:, 1] ** 2).max()))

    # Keep cant_fit objects at their target positions and treat them as placed.
    # Clip their polygon to the region boundary before recording — they are larger
    # than the region, so recording the full polygon would block all valid slots
    # for other objects, making overlaps with cant_fit objects irresolvable.
    for idx, mov in cant_fit:
        target = np.array(mov["target"])
        result[idx * 2] = target[0]
        result[idx * 2 + 1] = target[1]
        mov_poly = get_movable_polygon(mov, target)
        if mov_poly is not None:
            clipped = mov_poly.intersection(region_poly)
            if not clipped.is_empty:
                # Record the region-clipped footprint so other objects avoid it
                # without being blocked by the portion outside the region.
                placed_polys.append(clipped.buffer(min_separation / 2))
                placed_positions.append([float(target[0]), float(target[1])])
                bounds = clipped.bounds  # (minx, miny, maxx, maxy)
                r = max(
                    abs(bounds[0] - target[0]), abs(bounds[2] - target[0]),
                    abs(bounds[1] - target[1]), abs(bounds[3] - target[1]),
                )
                placed_max_radii.append(float(r))
            # If clipped is empty the object is fully outside the region — nothing to block

    # Lock in objects already at valid positions (largest first — priority to stay).
    # Each must be checked against already-locked objects: two objects can both be
    # valid vs fixed obstacles yet still overlap each other.
    valid_at_target.sort(key=lambda im: (np.array(im[1]["verts"])[:, 0].ptp() * np.array(im[1]["verts"])[:, 1].ptp()), reverse=True)
    for idx, mov in valid_at_target:
        target = np.array(mov["target"])
        mov_poly = get_movable_polygon(mov, target)
        if mov_poly is None:
            need_placement.append((idx, mov))
            continue

        if placed_polys:
            verts = np.array(mov["verts"])
            obj_max_radius = float(np.sqrt(verts[:, 0] ** 2 + verts[:, 1] ** 2).max())
            placed_pos_arr = np.array(placed_positions)
            placed_rad_arr = np.array(placed_max_radii)
            dists = np.linalg.norm(placed_pos_arr - target, axis=1)
            nearby = np.where(dists < obj_max_radius + placed_rad_arr + min_separation)[0]
            if len(nearby) > 0:
                mov_buffered = mov_poly.buffer(min_separation / 2)
                if np.any(_shp.intersects(mov_buffered, np.array([placed_polys[i] for i in nearby]))):
                    need_placement.append((idx, mov))
                    continue

        result[idx * 2] = target[0]
        result[idx * 2 + 1] = target[1]
        _record_placed(mov, target, mov_poly)

    # PHASE 2: Place movable-fixed conflicts first (hardest — they must move).
    # Then Phase 3 places everything else that still needs a slot.

    def _size_key(im):
        v = np.array(im[1]["verts"])
        return (v[:, 0].max() - v[:, 0].min()) * (v[:, 1].max() - v[:, 1].min())

    movable_fixed_conflicts = []
    movable_other = []
    for idx, mov in need_placement:
        target = np.array(mov["target"])
        mov_poly = get_movable_polygon(mov, target)
        if mov_poly is None:
            movable_other.append((idx, mov))
            continue
        overlaps_fixed = (not obstacle_union.is_empty and
                          mov_poly.buffer(min_separation / 2).intersects(obstacle_union))
        if overlaps_fixed:
            movable_fixed_conflicts.append((idx, mov))
        else:
            movable_other.append((idx, mov))

    movable_fixed_conflicts.sort(key=_size_key, reverse=True)
    print(f"    {len(movable_fixed_conflicts)} movable-fixed conflicts to place first")

    still_need_placement = list(movable_other)
    failed_placements = []

    for idx, mov in movable_fixed_conflicts:
        target = np.array(mov["target"])
        new_pos = find_nearest_valid_position(
            mov, target, obstacle_union, placed_polys,
            placed_positions, placed_max_radii,
            region_poly, valid_slots, slot_tree,
            min_separation, max_search_radius, search_step,
        )
        if new_pos is None:
            still_need_placement.append((idx, mov))
        else:
            result[idx * 2] = new_pos[0]
            result[idx * 2 + 1] = new_pos[1]
            placed_poly = get_movable_polygon(mov, new_pos)
            if placed_poly is not None:
                _record_placed(mov, new_pos, placed_poly)

    # PHASE 3: Place remaining objects (largest first — harder to fit).
    still_need_placement.sort(key=_size_key, reverse=True)
    for idx, mov in still_need_placement:
        target = np.array(mov["target"])
        new_pos = find_nearest_valid_position(
            mov, target, obstacle_union, placed_polys,
            placed_positions, placed_max_radii,
            region_poly, valid_slots, slot_tree,
            min_separation, max_search_radius, search_step,
        )
        if new_pos is None:
            result[idx * 2] = target[0]
            result[idx * 2 + 1] = target[1]
            failed_placements.append(idx)
            mov_poly = get_movable_polygon(mov, target)
            if mov_poly is not None:
                _record_placed(mov, target, mov_poly)
        else:
            result[idx * 2] = new_pos[0]
            result[idx * 2 + 1] = new_pos[1]
            placed_poly = get_movable_polygon(mov, new_pos)
            if placed_poly is not None:
                _record_placed(mov, new_pos, placed_poly)

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
    n_jobs=-1,
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

    all_results = [None] * len(regions_info)
    all_x0s = [None] * len(regions_info)
    region_indices = [None] * len(regions_info)

    def optimize_one(i):
        boundary = regions_info[i].get("boundary")
        return i, greedy_optimize_region(
            movables_per_region[i],
            fixed_per_region[i],
            region_boundary=boundary,
            region_index=i,
            min_separation=min_separation,
            search_step=search_step,
            max_search_radius=max_search_radius,
        )

    with ThreadPoolExecutor(max_workers=n_jobs if n_jobs > 1 else None) as executor:
        futures = {executor.submit(optimize_one, i): i for i in range(len(regions_info))}
        for future in as_completed(futures):
            i, (result, x0, idx) = future.result()
            all_results[i] = result
            all_x0s[i] = x0
            region_indices[i] = idx

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
