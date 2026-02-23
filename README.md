# Annotation Placement Optimizer

A collision-free placement optimization system for BIM/MEP annotation objects. The system resolves overlaps between movable annotation objects while respecting fixed obstacles (pipes, equipment) and region boundaries.

## Overview

The optimizer uses a **region-based greedy placement** approach that guarantees zero overlaps by construction. The space is divided into regions using pipe networks as natural dividers, then each region is optimized independently.

## Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            OPTIMIZATION PIPELINE                             │
└─────────────────────────────────────────────────────────────────────────────┘

    ┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
    │   STAGE 1    │     │  STAGE 1.5   │     │   STAGE 2    │     │   STAGE 3    │
    │    Region    │ ──▶ │     Pull     │ ──▶ │    Greedy    │ ──▶ │   Combine    │
    │   Splitting  │     │    Inside    │     │  Placement   │     │   & Verify   │
    └──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
          │                    │                    │                    │
          ▼                    ▼                    ▼                    ▼
    Divide space by      Ensure all movables   Place objects one    Merge results,
    subtracting pipes    are fully inside      by one, avoiding     verify 0 overlaps
    from boundary        their assigned region all collisions
```

## Key Components

### 1. Region Splitting (`region_simple.py`)

Divides the placement space into independent regions using the **pipe subtraction method**:

```python
fixed_per_region, movables_per_region, regions_info = split_into_regions(
    movables, 
    fixed_obstacles, 
    placement_bounds,
    pipe_buffer=0.5,       # Buffer for detecting pipe connections
    min_region_area=10.0,  # Filter tiny regions
)
```

**How it works:**
1. Create boundary polygon from placement bounds
2. Buffer all pipes and merge connected ones
3. Subtract buffered pipes from boundary
4. Resulting pieces become separate regions
5. Assign movables to regions based on their center point
![Region Splitting](regions_after.png)
### 2. Pull Into Region (`greedy_optimizer.py`)

Ensures all movables are **fully contained** within their assigned region before optimization.

**Problem:** A movable's center may be in Region A, but its polygon extends into Region B.

**Solution:** Find the minimum displacement to pull the entire polygon inside:

```python
# Find the part of the polygon that's INSIDE the region
inside_part = mov_poly.intersection(region_poly)

# Direction: from target toward inside part centroid
direction = inside_centroid - target

# Move along direction until entire polygon fits
for dist in np.arange(0.1, 50, 0.1):
    test_pos = target + direction * dist
    if inner_region.contains(get_movable_polygon(mov, test_pos)):
        mov["target"] = test_pos
        break
```

### 3. Greedy Placement (`greedy_optimizer.py`)

Places objects one-by-one, guaranteeing zero overlaps by construction.

**Key insight:** Objects already at valid positions should stay there.

```python
# PHASE 1: Identify objects already at valid positions
for mov in movables:
    if is_inside_region(mov) and not overlaps_fixed_obstacles(mov):
        valid_at_target.append(mov)
    else:
        need_placement.append(mov)

# PHASE 2: Place valid objects first (largest first for priority)
for mov in sorted(valid_at_target, by_size, largest_first):
    if not overlaps_already_placed(mov):
        place_at_target(mov)  # No movement needed!
    else:
        need_placement.append(mov)

# PHASE 3: Place remaining objects using spiral search
for mov in sorted(need_placement, by_size, largest_first):
    new_pos = find_nearest_valid_position(mov)
    place_at(mov, new_pos)
```

### 4. Position Search

When an object needs to move, find the nearest valid position using **spiral search**:

```python
def spiral_search(center, max_radius, step=0.5, region_poly=None):
    """Search outward from center in spiral pattern."""
    yield center  # Try original position first
    
    for radius in np.arange(step, max_radius, step):
        n_points = max(8, int(2 * π * radius / step))
        for i in range(n_points):
            angle = 2 * π * i / n_points
            pos = center + radius * [cos(angle), sin(angle)]
            
            # Only yield positions inside region
            if region_poly is None or region_poly.contains(Point(pos)):
                yield pos
```

### 5. Collision Detection

Uses **Shapely** for robust polygon operations:

```python
def check_position_valid(mov, position, obstacle_union, placed_polys, region_poly):
    mov_poly = get_movable_polygon(mov, position)
    
    # Must be inside region
    if not region_poly.contains(mov_poly):
        return False
    
    # Must not overlap fixed obstacles
    if mov_poly.buffer(min_separation/2).intersects(obstacle_union):
        return False
    
    # Must not overlap already-placed movables
    for placed in placed_polys:
        if mov_poly.buffer(min_separation/2).intersects(placed):
            return False
    
    return True
```

## Geometry Handling

### Movable Structure

```python
movable = {
    "ElementId": "12345",           # Unique identifier
    "target": np.array([x, y]),     # Position (local origin placement)
    "verts": [[-2,-2], [2,-2], ...], # Vertices in local coordinates
    "RotationAngle": 0.0,           # Rotation in radians
}
```

### Coordinate System

- `target` is where the **local origin (0,0)** is placed in world coordinates
- `verts` are in **local coordinates** relative to origin
- The **geometric center** may differ from `target` if vertices aren't centered at origin

```python
# Get actual geometric center in world coordinates
local_center = np.array(mov["verts"]).mean(axis=0)
world_center = target + rotate(local_center, rotation_angle)
```

### Polygon Creation

```python
def get_movable_polygon(mov, position):
    """Create Shapely polygon for movable at given position."""
    # Transform vertices to world coordinates
    verts = translate_polygon(mov["verts"], position, mov["RotationAngle"])
    hull = get_convex_hull_vertices(verts)
    return ShapelyPolygon(hull)
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_separation` | 0.1 | Minimum gap between objects |
| `search_step` | 0.3 | Step size for spiral search |
| `max_search_radius` | 50.0 | Maximum search distance from target |
| `pipe_buffer` | 0.5 | Buffer for detecting pipe connections |
| `min_region_area` | 10.0 | Filter regions smaller than this |
| `min_margin` | 0.1 | Margin from region boundary |

## Output

```python
# Final positions as flat array [x0, y0, x1, y1, ...]
combined_result = np.array([...])

# Metrics
displacement = np.linalg.norm(final - original, axis=1)
avg_displacement = displacement.mean()
max_displacement = displacement.max()

# Overlap verification (should be 0)
overlaps = find_all_overlaps(result, movables, fixed_obstacles)
```

## Visualization

The system generates several visualization files:

| File | Description |
|------|-------------|
| `regions_before.png` | Two-panel view showing regions and movables before optimization |
| `regions_after.png` | Two-panel view after optimization |
| `before.png` | Standard view with original positions |
| `after.png` | Standard view with optimized positions |
| `output.json` | Final positions in JSON format |

## Algorithm Guarantees

1. **Zero overlaps** - By construction, each object is only placed if it doesn't overlap anything already placed

2. **Region containment** - All movables stay within their assigned region boundaries

3. **Minimum displacement** - Objects at valid positions stay in place; others move the minimum distance needed

4. **Fixed obstacle avoidance** - No movable overlaps pipes or other fixed obstacles

## Files

| File | Description |
|------|-------------|
| `main.py` | Main execution script |
| `greedy_optimizer.py` | Greedy placement algorithm |
| `region_simple.py` | Region splitting using pipe subtraction |
| `polygon_utils.py` | Geometry utilities (SAT, convex hull, etc.) |
| `json_helper.py` | JSON loading/saving |
| `plotting.py` | Visualization functions |

## Usage

```python
from main import main

# Run optimization
main()

# Or use components directly:
from region_simple import split_into_regions
from greedy_optimizer import greedy_optimize_with_regions, pull_movables_into_region

# Split into regions
fixed_per_region, movables_per_region, regions_info = split_into_regions(
    movables, fixed_obstacles, placement_bounds
)

# Pull movables inside their regions
for movs, region_info in zip(movables_per_region, regions_info):
    pull_movables_into_region(movs, region_info["shapely_polygon"])

# Run optimization
results, x0s, indices = greedy_optimize_with_regions(
    movables_per_region, fixed_per_region, regions_info
)
```

## Performance

- **Region splitting:** O(n) where n = number of pipes
- **Greedy placement:** O(m² × s) where m = movables per region, s = search positions
- **Typical runtime:** 1-5 seconds for ~100 movables

## Limitations

1. Greedy approach may not find globally optimal solution
2. Very crowded regions may require large displacements
3. Non-convex movables are approximated by convex hull

## Future Improvements

- [ ] Parallel region optimization
- [ ] Simulated annealing for local refinement
- [ ] Support for non-convex polygons
- [ ] Weighted displacement (prefer certain objects to stay in place)