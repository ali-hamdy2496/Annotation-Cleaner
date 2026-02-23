"""
Test the greedy placement optimizer.
"""

import numpy as np
import sys
sys.path.insert(0, '/home/claude')

from greedy_optimizer import (
    greedy_optimize_region,
    grid_optimize_region,
    create_obstacle_union,
    check_position_valid,
)
from polygon_utils import get_precomputed_geometry, separating_distance_SAT_precomputed, translate_polygon
from shapely.geometry import Polygon as ShapelyPolygon


def check_overlaps_detailed(result, movables, fixed_obstacles, min_separation=0.0):
    """Check for overlaps using Shapely (ground truth)."""
    from shapely.ops import unary_union
    
    n = len(movables)
    pts = result.reshape(-1, 2)
    
    overlaps = []
    
    # Get movable polygons
    mov_polys = []
    for i, mov in enumerate(movables):
        verts = translate_polygon(mov["verts"], pts[i], mov.get("RotationAngle", 0.0))
        try:
            poly = ShapelyPolygon(verts)
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
        try:
            poly = ShapelyPolygon(verts)
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
            
            if min_separation > 0:
                # Check with buffer
                if mov_polys[i].buffer(min_separation/2).intersects(mov_polys[j].buffer(min_separation/2)):
                    overlaps.append(("mov-mov", i, j))
            else:
                if mov_polys[i].intersects(mov_polys[j]):
                    # Check if it's more than just touching
                    intersection = mov_polys[i].intersection(mov_polys[j])
                    if hasattr(intersection, 'area') and intersection.area > 1e-6:
                        overlaps.append(("mov-mov", i, j))
    
    # Check movable-fixed overlaps
    for i in range(n):
        if mov_polys[i] is None:
            continue
        for obs, obs_poly in fixed_polys:
            if min_separation > 0:
                if mov_polys[i].buffer(min_separation/2).intersects(obs_poly.buffer(min_separation/2)):
                    overlaps.append(("mov-fix", i, obs.get("ElementId", "?")))
            else:
                if mov_polys[i].intersects(obs_poly):
                    intersection = mov_polys[i].intersection(obs_poly)
                    if hasattr(intersection, 'area') and intersection.area > 1e-6:
                        overlaps.append(("mov-fix", i, obs.get("ElementId", "?")))
    
    return overlaps


def test_greedy_basic():
    """Test greedy placement with a simple scenario."""
    print("=" * 60)
    print("TEST: Greedy placement - basic")
    print("=" * 60)
    
    # Create a region
    region_boundary = [(0, 0), (50, 0), (50, 50), (0, 50), (0, 0)]
    
    # Create some fixed obstacles
    fixed_obstacles = [
        {
            "ElementId": "fixed_1",
            "center": [25, 25],
            "verts": [[-5, -5], [5, -5], [5, 5], [-5, 5]],
            "RotationAngle": 0.0,
        },
    ]
    
    # Create movables - some overlapping with each other and the fixed obstacle
    movables = [
        {"ElementId": "mov_0", "target": np.array([24, 24]), "verts": [[-2, -2], [2, -2], [2, 2], [-2, 2]], "RotationAngle": 0.0},
        {"ElementId": "mov_1", "target": np.array([26, 26]), "verts": [[-2, -2], [2, -2], [2, 2], [-2, 2]], "RotationAngle": 0.0},
        {"ElementId": "mov_2", "target": np.array([25, 25]), "verts": [[-2, -2], [2, -2], [2, 2], [-2, 2]], "RotationAngle": 0.0},
        {"ElementId": "mov_3", "target": np.array([10, 10]), "verts": [[-2, -2], [2, -2], [2, 2], [-2, 2]], "RotationAngle": 0.0},
        {"ElementId": "mov_4", "target": np.array([10, 10]), "verts": [[-2, -2], [2, -2], [2, 2], [-2, 2]], "RotationAngle": 0.0},  # Same as mov_3!
    ]
    
    print(f"Movables: {len(movables)}")
    print(f"Fixed obstacles: {len(fixed_obstacles)}")
    
    # Run greedy optimization
    result, x0, idx = greedy_optimize_region(
        movables,
        fixed_obstacles,
        region_boundary=region_boundary,
        region_index=0,
        min_separation=0.5,
        search_step=0.5,
        max_search_radius=30.0,
    )
    
    # Check for overlaps
    overlaps = check_overlaps_detailed(result, movables, fixed_obstacles, min_separation=0.0)
    
    print(f"\nResult positions:")
    pts = result.reshape(-1, 2)
    for i, mov in enumerate(movables):
        orig = mov["target"]
        final = pts[i]
        disp = np.linalg.norm(final - orig)
        print(f"  {mov['ElementId']}: ({orig[0]:.1f}, {orig[1]:.1f}) -> ({final[0]:.1f}, {final[1]:.1f}), disp={disp:.2f}")
    
    print(f"\nOverlaps found: {len(overlaps)}")
    for ov in overlaps:
        print(f"  {ov}")
    
    if len(overlaps) == 0:
        print("✓ TEST PASSED - No overlaps!")
    else:
        print("✗ TEST FAILED - Overlaps remain!")
    
    return len(overlaps) == 0


def test_greedy_crowded():
    """Test greedy placement in a crowded region."""
    print("\n" + "=" * 60)
    print("TEST: Greedy placement - crowded region")
    print("=" * 60)
    
    # Small region
    region_boundary = [(0, 0), (30, 0), (30, 30), (0, 30), (0, 0)]
    
    # Pipe through the middle
    fixed_obstacles = [
        {
            "ElementId": "pipe_h",
            "ElementType": "Pipe",
            "center": [15, 15],
            "verts": [[-15, -1], [15, -1], [15, 1], [-15, 1]],
            "RotationAngle": 0.0,
        },
    ]
    
    # Many movables, all targeting positions near the pipe
    movables = []
    for i in range(10):
        movables.append({
            "ElementId": f"mov_{i}",
            "target": np.array([5 + i * 2, 15]),  # All along the pipe!
            "verts": [[-1.5, -1.5], [1.5, -1.5], [1.5, 1.5], [-1.5, 1.5]],
            "RotationAngle": 0.0,
        })
    
    print(f"Movables: {len(movables)}")
    print(f"All movables target positions along the pipe!")
    
    # Run greedy optimization
    result, x0, idx = greedy_optimize_region(
        movables,
        fixed_obstacles,
        region_boundary=region_boundary,
        region_index=0,
        min_separation=0.5,
        search_step=0.5,
        max_search_radius=20.0,
    )
    
    # Check for overlaps
    overlaps = check_overlaps_detailed(result, movables, fixed_obstacles, min_separation=0.0)
    
    print(f"\nOverlaps found: {len(overlaps)}")
    
    if len(overlaps) == 0:
        print("✓ TEST PASSED - No overlaps!")
    else:
        print("✗ TEST FAILED - Overlaps remain!")
        for ov in overlaps[:5]:
            print(f"  {ov}")
    
    return len(overlaps) == 0


def test_grid_placement():
    """Test grid-based placement."""
    print("\n" + "=" * 60)
    print("TEST: Grid-based placement")
    print("=" * 60)
    
    region_boundary = [(0, 0), (40, 0), (40, 40), (0, 40), (0, 0)]
    
    fixed_obstacles = [
        {
            "ElementId": "obs_1",
            "center": [20, 20],
            "verts": [[-8, -8], [8, -8], [8, 8], [-8, 8]],
            "RotationAngle": 0.0,
        },
    ]
    
    # Movables clustered at center (all overlapping)
    movables = []
    for i in range(8):
        movables.append({
            "ElementId": f"mov_{i}",
            "target": np.array([20 + np.random.randn(), 20 + np.random.randn()]),
            "verts": [[-2, -2], [2, -2], [2, 2], [-2, 2]],
            "RotationAngle": 0.0,
        })
    
    print(f"Movables: {len(movables)}, all clustered at center around obstacle")
    
    # Run grid optimization
    result, x0, idx = grid_optimize_region(
        movables,
        fixed_obstacles,
        region_boundary=region_boundary,
        region_index=0,
        min_separation=0.5,
        grid_step=1.0,
    )
    
    # Check for overlaps
    overlaps = check_overlaps_detailed(result, movables, fixed_obstacles, min_separation=0.0)
    
    print(f"\nOverlaps found: {len(overlaps)}")
    
    if len(overlaps) == 0:
        print("✓ TEST PASSED - No overlaps!")
    else:
        print("✗ TEST FAILED - Overlaps remain!")
    
    return len(overlaps) == 0


if __name__ == "__main__":
    np.random.seed(42)
    
    t1 = test_greedy_basic()
    t2 = test_greedy_crowded()
    t3 = test_grid_placement()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Basic test: {'PASS' if t1 else 'FAIL'}")
    print(f"Crowded test: {'PASS' if t2 else 'FAIL'}")
    print(f"Grid test: {'PASS' if t3 else 'FAIL'}")
    
    if t1 and t2 and t3:
        print("\n✓ ALL TESTS PASSED!")
    else:
        print("\n✗ SOME TESTS FAILED!")