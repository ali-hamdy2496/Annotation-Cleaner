import numpy as np
from joblib import Parallel, delayed

from polygon_utils import (
    unpack_xy,
    get_precomputed_geometry,
    separating_distance_SAT_precomputed,

)

def calculate_displacement_metric(x_initial, x_final):
    """
    Calculate the average displacement of objects from their original positions.
    D = (1/N) * sum(|x_final_i - x_original_i|)

    Args:
        x_initial: Initial positions (flattened)
        x_final: Final positions (flattened)

    Returns:
        Average displacement value
    """
    pts_initial = unpack_xy(x_initial)
    pts_final = unpack_xy(x_final)

    displacements = np.linalg.norm(pts_final - pts_initial, axis=1)
    avg_displacement = np.mean(displacements)
    return avg_displacement


def find_all_overlaps(x, movables, fixed_obstacles, min_separation=0.0):
    """
    Find all overlapping pairs between movables and between movables and fixed obstacles.
    Optimized with precomputed geometry and parallelization.
    """
    pts = unpack_xy(x)
    overlaps = []

    # Ensure precomputed geometry exists
    for m in movables:
        if "precomputed" not in m:
            m["precomputed"] = get_precomputed_geometry(
                m["verts"], m.get("RotationAngle", 0.0)
            )
    for obs in fixed_obstacles:
        if obs.get("center") is not None and "precomputed" not in obs:
            obs["precomputed"] = get_precomputed_geometry(
                obs["verts"], obs.get("RotationAngle", 0.0)
            )

    n_movables = len(movables)

    # Helper for movable-movable check
    def check_movable_movable(i, j, min_sep):
        hullA = movables[i]["precomputed"]["hull"] + pts[i]
        hullB = movables[j]["precomputed"]["hull"] + pts[j]

        sep, normal, penetration = separating_distance_SAT_precomputed(
            hullA,
            hullB,
            movables[i]["precomputed"]["normals"],
            movables[j]["precomputed"]["normals"],
            return_normal=True,
        )

        if sep < min_sep - 1e-6:
            return (i, j, min_sep - sep, normal)
        return None

    # Helper for movable-fixed check
    def check_movable_fixed(i, obs_idx, min_sep):
        obs = fixed_obstacles[obs_idx]
        if obs.get("center") is None:
            return None

        hullA = movables[i]["precomputed"]["hull"] + pts[i]
        hullB = obs["precomputed"]["hull"] + obs["center"]

        sep, normal, penetration = separating_distance_SAT_precomputed(
            hullA,
            hullB,
            movables[i]["precomputed"]["normals"],
            obs["precomputed"]["normals"],
            return_normal=True,
        )

        if sep < min_sep - 1e-6:
            return (i, None, min_sep - sep, normal)
        return None

    # Parallelize movable-movable checks
    if n_movables > 1:
        num_mm_pairs = n_movables * (n_movables - 1) // 2
        if num_mm_pairs > 50:
            mm_results = Parallel(n_jobs=-1, backend="threading")(
                delayed(check_movable_movable)(i, j, min_separation)
                for i in range(n_movables)
                for j in range(i + 1, n_movables)
            )
            overlaps.extend([r for r in mm_results if r is not None])
        else:
            for i in range(n_movables):
                for j in range(i + 1, n_movables):
                    res = check_movable_movable(i, j, min_separation)
                    if res:
                        overlaps.append(res)

    # Parallelize movable-fixed checks
    if n_movables > 0 and len(fixed_obstacles) > 0:
        num_mf_pairs = n_movables * len(fixed_obstacles)
        if num_mf_pairs > 50:
            mf_results = Parallel(n_jobs=-1, backend="threading")(
                delayed(check_movable_fixed)(i, obs_idx, min_separation)
                for i in range(n_movables)
                for obs_idx in range(len(fixed_obstacles))
            )
            overlaps.extend([r for r in mf_results if r is not None])
        else:
            for i in range(n_movables):
                for obs_idx in range(len(fixed_obstacles)):
                    res = check_movable_fixed(i, obs_idx, min_separation)
                    if res:
                        overlaps.append(res)

    return overlaps

