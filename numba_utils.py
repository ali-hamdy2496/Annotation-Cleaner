"""
Numba-accelerated utility functions for collision detection.
"""

import numpy as np
from numba import njit


def pack_geometry(polygons):
    """
    Pack a list of polygon dictionaries into flat NumPy arrays for Numba.

    Args:
        polygons: List of dicts, each with "verts" (N, 2) and "RotationAngle".
                  Note: "verts" here should be the LOCAL vertices (relative to center/target).

    Returns:
        Tuple of:
        - all_verts: (TotalVerts, 2) array of all vertices
        - all_normals: (TotalVerts, 2) array of all edge normals
        - offsets: (N_polys,) array of start indices in all_verts/all_normals
        - counts: (N_polys,) array of vertex counts per polygon
    """
    all_verts_list = []
    all_normals_list = []
    offsets = []
    counts = []

    current_offset = 0

    for poly in polygons:
        verts = poly["verts"]
        n_verts = len(verts)
        offsets.append(current_offset)
        counts.append(n_verts)

        all_verts_list.append(verts)

        # Compute normals
        # Edges: v[i+1] - v[i]
        # Normal: (dy, -dx)
        # We assume vertices are CCW

        verts_next = np.roll(verts, -1, axis=0)
        edges = verts_next - verts

        normals = np.zeros_like(edges)
        normals[:, 0] = -edges[:, 1]
        normals[:, 1] = edges[:, 0]

        # Normalize
        norms = np.sqrt(normals[:, 0] ** 2 + normals[:, 1] ** 2)
        norms[norms < 1e-12] = 1.0
        normals /= norms[:, np.newaxis]

        all_normals_list.append(normals)

        current_offset += n_verts

    if not all_verts_list:
        return (
            np.zeros((0, 2), dtype=np.float64),
            np.zeros((0, 2), dtype=np.float64),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
        )

    return (
        np.vstack(all_verts_list).astype(np.float64),
        np.vstack(all_normals_list).astype(np.float64),
        np.array(offsets, dtype=np.int64),
        np.array(counts, dtype=np.int64),
    )


@njit(fastmath=True)
def get_min_max_proj(verts, axis):
    """Project vertices onto axis and return min, max."""
    min_proj = np.inf
    max_proj = -np.inf

    for i in range(len(verts)):
        proj = verts[i, 0] * axis[0] + verts[i, 1] * axis[1]
        if proj < min_proj:
            min_proj = proj
        if proj > max_proj:
            max_proj = proj

    return min_proj, max_proj


@njit(fastmath=True)
def separating_distance_SAT_numba(vertsA, normalsA, vertsB, normalsB):
    """
    JIT-compiled SAT separating distance.
    Returns:
        max_sep: float (positive = separation, negative = overlap depth)
        normal: (2,) array (direction to move A to resolve overlap/maintain separation)
                Always points AWAY from B.
    """
    max_sep = -np.inf
    best_normal = np.zeros(2)

    # Check axes from A
    nA = len(normalsA)
    for i in range(nA):
        axis = normalsA[i]

        minA, maxA = get_min_max_proj(vertsA, axis)
        minB, maxB = get_min_max_proj(vertsB, axis)

        # Separation on this axis
        d1 = minB - maxA  # A is left of B
        d2 = minA - maxB  # A is right of B
        sep = max(d1, d2)

        if sep > max_sep:
            max_sep = sep
            # Determine correct push direction (away from B)
            if d1 > d2:
                # A is left of B (or overlapping on left)
                # We want to push A left (negative axis direction)
                best_normal = -axis
            else:
                # A is right of B
                # We want to push A right (positive axis direction)
                best_normal = axis

            # Optimization: if separated, we can exit early
            if max_sep > 0:
                return max_sep, best_normal

    # Check axes from B
    nB = len(normalsB)
    for i in range(nB):
        axis = normalsB[i]

        minA, maxA = get_min_max_proj(vertsA, axis)
        minB, maxB = get_min_max_proj(vertsB, axis)

        d1 = minB - maxA
        d2 = minA - maxB
        sep = max(d1, d2)

        if sep > max_sep:
            max_sep = sep
            if d1 > d2:
                best_normal = -axis
            else:
                best_normal = axis

            if max_sep > 0:
                return max_sep, best_normal

    return max_sep, best_normal


@njit(fastmath=True)
def check_collisions_numba(
    current_verts,
    current_normals,
    others_verts,
    others_normals,
    others_offsets,
    others_counts,
    min_separation,
):
    """
    Check one polygon against many others using Numba.
    """
    n_others = len(others_counts)

    # Pre-allocate output (assume max 100 collisions)
    max_collisions = 100
    collisions = np.zeros((max_collisions, 3), dtype=np.float64)
    count = 0

    for i in range(n_others):
        start = others_offsets[i]
        n = others_counts[i]
        end = start + n

        other_v = others_verts[start:end]
        other_n = others_normals[start:end]

        sep, normal = separating_distance_SAT_numba(
            current_verts, current_normals, other_v, other_n
        )

        # Check against min_separation
        if sep < min_separation:
            # We have a violation
            # sep is e.g. -0.1 (overlap) or 0.1 (separated but too close)
            push_dist = min_separation - sep

            if count < max_collisions:
                collisions[count, 0] = push_dist
                collisions[count, 1] = normal[0]
                collisions[count, 2] = normal[1]
                count += 1

    return count, collisions
