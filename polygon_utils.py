"""
Minimal polygon utilities for testing region splitting.
"""

import numpy as np
from scipy.spatial import ConvexHull


def translate_polygon(verts, center, rotation_angle):
    """
    Translate polygon vertices from local to world coordinates.
    
    Args:
        verts: Local vertices (N, 2) array
        center: Center position [x, y]
        rotation_angle: Rotation angle in radians
        
    Returns:
        World vertices as numpy array (N, 2)
    """
    verts = np.array(verts, dtype=float)
    center = np.array(center, dtype=float)
    
    if rotation_angle != 0:
        cos_r = np.cos(rotation_angle)
        sin_r = np.sin(rotation_angle)
        rot_matrix = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
        verts = verts @ rot_matrix.T
    
    return verts + center


def get_convex_hull_vertices(verts, closed=False):
    """
    Get convex hull vertices in order.
    
    Args:
        verts: Vertices (N, 2) array
        closed: If True, append first vertex at end
        
    Returns:
        Hull vertices as numpy array
    """
    verts = np.array(verts, dtype=float)
    
    if len(verts) < 3:
        if closed:
            return np.vstack([verts, verts[0:1]])
        return verts
    
    try:
        hull = ConvexHull(verts)
        hull_verts = verts[hull.vertices]
    except:
        hull_verts = verts
    
    if closed:
        hull_verts = np.vstack([hull_verts, hull_verts[0:1]])
    
    return hull_verts


def unpack_xy(x):
    """
    Unpack flattened position array to (N, 2) array.
    
    Args:
        x: Flattened array [x0, y0, x1, y1, ...]
        
    Returns:
        (N, 2) array of positions
    """
    x = np.array(x, dtype=float)
    return x.reshape(-1, 2)


def polygon_characteristic_size(verts):
    """
    Get characteristic size of polygon (max distance from centroid).
    """
    verts = np.array(verts, dtype=float)
    centroid = np.mean(verts, axis=0)
    distances = np.linalg.norm(verts - centroid, axis=1)
    return np.max(distances)


def polygon_edges(verts):
    """
    Get edges of polygon as (N, 2, 2) array.
    """
    verts = np.array(verts, dtype=float)
    n = len(verts)
    edges = np.zeros((n, 2, 2))
    for i in range(n):
        edges[i, 0] = verts[i]
        edges[i, 1] = verts[(i + 1) % n]
    return edges


def normals_from_edges(edges):
    """
    Get outward normals from edges.
    """
    normals = []
    for edge in edges:
        d = edge[1] - edge[0]
        normal = np.array([-d[1], d[0]])
        norm = np.linalg.norm(normal)
        if norm > 1e-9:
            normal = normal / norm
        normals.append(normal)
    return np.array(normals)


def get_precomputed_geometry(verts, rotation_angle=0.0):
    """
    Precompute geometry for faster collision detection.
    """
    verts = np.array(verts, dtype=float)
    
    if rotation_angle != 0:
        cos_r = np.cos(rotation_angle)
        sin_r = np.sin(rotation_angle)
        rot_matrix = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
        verts = verts @ rot_matrix.T
    
    hull = get_convex_hull_vertices(verts, closed=False)
    edges = polygon_edges(hull)
    normals = normals_from_edges(edges)
    
    return {
        "hull": hull,
        "normals": normals,
        "edges": edges
    }


def separating_distance_SAT_precomputed(hullA, hullB, normalsA, normalsB, return_normal=False):
    """
    Separating Axis Theorem for collision detection.
    
    Returns:
        If return_normal=False: separation distance (negative = overlap, positive = gap)
        If return_normal=True: (separation, normal, penetration)
        
    The separation is the MAXIMUM across all tested axes:
    - If positive: objects are separated by that distance
    - If negative: objects overlap by that amount (penetration depth)
    """
    max_sep = float('-inf')  # We want the MAXIMUM separation
    best_normal = np.array([1.0, 0.0])
    
    all_normals = np.vstack([normalsA, normalsB])
    
    for normal in all_normals:
        # Project both hulls onto the normal
        projA = hullA @ normal
        projB = hullB @ normal
        
        minA, maxA = projA.min(), projA.max()
        minB, maxB = projB.min(), projB.max()
        
        # Calculate separation on this axis
        # Positive = gap, Negative = overlap
        sep = max(minA - maxB, minB - maxA)
        
        # We want the axis with the LARGEST separation
        # If ANY axis has positive separation, objects don't overlap
        if sep > max_sep:
            max_sep = sep
            best_normal = normal.copy()
            if sep > 0:
                # Early exit - found separating axis, no overlap
                if return_normal:
                    return max_sep, best_normal, -max_sep
                return max_sep
    
    # Ensure normal points from A to B
    centerA = np.mean(hullA, axis=0)
    centerB = np.mean(hullB, axis=0)
    direction = centerB - centerA
    if np.dot(best_normal, direction) < 0:
        best_normal = -best_normal
    
    if return_normal:
        return max_sep, best_normal, -max_sep
    return max_sep