"""
JSON helper functions for reading problem data and saving optimization results.
"""

import json
import numpy as np
from polygon_utils import translate_polygon, unpack_xy


def load_problem_data(json_path):
    """
    Load movables and fixed obstacles from JSON file.

    Args:
        json_path: Path to the JSON file

    Returns:
        Tuple of (movables, fixed_obstacles, placement_bounds)
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    movables = []
    fixed_obstacles = []

    for item in data:
        origin = np.array([item["Origin"]["X"], item["Origin"]["Y"]])
        origin_z = item["Origin"]["Z"]
        verts_absolute = np.array([[v["x"], v["y"]] for v in item["Vertices"]])
        # verts_z = [v["Z"] for v in item["Vertices"]]
        verts_local = verts_absolute - origin

        element = {
            "verts": verts_local,
            "ElementId": item["ElementId"],
            "RotationAngle": item["RotationAngle"],
            "ElementType": item["ElementType"],
            "CategoryName": item["CategoryName"],
            "BuiltInCategory": item["BuiltInCategory"],
            "SegmentIndex": item["SegmentIndex"]
        }

        if item.get("IsMovable"):
            element["target"] = origin
            element["origin_z"] = origin_z
            # element["verts_z"] = verts_z
            movables.append(element)
        else:
            element["center"] = origin
            element["ElementType"] = item["ElementType"]
            fixed_obstacles.append(element)

    # Compute placement bounds from the data
    all_coords = []
    for item in data:
        for v in item["Vertices"]:
            all_coords.append([v["x"], v["y"]])

    all_coords = np.array(all_coords)
    x_min, y_min = all_coords.min(axis=0)
    x_max, y_max = all_coords.max(axis=0)

    # Add margin (20% of the range on each side)
    x_range = x_max - x_min
    y_range = y_max - y_min
    margin_x = 0.2 * x_range
    margin_y = 0.2 * y_range

    placement_bounds = (
        (x_min - margin_x, x_max + margin_x),
        (y_min - margin_y, y_max + margin_y),
    )

    return movables, fixed_obstacles, placement_bounds


def save_optimized_output(xvec, movables, output_path="output.json"):
    """
    Save the optimized movables to a JSON file with the same format as the input.

    Args:
        xvec: Optimized positions vector
        movables: List of movable object dictionaries
        output_path: Path to save the output JSON file
    """
    pts = unpack_xy(xvec)
    output_data = []

    for i, p in enumerate(pts):
        movable = movables[i]

        # Convert local vertices to absolute coordinates using optimized position
        verts_absolute = translate_polygon(
            movable["verts"], p, movable["RotationAngle"]
        )

        # Create output element
        element = {
            "ElementId": movable["ElementId"],
            "IsMovable": True,
            "RotationAngle": movable["RotationAngle"],
            "Origin": {
                "X": float(p[0]),
                "Y": float(p[1]),
                "Z": float(movable["origin_z"]),
            },
            "Vertices": [
                {"x": float(v[0]), "y": float(v[1])}
                for j, v in enumerate(verts_absolute)
            ],
            "ElementType": movable["ElementType"],
            "CategoryName": movable["CategoryName"],
            "BuiltInCategory": movable["BuiltInCategory"],
            "SegmentIndex": movable["SegmentIndex"]
        }

        output_data.append(element)

    # Save to JSON file
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"Saved optimized output to {output_path}")
