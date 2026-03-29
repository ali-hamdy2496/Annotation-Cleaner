"""
Test client for the annotation optimization server.

Usage:
    # Local server (default)
    python3 test_client.py

    # Remote Heroku server
    python3 test_client.py https://your-app.herokuapp.com

Steps:
    1. Submits the job with AnnotationCleaner_CurveLoops.json data
    2. Polls /status every 2 seconds until the job completes
    3. Fetches full result from /result/<job_id>
    4. Plots before/after visualizations
"""

import sys
import json
import time
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon

SERVER_URL = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://localhost:5000"
JOB_ID = 1001
INPUT_FILE = "AnnotationCleaner_CurveLoops.json"


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_before_after(input_data, output_data):
    """Plot movable positions before (input) and after (output) optimization."""
    movables_before = [e for e in input_data if e.get("IsMovable", False)]
    movables_after = [e for e in output_data if e.get("IsMovable", False)]
    fixed = [e for e in input_data if not e.get("IsMovable", False)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    for ax, movables, title in [
        (ax1, movables_before, "Before Optimization"),
        (ax2, movables_after, "After Optimization"),
    ]:
        # Draw fixed obstacles
        for obs in fixed:
            verts = obs.get("Vertices", [])
            if len(verts) >= 3:
                world_verts = [[v["X"], v["Y"]] for v in verts]
                patch = MplPolygon(world_verts, closed=True, fc="lightgray", ec="gray", lw=0.3, alpha=0.5)
                ax.add_patch(patch)

        # Draw movables
        for mov in movables:
            verts = mov.get("Vertices", [])
            if len(verts) >= 3:
                world_verts = [[v["X"], v["Y"]] for v in verts]
                patch = MplPolygon(world_verts, closed=True, fc="cornflowerblue", ec="navy", lw=0.5, alpha=0.6)
                ax.add_patch(patch)

        ax.set_title(title, fontsize=14)
        ax.set_aspect("equal")
        ax.autoscale_view()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = "test_before_after.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[Plot] Saved {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print(f"Server: {SERVER_URL}")

    # Load input data
    print(f"Loading {INPUT_FILE}...")
    with open(INPUT_FILE, "r") as f:
        input_data = json.load(f)
    print(f"Loaded {len(input_data)} elements")

    # Submit job (no callback_url — we'll poll for the result)
    payload = {
        "job_id": JOB_ID,
        "data": input_data,
    }

    print(f"\nSubmitting job {JOB_ID} to {SERVER_URL}/submit ...")
    resp = requests.post(f"{SERVER_URL}/submit", json=payload, timeout=60)
    print(f"Submit response ({resp.status_code}): {resp.json()}")

    if resp.status_code != 202:
        print("Submit failed, exiting.")
        return

    # Poll /status every 2 seconds until completed/failed
    print("\nPolling /status every 2 seconds...\n")
    job_info = None
    while True:
        time.sleep(2)
        try:
            status_resp = requests.get(f"{SERVER_URL}/status", timeout=10)
            status_data = status_resp.json()
            job_info = status_data.get("jobs", {}).get(str(JOB_ID))

            if job_info is None:
                print(f"  Job {JOB_ID}: not found in status")
                continue

            status = job_info["status"]
            elapsed = ""
            if job_info.get("started_at") and status == "running":
                elapsed = f" ({time.time() - job_info['started_at']:.1f}s elapsed)"
            print(f"  Job {JOB_ID}: {status}{elapsed}")

            if status in ("completed", "failed"):
                break

        except Exception as e:
            print(f"  Status poll error: {e}")

    print(f"\n{'=' * 60}")
    print(f"Job {JOB_ID} — {job_info['status']}")

    if job_info["status"] == "completed":
        print(f"  Overlaps:         {job_info['num_overlaps']}")
        print(f"  Avg displacement: {job_info['avg_displacement']:.4f}")
        print(f"  Total time:       {job_info['total_time']:.2f}s")

        output_data = job_info["output_data"]
        print(f"  Output elements:  {len(output_data)}")

        # Plot before/after
        plot_before_after(input_data, output_data)
    else:
        print(f"  Error: {job_info.get('error', 'unknown')}")

    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
