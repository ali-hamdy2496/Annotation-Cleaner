"""
Flask API server for annotation placement optimization.

Endpoints:
    POST /submit          - Submit a new optimization job
    GET  /status          - Get status of all ongoing jobs
    GET  /result/<job_id> - Get full result (including output_data) for a completed job
"""

import os
import threading
import time
import psutil
import requests as req_lib
from flask import Flask, request, jsonify

from main import run_optimization

app = Flask(__name__)

# In-memory job store: job_id -> {status, submitted_at, started_at, finished_at, error, ...}
jobs = {}
jobs_lock = threading.Lock()


def _monitor_performance(job_id, stop_event):
    """Log CPU and RAM usage every 2 seconds until stop_event is set."""
    logs = []
    total_cores = psutil.cpu_count(logical=True)
    total_ram = psutil.virtual_memory().total
    while not stop_event.is_set():
        cpu_percent = psutil.cpu_percent(interval=2)
        mem = psutil.virtual_memory()
        logs.append({
            "timestamp": time.time(),
            "cpu_percent": cpu_percent,
            "cpu_cores_total": total_cores,
            "cpu_cores_used": round(cpu_percent / 100 * total_cores, 2),
            "ram_total_bytes": total_ram,
            "ram_used_bytes": mem.used,
            "ram_percent": mem.percent,
        })
    with jobs_lock:
        jobs[job_id]["performance"] = logs


def _run_job(job_id, input_data, callback_url):
    """Run optimization in a background thread and optionally POST results to callback_url."""
    with jobs_lock:
        jobs[job_id]["status"] = "running"
        jobs[job_id]["started_at"] = time.time()

    stop_event = threading.Event()
    monitor_thread = threading.Thread(
        target=_monitor_performance, args=(job_id, stop_event), daemon=True
    )
    monitor_thread.start()

    try:
        result = run_optimization(input_data)

        stop_event.set()
        monitor_thread.join()

        with jobs_lock:
            jobs[job_id]["status"] = "completed"
            jobs[job_id]["finished_at"] = time.time()
            jobs[job_id]["num_overlaps"] = result["num_overlaps"]
            jobs[job_id]["avg_displacement"] = result["avg_displacement"]
            jobs[job_id]["total_time"] = result["total_time"]
            jobs[job_id]["output_data"] = result["output_data"]

        # Send results back to the caller if callback_url was provided
        if callback_url:
            callback_payload = {
                "job_id": job_id,
                "status": "completed",
                "num_overlaps": result["num_overlaps"],
                "avg_displacement": result["avg_displacement"],
                "total_time": result["total_time"],
                "output_data": result["output_data"],
            }
            try:
                resp = req_lib.post(callback_url, json=callback_payload, timeout=30)
                print(f"[Job {job_id}] Callback sent to {callback_url} — HTTP {resp.status_code}")
            except Exception as e:
                print(f"[Job {job_id}] Callback to {callback_url} failed: {e}")
                with jobs_lock:
                    jobs[job_id]["callback_error"] = str(e)

    except Exception as e:
        stop_event.set()
        monitor_thread.join()

        with jobs_lock:
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["finished_at"] = time.time()
            jobs[job_id]["error"] = str(e)

        if callback_url:
            try:
                req_lib.post(callback_url, json={
                    "job_id": job_id,
                    "status": "failed",
                    "error": str(e),
                }, timeout=30)
            except Exception:
                pass

        print(f"[Job {job_id}] Failed: {e}")


@app.route("/submit", methods=["POST"])
def submit_job():
    """
    Submit a new optimization job.

    Request JSON:
        {
            "job_id": <unique identifier>,
            "data": [ ... ],                    # input elements (same format as input JSON file)
            "callback_url": "<url>" (optional)   # URL to POST results to when done
        }

    Response JSON:
        {
            "message": "Working on job <job_id>",
            "job_id": <job_id>
        }
    """
    body = request.get_json(force=True)

    job_id = body.get("job_id")
    input_data = body.get("data")
    callback_url = body.get("callback_url")

    if job_id is None:
        return jsonify({"error": "Missing required field: job_id"}), 400
    if input_data is None:
        return jsonify({"error": "Missing required field: data"}), 400

    with jobs_lock:
        if job_id in jobs and jobs[job_id]["status"] == "running":
            return jsonify({"error": f"Job {job_id} is already running"}), 409

        jobs[job_id] = {
            "status": "queued",
            "submitted_at": time.time(),
            "started_at": None,
            "finished_at": None,
            "callback_url": callback_url,
            "output_data": None,
        }

    thread = threading.Thread(target=_run_job, args=(job_id, input_data, callback_url), daemon=True)
    thread.start()

    return jsonify({"message": f"Working on job {job_id}", "job_id": job_id}), 202


@app.route("/status", methods=["GET"])
def get_status():
    """
    Get status of all jobs.
    output_data is included only for completed jobs; null otherwise.
    """
    with jobs_lock:
        snapshot = {}
        for jid, info in jobs.items():
            completed = info["status"] == "completed"
            snapshot[jid] = {
                "status": info["status"],
                "submitted_at": info.get("submitted_at"),
                "started_at": info.get("started_at"),
                "finished_at": info.get("finished_at"),
                "num_overlaps": info.get("num_overlaps"),
                "avg_displacement": info.get("avg_displacement"),
                "total_time": info.get("total_time"),
                "error": info.get("error"),
                "output_data": info.get("output_data") if completed else None,
                "performance": info.get("performance", []) if completed else [],
            }

    return jsonify({"jobs": snapshot}), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
