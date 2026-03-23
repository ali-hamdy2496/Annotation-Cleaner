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
import requests as req_lib
from flask import Flask, request, jsonify

from main import run_optimization

app = Flask(__name__)

# In-memory job store: job_id -> {status, submitted_at, started_at, finished_at, error, ...}
jobs = {}
jobs_lock = threading.Lock()


def _run_job(job_id, input_data, callback_url):
    """Run optimization in a background thread and optionally POST results to callback_url."""
    with jobs_lock:
        jobs[job_id]["status"] = "running"
        jobs[job_id]["started_at"] = time.time()

    try:
        result = run_optimization(input_data)

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
    Get status of all jobs (without output_data).
    """
    with jobs_lock:
        snapshot = {}
        for jid, info in jobs.items():
            snapshot[jid] = {
                "status": info["status"],
                "submitted_at": info.get("submitted_at"),
                "started_at": info.get("started_at"),
                "finished_at": info.get("finished_at"),
                "num_overlaps": info.get("num_overlaps"),
                "avg_displacement": info.get("avg_displacement"),
                "total_time": info.get("total_time"),
                "error": info.get("error"),
            }

    return jsonify({"jobs": snapshot}), 200


@app.route("/result/<job_id>", methods=["GET"])
def get_result(job_id):
    """
    Get full result for a completed job, including output_data.
    """
    with jobs_lock:
        info = jobs.get(job_id) or jobs.get(int(job_id) if job_id.isdigit() else job_id)

    if info is None:
        return jsonify({"error": f"Job {job_id} not found"}), 404

    if info["status"] not in ("completed", "failed"):
        return jsonify({"error": f"Job {job_id} is still {info['status']}"}), 202

    if info["status"] == "failed":
        return jsonify({"job_id": job_id, "status": "failed", "error": info.get("error")}), 200

    return jsonify({
        "job_id": job_id,
        "status": "completed",
        "num_overlaps": info.get("num_overlaps"),
        "avg_displacement": info.get("avg_displacement"),
        "total_time": info.get("total_time"),
        "output_data": info.get("output_data"),
    }), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
