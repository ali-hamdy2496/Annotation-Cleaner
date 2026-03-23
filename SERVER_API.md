# Annotation Cleaner — HTTP API Reference

Flask server exposing a job-based REST API for annotation placement optimization.

---

## Base URL

```
http://<host>:<port>
```

When running locally (default port 5000):

```
http://localhost:5000
```

---

## Endpoints

### 1. Submit Job

**`POST /submit`**

Submit a new optimization job. The job runs asynchronously in a background thread.

#### Request

**Headers:**
```
Content-Type: application/json
```

**Body:**
```json
{
    "job_id": "job_001",
    "data": [ ... ],
    "callback_url": "https://your-service.com/callback"
}
```

| Field          | Type   | Required | Description                                              |
|----------------|--------|----------|----------------------------------------------------------|
| `job_id`       | string | Yes      | Unique identifier for this job                           |
| `data`         | array  | Yes      | Input annotation elements (same format as input JSON file) |
| `callback_url` | string | No       | URL to POST results to when the job completes or fails   |

**Example `curl`:**
```bash
curl -X POST http://localhost:5000/submit \
  -H "Content-Type: application/json" \
  -d '{
    "job_id": "job_001",
    "data": [
      {
        "id": 1,
        "x": 100.0,
        "y": 200.0,
        "width": 50.0,
        "height": 20.0
      },
      {
        "id": 2,
        "x": 120.0,
        "y": 210.0,
        "width": 50.0,
        "height": 20.0
      }
    ],
    "callback_url": "https://your-service.com/callback"
  }'
```

#### Responses

**`202 Accepted` — Job started:**
```json
{
    "message": "Working on job job_001",
    "job_id": "job_001"
}
```

**`400 Bad Request` — Missing `job_id`:**
```json
{
    "error": "Missing required field: job_id"
}
```

**`400 Bad Request` — Missing `data`:**
```json
{
    "error": "Missing required field: data"
}
```

**`409 Conflict` — Job already running:**
```json
{
    "error": "Job job_001 is already running"
}
```

---

### 2. Get Status

**`GET /status`**

Returns the status of all known jobs (does not include `output_data` to keep the response lightweight).

#### Request

No body or parameters required.

**Example `curl`:**
```bash
curl http://localhost:5000/status
```

#### Response

**`200 OK`:**
```json
{
    "jobs": {
        "job_001": {
            "status": "completed",
            "submitted_at": 1710000000.123,
            "started_at": 1710000000.456,
            "finished_at": 1710000012.789,
            "num_overlaps": 0,
            "avg_displacement": 3.42,
            "total_time": 12.333,
            "error": null
        },
        "job_002": {
            "status": "running",
            "submitted_at": 1710000015.000,
            "started_at": 1710000015.200,
            "finished_at": null,
            "num_overlaps": null,
            "avg_displacement": null,
            "total_time": null,
            "error": null
        },
        "job_003": {
            "status": "failed",
            "submitted_at": 1710000020.000,
            "started_at": 1710000020.100,
            "finished_at": 1710000020.500,
            "num_overlaps": null,
            "avg_displacement": null,
            "total_time": null,
            "error": "Invalid input format"
        }
    }
}
```

**Job `status` values:**

| Value       | Meaning                              |
|-------------|--------------------------------------|
| `queued`    | Accepted, not yet started            |
| `running`   | Currently being processed            |
| `completed` | Finished successfully                |
| `failed`    | Finished with an error               |

**Timestamp fields** (`submitted_at`, `started_at`, `finished_at`) are Unix epoch seconds (`float`). `null` means that stage has not been reached yet.

---

### 3. Get Result

**`GET /result/<job_id>`**

Get the full result for a specific job, including the optimized `output_data`.

#### Request

No body required. The job ID is in the URL path.

**Example `curl`:**
```bash
curl http://localhost:5000/result/job_001
```

#### Responses

**`200 OK` — Job completed:**
```json
{
    "job_id": "job_001",
    "status": "completed",
    "num_overlaps": 0,
    "avg_displacement": 3.42,
    "total_time": 12.333,
    "output_data": [
      {
        "id": 1,
        "x": 100.0,
        "y": 200.0,
        "width": 50.0,
        "height": 20.0
      },
      {
        "id": 2,
        "x": 125.0,
        "y": 225.0,
        "width": 50.0,
        "height": 20.0
      }
    ]
}
```

**Result fields:**

| Field              | Type   | Description                                              |
|--------------------|--------|----------------------------------------------------------|
| `job_id`           | string | The job identifier                                       |
| `status`           | string | Always `"completed"` in this response                    |
| `num_overlaps`     | int    | Number of remaining overlaps after optimization          |
| `avg_displacement` | float  | Average displacement of annotations from original positions |
| `total_time`       | float  | Total processing time in seconds                         |
| `output_data`      | array  | Optimized annotation elements in the same format as input |

**`200 OK` — Job failed:**
```json
{
    "job_id": "job_001",
    "status": "failed",
    "error": "Invalid input format"
}
```

**`202 Accepted` — Job still in progress:**
```json
{
    "error": "Job job_001 is still running"
}
```

**`404 Not Found` — Job does not exist:**
```json
{
    "error": "Job job_001 not found"
}
```

---

## Callback (Webhook)

If a `callback_url` was provided in the `/submit` request, the server will `POST` the result to that URL when the job finishes.

**On success:**
```json
{
    "job_id": "job_001",
    "status": "completed",
    "num_overlaps": 0,
    "avg_displacement": 3.42,
    "total_time": 12.333,
    "output_data": [ ... ]
}
```

**On failure:**
```json
{
    "job_id": "job_001",
    "status": "failed",
    "error": "Description of what went wrong"
}
```

The callback request has a **30-second timeout**. If delivery fails, the error is logged and stored in the job under `callback_error`, but it does not affect the job status.

---

## Typical Workflow

```
1.  POST /submit          →  202  { "job_id": "job_001" }
2.  GET  /status          →  200  { "jobs": { "job_001": { "status": "running", ... } } }
3.  GET  /status          →  200  { "jobs": { "job_001": { "status": "completed", ... } } }
4.  GET  /result/job_001  →  200  { "status": "completed", "output_data": [...], ... }
```

Or, skip polling entirely by providing a `callback_url` in step 1.

---

## Running Locally

```bash
python server.py
```

Server starts on port `5000` by default. Override with the `PORT` environment variable:

```bash
PORT=8080 python server.py
```

## Running with Gunicorn (production / Heroku)

```bash
gunicorn server:app --bind 0.0.0.0:$PORT
```
