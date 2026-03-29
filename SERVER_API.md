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

Returns the status of all known jobs. `output_data` is included for completed jobs and `null` for all others.

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
            "error": null,
            "output_data": [
                {
                    "ElementId": 11755664,
                    "IsMovable": true,
                    "RotationAngle": 0,
                    "Origin": { "X": 722.4, "Y": -485.7, "Z": 207.9 },
                    "Vertices": [
                        { "X": 721.5, "Y": -487.6 },
                        { "X": 723.3, "Y": -487.6 },
                        { "X": 723.3, "Y": -483.7 },
                        { "X": 721.5, "Y": -483.7 }
                    ],
                    "ElementType": "...",
                    "SegmentIndex": 0,
                    "state": "solved"
                }
            ]
        },
        "job_002": {
            "status": "running",
            "submitted_at": 1710000015.000,
            "started_at": 1710000015.200,
            "finished_at": null,
            "num_overlaps": null,
            "avg_displacement": null,
            "total_time": null,
            "error": null,
            "output_data": null
        },
        "job_003": {
            "status": "failed",
            "submitted_at": 1710000020.000,
            "started_at": 1710000020.100,
            "finished_at": 1710000020.500,
            "num_overlaps": null,
            "avg_displacement": null,
            "total_time": null,
            "error": "Invalid input format",
            "output_data": null
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

**Response fields per job:**

| Field              | Type         | Description                                                      |
|--------------------|--------------|------------------------------------------------------------------|
| `status`           | string       | Current job state (see table above)                              |
| `submitted_at`     | float / null | Unix timestamp when the job was received                         |
| `started_at`       | float / null | Unix timestamp when processing began                             |
| `finished_at`      | float / null | Unix timestamp when the job finished                             |
| `num_overlaps`     | int / null   | Remaining overlaps after optimization (`completed` only)         |
| `avg_displacement` | float / null | Average annotation displacement from original (`completed` only) |
| `total_time`       | float / null | Processing time in seconds (`completed` only)                    |
| `error`            | string / null| Error message (`failed` only)                                    |
| `output_data`      | array / null | Optimized annotation elements (`completed` only, `null` otherwise) |

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
1.  POST /submit  →  202  { "job_id": "job_001" }
2.  GET  /status  →  200  { "jobs": { "job_001": { "status": "running",   "output_data": null, ... } } }
3.  GET  /status  →  200  { "jobs": { "job_001": { "status": "completed", "output_data": [...], ... } } }
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
