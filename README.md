# Robot Arena Backend

FastAPI backend for the Robot Policy Comparison Study. Collects participant responses, persists them to PostgreSQL, and serves a live dashboard with policy rankings computed by multiple algorithms.

## Features

- **Annotation collection** — stores per-participant responses with quiz scores, sanity-check results, and failure tracking
- **Multiple ranking algorithms** — Bradley-Terry MLE, EM with latent task-difficulty buckets, Elo, and points-based
- **Live dashboard** — view policy standings at `GET /` with a 60-second server-side cache
- **Version-aware pairs** — each config version maps to its own `pairs*.json` file
- **Legacy support** — `/annotate/legacy` endpoint for older frontend versions

## Setup

### 1. Database

```bash
# Install PostgreSQL (if not already installed)
sudo apt install postgresql postgresql-contrib

# Create the database
sudo -u postgres psql -c "CREATE DATABASE robotarenainf;"

# Apply the schema
sudo -u postgres psql -d robotarenainf -f schema.sql
```

### 2. Credentials

Edit `main.py` with your database credentials:

```python
def get_db_connection():
    return psycopg2.connect(
        host="localhost",
        port=5432,
        database="robotarenainf",
        user="your_username",
        password="your_password"
    )
```

### 3. Install Dependencies

```bash
pip install fastapi uvicorn psycopg2-binary pydantic numpy scipy pandas statsmodels
```

### 4. Add Pairs Files

Copy the generated pairs file(s) from the frontend into the backend directory:

```bash
cp ../videos/pairs.json .
```

The version-to-file mapping is defined in `main.py`:

```python
PAIRS_FILES = {
    None:   'pairs_v1.json',   # Legacy / no version specified
    'v1.0': 'pairs_v1.json',
    'v2.0': 'pairs_v2.json',
    'v5.0': 'pairs.json',      # Current version
}
```

### 5. Run the Server

```bash
# Development (with auto-reload)
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Production
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

---

## API Reference

### `POST /annotate`

Save a completed study session.

**Request body:**
```json
{
  "participant_id": "worker_123",
  "participant_type": "paid",
  "completion_code": "ABC12345",
  "total_time_ms": 450000,
  "response_length": 35,
  "quiz_score": 8,
  "quiz_total": 10,
  "sanity_check_results": [
    { "index": 360, "correct": true, "position": 12, "userAnswer": "right", "correctAnswer": "right" }
  ],
  "sanity_checks_passed": 5,
  "sanity_checks_total": 5,
  "failed": false,
  "failure_reason": null,
  "responses": [ ... ],
  "timestamp": "2024-12-08T10:30:00Z",
  "config_version": "v5.0"
}
```

**Response:**
```json
{ "status": "success", "message": "Annotation saved successfully" }
```

### `GET /`

Live HTML dashboard showing policy rankings from all four algorithms. Results are cached for 60 seconds and invalidated when new annotations arrive.

### `POST /annotate/legacy`

Accepts the older payload format (no quiz/sanity fields) for backwards compatibility.

### `GET /versions`

Returns metadata about all configured pairs versions and their load status.

### `POST /versions/reload`

Clears the in-memory pairs cache and reloads all files from disk. Also invalidates the dashboard cache. Useful after updating pairs files without restarting the server.

### `POST /dashboard/refresh`

Forces the dashboard cache to regenerate on the next `GET /` request.

### `GET /stats/by-version`

Returns annotation counts and timestamps grouped by `config_version`.

---

## Ranking Algorithms

Only `"regular"` type responses are used for ranking — quiz and sanity-check responses are excluded.

| Algorithm | Description |
|-----------|-------------|
| **Bradley-Terry MLE** | Maximum likelihood estimation of pairwise win probabilities, with sandwich (HC0) standard errors and 95% confidence intervals |
| **EM (Latent Buckets)** | Expectation-maximization over latent task-difficulty buckets (60 buckets, 60 iterations); accounts for varying task complexity |
| **Elo** | Sequential rating updates with K=32; some policies initialised with lower starting ratings to reflect prior expectations |
| **Points-based** | Win = 2 pts, Tie = 1 pt, Loss = 0 pts; ranked by average score per comparison |

---

## Database Schema

### Table: `annotations`

| Column                  | Type      | Description                                              |
|-------------------------|-----------|----------------------------------------------------------|
| `participant_id`        | VARCHAR   | Worker / user identifier                                 |
| `participant_type`      | VARCHAR   | `'paid'`, `'free'`, or `'unknown'`                       |
| `completion_code`       | VARCHAR   | Code displayed to the participant on completion          |
| `total_time_ms`         | INTEGER   | Total time spent in the study (milliseconds)             |
| `response_length`       | INTEGER   | Number of responses submitted                            |
| `quiz_score`            | INTEGER   | Correct answers on the initial quiz (paid workers only)  |
| `quiz_total`            | INTEGER   | Total initial quiz questions shown                       |
| `sanity_checks_passed`  | INTEGER   | Number of sanity checks answered correctly               |
| `sanity_checks_total`   | INTEGER   | Total sanity checks encountered                          |
| `sanity_check_results`  | JSONB     | Per-check detail: index, position, answers, correctness  |
| `failed`                | BOOLEAN   | Whether the participant was disqualified                 |
| `failure_reason`        | VARCHAR   | `'quiz_failed'` or `'sanity_failed'`                     |
| `response_data`         | JSONB     | Full array of responses (quiz, sanity_check, regular)    |
| `timestamp`             | TIMESTAMP | Submission time                                          |
| `config_version`        | VARCHAR   | Frontend version that generated the submission           |

### Views

| View                       | Description                                     |
|----------------------------|-------------------------------------------------|
| `participant_summary`      | Completion and failure counts by participant type |
| `quiz_performance`         | Quiz scores for paid workers                    |
| `sanity_check_performance` | Sanity check pass rates per participant         |

---

## Filtering Bad Video Pairs

Known-problematic pairs are excluded from all ranking computations. Add substrings to the set in `main.py`:

```python
BAD_VIDEO_SUBSTRINGS = {
    "test_put blue cube to left corner of the table_test",
    # add more as needed
}
```

Any pair whose `videoA` or `videoB` path contains a listed substring is silently skipped.

---

## Deployment with Cloudflare Tunnel

To expose the local server without port-forwarding or a dedicated IP:

```bash
# 1. Install cloudflared
# https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/

# 2. Authenticate
cloudflared tunnel login

# 3. Create a tunnel and route it to the local server
cloudflared tunnel create robot-arena
cloudflared tunnel route dns robot-arena api.yourdomain.com

# 4. Run the tunnel
cloudflared tunnel run robot-arena
```

To keep the tunnel running as a system service:

```bash
sudo cloudflared service install
sudo systemctl enable --now cloudflared
```

If you don't have a custom domain, Cloudflare can assign a temporary `*.trycloudflare.com` URL — no account required.

---

## Troubleshooting

**Database connection refused**
```bash
sudo systemctl status postgresql   # Check the service is running
psql -U your_username -d robotarenainf -c "SELECT 1;"  # Verify credentials
```

**Dashboard shows no data**
- Confirm `pairs.json` (or the appropriate versioned file) exists in the backend directory.
- Check that the frontend is posting to the correct backend URL.
- Review server logs for JSON decode or index errors.

**CORS errors in the browser**
The allowed origins are set in `main.py`. Add your frontend's origin:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],
    allow_methods=["OPTIONS", "GET", "POST"],
    allow_headers=["*"],
)
```

**Ranking algorithms raise errors**
- Ensure there are enough responses (at least a few per policy pair) before rankings converge.
- Verify that `pairs.json` indices in the frontend responses match the file loaded by the backend.
- Check for `NaN` values or missing fields in stored `response_data`.
