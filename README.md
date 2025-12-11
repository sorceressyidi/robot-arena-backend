# Robot Arena Backend

FastAPI backend for the Robot Policy Comparison Study. Handles data collection, storage, and provides a real-time ranking dashboard.

## Features

- **Data collection**: Stores participant responses with quiz and sanity check tracking
- **Multiple ranking algorithms**: Bradley-Terry MLE, EM, Elo, Points-based
- **Real-time dashboard**: View policy rankings as data comes in
- **Legacy support**: Compatible with older frontend versions

## Quick Start

### 1. Database Setup

```bash
# Install PostgreSQL if needed
sudo apt install postgresql postgresql-contrib

# Create database
sudo -u postgres psql -c "CREATE DATABASE robotarenainf;"

# Run schema
sudo -u postgres psql -d robotarenainf -f schema.sql
```

### 2. Update Credentials

Edit `main.py`:

```python
def get_db_connection():
    return psycopg2.connect(
        host="localhost",
        port=5432,
        database="robotarenainf",
        user="your_username",      # Change this
        password="your_password"   # Change this
    )
```

### 3. Install Dependencies

```bash
pip install fastapi uvicorn psycopg2-binary pydantic numpy scipy pandas statsmodels
```

### 4. Add Video Pairs

Copy `pairs.json` (and optionally `pairs_scaled.json`) to the backend directory:

```bash
cp ../videos/pairs.json .
```

### 5. Run Server

```bash
# Development
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Production
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

## Deployment with Cloudflare Tunnel

To expose your local backend to the internet without port forwarding:

### 1. Create Cloudflare Account

Sign up at [dash.cloudflare.com](https://dash.cloudflare.com)

### 2. Set Up Tunnel

1. Go to **Networks → Tunnels** in Cloudflare dashboard
2. Create a new tunnel
3. Configure:
   - Local address: `http://localhost:8000`
   - Subdomain: your choice (e.g., `robotarena.yourdomain.com`)
4. Follow the provided instructions to install `cloudflared`

### 3. Run the Tunnel

```bash
# Run the command provided by Cloudflare, e.g.:
cloudflared tunnel run your-tunnel-name
```

### 4. Keep Running 24/7 (Optional)

Set up as a system service for automatic restart:

```bash
# Install as service (Linux)
sudo cloudflared service install
sudo systemctl enable cloudflared
sudo systemctl start cloudflared
```

See [Cloudflare's guide](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/do-more-with-tunnels/local-management/as-a-service/) for detailed service setup.

### Without Custom Domain

Cloudflare can assign a temporary URL like `https://random-name.trycloudflare.com` if you don't have your own domain.

**Resources:**
- [Cloudflare Tunnel Documentation](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/)
- [Running as a Service](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/do-more-with-tunnels/local-management/as-a-service/)

## API Endpoints

### POST `/annotate`

Save annotation data from the study frontend.

**Request:**
```json
{
  "participant_id": "worker_123",
  "participant_type": "paid",
  "completion_code": "ABC12345",
  "total_time_ms": 450000,
  "response_length": 35,
  "quiz_score": 4,
  "quiz_total": 5,
  "sanity_check_results": [
    {
      "index": 360,
      "correct": true,
      "position": 12,
      "userAnswer": "right",
      "correctAnswer": "right"
    }
  ],
  "sanity_checks_passed": 5,
  "sanity_checks_total": 5,
  "failed": false,
  "failure_reason": null,
  "responses": [...],
  "timestamp": "2024-12-08T10:30:00Z",
  "config_version": "v3.0"
}
```

**Response:**
```json
{"status": "success", "message": "Annotation saved successfully"}
```

### GET `/`

Dashboard showing policy rankings with four algorithms:
- Bradley-Terry MLE with confidence intervals
- EM-based ranking with latent task difficulty buckets
- Elo rating (K=32)
- Points-based (Win=2, Tie=1, Loss=0)

### POST `/annotate/legacy`

Legacy endpoint for older frontend versions.

## Database Schema

### Main Table: `annotations`

| Column | Type | Description |
|--------|------|-------------|
| `participant_id` | VARCHAR | Worker/user identifier |
| `participant_type` | VARCHAR | 'paid', 'free', or 'unknown' |
| `completion_code` | VARCHAR | Code shown on completion |
| `total_time_ms` | INTEGER | Total study time |
| `response_length` | INTEGER | Number of responses |
| `quiz_score` | INTEGER | Quiz correct answers (paid only) |
| `quiz_total` | INTEGER | Total quiz questions |
| `sanity_checks_passed` | INTEGER | Sanity checks passed |
| `sanity_checks_total` | INTEGER | Total sanity checks |
| `sanity_check_results` | JSONB | Detailed sanity check data |
| `failed` | BOOLEAN | Whether participant failed |
| `failure_reason` | VARCHAR | 'quiz_failed' or 'sanity_failed' |
| `response_data` | JSONB | All responses |
| `timestamp` | TIMESTAMP | Submission time |
| `config_version` | VARCHAR | Frontend version |

### Useful Views

- `participant_summary`: Aggregated stats by participant type
- `quiz_performance`: Quiz scores for paid workers
- `sanity_check_performance`: Sanity check pass rates

## Ranking Algorithms

Only `"regular"` type responses are used for ranking (quiz and sanity check responses are excluded).

### Bradley-Terry MLE
Maximum likelihood estimation with sandwich standard errors and confidence intervals.

### EM Algorithm
Expectation-maximization with latent task difficulty buckets. Accounts for varying task complexity.

### Elo Rating
Sequential updates with K-factor of 32. Some policies start with lower initial ratings.

### Points-Based
Simple scoring: Win=2 points, Tie=1 point, Loss=0 points.

## Filtering Bad Videos

The backend filters out known problematic video pairs. Edit `BAD_VIDEO_SUBSTRINGS` in `main.py` to add/remove filters:

```python
BAD_VIDEO_SUBSTRINGS = {
    "test_Move the can to the right of the pot_test",
    # ... add more as needed
}
```

## Troubleshooting

### Database connection failed
```bash
# Check PostgreSQL is running
sudo systemctl status postgresql

# Verify credentials
psql -U your_username -d robotarenainf -c "SELECT 1;"
```

### No data in dashboard
- Verify `pairs.json` exists in backend directory
- Check frontend is posting to correct URL
- Look for errors in server logs

### CORS errors
The backend allows all origins by default. If needed, restrict in `main.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],
    # ...
)
```

### Ranking errors
- Ensure enough data points (at least a few responses per policy pair)
- Check for NaN values in response data
- Verify `pairs.json` indices match frontend