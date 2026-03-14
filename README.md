# Backend

FastAPI server for the Robot Policy Comparison Study — collects participant responses, persists them to PostgreSQL, and serves a live ranking dashboard.

---

## ⚙️ Setup

### 1. Database

```bash
sudo apt install postgresql postgresql-contrib
sudo -u postgres psql -c "CREATE DATABASE robotarenainf;"
sudo -u postgres psql -d robotarenainf -f schema.sql
```

### 2. Configure

Copy the example config and fill in your values:

```bash
cp config.example.py config.py
```

Edit **`config.py`** — this is the only file you need to touch before running:

```python
# Database
DB_NAME     = "robotarenainf"
DB_USER     = "your_username"
DB_PASSWORD = "your_password"

# CORS — add your frontend origin
ALLOWED_ORIGINS = ["https://yourdomain.com"]

# Pairs versioning — maps config_version → pairs file in this directory
PAIRS_FILES = {
    None:   "pairs.json",   # legacy (no version field)
    "v5.0": "pairs.json",   # current
}
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Copy Pairs File

```bash
cp ../videos/pairs.json .
```

### 5. Run

```bash
# Development
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Production
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

---

## 🔌 API Reference

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/annotate` | Save a completed study session |
| `GET` | `/` | Live HTML ranking dashboard *(60 s cache)* |
| `POST` | `/annotate/legacy` | Backwards-compatible endpoint for older frontend versions |
| `GET` | `/versions` | List configured pairs versions and their load status |
| `POST` | `/versions/reload` | Clear pairs + dashboard cache and reload files from disk |
| `POST` | `/dashboard/refresh` | Invalidate dashboard cache *(regenerates on next `GET /`)* |
| `GET` | `/stats/by-version` | Annotation counts grouped by `config_version` |

---

## 🏆 Ranking Algorithms

Only `"regular"` responses are used for ranking — quiz and sanity-check responses are excluded.

| Algorithm | Description |
|-----------|-------------|
| **Bradley-Terry MLE** | MLE of pairwise win probabilities with HC0 sandwich standard errors and 95% CI |
| **EM (Latent Buckets)** | EM over 60 latent task-difficulty buckets; accounts for varying task complexity |
| **Elo** | Sequential updates with K=32; configurable per-policy starting ratings |
| **Points-based** | Win = 2 pts, Tie = 1 pt, Loss = 0 pts; ranked by average score |

---

## 🗄️ Database Schema

### Table: `annotations`

| Column | Type | Description |
|--------|------|-------------|
| `participant_id` | VARCHAR | Worker / user identifier |
| `participant_type` | VARCHAR | `'paid'`, `'free'`, or `'unknown'` |
| `completion_code` | VARCHAR | Code shown to participant on completion |
| `total_time_ms` | INTEGER | Total time in the study (ms) |
| `response_length` | INTEGER | Number of responses submitted |
| `quiz_score` | INTEGER | Correct answers on the initial quiz |
| `quiz_total` | INTEGER | Total initial quiz questions shown |
| `sanity_checks_passed` | INTEGER | Sanity checks answered correctly |
| `sanity_checks_total` | INTEGER | Total sanity checks encountered |
| `sanity_check_results` | JSONB | Per-check detail: index, position, answers, correctness |
| `failed` | BOOLEAN | Whether the participant was disqualified |
| `failure_reason` | VARCHAR | `'quiz_failed'` or `'sanity_failed'` |
| `response_data` | JSONB | Full response array (regular, quiz, sanity_check) |
| `timestamp` | TIMESTAMP | Submission time |
| `config_version` | VARCHAR | Frontend version that generated the submission |

### Views

| View | Description |
|------|-------------|
| `participant_summary` | Completion and failure counts by participant type |
| `quiz_performance` | Quiz scores for paid workers |
| `sanity_check_performance` | Sanity check pass rates per participant |

---

## 🚀 Deployment with Cloudflare Tunnel

```bash
cloudflared tunnel login
cloudflared tunnel create robot-arena
cloudflared tunnel route dns robot-arena api.yourdomain.com
cloudflared tunnel run robot-arena
```

To run as a system service:

```bash
sudo cloudflared service install
sudo systemctl enable --now cloudflared
```

---

## 🔧 Troubleshooting

**Database connection refused**
```bash
sudo systemctl status postgresql
psql -U your_username -d robotarenainf -c "SELECT 1;"
```

**Dashboard shows no data**
- Confirm `pairs.json` exists in the backend directory.
- Check the frontend is posting to the correct backend URL.
- Review server logs for JSON decode or index errors.

**CORS errors** — add your frontend origin to `ALLOWED_ORIGINS` in `config.py`.

**Ranking errors** — ensure there are enough responses per policy pair for the algorithms to converge, and that pairs file indices match between frontend and backend.
