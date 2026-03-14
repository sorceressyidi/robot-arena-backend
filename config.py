# ============================================================================
# CONFIGURATION — edit this file to customise the backend
# ============================================================================

# --- Database -----------------------------------------------------------------
DB_HOST     = "localhost"
DB_PORT     = 5432
DB_NAME     = ""          # e.g. "robotarenainf"
DB_USER     = ""          # e.g. "postgres"
DB_PASSWORD = ""          # your database password

# --- CORS ---------------------------------------------------------------------
# List the origin(s) your frontend is served from.
ALLOWED_ORIGINS = ["https://zhangyidi.tech"]

# --- Pairs file versioning ----------------------------------------------------
# Maps config_version (sent by the frontend) to a pairs JSON file in this
# directory. Add entries as you deploy new versions.
PAIRS_FILES: dict = {
    None:   "pairs.json",    # legacy — no version field in payload
    "v5.0": "pairs.json",    # current
}

# Fallback for unrecognised version strings.
DEFAULT_PAIRS_FILE = "pairs.json"

# --- Dashboard ----------------------------------------------------------------
CACHE_TTL_SECONDS = 60   # seconds between dashboard recalculations

# --- Elo ratings --------------------------------------------------------------
ELO_K_FACTOR = 32

# Policies listed here start with a lower-than-default rating to reflect
# prior expectations. Omitted policies start at ELO_DEFAULT_RATING.
ELO_STARTING_RATINGS: dict = {
    "cogact":  800,
    "spatial": 800,
    "octo":    800,
    "robovlm": 800,
}
ELO_DEFAULT_RATING = 1200
