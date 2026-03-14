# ============================================================================
# CONFIGURATION TEMPLATE — copy this file to config.py and fill in your values
#   cp config.example.py config.py
# ============================================================================

# --- Database -----------------------------------------------------------------
DB_HOST     = "localhost"
DB_PORT     = 5432
DB_NAME     = "robotarenainf"   # your database name
DB_USER     = "your_username"
DB_PASSWORD = "your_password"

# --- CORS ---------------------------------------------------------------------
ALLOWED_ORIGINS = ["https://yourdomain.com"]

# --- Pairs file versioning ----------------------------------------------------
# Maps the config_version string (sent by the frontend) to a pairs JSON file
# that must exist in this directory. Add a new entry each time you deploy a
# new pairs file.
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
