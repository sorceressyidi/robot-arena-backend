import os
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import psycopg2
from psycopg2.extras import Json
from datetime import datetime
import json
from collections import defaultdict
import numpy as np
from typing import Optional, List
from scipy.special import expit
import pandas as pd

# ============================================================================
# BAD VIDEO FILTERING
# ============================================================================
BAD_VIDEO_SUBSTRINGS = {
    "test_Move the can to the right of the pot_test",
    "test_Move the cheese into the silver pot_test",
    "move the cloth to the top right of the bowl_test",
    "test_Move the green cloth to the bottom left of the table below the yellow pepper_test",
    "test_move the silver pot to the lower right of the cooker_test",
    "test_Place the metal pot on the front left corner of the purple towel_test0_2",
    "test_put blue cube to left corner of the table_test",
    "test_put brown toy at the bottom center of the table_test",
    "test_put pot or pan on stove and put cheese in pot or pan_test",
    "test_put spatula on cutting board_test",
    "test_Put the Blue spoon on top of the green cloth_test",
    "test_put the yellow knife inside the orange pot_test",
    "test_take the blue object and place between two silvered pots_test",
}

def is_bad_video(pair_info):
    path_a = pair_info['videoA']['path']
    path_b = pair_info['videoB']['path']
    for substring in BAD_VIDEO_SUBSTRINGS:
        if substring in path_a or substring in path_b:
            return True
    return False

# ============================================================================
# RANKING ALGORITHMS (unchanged from original)
# ============================================================================
def sigmoid(z):
    return expit(z)

def clip_update(g, h, clip_val=1.0):
    """Clipped-Newton update step."""
    h_safe = h - 1e-6
    update = np.divide(g, h_safe, out=np.zeros_like(g), where=h_safe!=0)
    return np.clip(update, -clip_val, clip_val)

def run_em_ranking(all_responses, all_scaled_versions,
                   policy_map, all_pairs, all_pairs_scaled):
    """
    Runs the EM algorithm to fit model parameters.
    """
    EM_ITERS = 60
    N_BUCKETS = 60 
    STEP_CLIP = 1.0
    L2_THETA, L2_PSI, L2_TAU = 1e-2, 1e-2, 0.0 
    TOL = 1e-4

    N_POLICIES = len(policy_map)
    trials = []
    
    for response, scaled_version in zip(all_responses, all_scaled_versions):
        try:
            choice, pair_index_str = response['answer'].split('_')
            if scaled_version is not None:
                pair_info = all_pairs_scaled[int(pair_index_str)]
            else:
                pair_info = all_pairs[int(pair_index_str)]
            if is_bad_video(pair_info):
                continue
            p_left, p_right = pair_info['videoA']['source'], pair_info['videoB']['source']
            i_n, j_n = policy_map[p_left], policy_map[p_right]
            y_n = 2 if choice == 'left' else 0 if choice == 'right' else 1
            trials.append({'i_n': i_n, 'j_n': j_n, 'y_n': y_n})
        except (ValueError, IndexError, KeyError):
            continue
    
    M_TRIALS = len(trials)
    
    if M_TRIALS == 0: return []

    trials_i_n = np.array([t['i_n'] for t in trials])
    trials_j_n = np.array([t['j_n'] for t in trials])
    trials_y_n = np.array([t['y_n'] for t in trials])

    theta = np.random.normal(0, 0.1, N_POLICIES)
    psi = np.zeros((N_POLICIES, N_BUCKETS))
    tau = np.random.normal(0, 0.1, N_BUCKETS)
    nu = np.full(N_BUCKETS, 1.0 / N_BUCKETS)
    nu_tie = 0.5

    for _ in range(EM_ITERS):
        theta_old = theta.copy()
        
        q_i = np.zeros((M_TRIALS, N_BUCKETS))
        q_j = np.zeros((M_TRIALS, N_BUCKETS))
        likelihoods = np.zeros((M_TRIALS, N_BUCKETS))

        for t in range(N_BUCKETS):
            z_i = theta[trials_i_n] + psi[trials_i_n, t] - tau[t]
            z_j = theta[trials_j_n] + psi[trials_j_n, t] - tau[t]
            q_i[:, t], q_j[:, t] = sigmoid(z_i), sigmoid(z_j)
            
            p_y_win = q_i[:, t] * (1 - q_j[:, t])
            p_y_tie = 2 * nu_tie * np.sqrt(q_i[:, t] * (1 - q_i[:, t]) * q_j[:, t] * (1 - q_j[:, t]))
            p_y_loss = (1 - q_i[:, t]) * q_j[:, t]
            
            p_y_given_t = np.zeros(M_TRIALS)
            p_y_given_t[trials_y_n == 2] = p_y_win[trials_y_n == 2]
            p_y_given_t[trials_y_n == 1] = p_y_tie[trials_y_n == 1]
            p_y_given_t[trials_y_n == 0] = p_y_loss[trials_y_n == 0]
            likelihoods[:, t] = p_y_given_t 

        log_nu = np.log(nu + 1e-9)
        log_likelihoods = np.log(likelihoods + 1e-9)
        log_gamma_unnorm = log_nu[np.newaxis, :] + log_likelihoods
        log_denominator = np.logaddexp.reduce(log_gamma_unnorm, axis=1, keepdims=True)
        gamma = np.exp(log_gamma_unnorm - log_denominator)

        d_zi = np.zeros((M_TRIALS, N_BUCKETS))
        d_zj = np.zeros((M_TRIALS, N_BUCKETS))
        d_zi[trials_y_n == 2, :] = 1 - q_i[trials_y_n == 2, :]
        d_zj[trials_y_n == 2, :] = -q_j[trials_y_n == 2, :]
        d_zi[trials_y_n == 1, :] = 0.5 - q_i[trials_y_n == 1, :]
        d_zj[trials_y_n == 1, :] = 0.5 - q_j[trials_y_n == 1, :]
        d_zi[trials_y_n == 0, :] = -q_i[trials_y_n == 0, :]
        d_zj[trials_y_n == 0, :] = 1 - q_j[trials_y_n == 0, :]

        g_theta = np.zeros(N_POLICIES)
        h_theta = np.zeros(N_POLICIES)
        np.add.at(g_theta, trials_i_n, np.sum(gamma * d_zi, axis=1))
        np.add.at(g_theta, trials_j_n, np.sum(gamma * d_zj, axis=1))
        h_theta_i_contrib = np.sum(gamma * q_i * (1 - q_i), axis=1)
        h_theta_j_contrib = np.sum(gamma * q_j * (1 - q_j), axis=1)
        np.add.at(h_theta, trials_i_n, h_theta_i_contrib)
        np.add.at(h_theta, trials_j_n, h_theta_j_contrib)
        theta -= clip_update(g_theta - L2_THETA * theta, -h_theta - L2_THETA, STEP_CLIP)
        
        for t in range(N_BUCKETS):
            g_psi_t = np.zeros(N_POLICIES)
            h_psi_t = np.zeros(N_POLICIES)
            np.add.at(g_psi_t, trials_i_n, gamma[:, t] * d_zi[:, t])
            np.add.at(g_psi_t, trials_j_n, gamma[:, t] * d_zj[:, t])
            h_psi_t_i_contrib = gamma[:, t] * q_i[:, t] * (1 - q_i[:, t])
            h_psi_t_j_contrib = gamma[:, t] * q_j[:, t] * (1 - q_j[:, t])
            np.add.at(h_psi_t, trials_i_n, h_psi_t_i_contrib)
            np.add.at(h_psi_t, trials_j_n, h_psi_t_j_contrib)
            psi[:, t] -= clip_update(g_psi_t - L2_PSI * psi[:, t], -h_psi_t - L2_PSI, STEP_CLIP)
                
        g_tau = -np.sum(gamma * (d_zi + d_zj), axis=0)
        h_tau_pos = np.sum(gamma * (q_i * (1 - q_i) + q_j * (1 - q_j)), axis=0)
        tau -= clip_update(g_tau - L2_TAU * tau, -h_tau_pos - L2_TAU, STEP_CLIP)
        
        nu = np.mean(gamma, axis=0)
        
        tie_numerator = np.sum(gamma[trials_y_n == 1, :])
        win_denominator = np.sum(gamma[trials_y_n == 2, :])
        nu_tie = 0.5 * tie_numerator / (win_denominator + 1e-9)
        
        if np.max(np.abs(theta - theta_old)) < TOL: 
            break
            
    inverse_policy_map = {v: k for k, v in policy_map.items()}
    return sorted([(inverse_policy_map[i], score) for i, score in enumerate(theta)], key=lambda item: item[1], reverse=True)


def bradley_terry_offline(wins, max_iter=1000, tolerance=1e-6):
    num_players = wins.shape[0]
    
    if wins.shape[0] != wins.shape[1]:
        raise ValueError("Win matrix must be a square matrix.")
    if not np.allclose(np.diag(wins), 0):
        raise ValueError("The diagonal of the win matrix must be all zeros.")

    total_wins = np.sum(wins, axis=1)
    num_games = wins + wins.T
    abilities = np.ones(num_players)

    for i in range(max_iter):
        old_abilities = abilities.copy()
        abilities_row = abilities[:, np.newaxis]
        abilities_col = abilities[np.newaxis, :]
        pairwise_sums = abilities_row + abilities_col
        pairwise_sums[pairwise_sums == 0] = 1
        
        update_fractions = num_games / pairwise_sums
        
        denominator = np.sum(update_fractions, axis=1) - np.diag(update_fractions)
        denominator[denominator == 0] = 1e-9

        abilities = total_wins / denominator
        abilities /= np.sum(abilities)

        change = np.max(np.abs(abilities - old_abilities))
        if change < tolerance:
            break
    return abilities

import statsmodels.api as sm
from scipy.stats import norm

def get_ranking_with_confidence(wins, model_names=None, alpha=0.05):
    num_models = wins.shape[0]
    if model_names is None:
        model_names = [f"Model_{i+1}" for i in range(num_models)]

    battles = []
    for i in range(num_models):
        for j in range(num_models):
            if wins[i, j] > 0:
                battles.extend([(model_names[i], model_names[j])] * int(wins[i, j]))

    model_to_idx = {name: i for i, name in enumerate(model_names)}
    
    X, y = [], []
    for winner, loser in battles:
        row1 = np.zeros(num_models)
        row1[model_to_idx[winner]] = 1
        row1[model_to_idx[loser]] = -1
        X.append(row1)
        y.append(1)
        
        row2 = np.zeros(num_models)
        row2[model_to_idx[loser]] = 1
        row2[model_to_idx[winner]] = -1
        X.append(row2)
        y.append(0)

    X = np.array(X)[:, :-1]

    logit_model = sm.Logit(np.array(y), X)
    result = logit_model.fit(disp=0, cov_type='HC0')

    params = np.append(result.params, 0)
    cov_matrix_reduced = result.cov_params()
    cov_matrix = np.zeros((num_models, num_models))
    cov_matrix[:-1, :-1] = cov_matrix_reduced
    
    centered_scores = params - np.mean(params)
    
    A = np.eye(num_models) - (1 / num_models)
    centered_cov = A @ cov_matrix @ A.T
    std_errors = np.sqrt(np.diag(centered_cov))

    z_critical = norm.ppf(1 - alpha / 2)
    margin_of_error = z_critical * std_errors
    ci_lower = centered_scores - margin_of_error
    ci_upper = centered_scores + margin_of_error
    
    approx_rankings = []
    for i in range(num_models):
        num_better = np.sum(ci_lower > ci_upper[i])
        approx_rankings.append(1 + num_better)

    results_df = pd.DataFrame({
        'model': model_names,
        'bt_score (log)': centered_scores,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'approximate_ranking': approx_rankings
    })
    
    return results_df.sort_values(by='bt_score (log)', ascending=False)


def bt_sandwich_std(wins, abilities):
    num_players = wins.shape[0]
    
    comparisons = []
    weights = []
    outcomes = []
    probabilities = []
    
    for i in range(num_players):
        for j in range(num_players):
            if i != j:
                n_ij = wins[i, j] + wins[j, i]
                if n_ij > 0:
                    p_ij = abilities[i] / (abilities[i] + abilities[j])
                    y_ij = wins[i, j] / n_ij
                    
                    x_ij = np.zeros(num_players)
                    x_ij[i] = 1.0
                    x_ij[j] = -1.0
                    
                    comparisons.append(x_ij)
                    weights.append(n_ij)
                    outcomes.append(y_ij)
                    probabilities.append(p_ij)
    
    if len(comparisons) == 0:
        return np.full(num_players, np.nan)
    
    X = np.array(comparisons)
    w = np.array(weights)
    y = np.array(outcomes)
    p = np.array(probabilities)
    
    H = np.zeros((num_players, num_players))
    for t in range(len(comparisons)):
        x_t = X[t]
        w_t = w[t]
        p_t = p[t]
        H += w_t * p_t * (1 - p_t) * np.outer(x_t, x_t)
    
    B = np.zeros((num_players, num_players))
    for t in range(len(comparisons)):
        x_t = X[t]
        w_t = w[t]
        y_t = y[t]
        p_t = p[t]
        residual = y_t - p_t
        B += (w_t ** 2) * (residual ** 2) * np.outer(x_t, x_t)
    
    H_red = H[:-1, :-1]
    B_red = B[:-1, :-1]
    
    try:
        if np.linalg.det(H_red) != 0:
            H_inv = np.linalg.inv(H_red)
            V_sandwich = H_inv @ B_red @ H_inv
            std_sandwich = np.sqrt(np.diag(V_sandwich))
            std_sandwich = np.append(std_sandwich, 0.0)
            return std_sandwich
        else:
            print("Warning: Information matrix is singular")
            return np.full(num_players, np.nan)
    except np.linalg.LinAlgError:
        print("Warning: Failed to compute sandwich variance")
        return np.full(num_players, np.nan)


def run_bt_mle_ranking(all_responses, all_scaled_versions, policy_map, all_pairs, all_pairs_scaled):
    N_POLICIES = len(policy_map)
    wins = np.zeros((N_POLICIES, N_POLICIES))

    for response, scaled_version in zip(all_responses, all_scaled_versions):
        try:
            choice, pair_index_str = response['answer'].split('_')
            pair_index = int(pair_index_str)

            pair_info = all_pairs_scaled[pair_index] if scaled_version else all_pairs[pair_index]
            if is_bad_video(pair_info):
                continue

            p_left = pair_info['videoA']['source']
            p_right = pair_info['videoB']['source']
            i_n = policy_map[p_left]
            j_n = policy_map[p_right]
            
            if choice == 'left':
                wins[i_n, j_n] += 1.0
            elif choice == 'right':
                wins[j_n, i_n] += 1.0
            elif choice == 'same':
                wins[i_n, j_n] += 0.5
                wins[j_n, i_n] += 0.5
        except (ValueError, IndexError, KeyError):
            continue

    if np.sum(wins) == 0:
        print("No valid trials found. Returning empty ranking.")
        return []

    abilities = bradley_terry_offline(wins)
    theta = abilities
    std = bt_sandwich_std(wins, abilities)
    
    inverse_policy_map = {v: k for k, v in policy_map.items()}
    ranking_with_std = sorted(
        [(inverse_policy_map[i], theta[i], std[i]) for i in range(N_POLICIES)],
        key=lambda item: item[1],
        reverse=True
    )
    full_ranking_df = get_ranking_with_confidence(wins, model_names=list(policy_map.keys()))
    full_ranking_df.to_csv('bt_ranking_with_ci.csv', index=False)
    print(full_ranking_df)
    print("Ranking with std:", ranking_with_std)
    return ranking_with_std


# ============================================================================
# FASTAPI APPLICATION
# ============================================================================
app = FastAPI(title="Robot Arena API ", description="Backend for robot policy comparison study with quiz and sanity checks")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["OPTIONS", "GET", "POST"],
    allow_headers=["*"],
)

# --- Configuration ---
PAIRS_FILE_PATH = 'pairs.json'
PAIRS_FILE_PATH_SCALED = 'pairs_scaled.json'

def get_db_connection():
    return psycopg2.connect(
        host="localhost",
        port=5432,
        database="robotarenainf",
        user="postgres",
        password="sorceressyidi"  # Change this!
    )


# ============================================================================
# PYDANTIC MODELS - Updated for new response format
# ============================================================================
class SanityCheckResult(BaseModel):
    index: int
    correct: bool
    position: int
    userAnswer: Optional[str] = None
    correctAnswer: Optional[str] = None

class Annotation(BaseModel):
    """New annotation model for  with quiz and sanity check support"""
    participant_id: str
    participant_type: str  # 'paid' or 'free'
    completion_code: str
    total_time_ms: int
    response_length: int
    
    # Quiz tracking (for paid workers)
    quiz_score: Optional[int] = None
    quiz_total: Optional[int] = None
    
    # Sanity check tracking
    sanity_check_results: Optional[List[SanityCheckResult]] = []
    sanity_checks_passed: Optional[int] = None
    sanity_checks_total: Optional[int] = None
    
    # Failure tracking
    failed: bool = False
    failure_reason: Optional[str] = None  # 'quiz_failed' or 'sanity_failed'
    
    # Response data
    responses: list
    timestamp: str
    config_version: Optional[str] = None


# ============================================================================
# API ENDPOINTS
# ============================================================================
@app.post("/annotate")
def save_annotation(data: Annotation):
    """Save annotation data with full quiz and sanity check tracking"""
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            query = """
            INSERT INTO annotations (
                participant_id, participant_type, completion_code, total_time_ms,
                response_length, quiz_score, quiz_total, 
                sanity_checks_passed, sanity_checks_total, sanity_check_results,
                failed, failure_reason, response_data, timestamp, config_version
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            # Convert sanity check results to JSON-serializable format
            sanity_results_json = [r.dict() for r in data.sanity_check_results] if data.sanity_check_results else []
            
            cursor.execute(query, (
                data.participant_id,
                data.participant_type,
                data.completion_code,
                data.total_time_ms,
                data.response_length,
                data.quiz_score,
                data.quiz_total,
                data.sanity_checks_passed,
                data.sanity_checks_total,
                Json(sanity_results_json),
                data.failed,
                data.failure_reason,
                Json(data.responses),
                datetime.fromisoformat(data.timestamp.replace("Z", "+00:00")),
                data.config_version
            ))
            conn.commit()
        return {"status": "success", "message": "Annotation saved successfully"}
    except Exception as e:
        conn.rollback()
        print(f"Error saving annotation: {e}")
        return {"status": "error", "message": str(e)}
    finally:
        conn.close()


@app.get("/", response_class=HTMLResponse)
def read_root_and_show_stats():
    """
    Dashboard showing policy rankings (without participant statistics).
    """
    # Load video pairs
    try:
        with open(PAIRS_FILE_PATH, 'r') as f:
            all_pairs = json.load(f)
    except FileNotFoundError:
        return "<h1>Error: Could not find 'pairs.json'.</h1>"
    
    try:
        with open(PAIRS_FILE_PATH_SCALED, 'r') as f:
            all_pairs_scaled = json.load(f)
    except FileNotFoundError:
        all_pairs_scaled = all_pairs  # Fallback to regular pairs
    
    # Fetch data from database
    conn = get_db_connection()
    all_records = []
    
    try:
        with conn.cursor() as cursor:
            # Get all annotations
            cursor.execute("""
                SELECT response_data, timestamp, config_version, participant_type, failed, failure_reason
                FROM annotations 
                ORDER BY timestamp ASC
            """)
            all_records = cursor.fetchall()
    except Exception as e:
        return f"<h1>Database Error</h1><p>{e}</p>"
    finally:
        conn.close()
        
    if not all_records:
        return """
        <html><head><title>Robot Arena </title>
        <style>body{font-family:sans-serif;max-width:800px;margin:2rem auto;padding:1rem}</style>
        </head><body>
        <h1>🤖 Robot Arena </h1>
        <p>No annotation data found yet. Waiting for participants...</p>
        </body></html>
        """
    
    # Process responses - only include regular responses (not quiz or sanity checks)
    all_responses = []
    all_scaled_versions = []
    
    for record in all_records:
        if record and record[0]:
            responses = record[0]
            config_version = record[2]
            
            # Filter to only include 'regular' type responses for ranking
            for resp in responses:
                resp_type = resp.get('type', 'regular')
                if resp_type == 'regular':
                    all_responses.append(resp)
                    all_scaled_versions.append(config_version)
    
    # Calculate policy stats
    policy_stats = defaultdict(lambda: {'wins': 0, 'losses': 0, 'ties': 0, 'points': 0, 'comparisons': 0})
    
    K_FACTOR = 32
    STARTING_ELO_RATINGS = {'cogact': 800, 'spatial': 800, 'octo': 800, 'robovlm': 800}
    DEFAULT_STARTING_ELO = 1200
    elo_ratings = defaultdict(lambda: DEFAULT_STARTING_ELO)
    elo_ratings.update(STARTING_ELO_RATINGS)

    for response, scaled_version in zip(all_responses, all_scaled_versions):
        try:
            choice, pair_index_str = response['answer'].split('_')
            pair_index = int(pair_index_str)
            if scaled_version is not None:
                pair_info = all_pairs_scaled[pair_index]
            else:
                pair_info = all_pairs[pair_index]
            if is_bad_video(pair_info):
                continue
            policy_left = pair_info['videoA']['source']
            policy_right = pair_info['videoB']['source']
        except (ValueError, IndexError, KeyError) as e:
            continue

        policy_stats[policy_left]['comparisons'] += 1
        policy_stats[policy_right]['comparisons'] += 1
        
        if choice == 'left':
            policy_stats[policy_left]['wins'] += 1
            policy_stats[policy_left]['points'] += 2
            policy_stats[policy_right]['losses'] += 1
            actual_left, actual_right = 1.0, 0.0
        elif choice == 'right':
            policy_stats[policy_right]['wins'] += 1
            policy_stats[policy_right]['points'] += 2
            policy_stats[policy_left]['losses'] += 1
            actual_left, actual_right = 0.0, 1.0
        else:
            policy_stats[policy_left]['ties'] += 1
            policy_stats[policy_left]['points'] += 1
            policy_stats[policy_right]['ties'] += 1
            policy_stats[policy_right]['points'] += 1
            actual_left, actual_right = 0.5, 0.5
                 
        rating_left = elo_ratings[policy_left]
        rating_right = elo_ratings[policy_right]
        expected_left = 1 / (1 + 10**((rating_right - rating_left) / 400))
        expected_right = 1 - expected_left
        elo_ratings[policy_left] = rating_left + K_FACTOR * (actual_left - expected_left)
        elo_ratings[policy_right] = rating_right + K_FACTOR * (actual_right - expected_right)

    sorted_policies_by_score = sorted(policy_stats.items(), key=lambda i: (i[1]['points'] / i[1]['comparisons']) if i[1]['comparisons'] > 0 else 0, reverse=True)
    sorted_policies_by_elo = sorted(elo_ratings.items(), key=lambda i: i[1], reverse=True)
    
    # Run ranking algorithms
    em_ranking = []
    bt_mle_ranking = []
    all_policy_names = sorted(list(policy_stats.keys()))
    
    if all_policy_names:
        policy_map = {name: i for i, name in enumerate(all_policy_names)}
        try:
            em_ranking = run_em_ranking(all_responses, all_scaled_versions, policy_map, all_pairs, all_pairs_scaled)
        except Exception as e:
            em_ranking = [("Error running EM", f"{e}")]
        try:
            bt_mle_ranking = run_bt_mle_ranking(all_responses, all_scaled_versions, policy_map, all_pairs, all_pairs_scaled)
        except Exception as e:
            bt_mle_ranking = [("Error running BT MLE", f"{e}")]

    # Generate HTML - Original layout without participant stats
    total_comparisons = len(all_responses)
    
    html = f"""
    <html>
    <head>
        <title>Robot Arena  Dashboard</title>
        <style>
            body {{ font-family: 'Segoe UI', sans-serif; background-color: #f8f9fa; margin: 0; padding: 2rem; }}
            .container {{ max-width: 900px; margin: auto; }}
            h1 {{ font-size: 2.5rem; text-align: center; margin-bottom: 0.5rem; }}
            h2 {{ font-size: 1.5rem; margin-top: 2.5rem; border-bottom: 2px solid #dee2e6; padding-bottom: 0.5rem; }}
            .summary {{ text-align: center; font-size: 1.1rem; color: #555; margin-bottom: 2rem; }}
            table {{ width: 100%; margin: 1rem 0; border-collapse: collapse; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
            th, td {{ padding: 0.75rem 1rem; text-align: left; border-bottom: 1px solid #dee2e6; }}
            th {{ background: #f1f3f5; font-weight: 600; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em; }}
            td.center, th.center {{ text-align: center; }}
            td.policy-name {{ font-weight: 600; color: #005fcc; }}
        </style>
    </head>
    <body>
    <div class="container">
        <h1>🤖 Robot Arena  Dashboard</h1>
        <p class="summary">Total Comparisons: <strong>{total_comparisons}</strong></p>
    """
    
    # EM Ranking
    html += "<h2>🧠 EM-Based Ranking (Latent Buckets)</h2><table><thead><tr><th class='center'>Rank</th><th>Policy</th><th class='center'>Theta (Skill)</th></tr></thead><tbody>"
    for rank, item in enumerate(em_ranking, 1):
        if len(item) >= 2:
            name, score = item[0], item[1]
            score_str = f"{score:.3f}" if isinstance(score, (int, float)) else str(score)
            html += f"<tr><td class='center'>{rank}</td><td class='policy-name'>{name}</td><td class='center'><strong>{score_str}</strong></td></tr>"
    html += "</tbody></table>"
    
    # BT MLE Ranking
    html += "<h2>📈 Bradley-Terry MLE Ranking</h2><table><thead><tr><th class='center'>Rank</th><th>Policy</th><th class='center'>Score</th><th class='center'>Std Error</th></tr></thead><tbody>"
    for rank, item in enumerate(bt_mle_ranking, 1):
        if len(item) >= 3:
            name, score, std_val = item[0], item[1], item[2]
            score_str = f"{score:.4f}" if isinstance(score, (int, float)) else str(score)
            std_str = f"±{std_val:.4f}" if isinstance(std_val, (int, float)) and not np.isnan(std_val) else "N/A"
            html += f"<tr><td class='center'>{rank}</td><td class='policy-name'>{name}</td><td class='center'><strong>{score_str}</strong></td><td class='center'>{std_str}</td></tr>"
    html += "</tbody></table>"
    
    # Points-Based Ranking
    html += "<h2>🏆 Points-Based Ranking</h2><table><thead><tr><th class='center'>Rank</th><th>Policy</th><th class='center'>Avg Score</th><th class='center'>Win %</th><th class='center'>W</th><th class='center'>T</th><th class='center'>L</th><th class='center'>Total</th></tr></thead><tbody>"
    for rank, (name, stats) in enumerate(sorted_policies_by_score, 1):
        comps = stats['comparisons']
        avg_s = stats['points'] / comps if comps > 0 else 0
        win_r = stats['wins'] / comps * 100 if comps > 0 else 0
        html += f"<tr><td class='center'>{rank}</td><td class='policy-name'>{name}</td><td class='center'>{avg_s:.2f}</td><td class='center'>{win_r:.1f}%</td><td class='center'>{stats['wins']}</td><td class='center'>{stats['ties']}</td><td class='center'>{stats['losses']}</td><td class='center'>{comps}</td></tr>"
    html += "</tbody></table>"
    
    # Elo Ranking
    html += "<h2>⚡ Elo Rating Ranking</h2><table><thead><tr><th class='center'>Rank</th><th>Policy</th><th class='center'>Elo Rating</th><th class='center'>Comparisons</th></tr></thead><tbody>"
    for rank, (name, rating) in enumerate(sorted_policies_by_elo, 1):
        html += f"<tr><td class='center'>{rank}</td><td class='policy-name'>{name}</td><td class='center'><strong>{round(rating)}</strong></td><td class='center'>{policy_stats[name]['comparisons']}</td></tr>"
    html += "</tbody></table>"
    
    html += "</div></body></html>"
    
    return HTMLResponse(content=html)


# ============================================================================
# LEGACY ENDPOINT - For backwards compatibility with old frontend
# ============================================================================
class AnnotationLegacy(BaseModel):
    participant_id: str
    completion_code: str
    total_time_ms: int  
    response_length: int 
    responses: list
    timestamp: str
    scaled_version: Optional[str] = None

@app.post("/annotate/legacy")
def save_annotation_legacy(data: AnnotationLegacy):
    """Legacy endpoint for old frontend compatibility"""
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            query = """
            INSERT INTO annotations (
                participant_id, participant_type, completion_code, total_time_ms,
                response_length, quiz_score, quiz_total, 
                sanity_checks_passed, sanity_checks_total, sanity_check_results,
                failed, failure_reason, response_data, timestamp, config_version
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            cursor.execute(query, (
                data.participant_id,
                'unknown',  # Legacy doesn't have participant type
                data.completion_code,
                data.total_time_ms,
                data.response_length,
                None, None,  # No quiz tracking
                None, None, Json([]),  # No sanity check tracking
                False, None,  # Not failed
                Json(data.responses),
                datetime.fromisoformat(data.timestamp.replace("Z", "+00:00")),
                data.scaled_version
            ))
            conn.commit()
        return {"status": "success", "message": "Legacy annotation saved"}
    except Exception as e:
        conn.rollback()
        return {"status": "error", "message": str(e)}
    finally:
        conn.close()