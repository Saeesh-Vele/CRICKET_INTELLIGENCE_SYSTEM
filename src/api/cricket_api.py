# ==========================================================
# CRICKET DATA API – LIVE MATCH UTILITIES
# ==========================================================
# Provides functions to fetch live matches and scores from
# CricketData API (https://api.cricketdata.org/v1).
# Transforms API response to the exact feature format
# expected by the existing ML pipelines.
# ==========================================================

import requests
import time

API_BASE_URL = "https://api.cricapi.com/v1"
API_KEY = "41c57219-ae5d-4921-88c0-1aca3fa82ef0"

# ----------------------------------------------------------
# 1. FETCH LIVE MATCHES
# ----------------------------------------------------------
def fetch_live_matches():
    """
    Calls the /matches endpoint with status=live to get
    all currently live cricket matches.

    Returns:
        list[dict] | None: List of live match objects, or None on error.
    """
    try:
        url = f"{API_BASE_URL}/currentMatches"
        params = {"apikey": API_KEY, "offset": 0, "t": int(time.time() * 1000)}
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()

        data = response.json()

        if data.get("status") != "success":
            return None

        matches = data.get("data", [])
        if not matches or not isinstance(matches, list):
            return None

        # Filter for truly live matches (status string heuristic)
        live_matches = []
        for m in matches:
            if not isinstance(m, dict):
                continue
            status = (m.get("status") or "").lower()
            match_started = m.get("matchStarted", False)
            match_ended = m.get("matchEnded", False)

            # A match is "live" if it has started but not ended
            if match_started and not match_ended:
                live_matches.append(m)
            # Fallback: status contains keywords indicating live play
            elif any(kw in status for kw in ["innings break", "batting", "need", "trail", "lead", "require"]):
                live_matches.append(m)

        return live_matches if live_matches else None

    except (requests.RequestException, ValueError, KeyError):
        return None


# ----------------------------------------------------------
# 1b. FILTER IPL MATCHES ONLY
# ----------------------------------------------------------
_IPL_KEYWORDS = ["ipl", "indian premier league", "tata ipl"]


def filter_ipl_matches(matches):
    """
    Filters a list of match objects to return ONLY IPL matches.

    Checks the 'series', 'name', and 'competition' fields for
    IPL-related keywords (case-insensitive).

    Args:
        matches (list[dict]): Live match objects from fetch_live_matches().

    Returns:
        list[dict]: Only IPL matches. Empty list if none found.
    """
    if not matches:
        return []

    ipl_matches = []
    for m in matches:
        if not isinstance(m, dict):
            continue

        # Check multiple fields where series/competition name may appear
        series = (m.get("series") or m.get("series_id") or "").lower()
        name = (m.get("name") or "").lower()
        competition = (m.get("competition") or "").lower()
        match_type = (m.get("matchType") or "").lower()

        # Combine all searchable text
        searchable = f"{series} {name} {competition}"

        if any(kw in searchable for kw in _IPL_KEYWORDS):
            ipl_matches.append(m)

    return ipl_matches


# ----------------------------------------------------------
# 2. FETCH MATCH SCORE
# ----------------------------------------------------------
def fetch_match_score(match_id):
    """
    Calls the /scores endpoint for a specific match_id
    to get the latest scorecard.

    If the match object already contains score data (from
    the matches response), pass the full match dict directly
    to extract_score_from_match() instead.

    Returns:
        dict | None: Parsed score data or None on error.
    """
    try:
        url = f"{API_BASE_URL}/match_info"
        params = {"apikey": API_KEY, "id": match_id, "t": int(time.time() * 1000)}
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()

        data = response.json()

        if data.get("status") != "success":
            return None

        match_data = data.get("data", {})
        if not match_data:
            return None

        return extract_score_from_match(match_data)

    except (requests.RequestException, ValueError, KeyError):
        return None


def extract_score_from_match(match_data):
    """
    Given a match object (from currentMatches or match_info),
    extracts a clean score dict with the fields our app needs.

    Returns:
        dict with keys: team1, team2, venue, runs, wickets,
                        overs, match_id, status, inning_label
    """
    try:
        team_info = match_data.get("teamInfo") or []
        team1 = team_info[0].get("name", "Team A") if len(team_info) > 0 else "Team A"
        team2 = team_info[1].get("name", "Team B") if len(team_info) > 1 else "Team B"

        venue = match_data.get("venue", "Unknown Venue")
        match_id = match_data.get("id", "")
        status = match_data.get("status", "")

        # Extract the latest innings score
        scores = match_data.get("score") or []
        runs = 0
        wickets_val = 0
        overs = 0.0
        inning_label = ""

        if scores and isinstance(scores, list):
            # Use the LAST innings entry (most recent)
            latest = scores[-1] if isinstance(scores[-1], dict) else {}
            runs = latest.get("r", 0) or 0
            wickets_val = latest.get("w", 0) or 0
            overs_raw = latest.get("o", 0) or 0
            overs = float(overs_raw)
            inning_label = latest.get("inning", "")

        return {
            "team1": team1,
            "team2": team2,
            "venue": venue,
            "runs": int(runs),
            "wickets": int(wickets_val),
            "overs": float(overs),
            "match_id": match_id,
            "status": status,
            "inning_label": inning_label,
        }

    except (IndexError, TypeError, KeyError):
        return None


# ----------------------------------------------------------
# 3. TRANSFORM API DATA → MODEL FEATURES
# ----------------------------------------------------------
def transform_api_to_features(api_score):
    """
    Converts the API score dict into the exact match-context
    features expected by the existing prediction pipeline.

    The model's match-context features are:
        - match_run_rate
        - wickets_fallen
        - wickets_left
        - pressure_index

    These are the ONLY features we inject from live data.
    All other features (rolling averages, venue stats, etc.)
    come from the historical dataset, exactly as in manual mode.

    Args:
        api_score (dict): Output of extract_score_from_match()

    Returns:
        dict with keys: current_score, overs, wickets,
                        match_run_rate, wickets_fallen,
                        wickets_left, pressure_index
    """
    if not api_score:
        return None

    try:
        runs = api_score.get("runs", 0) or 0
        overs = api_score.get("overs", 0) or 0
        wickets_val = api_score.get("wickets", 0) or 0

        # Ensure safe values
        runs = max(0, int(runs))
        overs = max(0.0, float(overs))
        wickets_val = max(0, min(10, int(wickets_val)))

        # Compute derived features (same logic as generate_match_context)
        if overs == 0:
            run_rate = 0.0
        else:
            run_rate = runs / overs

        wickets_left = max(10 - wickets_val, 1)
        pressure_index = run_rate / wickets_left

        return {
            "current_score": runs,
            "overs": overs,
            "wickets": wickets_val,
            "match_run_rate": round(run_rate, 4),
            "wickets_fallen": wickets_val,
            "wickets_left": wickets_left,
            "pressure_index": round(pressure_index, 4),
        }

    except (TypeError, ValueError, ZeroDivisionError):
        return None


# ----------------------------------------------------------
# 4. HELPERS
# ----------------------------------------------------------
def format_match_label(match):
    """
    Creates a human-readable label for a match dropdown.
    e.g. "CSK vs MI (T20) — IPL 2026"
    """
    name = match.get("name", "Unknown Match")
    match_type = (match.get("matchType") or "").upper()
    series = match.get("series") or match.get("competition") or ""

    parts = [name]
    if match_type:
        parts[0] = f"{name} ({match_type})"
    if series:
        parts.append(f"— {series}")

    return " ".join(parts)
