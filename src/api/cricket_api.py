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
    Calls the /currentMatches endpoint to get all currently
    live cricket matches. Paginates through all pages (the API
    returns 25 matches per page).

    Returns:
        list[dict] | None: List of live match objects, or None on error.
    """
    try:
        all_matches = []
        offset = 0
        page_size = 25

        while True:
            url = f"{API_BASE_URL}/currentMatches"
            params = {"apikey": API_KEY, "offset": offset, "t": int(time.time() * 1000)}
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()

            if data.get("status") != "success":
                break

            page_matches = data.get("data", [])
            if not page_matches or not isinstance(page_matches, list):
                break

            all_matches.extend(page_matches)

            # Check if there are more pages
            info = data.get("info", {})
            total_rows = info.get("totalRows", 0)
            if offset + page_size >= total_rows:
                break
            offset += page_size

        if not all_matches:
            return None

        # Filter for truly live matches (status string heuristic)
        live_matches = []
        for m in all_matches:
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
_IPL_KEYWORDS = ["ipl", "indian premier league", "tata ipl", "premier league"]


def filter_ipl_matches(matches):
    """
    Filters a list of match objects to return ONLY IPL matches.

    The CricAPI v1 currentMatches endpoint often returns series=None,
    so we primarily check the 'name' field where the series info is
    embedded (e.g. "CSK vs MI, 42nd Match, Indian Premier League 2025").

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

        # The API often has series=None; the series name is in 'name'
        name = (m.get("name") or "").lower()
        series = (m.get("series") or "").lower()
        competition = (m.get("competition") or "").lower()

        # Combine all searchable text
        searchable = f"{name} {series} {competition}"

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
        
        team1_ordered = team1
        team2_ordered = team2
        team1_score_str = ""
        team2_score_str = "Yet to bat"
        
        runs = 0
        wickets_val = 0
        overs = 0.0
        inning_label = ""

        if scores and isinstance(scores, list):
            # Sort out who is Team 1 based on first inning
            s1 = scores[0] if isinstance(scores[0], dict) else {}
            i1_label = s1.get("inning", "")
            
            if team2.lower() in i1_label.lower():
                team1_ordered = team2
                team2_ordered = team1
            
            # Format Team 1 score
            runs1 = s1.get("r", 0) or 0
            w1 = s1.get("w", 0) or 0
            o1 = float(s1.get("o", 0) or 0)
            team1_score_str = f"{runs1}/{w1} ({o1})"
            
            if len(scores) == 1:
                runs = runs1
                wickets_val = w1
                overs = o1
                inning_label = i1_label
            elif len(scores) > 1:
                s2 = scores[1] if isinstance(scores[1], dict) else {}
                i2_label = s2.get("inning", "")
                runs2 = s2.get("r", 0) or 0
                w2 = s2.get("w", 0) or 0
                o2 = float(s2.get("o", 0) or 0)
                team2_score_str = f"{runs2}/{w2} ({o2})"
                
                latest = scores[-1] if isinstance(scores[-1], dict) else {}
                runs = latest.get("r", 0) or 0
                wickets_val = latest.get("w", 0) or 0
                overs = float(latest.get("o", 0) or 0)
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
            "team1_ordered": team1_ordered,
            "team2_ordered": team2_ordered,
            "team1_score_str": team1_score_str,
            "team2_score_str": team2_score_str,
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
