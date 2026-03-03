import pandas as pd
from pathlib import Path
import logging

# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
RAW_DIR = Path("../data/raw")
CLEANED_DIR = Path("../data/cleaned")
CLEANED_DIR.mkdir(parents=True, exist_ok=True)

MATCHES_FILE = RAW_DIR / "matches.csv"
BALLS_FILE = RAW_DIR / "deliveries.csv"

# ------------------------------------------------------------------
# Load Data
# ------------------------------------------------------------------
logging.info("Loading raw datasets...")
matches = pd.read_csv(MATCHES_FILE)
balls = pd.read_csv(BALLS_FILE)

# ------------------------------------------------------------------
# Standardize Column Names
# ------------------------------------------------------------------
matches.columns = matches.columns.str.lower().str.strip()
balls.columns = balls.columns.str.lower().str.strip()

# ------------------------------------------------------------------
# Date Handling (Robust)
# ------------------------------------------------------------------
date_col = "date" if "date" in matches.columns else "match_date"
matches["date"] = pd.to_datetime(matches[date_col], errors="coerce")

# ------------------------------------------------------------------
# Missing Values
# ------------------------------------------------------------------
for col in ["venue", "winner", "team1", "team2"]:
    if col in matches.columns:
        matches[col] = matches[col].fillna("Unknown")

for col in ["batsman_runs", "extra_runs", "total_runs", "is_wicket"]:
    if col in balls.columns:
        balls[col] = balls[col].fillna(0)

# Remove invalid player rows
balls.dropna(subset=["batter", "bowler"], inplace=True)

# ------------------------------------------------------------------
# Standardize Team Names (Canonical Form)
# ------------------------------------------------------------------
TEAM_MAP = {
    "delhi daredevils": "delhi capitals",
    "rising pune supergiants": "rising pune supergiant",
    "royal challengers bangalore": "royal challengers bengaluru",
    "kings xi punjab": "punjab kings"
}

for col in ["batting_team", "bowling_team"]:
    if col in balls.columns:
        balls[col] = balls[col].str.lower().replace(TEAM_MAP)

for col in ["team1", "team2", "winner"]:
    if col in matches.columns:
        matches[col] = matches[col].str.lower().replace(TEAM_MAP)

# ------------------------------------------------------------------
# Venue Cleaning
# ------------------------------------------------------------------
matches["venue"] = matches["venue"].str.strip().str.title()


import re

# Complete mapping of all stadium names to canonical names
venue_map = {
    "arun jaitley stadium": "Arun Jaitley Stadium",
    "arun jaitley stadium delhi": "Arun Jaitley Stadium",
    "feroz shah kotla": "Arun Jaitley Stadium",

    "barabati stadium": "Barabati Stadium",
    "barsapara cricket stadium": "Barsapara Cricket Stadium",
    "barsapara cricket stadium guwahati": "Barsapara Cricket Stadium",

    "bharat ratna shri atal bihari vajpayee ekana cricket stadium": "Ekana Cricket Stadium",
    "bharat ratna shri atal bihari vajpayee ekana cricket stadium lucknow": "Ekana Cricket Stadium",

    "brabourne stadium": "Brabourne Stadium",
    "brabourne stadium mumbai": "Brabourne Stadium",

    "buffalo park": "Buffalo Park",
    "de beers diamond oval": "De Beers Diamond Oval",

    "dr dy patil sports academy": "Dr DY Patil Sports Academy",
    "dr dy patil sports academy mumbai": "Dr DY Patil Sports Academy",
    "dr ys rajasekhara reddy acavdca cricket stadium": "Dr Y.S. Rajasekhara Reddy Stadium",
    "dr ys rajasekhara reddy acavdca cricket stadium visakhapatnam": "Dr Y.S. Rajasekhara Reddy Stadium",

    "dubai international cricket stadium": "Dubai International Cricket Stadium",

    "eden gardens": "Eden Gardens",
    "eden gardens kolkata": "Eden Gardens",

    "green park": "Green Park",

    "himachal pradesh cricket association stadium": "HPCA Stadium",
    "himachal pradesh cricket association stadium dharamsala": "HPCA Stadium",

    "holkar cricket stadium": "Holkar Cricket Stadium",
    "jsca international stadium complex": "JSCA International Stadium Complex",

    "kingsmead": "Kingsmead",

    "m chinnaswamy stadium": "M Chinnaswamy Stadium",
    "m chinnaswamy stadium bengaluru": "M Chinnaswamy Stadium",
    "m.chinnaswamy stadium": "M Chinnaswamy Stadium",

    "ma chidambaram stadium": "MA Chidambaram Stadium",
    "ma chidambaram stadium chepauk": "MA Chidambaram Stadium",
    "ma chidambaram stadium chepauk chennai": "MA Chidambaram Stadium",

    "maharaja yadavindra singh international cricket stadium mullanpur": "Maharaja Yadavindra Singh Stadium",

    "maharashtra cricket association stadium": "Maharashtra Cricket Association Stadium",
    "maharashtra cricket association stadium pune": "Maharashtra Cricket Association Stadium",

    "narendra modi stadium ahmedabad": "Narendra Modi Stadium",

    "nehru stadium": "Nehru Stadium",

    "new wanderers stadium": "New Wanderers Stadium",
    "newlands": "Newlands",
    "outsurance oval": "Outsurance Oval",
    "supersport park": "Supersport Park",
    "st georges park": "St George's Park",

    "punjab cricket association is bindra stadium": "PCA IS Bindra Stadium",
    "punjab cricket association is bindra stadium mohali": "PCA IS Bindra Stadium",
    "punjab cricket association is bindra stadium mohali chandigarh": "PCA IS Bindra Stadium",
    "punjab cricket association stadium mohali": "PCA IS Bindra Stadium",

    "rajiv gandhi international stadium": "Rajiv Gandhi International Stadium",
    "rajiv gandhi international stadium uppal": "Rajiv Gandhi International Stadium",
    "rajiv gandhi international stadium uppal hyderabad": "Rajiv Gandhi International Stadium",

    "sardar patel stadium motera": "Sardar Patel Stadium",

    "saurashtra cricket association stadium": "Saurashtra Cricket Association Stadium",

    "sawai mansingh stadium": "Sawai Mansingh Stadium",
    "sawai mansingh stadium jaipur": "Sawai Mansingh Stadium",

    "shaheed veer narayan singh international stadium": "Shaheed Veer Narayan Singh International Stadium",
    "sharjah cricket stadium": "Sharjah Cricket Stadium",
    "sheikh zayed stadium": "Sheikh Zayed Stadium",
    "zayed cricket stadium abu dhabi": "Zayed Cricket Stadium",

    "subrata roy sahara stadium": "Subrata Roy Sahara Stadium",
    "vidarbha cricket association stadium jamtha": "Vidarbha Cricket Association Stadium",

    "wankhede stadium": "Wankhede Stadium",
    "wankhede stadium mumbai": "Wankhede Stadium"
}

def normalize_venue(name):
    if not isinstance(name, str):
        return name

    # Lowercase
    name = name.lower()

    # Remove common city suffixes not included in map
    remove_words = [
        "delhi", "mohali", "chandigarh", "kolkata",
        "bengaluru", "bangalore", "mumbai",
        "chennai", "hyderabad", "ahmedabad",
        "jaipur", "guwahati", "lucknow", "pune",
        "visakhapatnam", "uppal", "motera", "jamtha", "abu dhabi"
    ]
    for word in remove_words:
        name = name.replace(word, "")

    # Remove punctuation
    name = re.sub(r"[^\w\s]", "", name)

    # Normalize spaces
    name = " ".join(name.split())

    # Map to canonical name if exists
    return venue_map.get(name, name.title())

# Apply normalization to DataFrame
if "venue" in matches.columns:
    matches["venue"] = matches["venue"].apply(normalize_venue)

# ------------------------------------------------------------------
# Remove Duplicates
# ------------------------------------------------------------------
matches.drop_duplicates(subset=["id"], inplace=True)
balls.drop_duplicates(inplace=True)

# ------------------------------------------------------------------
# Remove Super Overs (Important)
# ------------------------------------------------------------------
if "inning" in balls.columns:
    balls = balls[balls["inning"] <= 2]

# ------------------------------------------------------------------
# Merge Datasets (Safe)
# ------------------------------------------------------------------
match_id_col = "match_id" if "match_id" in balls.columns else "id"

ipl = balls.merge(
    matches[["id", "date", "venue", "team1", "team2"]],
    left_on=match_id_col,
    right_on="id",
    how="left",
    validate="many_to_one"
)

# ------------------------------------------------------------------
# Sorting for Time-Series Safety
# ------------------------------------------------------------------
sort_cols = ["date"]
for col in ["id", "inning", "over", "ball"]:
    if col in ipl.columns:
        sort_cols.append(col)

ipl = ipl.sort_values(sort_cols)

# ------------------------------------------------------------------
# Validation Checks
# ------------------------------------------------------------------
assert ipl["date"].isna().sum() == 0, "❌ Missing match dates after merge"
assert ipl["batter"].isna().sum() == 0, "❌ Null batter values"
assert ipl["bowler"].isna().sum() == 0, "❌ Null bowler values"

logging.info("Validation checks passed")

# ------------------------------------------------------------------
# Save Cleaned Data
# ------------------------------------------------------------------
output_path = CLEANED_DIR / "ipl_cleaned_ball_by_ball.csv"
ipl.to_csv(output_path, index=False)

logging.info(f"✅ Cleaned data saved to {output_path}")
logging.info(f"Final dataset shape: {ipl.shape}")
