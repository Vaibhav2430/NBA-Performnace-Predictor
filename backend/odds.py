import os
import time
import requests
from datetime import datetime
from difflib import get_close_matches
from zoneinfo import ZoneInfo
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ["ODDS_API_KEY"]
BASE_URL = "https://api.the-odds-api.com/v4"

SPORTS = {"NBA": "basketball_nba", "WNBA": "basketball_wnba"}
MARKET_MAP = {
    "player_points":   "PTS",
    "player_assists":  "AST",
    "player_rebounds": "REB",
}

# NBA/WNBA schedules are published in US Eastern time; ESPN's game logs
# date games by that same local "gameday", so game dates must be derived
# in this zone rather than the server's own local time or UTC.
GAME_DATE_TZ = ZoneInfo("America/New_York")

_cache:       dict = {}
_cache_time:  dict = {}
_game_dates:  dict = {}   # {league: {player_name: "YYYY-MM-DD"}}
CACHE_TTL = 1800  # 30 minutes


def _commence_time_to_game_date(commence_time: str) -> str | None:
    if not commence_time:
        return None
    try:
        dt = datetime.fromisoformat(commence_time.replace("Z", "+00:00"))
        return dt.astimezone(GAME_DATE_TZ).date().isoformat()
    except Exception:
        return None


def _normalize(name: str) -> str:
    return name.lower().replace(".", "").replace("'", "").replace("-", " ").strip()


def fetch_lines(league: str) -> dict:
    """Fetch and cache player prop lines for the given league.

    Returns {player_name: {PTS: float, AST: float, REB: float}}.
    """
    now = time.time()
    key = league.upper()
    if key in _cache and now - _cache_time.get(key, 0) < CACHE_TTL:
        return _cache[key]

    sport = SPORTS.get(key)
    if not sport:
        return {}

    # Get today's events
    try:
        events_resp = requests.get(
            f"{BASE_URL}/sports/{sport}/events",
            params={"apiKey": API_KEY},
            timeout=10,
        )
        events_resp.raise_for_status()
        events = events_resp.json()
    except Exception:
        return _cache.get(key, {})

    result:     dict = {}
    game_dates: dict = {}

    for event in events:
        event_id = event.get("id")
        if not event_id:
            continue
        game_date = _commence_time_to_game_date(event.get("commence_time"))
        try:
            odds_resp = requests.get(
                f"{BASE_URL}/sports/{sport}/events/{event_id}/odds",
                params={
                    "apiKey":      API_KEY,
                    "regions":     "us",
                    "markets":     "player_points,player_assists,player_rebounds",
                    "bookmakers":  "draftkings",
                },
                timeout=10,
            )
            if odds_resp.status_code != 200:
                continue
            data = odds_resp.json()
        except Exception:
            continue

        for bookmaker in data.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                stat = MARKET_MAP.get(market.get("key", ""))
                if not stat:
                    continue
                seen: set = set()
                for outcome in market.get("outcomes", []):
                    if outcome.get("name") != "Over":
                        continue
                    player = outcome.get("description", "")
                    line   = outcome.get("point")
                    if player and line is not None and player not in seen:
                        if player not in result:
                            result[player] = {}
                        result[player][stat] = float(line)
                        seen.add(player)
                        if game_date:
                            game_dates[player] = game_date

    _cache[key]       = result
    _cache_time[key]  = now
    _game_dates[key]  = game_dates
    return result


def _resolve_player_key(player_name: str, all_lines: dict) -> str | None:
    if player_name in all_lines:
        return player_name

    norm_target = _normalize(player_name)
    norm_map    = {_normalize(k): k for k in all_lines}

    if norm_target in norm_map:
        return norm_map[norm_target]

    matches = get_close_matches(norm_target, norm_map.keys(), n=1, cutoff=0.82)
    if matches:
        return norm_map[matches[0]]

    return None


def get_player_lines(player_name: str, league: str) -> dict:
    """Return {PTS: float, AST: float, REB: float} for the player, or {}."""
    all_lines = fetch_lines(league)
    if not all_lines:
        return {}

    key = _resolve_player_key(player_name, all_lines)
    return all_lines[key] if key else {}


def get_player_game_date(player_name: str, league: str) -> str | None:
    """Return the scheduled game date ('YYYY-MM-DD', US Eastern) for the
    player's current prop, or None if unknown."""
    all_lines = fetch_lines(league)
    if not all_lines:
        return None

    key = _resolve_player_key(player_name, all_lines)
    if not key:
        return None

    return _game_dates.get(league.upper(), {}).get(key)
