import time
import requests
from difflib import get_close_matches

API_KEY  = "dcda49044b41f09a272ab6857f81d2e9"
BASE_URL = "https://api.the-odds-api.com/v4"

SPORTS = {"NBA": "basketball_nba", "WNBA": "basketball_wnba"}
MARKET_MAP = {
    "player_points":   "PTS",
    "player_assists":  "AST",
    "player_rebounds": "REB",
}

_cache:      dict = {}
_cache_time: dict = {}
CACHE_TTL = 1800  # 30 minutes


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

    result: dict = {}

    for event in events:
        event_id = event.get("id")
        if not event_id:
            continue
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

    _cache[key]      = result
    _cache_time[key] = now
    return result


def get_player_lines(player_name: str, league: str) -> dict:
    """Return {PTS: float, AST: float, REB: float} for the player, or {}."""
    all_lines = fetch_lines(league)
    if not all_lines:
        return {}

    if player_name in all_lines:
        return all_lines[player_name]

    norm_target = _normalize(player_name)
    norm_map    = {_normalize(k): k for k in all_lines}

    if norm_target in norm_map:
        return all_lines[norm_map[norm_target]]

    matches = get_close_matches(norm_target, norm_map.keys(), n=1, cutoff=0.82)
    if matches:
        return all_lines[norm_map[matches[0]]]

    return {}
