import time
import json
import os
import difflib
import requests

CACHE_TTL = 1800  # 30 minutes

_cache: dict = {}
_wnba_team_id_to_abbr: dict = {}  # ESPN numeric team ID (str) -> abbreviation

HISTORY_FILE = os.path.join(os.path.dirname(__file__), "injury_history.json")
HISTORY_STALE_DAYS = 30  # drop entries untouched this long


def _load_history() -> dict:
    if not os.path.exists(HISTORY_FILE):
        return {}
    try:
        with open(HISTORY_FILE) as f:
            return json.load(f)
    except Exception:
        return {}


def _save_history(history: dict) -> None:
    try:
        with open(HISTORY_FILE, "w") as f:
            json.dump(history, f, indent=2)
    except Exception:
        pass


def _update_history(league: str, current: dict) -> None:
    """Track Out -> returned transitions per player so we can tell when a
    teammate who was recently Out has come back (see get_recently_returned_players).
    """
    history = _load_history()
    lg_hist = history.setdefault(league, {})
    now = time.time()

    currently_out = {name for name, rec in current.items() if rec["status"].lower() == "out"}

    for name in currently_out:
        entry = lg_hist.get(name, {})
        entry["name"]         = current[name]["name"]
        entry["team_abbr"]    = current[name]["team_abbr"]
        entry["was_out"]      = True
        entry["out_last_seen"] = now
        entry.pop("returned_at", None)  # still out (or relapsed) - clear any stale return flag
        lg_hist[name] = entry

    for name, entry in lg_hist.items():
        if entry.get("was_out") and name not in currently_out and "returned_at" not in entry:
            entry["returned_at"] = now

    stale_cutoff = now - HISTORY_STALE_DAYS * 86400
    for name in list(lg_hist.keys()):
        entry = lg_hist[name]
        last_touch = max(entry.get("out_last_seen", 0), entry.get("returned_at", 0))
        if last_touch < stale_cutoff:
            del lg_hist[name]

    history[league] = lg_hist
    _save_history(history)


def get_recently_returned_players(team_abbr: str, league: str, window_days: float = 5) -> list:
    """Players on this team who were Out and have since come off the Out list,
    within the last `window_days`. Used to taper teammate stat boosts back down
    as the returning player reclaims touches."""
    history = _load_history()
    lg_hist = history.get(league, {})
    now = time.time()
    window = window_days * 86400

    out = []
    for entry in lg_hist.values():
        returned_at = entry.get("returned_at")
        if not returned_at or entry.get("team_abbr") != team_abbr:
            continue
        age = now - returned_at
        if age <= window:
            out.append({
                "name":        entry["name"],
                "days_since":  round(age / 86400, 2),
            })
    return out


def _get_wnba_team_id_map() -> dict:
    """Build a one-time map of ESPN WNBA team ID -> abbreviation."""
    global _wnba_team_id_to_abbr
    if _wnba_team_id_to_abbr:
        return _wnba_team_id_to_abbr
    try:
        r = requests.get(
            "https://site.api.espn.com/apis/site/v2/sports/basketball/wnba/teams",
            timeout=8, headers={"User-Agent": "Mozilla/5.0"},
        )
        teams = r.json()["sports"][0]["leagues"][0]["teams"]
        _wnba_team_id_to_abbr = {
            str(t["team"]["id"]): t["team"].get("abbreviation", "")
            for t in teams
        }
    except Exception:
        pass
    return _wnba_team_id_to_abbr


ESPN_URLS = {
    "NBA":  "https://site.web.api.espn.com/apis/site/v2/sports/basketball/nba/injuries",
    "WNBA": "https://site.web.api.espn.com/apis/site/v2/sports/basketball/wnba/injuries",
}


def fetch_injuries(league: str) -> dict:
    """Returns {lowercase_name: {name, status, injury_type, comment, team_abbr}} for all injured players."""
    now = time.time()
    cached = _cache.get(league)
    if cached and now - cached["ts"] < CACHE_TTL:
        return cached["data"]

    try:
        r = requests.get(ESPN_URLS[league], timeout=8, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        data = r.json()

        wnba_id_map = _get_wnba_team_id_map() if league == "WNBA" else {}

        result = {}
        by_team = {}  # team_abbr -> list of entry dicts (Out players only)
        for team in data.get("injuries", []):
            # NBA wraps team info under a "team" key; WNBA exposes it at the top level
            team_info = team.get("team", team)
            raw_abbr  = team_info.get("abbreviation", "")
            team_id   = str(team_info.get("id", ""))
            # For WNBA the top-level entry has no abbreviation — look it up via ID
            team_abbr = raw_abbr or wnba_id_map.get(team_id, team_id)
            for entry in team.get("injuries", []):
                athlete = entry.get("athlete", {})
                name = athlete.get("displayName", "").strip()
                if not name:
                    continue
                status = entry.get("status", "Unknown")
                details = entry.get("details", {})
                comment = entry.get("shortComment", "")
                rec = {
                    "name":        name,
                    "status":      status,
                    "injury_type": details.get("type", ""),
                    "comment":     comment,
                    "team_abbr":   team_abbr,
                }
                result[name.lower()] = rec
                if status.lower() == "out":
                    by_team.setdefault(team_abbr, []).append(rec)

        _cache[league] = {"data": result, "by_team": by_team, "ts": now}
        _update_history(league, result)
        return result

    except Exception:
        return cached["data"] if cached else {}


def get_out_players_for_team(team_abbr: str, league: str) -> list:
    """Return list of Out players for the given team abbreviation."""
    fetch_injuries(league)
    cached = _cache.get(league, {})
    return cached.get("by_team", {}).get(team_abbr, [])


def get_player_injury(player_name: str, league: str) -> dict | None:
    """Returns injury info for a player, or None if not on the report."""
    injuries = fetch_injuries(league)
    nl = player_name.lower().strip()

    if nl in injuries:
        return injuries[nl]

    # Normalize: strip dots/apostrophes for fuzzy match
    def normalize(s):
        return s.replace(".", "").replace("'", "").replace("-", " ").lower()

    norm_nl = normalize(nl)
    norm_map = {normalize(k): v for k, v in injuries.items()}

    if norm_nl in norm_map:
        return norm_map[norm_nl]

    matches = difflib.get_close_matches(norm_nl, norm_map.keys(), n=1, cutoff=0.82)
    if matches:
        return norm_map[matches[0]]

    return None
