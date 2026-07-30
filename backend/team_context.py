"""Cache current-season per-game averages for NBA and WNBA players/teams.

Used to estimate the scoring impact of an injured-out teammate so the model
can apply an opportunity boost to remaining players.
"""
import time
import requests
from difflib import get_close_matches

_nba_players: dict = {}   # lower_name -> {ppg, apg, rpg, team_abbr}
_nba_teams:   dict = {}   # team_abbr  -> ppg
_nba_ts:      float = 0.0

_wnba_players: dict = {}  # lower_name -> {ppg, apg, rpg, team_abbr}
_wnba_ts:      float = 0.0

# nba_api's WNBA team abbreviations differ from the ESPN ones used everywhere
# else in this app (e.g. injuries.py, odds.py) - map to keep team_abbr consistent.
_WNBA_ABBR_MAP = {
    "GSV": "GS", "LAS": "LA", "LVA": "LV", "NYL": "NY", "PDX": "POR", "WAS": "WSH",
}

_nba_mpg:     dict = {}   # lower_name -> {mpg, team_abbr} (full roster, blended w/ last season)
_nba_mpg_ts:  float = 0.0
_wnba_mpg:    dict = {}
_wnba_mpg_ts: float = 0.0

_MPG_MIN_GP = 5  # below this many current-season games, trust last season's role instead

TTL = 3600 * 6  # refresh every 6 hours


# ── NBA ──────────────────────────────────────────────────────────────────────

def _load_nba() -> None:
    global _nba_players, _nba_teams, _nba_ts
    if time.time() - _nba_ts < TTL and _nba_players:
        return
    try:
        from nba_api.stats.endpoints import leaguedashplayerstats, leaguedashteamstats
        from model import CURRENT_SEASON

        time.sleep(0.6)
        pdf = leaguedashplayerstats.LeagueDashPlayerStats(
            season=CURRENT_SEASON,
            per_mode_detailed="PerGame",
        ).get_data_frames()[0]
        _nba_players = {
            str(row["PLAYER_NAME"]).lower(): {
                "ppg": float(row["PTS"]),
                "apg": float(row["AST"]),
                "rpg": float(row["REB"]),
                "team_abbr": str(row["TEAM_ABBREVIATION"]),
            }
            for _, row in pdf.iterrows()
            if row["PLAYER_NAME"] and int(row.get("GP", 0)) >= 5
        }

        time.sleep(0.6)
        tdf = leaguedashteamstats.LeagueDashTeamStats(
            season=CURRENT_SEASON,
            per_mode_detailed="PerGame",
        ).get_data_frames()[0]
        _nba_teams = {
            str(row["TEAM_ABBREVIATION"]): float(row["PTS"])
            for _, row in tdf.iterrows()
        }
        _nba_ts = time.time()
    except Exception:
        pass


def get_nba_player_ppg(name: str) -> float | None:
    _load_nba()
    nl = name.lower().strip()
    if nl in _nba_players:
        return _nba_players[nl]["ppg"]
    matches = get_close_matches(nl, _nba_players.keys(), n=1, cutoff=0.80)
    return _nba_players[matches[0]]["ppg"] if matches else None


def get_nba_team_ppg(team_abbr: str) -> float:
    _load_nba()
    return _nba_teams.get(team_abbr, 112.0)  # league-average fallback


# ── WNBA ─────────────────────────────────────────────────────────────────────

def _load_wnba() -> None:
    """Load WNBA player averages from ESPN statistics leaders (single request)."""
    global _wnba_players, _wnba_ts
    if time.time() - _wnba_ts < TTL and _wnba_players:
        return
    try:
        r = requests.get(
            "https://site.web.api.espn.com/apis/site/v2/sports/basketball/wnba/statistics",
            timeout=10,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        r.raise_for_status()
        cats = r.json().get("stats", {}).get("categories", [])

        stat_lookup: dict[str, dict] = {}
        for cat in cats:
            cname = cat.get("name", "")
            if cname not in ("pointsPerGame", "assistsPerGame", "reboundsPerGame"):
                continue
            for leader in cat.get("leaders", []):
                athlete = leader.get("athlete", {})
                name    = athlete.get("displayName", "").strip()
                team    = leader.get("team", {}).get("abbreviation", "")
                val     = float(leader.get("value", 0) or 0)
                nl      = name.lower()
                if nl not in stat_lookup:
                    stat_lookup[nl] = {"team_abbr": team}
                stat_lookup[nl][cname] = val

        _wnba_players = {
            nl: {
                "ppg": vals.get("pointsPerGame", 0.0),
                "apg": vals.get("assistsPerGame", 0.0),
                "rpg": vals.get("reboundsPerGame", 0.0),
                "team_abbr": vals.get("team_abbr", ""),
            }
            for nl, vals in stat_lookup.items()
        }
        _wnba_ts = time.time()
    except Exception:
        pass


def get_wnba_player_ppg(name: str) -> float | None:
    _load_wnba()
    nl = name.lower().strip()
    if nl in _wnba_players:
        return _wnba_players[nl]["ppg"]
    matches = get_close_matches(nl, _wnba_players.keys(), n=1, cutoff=0.80)
    return _wnba_players[matches[0]]["ppg"] if matches else None


def get_wnba_team_ppg(team_abbr: str) -> float:
    # Leaders endpoint only covers top-50 per stat so team totals are incomplete;
    # use the 2026 WNBA league-average team score as a reliable fallback.
    return 82.0


def _blended_mpg(cur_df, prev_df, abbr_map: dict | None = None) -> dict:
    """{lower_name: {mpg, team_abbr}} using current-season minutes/game, but
    falling back to last season's minutes for anyone with fewer than
    _MPG_MIN_GP games so far.

    Without this, a star out with a long-term injury (the exact case this
    ranking exists to catch) would rank by a tiny, unrepresentative current-
    season sample - or vanish from the ranking/pool entirely if they haven't
    played at all yet - instead of by their normal role."""
    abbr_map = abbr_map or {}
    prev_mpg = {
        str(row["PLAYER_NAME"]).lower(): float(row["MIN"])
        for _, row in prev_df.iterrows() if row["PLAYER_NAME"]
    } if prev_df is not None and not prev_df.empty else {}

    result: dict = {}
    for _, row in cur_df.iterrows():
        name = str(row.get("PLAYER_NAME", ""))
        if not name:
            continue
        nl  = name.lower()
        gp  = int(row.get("GP", 0))
        mpg = float(row["MIN"])
        if gp < _MPG_MIN_GP:
            mpg = max(mpg, prev_mpg.get(nl, 0.0))
        team_abbr = abbr_map.get(str(row["TEAM_ABBREVIATION"]), str(row["TEAM_ABBREVIATION"]))
        result[nl] = {"mpg": mpg, "team_abbr": team_abbr}

    # Players who haven't appeared in the current-season feed at all yet
    # (out since day one) still get a role from last season.
    for nl, mpg in prev_mpg.items():
        result.setdefault(nl, {"mpg": mpg, "team_abbr": ""})

    return result


def _load_nba_mpg() -> None:
    global _nba_mpg, _nba_mpg_ts
    if time.time() - _nba_mpg_ts < TTL and _nba_mpg:
        return
    try:
        from nba_api.stats.endpoints import leaguedashplayerstats
        from model import CURRENT_SEASON, _prev_nba_season

        time.sleep(0.6)
        cur_df = leaguedashplayerstats.LeagueDashPlayerStats(
            season=CURRENT_SEASON, per_mode_detailed="PerGame",
        ).get_data_frames()[0]
        time.sleep(0.6)
        prev_df = leaguedashplayerstats.LeagueDashPlayerStats(
            season=_prev_nba_season(), per_mode_detailed="PerGame",
        ).get_data_frames()[0]
        _nba_mpg = _blended_mpg(cur_df, prev_df)
        _nba_mpg_ts = time.time()
    except Exception:
        pass


def _load_wnba_mpg() -> None:
    """Full-roster minutes/game for WNBA, via nba_api's stats.nba.com feed
    (the ESPN leaders endpoint used above only covers top-50 scorers/passers/
    rebounders, not the whole roster, so it can't rank a full team by MPG)."""
    global _wnba_mpg, _wnba_mpg_ts
    if time.time() - _wnba_mpg_ts < TTL and _wnba_mpg:
        return
    try:
        from nba_api.stats.endpoints import leaguedashplayerstats
        from wnba_model import _current_wnba_season, _prev_wnba_season

        time.sleep(0.6)
        cur_df = leaguedashplayerstats.LeagueDashPlayerStats(
            season=_current_wnba_season(), league_id_nullable="10", per_mode_detailed="PerGame",
        ).get_data_frames()[0]
        time.sleep(0.6)
        prev_df = leaguedashplayerstats.LeagueDashPlayerStats(
            season=_prev_wnba_season(), league_id_nullable="10", per_mode_detailed="PerGame",
        ).get_data_frames()[0]
        _wnba_mpg = _blended_mpg(cur_df, prev_df, abbr_map=_WNBA_ABBR_MAP)
        _wnba_mpg_ts = time.time()
    except Exception:
        pass


# ── Unified helpers ───────────────────────────────────────────────────────────

def get_player_ppg(name: str, league: str) -> float | None:
    if league == "NBA":
        return get_nba_player_ppg(name)
    return get_wnba_player_ppg(name)


def get_team_ppg(team_abbr: str, league: str) -> float:
    if league == "NBA":
        return get_nba_team_ppg(team_abbr)
    return get_wnba_team_ppg(team_abbr)


def get_team_top_mpg_names(team_abbr: str, league: str, n: int = 6) -> list[str]:
    """Lowercase names of the top-n players on team_abbr by minutes/game."""
    if league == "NBA":
        _load_nba_mpg()
        pool = _nba_mpg
    else:
        _load_wnba_mpg()
        pool = _wnba_mpg

    team_players = [
        (name, info["mpg"]) for name, info in pool.items() if info.get("team_abbr") == team_abbr
    ]
    team_players.sort(key=lambda x: x[1], reverse=True)
    return [name for name, _ in team_players[:n]]


def is_top_mpg_player(name: str, team_abbr: str, league: str, n: int = 6) -> bool:
    """True if `name` is one of team_abbr's top-n players by minutes/game -
    used to gate injury-driven prediction adjustments to key rotation players
    only, rather than reacting to every player on the injury report."""
    top = get_team_top_mpg_names(team_abbr, league, n)
    if not top:
        return False
    nl = name.lower().strip()
    if nl in top:
        return True
    matches = get_close_matches(nl, top, n=1, cutoff=0.80)
    return bool(matches)
