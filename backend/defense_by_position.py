"""
Positional defense ratings: how many pts/ast/reb each team allows to G/F/C per game.

NBA: ESPN rosters (positions) + nba_api LeagueDashPlayerStats filtered by opponent team.
WNBA: ESPN rosters (positions) + concurrent ESPN game log fetches aggregated by opponent.
"""
import time
import requests
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from nba_api.stats.endpoints import leaguedashplayerstats, commonplayerinfo
from nba_api.stats.static import teams as nba_teams

_ESPN_WNBA = "https://site.web.api.espn.com/apis/common/v3/sports/basketball/wnba"

_POS_MAP = {
    "G": "G", "PG": "G", "SG": "G", "GF": "G",
    "F": "F", "SF": "F", "PF": "F",
    "C": "C", "FC": "C", "C-F": "C", "CF": "C",
}
_STATS = ("pts", "ast", "reb")

_nba_name_to_pos: dict = {}
_nba_id_to_pos:   dict = {}
_nba_def_by_pos:  dict = {}   # season → {abbr: {G_pts, G_ast, G_reb, G_pts_rank, ...}}

_wnba_name_to_pos: dict = {}
_wnba_players:     list = []
_wnba_def_by_pos:  dict = {}


# ─── shared ranking helper ────────────────────────────────────────────────────

def _rank(result: dict, n_teams: int) -> None:
    """Add {pos}_{stat}_rank keys to result in-place for all pos/stat combos."""
    for pos in ("G", "F", "C"):
        for stat in _STATS:
            key = f"{pos}_{stat}"
            ordered = sorted(result.items(), key=lambda x: x[1].get(key, 0))
            for rank, (abbr, _) in enumerate(ordered, 1):
                result[abbr][f"{key}_rank"] = rank


def get_def_vs_pos(team_abbr: str, pos: str, stat: str, pos_def: dict, fallback: float) -> float:
    """stat is 'pts', 'ast', or 'reb'. Returns pts/ast/reb allowed to pos by team."""
    return pos_def.get(team_abbr, {}).get(f"{pos}_{stat}", fallback)


# ─── NBA ─────────────────────────────────────────────────────────────────────

def _build_nba_name_pos() -> dict:
    if _nba_name_to_pos:
        return _nba_name_to_pos
    try:
        r = requests.get(
            "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams",
            timeout=10,
        )
        for t in r.json()["sports"][0]["leagues"][0]["teams"]:
            tid = t["team"]["id"]
            time.sleep(0.15)
            r2 = requests.get(
                f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams/{tid}/roster",
                timeout=10,
            )
            if r2.status_code != 200:
                continue
            for p in r2.json().get("athletes", []):
                name = p.get("displayName", "").lower()
                abbr = p.get("position", {}).get("abbreviation", "G")
                _nba_name_to_pos[name] = _POS_MAP.get(abbr.upper(), "G")
    except Exception:
        pass
    return _nba_name_to_pos


def get_nba_player_pos(player_id: int, player_name: str = "") -> str:
    if player_id in _nba_id_to_pos:
        return _nba_id_to_pos[player_id]

    pos_map = _build_nba_name_pos()
    if player_name and player_name.lower() in pos_map:
        pos = pos_map[player_name.lower()]
        _nba_id_to_pos[player_id] = pos
        return pos

    try:
        time.sleep(0.35)
        info = commonplayerinfo.CommonPlayerInfo(player_id=player_id).get_data_frames()[0]
        raw = info["POSITION"].iloc[0] if not info.empty else ""
        _FULL = {
            "Guard": "G", "Point Guard": "G", "Shooting Guard": "G",
            "Forward": "F", "Small Forward": "F", "Power Forward": "F",
            "Center": "C", "Forward-Center": "C", "Center-Forward": "C",
        }
        pos = _FULL.get(raw.strip()) or _POS_MAP.get(raw.strip(), "G")
    except Exception:
        pos = "G"

    _nba_id_to_pos[player_id] = pos
    return pos


def fetch_nba_def_by_pos(season: str) -> dict:
    """
    {team_abbr: {G_pts, G_ast, G_reb, F_pts, ..., C_reb,
                 G_pts_rank, G_ast_rank, G_reb_rank, ...}}
    rank 1 = fewest allowed (best defence), 30 = most allowed (worst).
    """
    if season in _nba_def_by_pos:
        return _nba_def_by_pos[season]

    pos_map   = _build_nba_name_pos()
    team_list = nba_teams.get_teams()

    # accum[abbr][pos][stat] = [values...]
    accum: dict = {
        t["abbreviation"]: {p: {s: [] for s in _STATS} for p in ("G", "F", "C")}
        for t in team_list
    }

    for team in team_list:
        abbr = team["abbreviation"]
        try:
            time.sleep(0.65)
            df = leaguedashplayerstats.LeagueDashPlayerStats(
                season=season,
                opponent_team_id=str(team["id"]),
                per_mode_detailed="PerGame",
            ).get_data_frames()[0]
            for _, row in df.iterrows():
                name = str(row.get("PLAYER_NAME", "")).lower()
                pos  = pos_map.get(name, "G")
                pts  = float(row.get("PTS", 0) or 0)
                ast  = float(row.get("AST", 0) or 0)
                reb  = float(row.get("REB", 0) or 0)
                if pts > 0:
                    accum[abbr][pos]["pts"].append(pts)
                    accum[abbr][pos]["ast"].append(ast)
                    accum[abbr][pos]["reb"].append(reb)
        except Exception:
            continue

    result: dict = {}
    for abbr, groups in accum.items():
        entry: dict = {}
        for pos in ("G", "F", "C"):
            for stat in _STATS:
                vals = groups[pos][stat]
                entry[f"{pos}_{stat}"] = float(np.mean(vals)) if vals else 0.0
        result[abbr] = entry

    _rank(result, len(result))
    _nba_def_by_pos[season] = result
    return result


# ─── WNBA ────────────────────────────────────────────────────────────────────

def register_wnba_player(player_id: str, name: str, pos_abbr: str) -> None:
    pos = _POS_MAP.get(pos_abbr.upper(), "G")
    _wnba_name_to_pos[name.lower()] = pos
    _wnba_players.append({"id": player_id, "name": name, "pos": pos})


def get_wnba_player_pos(player_name: str) -> str:
    return _wnba_name_to_pos.get(player_name.lower(), "G")


def _fetch_wnba_player_games(player: dict, season: str) -> list[tuple]:
    """Return [(opp_abbr, pts, ast, reb), ...] for one WNBA player."""
    params = {"season": season} if season else {}
    try:
        r = requests.get(
            f"{_ESPN_WNBA}/athletes/{player['id']}/gamelog",
            params=params,
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
    except Exception:
        return []

    labels      = data.get("labels", [])
    events_meta = data.get("events", {})
    try:
        pts_idx = labels.index("PTS")
        ast_idx = labels.index("AST")
        reb_idx = labels.index("REB")
    except ValueError:
        return []

    def _num(v):
        try:
            return float(str(v).split("-")[0])
        except Exception:
            return 0.0

    rows = []
    for stype in data.get("seasonTypes", []):
        if "Regular" not in stype.get("displayName", ""):
            continue
        for cat in stype.get("categories", []):
            for event in cat.get("events", []):
                stats = event.get("stats", [])
                n = len(stats)
                if not stats or max(pts_idx, ast_idx, reb_idx) >= n:
                    continue
                eid  = event["eventId"]
                meta = events_meta.get(eid, {})
                opp  = meta.get("opponent", {}).get("abbreviation", "")
                if opp:
                    rows.append((opp, _num(stats[pts_idx]), _num(stats[ast_idx]), _num(stats[reb_idx])))
    return rows


def fetch_wnba_def_by_pos(season: str) -> dict:
    """
    {team_abbr: {G_pts, G_ast, G_reb, F_pts, ..., C_reb,
                 G_pts_rank, G_ast_rank, G_reb_rank, ...}}
    Built by concurrently fetching all WNBA player game logs from ESPN.
    rank 1 = fewest allowed (best defence), 15 = most allowed (worst).
    """
    if season in _wnba_def_by_pos:
        return _wnba_def_by_pos[season]

    if not _wnba_players:
        return {}

    accum: dict = {}   # {opp_abbr: {pos: {stat: [values]}}}

    with ThreadPoolExecutor(max_workers=10) as ex:
        futures = {ex.submit(_fetch_wnba_player_games, p, season): p for p in _wnba_players}
        for fut in as_completed(futures):
            player = futures[fut]
            pos    = player["pos"]
            try:
                for opp, pts, ast, reb in fut.result():
                    if opp not in accum:
                        accum[opp] = {p: {s: [] for s in _STATS} for p in ("G", "F", "C")}
                    accum[opp][pos]["pts"].append(pts)
                    accum[opp][pos]["ast"].append(ast)
                    accum[opp][pos]["reb"].append(reb)
            except Exception:
                continue

    result: dict = {}
    for abbr, groups in accum.items():
        entry: dict = {}
        for pos in ("G", "F", "C"):
            for stat in _STATS:
                vals = groups[pos][stat]
                entry[f"{pos}_{stat}"] = float(np.mean(vals)) if vals else 0.0
        result[abbr] = entry

    _rank(result, len(result))
    _wnba_def_by_pos[season] = result
    return result
