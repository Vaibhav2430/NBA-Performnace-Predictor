import time
from datetime import date
import numpy as np
import pandas as pd
from nba_api.stats.endpoints import (
    playergamelog, leaguedashteamstats, leaguedashplayerstats, playbyplayv3,
    scheduleleaguev2,
)
from nba_api.stats.static import players as nba_players, teams as nba_teams
from xgboost import XGBRegressor
import defense_by_position as dbp

STATS = ["PTS", "AST", "REB"]


def _current_nba_season() -> str:
    today = date.today()
    start_year = today.year if today.month >= 10 else today.year - 1
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def _prev_nba_season() -> str:
    today = date.today()
    start_year = today.year if today.month >= 10 else today.year - 1
    prev = start_year - 1
    return f"{prev}-{str(prev + 1)[-2:]}"


CURRENT_SEASON = _current_nba_season()
FEATURE_COLS = [
    "pts_l5", "pts_l10",
    "ast_l5", "ast_l10",
    "reb_l5", "reb_l10",
    "min_l5", "home", "days_rest",
    "opp_def_rtg", "opp_pace", "team_pace",
    "opp_def_pts_vs_pos", "opp_def_ast_vs_pos", "opp_def_reb_vs_pos",
    "pts_home_avg", "pts_away_avg",
    "ast_home_avg", "ast_away_avg",
    "reb_home_avg", "reb_away_avg",
]


def fetch_team_defense_stats() -> dict:
    id_to_abbr = {t["id"]: t["abbreviation"] for t in nba_teams.get_teams()}
    time.sleep(0.65)
    stats = leaguedashteamstats.LeagueDashTeamStats(
        season=CURRENT_SEASON,
        measure_type_detailed_defense="Advanced",
        per_mode_detailed="PerGame",
    )
    df = stats.get_data_frames()[0].copy()
    df["TEAM_ABBREVIATION"] = df["TEAM_ID"].map(id_to_abbr)
    df["off_rank"] = df["OFF_RATING"].rank(ascending=False, method="min").astype(int)
    df["def_rank"] = df["DEF_RATING"].rank(ascending=True,  method="min").astype(int)
    return {
        row["TEAM_ABBREVIATION"]: {
            "team_name": row["TEAM_NAME"],
            "def_rtg":   float(row["DEF_RATING"]),
            "pace":      float(row["PACE"]),
            "off_rtg":   float(row["OFF_RATING"]),
            "off_rank":  int(row["off_rank"]),
            "def_rank":  int(row["def_rank"]),
        }
        for _, row in df.iterrows()
        if pd.notna(row["TEAM_ABBREVIATION"])
    }


_ROSTER_TTL = 300  # seconds
_roster_cache: dict = {}  # team_abbr -> (timestamp, players)


def fetch_team_roster_averages(team_abbr: str) -> list[dict]:
    """Season-average PTS/AST/REB/MIN per player on team_abbr, sorted by minutes."""
    cached = _roster_cache.get(team_abbr)
    if cached and time.time() - cached[0] < _ROSTER_TTL:
        return cached[1]

    id_map = {t["abbreviation"]: t["id"] for t in nba_teams.get_teams()}
    team_id = id_map.get(team_abbr)
    if not team_id:
        return []

    time.sleep(0.5)
    df = leaguedashplayerstats.LeagueDashPlayerStats(
        season=CURRENT_SEASON,
        per_mode_detailed="PerGame",
        team_id_nullable=str(team_id),
    ).get_data_frames()[0]
    df = df[df["GP"] > 0].sort_values("MIN", ascending=False)

    players = [
        {
            "id":   int(row["PLAYER_ID"]),
            "name": row["PLAYER_NAME"],
            "pts":  round(float(row["PTS"]), 1),
            "ast":  round(float(row["AST"]), 1),
            "reb":  round(float(row["REB"]), 1),
            "min":  round(float(row["MIN"]), 1),
            "gp":   int(row["GP"]),
        }
        for _, row in df.iterrows()
    ]
    _roster_cache[team_abbr] = (time.time(), players)
    return players


def _parse_opponent(matchup: str) -> str:
    # MATCHUP is always "TEAM vs. OPP" or "TEAM @ OPP"
    sep = " vs. " if " vs. " in matchup else " @ "
    return matchup.split(sep)[1].strip()


_SCHEDULE_TTL = 3600  # seconds - the league schedule barely changes intraday
_schedule_cache: dict = {}


def _fetch_league_schedule() -> pd.DataFrame:
    hit = _schedule_cache.get("df")
    if hit is not None and time.time() - _schedule_cache.get("ts", 0) < _SCHEDULE_TTL:
        return hit
    try:
        time.sleep(0.65)
        df = scheduleleaguev2.ScheduleLeagueV2(timeout=30).get_data_frames()[0]
    except Exception:
        df = pd.DataFrame()
    _schedule_cache.update(df=df, ts=time.time())
    return df


def _next_game(team_abbr: str, last_game_date) -> dict | None:
    """Next scheduled (not-yet-played) game for `team_abbr`, as
    {"opp_abbr", "home", "days_rest"} - or None if the schedule can't be read
    or every game is already final (offseason)."""
    sched = _fetch_league_schedule()
    if sched.empty or not team_abbr:
        return None
    is_team  = (sched["homeTeam_teamTricode"] == team_abbr) | (sched["awayTeam_teamTricode"] == team_abbr)
    upcoming = sched[is_team & (sched["gameStatus"] == 1)].copy()  # 1 = scheduled
    if upcoming.empty:
        return None
    upcoming["_dt"] = pd.to_datetime(upcoming["gameDateEst"], errors="coerce", utc=True).dt.tz_localize(None)
    upcoming = upcoming.dropna(subset=["_dt"]).sort_values("_dt")
    if upcoming.empty:
        return None

    row     = upcoming.iloc[0]
    is_home = row["homeTeam_teamTricode"] == team_abbr
    opp     = row["awayTeam_teamTricode"] if is_home else row["homeTeam_teamTricode"]

    days_rest = 2.0
    if last_game_date is not None and pd.notna(last_game_date):
        gap = (row["_dt"].normalize() - pd.Timestamp(last_game_date).normalize()).days
        days_rest = float(np.clip(gap, 0, 10))
    return {"opp_abbr": opp, "home": 1.0 if is_home else 0.0, "days_rest": days_rest}


def find_player(name: str) -> dict | None:
    all_p = nba_players.get_players()
    nl = name.lower().strip()
    exact = [p for p in all_p if p["full_name"].lower() == nl]
    if exact:
        return exact[0]
    partial = [p for p in all_p if nl in p["full_name"].lower()]
    return partial[0] if partial else None


def fetch_game_log(player_id: int, season: str = CURRENT_SEASON) -> pd.DataFrame:
    time.sleep(0.65)
    log = playergamelog.PlayerGameLog(player_id=player_id, season=season)
    df = log.get_data_frames()[0]
    if df.empty:
        return pd.DataFrame()

    df = df[["GAME_DATE", "MATCHUP", "WL", "MIN", "PTS", "AST", "REB", "Game_ID"]].copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values("GAME_DATE").reset_index(drop=True)
    for col in ["PTS", "AST", "REB", "MIN"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    df["HOME"] = df["MATCHUP"].apply(lambda x: 0 if "@" in x else 1)
    df["DAYS_REST"] = df["GAME_DATE"].diff().dt.days.fillna(2).clip(0, 10)
    return df


def played_second_half(game_id: str, player_id: int) -> bool | None:
    """True/False if play-by-play shows whether the player has any event in
    period 3+ of this game; None if play-by-play data isn't available (in
    which case the caller should not assume anything either way)."""
    try:
        time.sleep(0.65)
        df = playbyplayv3.PlayByPlayV3(game_id=game_id).get_data_frames()[0]
        sub = df[df["personId"] == player_id]
        if sub.empty:
            return None
        return bool((sub["period"] >= 3).any())
    except Exception:
        return None


def _home_away_avgs(hist: pd.DataFrame) -> dict:
    home = hist[hist["HOME"] == 1]
    away = hist[hist["HOME"] == 0]
    fallback_pts = hist["PTS"].mean()
    fallback_ast = hist["AST"].mean()
    fallback_reb = hist["REB"].mean()
    return {
        "pts_home_avg": home["PTS"].mean() if len(home) else fallback_pts,
        "pts_away_avg": away["PTS"].mean() if len(away) else fallback_pts,
        "ast_home_avg": home["AST"].mean() if len(home) else fallback_ast,
        "ast_away_avg": away["AST"].mean() if len(away) else fallback_ast,
        "reb_home_avg": home["REB"].mean() if len(home) else fallback_reb,
        "reb_away_avg": away["REB"].mean() if len(away) else fallback_reb,
    }


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for i in range(10, len(df)):
        hist = df.iloc[:i]
        cur = df.iloc[i]
        row = {
            "pts_l5":  hist["PTS"].tail(5).mean(),
            "pts_l10": hist["PTS"].tail(10).mean(),
            "ast_l5":  hist["AST"].tail(5).mean(),
            "ast_l10": hist["AST"].tail(10).mean(),
            "reb_l5":  hist["REB"].tail(5).mean(),
            "reb_l10": hist["REB"].tail(10).mean(),
            "min_l5":  hist["MIN"].tail(5).mean(),
            "home":               cur["HOME"],
            "days_rest":          cur["DAYS_REST"],
            "opp_def_rtg":        cur["OPP_DEF_RTG"],
            "opp_pace":           cur["OPP_PACE"],
            "team_pace":          cur["TEAM_PACE"],
            "opp_def_pts_vs_pos": cur["OPP_DEF_PTS_VS_POS"],
            "opp_def_ast_vs_pos": cur["OPP_DEF_AST_VS_POS"],
            "opp_def_reb_vs_pos": cur["OPP_DEF_REB_VS_POS"],
            **_home_away_avgs(hist),
            "PTS": cur["PTS"],
            "AST": cur["AST"],
            "REB": cur["REB"],
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _next_features(
    df: pd.DataFrame,
    avg_def_rtg: float,
    avg_pace: float,
    avg_pos: dict,
    team_pace: float,
    team_stats: dict,
    player_pos: str,
    pos_def: dict,
    next_game: dict | None,
) -> np.ndarray:
    """Feature row for the upcoming game. When the real next opponent is known
    (next_game), plug in that team's actual defense / pace / defense-vs-position
    and the real home/away + days-rest; otherwise fall back to league averages
    and a neutral venue."""
    opp_abbr  = next_game["opp_abbr"] if next_game else None
    opp_stats = team_stats.get(opp_abbr, {}) if opp_abbr else {}
    feats = {
        "pts_l5":               df["PTS"].tail(5).mean(),
        "pts_l10":              df["PTS"].tail(10).mean(),
        "ast_l5":               df["AST"].tail(5).mean(),
        "ast_l10":              df["AST"].tail(10).mean(),
        "reb_l5":               df["REB"].tail(5).mean(),
        "reb_l10":              df["REB"].tail(10).mean(),
        "min_l5":               df["MIN"].tail(5).mean(),
        "home":                 next_game["home"]      if next_game else 0.5,
        "days_rest":            next_game["days_rest"] if next_game else 2.0,
        "opp_def_rtg":          opp_stats.get("def_rtg", avg_def_rtg),
        "opp_pace":             opp_stats.get("pace",    avg_pace),
        "team_pace":            team_pace,
        "opp_def_pts_vs_pos":   dbp.get_def_vs_pos(opp_abbr, player_pos, "pts", pos_def, avg_pos["pts"]) if opp_abbr else avg_pos["pts"],
        "opp_def_ast_vs_pos":   dbp.get_def_vs_pos(opp_abbr, player_pos, "ast", pos_def, avg_pos["ast"]) if opp_abbr else avg_pos["ast"],
        "opp_def_reb_vs_pos":   dbp.get_def_vs_pos(opp_abbr, player_pos, "reb", pos_def, avg_pos["reb"]) if opp_abbr else avg_pos["reb"],
        **_home_away_avgs(df),
    }
    return np.array([[feats[c] for c in FEATURE_COLS]])


def predict(player_name: str) -> dict:
    player = find_player(player_name)
    if not player:
        raise ValueError(f"Player '{player_name}' not found.")

    df = fetch_game_log(player["id"])
    current_season_n = len(df)
    if current_season_n < 15:
        df_prev = fetch_game_log(player["id"], season=_prev_nba_season())
        if not df_prev.empty:
            df = pd.concat([df_prev, df], ignore_index=True)
            df = df.sort_values("GAME_DATE").reset_index(drop=True)
            df["DAYS_REST"] = df["GAME_DATE"].diff().dt.days.fillna(2).clip(0, 10)
    if len(df) < 15:
        raise ValueError(f"Not enough game data for {player['full_name']}.")

    team_stats  = fetch_team_defense_stats()
    avg_def_rtg = float(np.mean([v["def_rtg"] for v in team_stats.values()]))
    avg_pace    = float(np.mean([v["pace"]    for v in team_stats.values()]))

    player_team_abbr = df["MATCHUP"].iloc[-1].split(" ")[0]
    player_team_info = team_stats.get(player_team_abbr, {})

    # Positional defense: pts/ast/reb each team allows to this player's position
    player_pos = dbp.get_nba_player_pos(player["id"], player["full_name"])
    pos_def    = dbp.fetch_nba_def_by_pos(CURRENT_SEASON)
    avg_pos: dict = {}
    for stat in ("pts", "ast", "reb"):
        vals = [pos_def[a][f"{player_pos}_{stat}"] for a in pos_def if pos_def[a].get(f"{player_pos}_{stat}", 0) > 0]
        avg_pos[stat] = float(np.mean(vals)) if vals else (10.0 if stat == "pts" else 3.0)

    team_pace = float(player_team_info.get("pace", avg_pace))

    df["OPP"]                = df["MATCHUP"].apply(_parse_opponent)
    df["OPP_DEF_RTG"]        = df["OPP"].apply(lambda x: team_stats.get(x, {}).get("def_rtg",  avg_def_rtg))
    df["OPP_PACE"]           = df["OPP"].apply(lambda x: team_stats.get(x, {}).get("pace",     avg_pace))
    df["TEAM_PACE"]          = team_pace
    df["OPP_OFF_RANK"]       = df["OPP"].apply(lambda x: team_stats.get(x, {}).get("off_rank"))
    df["OPP_DEF_RANK"]       = df["OPP"].apply(lambda x: team_stats.get(x, {}).get("def_rank"))
    df["OPP_NAME"]           = df["OPP"].apply(lambda x: team_stats.get(x, {}).get("team_name", x))
    df["OPP_DEF_PTS_VS_POS"] = df["OPP"].apply(lambda x: dbp.get_def_vs_pos(x, player_pos, "pts", pos_def, avg_pos["pts"]))
    df["OPP_DEF_AST_VS_POS"] = df["OPP"].apply(lambda x: dbp.get_def_vs_pos(x, player_pos, "ast", pos_def, avg_pos["ast"]))
    df["OPP_DEF_REB_VS_POS"] = df["OPP"].apply(lambda x: dbp.get_def_vs_pos(x, player_pos, "reb", pos_def, avg_pos["reb"]))

    next_game = _next_game(player_team_abbr, df["GAME_DATE"].iloc[-1])

    feature_df = _build_features(df)
    X = feature_df[FEATURE_COLS].values
    X_next = _next_features(df, avg_def_rtg, avg_pace, avg_pos, team_pace,
                            team_stats, player_pos, pos_def, next_game)

    # Projected minutes for the next game: recent workload, tilted toward the
    # most recent handful of games. Used below as a ceiling on each stat - a
    # player can't run far past his per-minute rate x the minutes he'll play.
    if len(df) >= 3:
        proj_min = float(0.65 * df["MIN"].tail(3).mean() + 0.35 * df["MIN"].tail(10).mean())
    else:
        proj_min = float(df["MIN"].mean())

    n = len(X)
    sample_weights = np.ones(n)
    sample_weights[-10:] = 2.25

    # Fewer training rows → fewer trees and shallower depth to prevent overfitting
    n_estimators = max(30, min(150, n * 8))
    max_depth    = 2 if n < 15 else 3

    predictions: dict = {}
    for stat in STATS:
        y = feature_df[stat].values
        model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=5.0 if n < 15 else 1.0,
            random_state=42,
            verbosity=0,
        )
        model.fit(X, y, sample_weight=sample_weights)
        pred = max(0.0, float(model.predict(X_next)[0]))

        curr_games   = df[stat].tail(current_season_n) if current_season_n > 0 else df[stat]
        curr_min     = df["MIN"].tail(current_season_n) if current_season_n > 0 else df["MIN"]
        season_avg   = float(curr_games.mean())
        std          = float(curr_games.std())
        curr_min_avg = float(curr_min.mean())

        # Symmetric sanity band around the season average, then a minutes-based
        # ceiling on top (per-minute rate x projected minutes, +15% headroom).
        lo = max(0.0, season_avg - 2 * std)
        hi = season_avg + 2 * std
        if curr_min_avg > 0:
            hi = min(hi, season_avg / curr_min_avg * proj_min * 1.15)
        hi   = max(hi, lo)
        pred = float(np.clip(pred, lo, hi))

        predictions[stat] = {
            "prediction": round(pred, 1),
            "last5_avg":  round(float(df[stat].tail(5).mean()), 1),
            "season_avg": round(season_avg, 1),
        }

    game_log = (
        df.tail(20)[["GAME_DATE", "MATCHUP", "WL", "MIN", "PTS", "AST", "REB", "OPP", "OPP_NAME", "OPP_OFF_RANK", "OPP_DEF_RANK"]]
        .copy()
        .iloc[::-1]
        .reset_index(drop=True)
    )
    game_log["GAME_DATE"] = game_log["GAME_DATE"].dt.strftime("%Y-%m-%d")
    game_log["MIN"] = game_log["MIN"].round(0).astype(int)
    game_log = game_log.rename(columns={"OPP": "OPP_ABBR"})

    return {
        "player":        player["full_name"],
        "team":          player_team_abbr,
        "team_name":     player_team_info.get("team_name", player_team_abbr),
        "team_off_rank": player_team_info.get("off_rank"),
        "team_def_rank": player_team_info.get("def_rank"),
        "season":        CURRENT_SEASON,
        "games_used":    len(df),
        "proj_min":      round(proj_min, 1),
        "next_opponent": (
            {
                "abbr":     next_game["opp_abbr"],
                "home":     bool(next_game["home"]),
                "def_rank": team_stats.get(next_game["opp_abbr"], {}).get("def_rank"),
                "off_rank": team_stats.get(next_game["opp_abbr"], {}).get("off_rank"),
            } if next_game else None
        ),
        "predictions":   predictions,
        "game_log":      game_log.to_dict(orient="records"),
    }
