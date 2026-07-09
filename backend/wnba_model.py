import time
from datetime import date
import requests
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
import defense_by_position as dbp


def _current_wnba_season() -> str:
    today = date.today()
    return str(today.year) if 5 <= today.month <= 9 else str(today.year + 1)


def _prev_wnba_season() -> str:
    return str(int(_current_wnba_season()) - 1)

STATS        = ["PTS", "AST", "REB"]
FEATURE_COLS = [
    "pts_l5", "pts_l10",
    "ast_l5", "ast_l10",
    "reb_l5", "reb_l10",
    "min_l5", "home", "days_rest",
    "opp_def_pts", "opp_off_pts",
    "opp_def_pts_vs_pos", "opp_def_ast_vs_pos", "opp_def_reb_vs_pos",
    "pts_home_avg", "pts_away_avg",
    "ast_home_avg", "ast_away_avg",
    "reb_home_avg", "reb_away_avg",
]

ESPN_BASE    = "https://site.api.espn.com/apis/site/v2/sports/basketball/wnba"
ESPN_WEB     = "https://site.web.api.espn.com/apis/common/v3/sports/basketball/wnba"

_player_cache: dict | None = None   # name (lower) -> {id, full_name, team}


def _build_player_cache() -> dict:
    global _player_cache
    if _player_cache is not None:
        return _player_cache
    teams_r = requests.get(f"{ESPN_BASE}/teams", timeout=10)
    teams   = teams_r.json()["sports"][0]["leagues"][0]["teams"]
    cache: dict = {}
    for t in teams:
        tid   = t["team"]["id"]
        tname = t["team"]["displayName"]
        tabbr = t["team"].get("abbreviation", "")
        time.sleep(0.2)
        r = requests.get(f"{ESPN_BASE}/teams/{tid}/roster", timeout=10)
        if r.status_code == 200:
            for p in r.json().get("athletes", []):
                key      = p["displayName"].lower()
                pos_abbr = p.get("position", {}).get("abbreviation", "G")
                cache[key] = {
                    "id":        p["id"],
                    "full_name": p["displayName"],
                    "team":      tname,
                    "team_abbr": tabbr,
                    "position":  pos_abbr,
                }
                dbp.register_wnba_player(p["id"], p["displayName"], pos_abbr)
    _player_cache = cache
    return cache


def search_players(query: str) -> list[str]:
    cache = _build_player_cache()
    q     = query.lower().strip()
    all_names = sorted(p["full_name"] for p in cache.values())
    if not q:
        return all_names
    return [name for name in all_names if q in name.lower()][:15]


def find_player(name: str) -> dict | None:
    cache = _build_player_cache()
    nl    = name.lower().strip()
    if nl in cache:
        return cache[nl]
    matches = [p for p in cache.values() if nl in p["full_name"].lower()]
    return matches[0] if matches else None


def fetch_wnba_team_stats() -> dict:
    """Returns {team_abbr: {team_name, off_pts, def_pts, off_rank, def_rank}}."""
    ESPN_WEB_V2 = "https://site.web.api.espn.com/apis/v2/sports/basketball/wnba"
    r = requests.get(f"{ESPN_WEB_V2}/standings", timeout=10, params={"seasontype": 2})
    r.raise_for_status()
    data = r.json()

    raw: dict = {}
    for conference in data.get("children", []):
        for entry in conference.get("standings", {}).get("entries", []):
            team  = entry.get("team", {})
            abbr  = team.get("abbreviation", "")
            name  = team.get("displayName", abbr)
            stats = {s["name"]: s.get("value") for s in entry.get("stats", [])}
            off_pts = stats.get("avgPointsFor")
            def_pts = stats.get("avgPointsAgainst")
            if abbr and off_pts is not None and def_pts is not None:
                raw[abbr] = {"team_name": name, "off_pts": float(off_pts), "def_pts": float(def_pts)}

    if not raw:
        return {}

    # Higher off pts = better offense (rank 1); lower def pts allowed = better defense (rank 1)
    off_sorted = sorted(raw.keys(), key=lambda a: raw[a]["off_pts"], reverse=True)
    def_sorted = sorted(raw.keys(), key=lambda a: raw[a]["def_pts"])
    for rank, abbr in enumerate(off_sorted, 1):
        raw[abbr]["off_rank"] = rank
    for rank, abbr in enumerate(def_sorted, 1):
        raw[abbr]["def_rank"] = rank
    return raw


def fetch_game_log(player_id: str, season: str = None) -> pd.DataFrame:
    params = {"season": season} if season else {}
    r = requests.get(f"{ESPN_WEB}/athletes/{player_id}/gamelog", params=params, timeout=15)
    r.raise_for_status()
    data = r.json()

    labels    = data["labels"]           # ['MIN','PTS','REB','AST',...]
    events_meta = data["events"]         # {eventId -> {date, homeAway, opponent, ...}}

    try:
        pts_idx = labels.index("PTS")
        ast_idx = labels.index("AST")
        reb_idx = labels.index("REB")
        min_idx = labels.index("MIN")
    except ValueError as e:
        raise ValueError(f"Missing stat column: {e}")

    rows = []
    for season_type in data.get("seasonTypes", []):
        if "Regular" not in season_type.get("displayName", ""):
            continue
        for category in season_type.get("categories", []):
            for event in category.get("events", []):
                eid   = event["eventId"]
                stats = event.get("stats", [])
                if not stats:
                    continue
                meta  = events_meta.get(eid, {})

                # Parse date
                raw_date = meta.get("gameDate", meta.get("date", ""))
                try:
                    game_date = pd.to_datetime(raw_date[:10]) if raw_date else pd.NaT
                except Exception:
                    game_date = pd.NaT

                # Home/away
                home_away = meta.get("homeAway", "")
                home      = 1 if home_away == "home" else 0

                # Opponent — lives at meta["opponent"], not in competitors
                opp_obj  = meta.get("opponent", {})
                opp      = opp_obj.get("abbreviation", "")
                opp_name = opp_obj.get("displayName", opp)
                opp_logo = opp_obj.get("logo", "")

                def _num(v):
                    try:
                        return float(str(v).split("-")[0])
                    except Exception:
                        return 0.0

                rows.append({
                    "GAME_DATE": game_date,
                    "MATCHUP":   opp,
                    "OPP_NAME":  opp_name,
                    "OPP_LOGO":  opp_logo,
                    "HOME":      home,
                    "MIN":  _num(stats[min_idx] if min_idx < len(stats) else 0),
                    "PTS":  _num(stats[pts_idx] if pts_idx < len(stats) else 0),
                    "AST":  _num(stats[ast_idx] if ast_idx < len(stats) else 0),
                    "REB":  _num(stats[reb_idx] if reb_idx < len(stats) else 0),
                })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["GAME_DATE"]).sort_values("GAME_DATE").reset_index(drop=True)
    df["DAYS_REST"] = df["GAME_DATE"].diff().dt.days.fillna(2).clip(0, 10)
    return df


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
        cur  = df.iloc[i]
        rows.append({
            "pts_l5":               hist["PTS"].tail(5).mean(),
            "pts_l10":              hist["PTS"].tail(10).mean(),
            "ast_l5":               hist["AST"].tail(5).mean(),
            "ast_l10":              hist["AST"].tail(10).mean(),
            "reb_l5":               hist["REB"].tail(5).mean(),
            "reb_l10":              hist["REB"].tail(10).mean(),
            "min_l5":               hist["MIN"].tail(5).mean(),
            "home":                 cur["HOME"],
            "days_rest":            cur["DAYS_REST"],
            "opp_def_pts":          cur["OPP_DEF_PTS"],
            "opp_off_pts":          cur["OPP_OFF_PTS"],
            "opp_def_pts_vs_pos":   cur["OPP_DEF_PTS_VS_POS"],
            "opp_def_ast_vs_pos":   cur["OPP_DEF_AST_VS_POS"],
            "opp_def_reb_vs_pos":   cur["OPP_DEF_REB_VS_POS"],
            **_home_away_avgs(hist),
            "PTS": cur["PTS"],
            "AST": cur["AST"],
            "REB": cur["REB"],
        })
    return pd.DataFrame(rows)


def _next_features(
    df: pd.DataFrame,
    avg_def_pts: float,
    avg_off_pts: float,
    avg_pos: dict,
) -> np.ndarray:
    feats = {
        "pts_l5":               df["PTS"].tail(5).mean(),
        "pts_l10":              df["PTS"].tail(10).mean(),
        "ast_l5":               df["AST"].tail(5).mean(),
        "ast_l10":              df["AST"].tail(10).mean(),
        "reb_l5":               df["REB"].tail(5).mean(),
        "reb_l10":              df["REB"].tail(10).mean(),
        "min_l5":               df["MIN"].tail(5).mean(),
        "home":                 0.5,
        "days_rest":            2.0,
        "opp_def_pts":          avg_def_pts,
        "opp_off_pts":          avg_off_pts,
        "opp_def_pts_vs_pos":   avg_pos["pts"],
        "opp_def_ast_vs_pos":   avg_pos["ast"],
        "opp_def_reb_vs_pos":   avg_pos["reb"],
        **_home_away_avgs(df),
    }
    return np.array([[feats[c] for c in FEATURE_COLS]])


def predict(player_name: str) -> dict:
    player = find_player(player_name)
    if not player:
        raise ValueError(f"WNBA player '{player_name}' not found.")

    df = fetch_game_log(player["id"])
    current_season_n = len(df)
    if current_season_n < 12:
        df_prev = fetch_game_log(player["id"], season=_prev_wnba_season())
        if not df_prev.empty:
            df = pd.concat([df_prev, df], ignore_index=True)
            df = df.dropna(subset=["GAME_DATE"]).sort_values("GAME_DATE").reset_index(drop=True)
            df["DAYS_REST"] = df["GAME_DATE"].diff().dt.days.fillna(2).clip(0, 10)
    if len(df) < 12:
        raise ValueError(f"Not enough game data for {player['full_name']}.")

    team_stats  = fetch_wnba_team_stats()
    avg_def_pts = float(np.mean([v["def_pts"] for v in team_stats.values()])) if team_stats else 80.0
    avg_off_pts = float(np.mean([v["off_pts"] for v in team_stats.values()])) if team_stats else 80.0

    # Positional defense: pts/ast/reb each team allows to this player's position
    player_pos = dbp.get_wnba_player_pos(player["full_name"])
    pos_def    = dbp.fetch_wnba_def_by_pos(_current_wnba_season())
    avg_pos: dict = {}
    for stat in ("pts", "ast", "reb"):
        vals = [pos_def[a][f"{player_pos}_{stat}"] for a in pos_def if pos_def.get(a, {}).get(f"{player_pos}_{stat}", 0) > 0]
        avg_pos[stat] = float(np.mean(vals)) if vals else avg_def_pts

    df["OPP_DEF_PTS"]        = df["MATCHUP"].apply(lambda x: team_stats.get(x, {}).get("def_pts",  avg_def_pts))
    df["OPP_OFF_PTS"]        = df["MATCHUP"].apply(lambda x: team_stats.get(x, {}).get("off_pts",  avg_off_pts))
    df["OPP_OFF_RANK"]       = df["MATCHUP"].apply(lambda x: team_stats.get(x, {}).get("off_rank"))
    df["OPP_DEF_RANK"]       = df["MATCHUP"].apply(lambda x: team_stats.get(x, {}).get("def_rank"))
    df["OPP_DEF_PTS_VS_POS"] = df["MATCHUP"].apply(lambda x: dbp.get_def_vs_pos(x, player_pos, "pts", pos_def, avg_pos["pts"]))
    df["OPP_DEF_AST_VS_POS"] = df["MATCHUP"].apply(lambda x: dbp.get_def_vs_pos(x, player_pos, "ast", pos_def, avg_pos["ast"]))
    df["OPP_DEF_REB_VS_POS"] = df["MATCHUP"].apply(lambda x: dbp.get_def_vs_pos(x, player_pos, "reb", pos_def, avg_pos["reb"]))
    # OPP_NAME and OPP_LOGO already populated per-row from ESPN opponent object

    feature_df = _build_features(df)
    X          = feature_df[FEATURE_COLS].values
    X_next     = _next_features(df, avg_def_pts, avg_off_pts, avg_pos)
    n          = len(X)
    weights    = np.ones(n)
    weights[-10:] = 2.25

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
        model.fit(X, y, sample_weight=weights)
        pred       = max(0.0, float(model.predict(X_next)[0]))
        curr_games = df[stat].tail(current_season_n) if current_season_n > 0 else df[stat]
        season_avg = float(curr_games.mean())
        std        = float(curr_games.std())
        pred       = min(pred, season_avg + std)

        predictions[stat] = {
            "prediction": round(pred, 1),
            "floor":      round(max(0.0, pred - std), 1),
            "ceiling":    round(pred + std, 1),
            "last5_avg":  round(float(df[stat].tail(5).mean()), 1),
            "season_avg": round(season_avg, 1),
        }

    game_log = (
        df.tail(20)[["GAME_DATE", "MATCHUP", "OPP_NAME", "OPP_LOGO", "HOME", "MIN", "PTS", "AST", "REB", "OPP_OFF_RANK", "OPP_DEF_RANK"]]
        .copy().iloc[::-1].reset_index(drop=True)
    )
    game_log["GAME_DATE"] = game_log["GAME_DATE"].dt.strftime("%Y-%m-%d")
    game_log["WL"]        = ""
    game_log["MIN"]       = game_log["MIN"].round(0).astype(int)
    game_log = game_log.rename(columns={"MATCHUP": "OPP_ABBR"})

    player_team_info = team_stats.get(player.get("team_abbr", ""), {})
    return {
        "player":        player["full_name"],
        "team":          player["team"],
        "team_abbr":     player.get("team_abbr", ""),
        "team_off_rank": player_team_info.get("off_rank"),
        "team_def_rank": player_team_info.get("def_rank"),
        "league":        "WNBA",
        "season":        _current_wnba_season(),
        "games_used":    len(df),
        "predictions":   predictions,
        "game_log":      game_log[["GAME_DATE", "OPP_ABBR", "OPP_NAME", "OPP_LOGO", "WL", "MIN", "PTS", "AST", "REB", "OPP_OFF_RANK", "OPP_DEF_RANK"]].to_dict(orient="records"),
    }


def games_today() -> list:
    try:
        r     = requests.get(f"{ESPN_BASE}/scoreboard", timeout=10)
        data  = r.json()
        result = []
        for event in data.get("events", []):
            comp       = event["competitions"][0]
            status     = comp["status"]["type"]
            state      = status["state"]          # pre / in / post
            status_code = 1 if state == "pre" else (2 if state == "in" else 3)

            if state == "pre":
                # e.g. "6/19 - 7:30 PM EDT" → "7:30 PM EDT"
                short = status.get("shortDetail", "")
                status_txt = short.split(" - ", 1)[-1] if " - " in short else short or "Scheduled"
            elif state == "in":
                period     = comp["status"].get("period", "")
                clock      = comp["status"].get("displayClock", "")
                status_txt = f"Q{period} {clock}"
            else:
                status_txt = status.get("description", "Final")

            competitors = comp["competitors"]
            home = next((c for c in competitors if c["homeAway"] == "home"), competitors[0])
            away = next((c for c in competitors if c["homeAway"] == "away"), competitors[1])

            def rec(c):
                rec_str = ""
                for r_item in c.get("records", []):
                    if r_item.get("type") == "total":
                        rec_str = r_item.get("summary", "")
                        break
                wins, losses = 0, 0
                if "-" in rec_str:
                    parts = rec_str.split("-")
                    try:
                        wins   = int(parts[0])
                        losses = int(parts[1])
                    except Exception:
                        pass
                return {
                    "tricode": c["team"].get("abbreviation", ""),
                    "name":    c["team"].get("displayName", ""),
                    "score":   int(c.get("score", 0) or 0),
                    "wins":    wins,
                    "losses":  losses,
                }

            result.append({
                "gameId":     event["id"],
                "status":     status_txt,
                "statusCode": status_code,
                "home": rec(home),
                "away": rec(away),
            })
        return result
    except Exception:
        return []
