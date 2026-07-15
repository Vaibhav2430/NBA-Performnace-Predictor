import time
import threading
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from nba_api.stats.static import players as nba_players
from nba_api.live.nba.endpoints import scoreboard as live_scoreboard
from model import predict, find_player as nba_find_player, fetch_game_log as nba_fetch_game_log
import wnba_model
import odds
import tracker
import injuries
import team_context

def _seed_if_needed():
    """Seed both leagues on startup, but only if we haven't seeded today.
    Stagger requests so we don't hit PrizePicks simultaneously."""
    from datetime import date
    today = str(date.today())
    log = tracker.load_log()
    already_today = any(e["date"] == today for e in log)
    if already_today:
        return
    _seed_all("WNBA")
    time.sleep(5)  # stagger to avoid simultaneous requests
    _seed_all("NBA")

def _apply_teammate_boosts(result: dict, league: str) -> None:
    """Mutate result['predictions'] in-place to account for Out teammates.

    Formula: each Out teammate's scoring gets 50% redistributed to remaining
    players. Boost pct = out_ppg * 0.5 / (team_ppg - out_ppg). Capped at 25%.
    AST gets 60% of the PTS boost; REB gets 20%.
    """
    team_abbr = result.get("team") or result.get("team_abbr")
    if not team_abbr:
        return

    out_players = injuries.get_out_players_for_team(team_abbr, league)
    if not out_players:
        return

    team_ppg = team_context.get_team_ppg(team_abbr, league)
    boosts_applied = []

    total_pts_boost = 0.0
    for op in out_players:
        ppg = team_context.get_player_ppg(op["name"], league)
        if ppg is None or ppg < 8:  # skip low-impact players
            continue
        denominator = max(team_ppg - ppg, 40)  # avoid division edge cases
        boost = (ppg * 0.5) / denominator
        total_pts_boost += boost
        boosts_applied.append({"player": op["name"], "ppg": round(ppg, 1), "boost_pct": round(boost * 100, 1)})

    if not boosts_applied:
        return

    total_pts_boost = min(total_pts_boost, 0.25)  # hard cap at 25%
    total_ast_boost = total_pts_boost * 0.6
    total_reb_boost = total_pts_boost * 0.2

    preds = result.get("predictions", {})
    for stat, multiplier in [("PTS", total_pts_boost), ("AST", total_ast_boost), ("REB", total_reb_boost)]:
        if stat not in preds or multiplier == 0:
            continue
        p = preds[stat]
        orig = float(p["prediction"])
        boosted = round(orig * (1 + multiplier), 1)
        p["prediction"]     = boosted
        p["season_avg"]     = p.get("season_avg", orig)   # keep original for display
        p["last5_avg"]      = p.get("last5_avg", orig)
        p["floor"]          = round(float(p.get("floor", orig * 0.7)) * (1 + multiplier * 0.5), 1)
        p["ceiling"]        = round(float(p.get("ceiling", orig * 1.3)) * (1 + multiplier), 1)
        p["teammate_boost"] = round(multiplier * 100, 1)

    result["teammate_boosts"] = boosts_applied


@asynccontextmanager
async def lifespan(_app):
    threading.Thread(target=_seed_if_needed, daemon=True).start()
    yield


app = FastAPI(title="NBA Stat Predictor API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/predict")
def predict_endpoint(player: str = Query(..., description="Player full name")):
    try:
        result = predict(player)
        lines = odds.get_player_lines(player, "NBA")
        result["lines"] = lines
        injury = injuries.get_player_injury(player, "NBA")
        result["injury"] = injury
        _apply_teammate_boosts(result, "NBA")
        p = nba_find_player(player)
        inj_status = injury["status"] if injury else None
        is_out = inj_status and inj_status.lower() == "out"
        if p and not is_out:
            tracker.log_prediction(player, p["id"], "NBA", lines, result["predictions"], injury_status=inj_status)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


@app.get("/search")
def search_endpoint(q: str = Query(default="")):
    all_players = sorted(p["full_name"] for p in nba_players.get_players() if p["is_active"])
    if not q:
        return all_players[:50]
    q_lower = q.lower()
    matches = [name for name in all_players if q_lower in name.lower()]
    return matches[:15]


@app.get("/wnba/search")
def wnba_search(q: str = Query(default="")):
    return wnba_model.search_players(q)


@app.get("/wnba/predict")
def wnba_predict(player: str = Query(...)):
    try:
        result = wnba_model.predict(player)
        lines = odds.get_player_lines(player, "WNBA")
        result["lines"] = lines
        injury = injuries.get_player_injury(player, "WNBA")
        result["injury"] = injury
        _apply_teammate_boosts(result, "WNBA")
        p = wnba_model.find_player(player)
        inj_status = injury["status"] if injury else None
        is_out = inj_status and inj_status.lower() == "out"
        if p and not is_out:
            tracker.log_prediction(player, p["id"], "WNBA", lines, result["predictions"], injury_status=inj_status)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"WNBA prediction failed: {e}")


@app.get("/wnba/games/today")
def wnba_games_today():
    return wnba_model.games_today()


def _resolve_all():
    """Fetch actual game results and mark predictions resolved. Runs in background."""
    from datetime import date
    log = tracker.load_log()
    today = str(date.today())
    changed = False

    for entry in log:
        if entry.get("resolved"):
            continue
        if entry["date"] >= today:
            continue

        try:
            if entry["league"] == "NBA":
                p = nba_find_player(entry["player"])
                if not p:
                    continue
                df = nba_fetch_game_log(p["id"])
            else:
                p = wnba_model.find_player(entry["player"])
                if not p:
                    continue
                df = wnba_model.fetch_game_log(p["id"])

            if df.empty:
                continue

            pred_date = pd.Timestamp(entry["date"])
            same_day = df[df["GAME_DATE"].dt.date == pred_date.date()]
            if same_day.empty:
                continue

            game = same_day.iloc[0]

            # Exclude if player barely played (injury / DNP-illness)
            inj_status = entry.get("injury_status")
            if inj_status:
                try:
                    raw_min = game["MIN"]
                    min_played = float(str(raw_min).split(":")[0]) if ":" in str(raw_min) else float(raw_min)
                    if min_played < 5:
                        tracker.mark_excluded(entry)
                        changed = True
                        continue
                except Exception:
                    pass

            actual = {
                stat: float(game[stat])
                for stat in entry.get("lines", {})
                if stat in game.index
            }
            if actual:
                tracker.mark_resolved(entry, actual)
                changed = True
        except Exception:
            continue

    if changed:
        tracker.save_log(log)


@app.get("/accuracy")
def accuracy_endpoint(background_tasks: BackgroundTasks, league: str = Query(default=None)):
    background_tasks.add_task(_resolve_all)
    return tracker.compute_stats(tracker.load_log(), league=league)


def _seed_all(league: str):
    from datetime import date
    all_lines = odds.fetch_lines(league)
    if not all_lines:
        return
    today = str(date.today())
    log = tracker.load_log()
    already_logged = {e["player"] for e in log if e["date"] == today}

    for pp_name, lines in all_lines.items():
        if pp_name in already_logged:
            continue
        try:
            if league == "NBA":
                result = predict(pp_name)
            else:
                result = wnba_model.predict(pp_name)

            # Skip players averaging under 15 min
            game_log = result.get("game_log", [])
            if game_log:
                recent_min = sum(g["MIN"] for g in game_log[:5]) / min(5, len(game_log))
                if recent_min < 15:
                    continue

            player_name = result["player"]
            p = nba_find_player(player_name) if league == "NBA" else wnba_model.find_player(player_name)
            if p:
                tracker.log_prediction(player_name, p["id"], league, lines, result["predictions"])
        except Exception:
            continue


@app.post("/accuracy/seed")
def seed_accuracy(background_tasks: BackgroundTasks, league: str = Query(default="WNBA")):
    background_tasks.add_task(_seed_all, league)
    return {"status": "started", "league": league}


@app.get("/games/today")
def games_today():
    try:
        board = live_scoreboard.ScoreBoard()
        data  = board.get_dict()
        games = data.get("scoreboard", {}).get("games", [])
        result = []
        for g in games:
            home = g["homeTeam"]
            away = g["awayTeam"]
            result.append({
                "gameId":     g["gameId"],
                "status":     g["gameStatusText"].strip(),
                "statusCode": g["gameStatus"],   # 1=scheduled 2=live 3=final
                "home": {
                    "tricode": home["teamTricode"],
                    "name":    home["teamCity"] + " " + home["teamName"],
                    "score":   home.get("score", 0) or 0,
                    "wins":    home.get("wins", 0),
                    "losses":  home.get("losses", 0),
                },
                "away": {
                    "tricode": away["teamTricode"],
                    "name":    away["teamCity"] + " " + away["teamName"],
                    "score":   away.get("score", 0) or 0,
                    "wins":    away.get("wins", 0),
                    "losses":  away.get("losses", 0),
                },
            })
        return result
    except Exception:
        return []
