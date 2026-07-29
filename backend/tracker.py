import json
import os
from datetime import date

LOG_FILE = os.path.join(os.path.dirname(__file__), "predictions_log.json")


def _load() -> list:
    if not os.path.exists(LOG_FILE):
        return []
    try:
        with open(LOG_FILE) as f:
            return json.load(f)
    except Exception:
        return []


def _save(log: list):
    with open(LOG_FILE, "w") as f:
        json.dump(log, f, indent=2)


def log_prediction(player: str, player_id, league: str, lines: dict, predictions: dict,
                    injury_status: str = None, game_date: str = None):
    if not lines:
        return
    game_date = game_date or str(date.today())
    log = _load()
    # Skip duplicate unresolved entry for same player + game date
    if any(e["player"] == player and e["date"] == game_date and not e["resolved"] for e in log):
        return
    entry = {
        "player":         player,
        "player_id":      str(player_id),
        "league":         league,
        "date":           game_date,
        "lines":          {k: float(v) for k, v in lines.items()},
        "predicted":      {k: round(float(predictions[k]["prediction"]), 1)
                           for k in predictions if k in lines},
        "actual":         {},
        "resolved":       False,
        "injury_status":  injury_status,
    }
    log.append(entry)
    _save(log)


def load_log() -> list:
    return _load()


def save_log(log: list):
    _save(log)


def mark_resolved(entry: dict, actual: dict):
    entry["actual"]   = actual
    entry["resolved"] = True


def mark_excluded(entry: dict):
    entry["excluded"] = True
    entry["resolved"] = True  # won't retry


def compute_stats(log: list, league: str = None) -> dict:
    resolved = [
        e for e in log
        if e.get("resolved") and not e.get("excluded")
        and (league is None or e.get("league") == league)
    ]
    total, correct = 0, 0
    by_stat = {
        "PTS": {"correct": 0, "total": 0},
        "AST": {"correct": 0, "total": 0},
        "REB": {"correct": 0, "total": 0},
    }

    for entry in resolved:
        early_exit = entry.get("early_exit", False)
        for stat in entry.get("lines", {}):
            pred = entry["predicted"].get(stat)
            act  = entry["actual"].get(stat)
            line = entry["lines"].get(stat)
            if pred is None or act is None or line is None:
                continue
            if stat not in by_stat:
                continue
            if pred == line:  # push — skip; prediction offers no directional signal
                continue
            if early_exit and pred > line and act < line:
                # Player left the game injured before reaching the line —
                # the "over" never got a fair chance to hit, so void it
                # rather than count it a miss. An "under" that already
                # cleared the line is graded normally below (it's already
                # locked in — more playing time could only push it up).
                continue
            is_correct = (pred > line) == (float(act) > line)
            total += 1
            by_stat[stat]["total"] += 1
            if is_correct:
                correct += 1
                by_stat[stat]["correct"] += 1

    return {
        "total":          total,
        "correct":        correct,
        "pct":            round(correct / total * 100, 1) if total else None,
        "resolved_count": len(resolved),
        "pending_count":  len(log) - len(resolved),
        "by_stat": {
            stat: {
                **v,
                "pct": round(v["correct"] / v["total"] * 100, 1) if v["total"] else None,
            }
            for stat, v in by_stat.items()
        },
    }
