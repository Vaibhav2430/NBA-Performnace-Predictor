"""Backing logic for the site chatbot's three canned questions:

  1. reasoning behind a given player's projections   -> build_reasoning()
  2. which players have the best hit rate            -> player_hit_rates()[:n]
  3. which players have the worst hit rate           -> player_hit_rates()[-n:]

Everything here is deterministic and template-based - no LLM call.
"""
from statistics import mean

GRADED_STATS = ("PTS", "AST", "REB")

# Known name splits in predictions_log.json (see project memory
# "data-player-name-aliases"). Map every variant's lower-cased form to the
# canonical display name so a single player's picks aren't counted twice.
_ALIASES = {
    "megan dileo":  "Megan Gustafson",
    "jackie young":  "Jackie Young",
}


def _canonical(raw_name: str) -> str:
    return _ALIASES.get(raw_name.lower().strip(), raw_name)


def _ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


# ── Q2 / Q3: per-player hit rate ──────────────────────────────────────────
def player_hit_rates(log: list, league: str | None = None, min_picks: int = 5) -> list[dict]:
    """Group resolved predictions by player and grade each stat over/under the
    line, using the same rules as tracker.compute_stats (skip pushes, void an
    'over' the player never got a fair shot at after an early injury exit).

    Returns a list of {player, correct, total, pct} sorted best -> worst,
    excluding players with fewer than `min_picks` graded picks (tiny samples
    produce meaningless 0% / 100% rates).
    """
    groups: dict[str, dict] = {}

    for e in log:
        if not e.get("resolved") or e.get("excluded"):
            continue
        if league and e.get("league") != league:
            continue

        display = _canonical(e.get("player", ""))
        g = groups.setdefault(display.lower(), {"player": display, "correct": 0, "total": 0})

        early_exit = e.get("early_exit", False)
        for stat in e.get("lines", {}):
            if stat not in GRADED_STATS:
                continue
            pred = e.get("predicted", {}).get(stat)
            act  = e.get("actual", {}).get(stat)
            line = e.get("lines", {}).get(stat)
            if pred is None or act is None or line is None:
                continue
            if pred == line:                       # push - no directional signal
                continue
            if early_exit and pred > line and act < line:
                continue                           # over voided by injury exit

            g["total"] += 1
            if (pred > line) == (float(act) > line):
                g["correct"] += 1

    rows = [
        {**g, "pct": round(g["correct"] / g["total"] * 100, 1)}
        for g in groups.values()
        if g["total"] >= min_picks
    ]
    rows.sort(key=lambda r: (r["pct"], r["total"]), reverse=True)
    return rows


def leaderboard(log: list, league: str | None = None, limit: int = 10, min_picks: int = 5) -> dict:
    rows = player_hit_rates(log, league=league, min_picks=min_picks)
    return {
        "league":         league or "ALL",
        "min_picks":      min_picks,
        "ranked_players": len(rows),
        "best":  rows[:limit],
        "worst": list(reversed(rows[-limit:])) if rows else [],
    }


# ── Q1: reasoning behind a player's projections ───────────────────────────
def _vs_avg(pred: float, avg: float | None, label: str) -> str | None:
    if avg is None:
        return None
    d = round(pred - avg, 1)
    if abs(d) < 0.6:
        return f"right around his {label} ({avg})"
    return f"{abs(d)} {'above' if d > 0 else 'below'} his {label} ({avg})"


def _minutes_trend(game_log: list) -> str | None:
    mins = [g["MIN"] for g in game_log if g.get("MIN") is not None]
    recent, prior = mins[:5], mins[5:10]
    if len(recent) < 3 or len(prior) < 3:
        return None
    r, p = mean(recent), mean(prior)
    if r - p >= 3:
        return f"Minutes trending up (~{r:.0f} over the last 5 vs ~{p:.0f} before) — more opportunity."
    if p - r >= 3:
        return f"Minutes trending down (~{r:.0f} over the last 5 vs ~{p:.0f} before) — fewer touches."
    return None


def build_reasoning(result: dict, league: str = "NBA") -> dict:
    """Turn an already-computed /predict (or /wnba/predict) payload into a
    plain-language explanation. Expects the keys that endpoint returns:
    predictions, game_log, and optionally lines, injury, teammate_boosts,
    return_dampening.
    """
    player      = result.get("player", "This player")
    preds       = result.get("predictions", {})
    game_log    = result.get("game_log", []) or []
    lines       = result.get("lines", {}) or {}
    injury      = result.get("injury")
    boosts      = result.get("teammate_boosts") or []
    dampening   = result.get("return_dampening") or []

    blocks: list[str] = [f"Here's what's driving the {player} projections for the next game:"]

    leans: list[str] = []
    for stat in GRADED_STATS:
        p = preds.get(stat)
        if not p:
            continue
        pred = float(p["prediction"])
        parts = [f"{stat}: model projects {pred}"]
        rel = [
            _vs_avg(pred, p.get("season_avg"), "season average"),
            _vs_avg(pred, p.get("last5_avg"),  "last-5 average"),
        ]
        rel = [r for r in rel if r]
        if rel:
            parts.append(" — " + ", ".join(rel))

        line = lines.get(stat)
        if line is not None:
            diff = round(pred - line, 1)
            if abs(diff) < 0.3:
                parts.append(f". Sits on the posted line ({line}).")
            else:
                side = "OVER" if diff > 0 else "UNDER"
                parts.append(f". Line is {line} → leans {side} by {abs(diff)}.")
                leans.append(f"{stat} {side}")
        else:
            parts.append(".")

        if p.get("teammate_boost"):
            parts.append(f" (Includes a +{p['teammate_boost']}% teammate-out boost.)")
        if p.get("return_dampen"):
            parts.append(f" (Includes a -{p['return_dampen']}% taper as a teammate returns.)")

        blocks.append("".join(parts))

    context: list[str] = []
    opp = result.get("next_opponent")
    if opp and opp.get("abbr"):
        where = "home vs" if opp.get("home") else "away at"
        rank  = f", {_ordinal(opp['def_rank'])} in defensive rating" if opp.get("def_rank") else ""
        context.append(f"Next up: {where} {opp['abbr']}{rank}.")
    mt = _minutes_trend(game_log)
    if mt:
        context.append(mt)
    if result.get("proj_min"):
        context.append(f"Projected ~{result['proj_min']} minutes, which caps the ceiling on each stat.")
    if result.get("questionable_gate"):
        g = result["questionable_gate"]
        context.append(f"Scaled down {abs(round((g['mult']-1)*100))}% because he's listed {g['status']}.")
    if boosts:
        names = ", ".join(b["player"] for b in boosts)
        context.append(f"Usage boost applied with {names} out — their touches redistribute toward {player}.")
    if dampening:
        names = ", ".join(d["player"] for d in dampening)
        context.append(f"Projection tapered because {names} recently returned and is reclaiming usage.")
    if injury and injury.get("status"):
        extra = f" ({injury['injury_type']})" if injury.get("injury_type") else ""
        context.append(f"Injury report: {injury['status']}{extra} — factored into the number.")
    if context:
        blocks.append("Context: " + " ".join(context))

    if leans:
        blocks.append("Net read vs the posted lines: " + ", ".join(leans) + ".")
    elif not lines:
        blocks.append("No prop lines are posted for this player right now, so there's no over/under lean — just the model projection.")

    return {"player": player, "league": league, "blocks": blocks}
