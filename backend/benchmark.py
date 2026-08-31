"""Offline sanity check: does the XGBoost model actually beat a dumb
recent-form baseline on the predictions we've already resolved?

Baseline  = 0.6 * last-5 average + 0.4 * last-10 average (as of the game date)
Blend     = 0.5 * XGB + 0.5 * baseline

For every resolved, non-excluded entry in predictions_log.json it re-derives
the baseline from the player's game log (games strictly before the predicted
date), grades all three methods over/under the posted line with the same rules
as tracker.compute_stats, and prints hit rate + mean absolute error.

Run:  python benchmark.py [--league NBA|WNBA] [--limit N]
Slow (one or two nba_api calls per player); --limit N does a quick sample.
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

import model as nba
import wnba_model as wnba

LOG_FILE = os.path.join(os.path.dirname(__file__), "predictions_log.json")
STATS = ("PTS", "AST", "REB")
METHODS = ("xgb", "baseline", "blend")


def _player_log(league: str, player_id: str) -> pd.DataFrame:
    """Current + previous season game log for one player, sorted by date."""
    if league == "NBA":
        cur  = nba.fetch_game_log(int(player_id))
        prev = nba.fetch_game_log(int(player_id), season=nba._prev_nba_season())
    else:
        cur  = wnba.fetch_game_log(str(player_id))
        prev = wnba.fetch_game_log(str(player_id), season=wnba._prev_wnba_season())

    frames = [f for f in (prev, cur) if f is not None and not f.empty]
    if not frames:
        return pd.DataFrame()
    full = pd.concat(frames, ignore_index=True)
    return full.dropna(subset=["GAME_DATE"]).sort_values("GAME_DATE").reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--league", choices=["NBA", "WNBA"], default=None)
    ap.add_argument("--limit", type=int, default=None, help="cap number of players (quick run)")
    args = ap.parse_args()

    log = json.load(open(LOG_FILE))
    entries = [
        e for e in log
        if e.get("resolved") and not e.get("excluded")
        and e.get("actual") and e.get("lines") and e.get("predicted")
        and (args.league is None or e.get("league") == args.league)
    ]

    by_player: dict = defaultdict(list)
    for e in entries:
        by_player[(e["league"], str(e["player_id"]), e["player"])].append(e)

    players = list(by_player.items())
    if args.limit:
        players = players[: args.limit]

    tally  = {m: {s: [0, 0] for s in STATS} for m in METHODS}   # method -> stat -> [correct, total]
    abserr = {m: [0.0, 0] for m in METHODS}                     # method -> [sum |err|, n]
    skipped = 0

    for i, ((league, pid, pname), elist) in enumerate(players, 1):
        print(f"[{i}/{len(players)}] {league} {pname}", file=sys.stderr)
        try:
            full = _player_log(league, pid)
        except Exception as ex:
            print(f"    skip ({ex})", file=sys.stderr)
            full = pd.DataFrame()
        if full.empty:
            skipped += 1
            continue

        for e in elist:
            gd   = pd.Timestamp(e["date"])
            hist = full[full["GAME_DATE"] < gd]
            if len(hist) < 5:
                continue
            early_exit = e.get("early_exit", False)

            for stat in e["lines"]:
                if stat not in STATS:
                    continue
                line = e["lines"].get(stat)
                act  = e["actual"].get(stat)
                xgb  = e["predicted"].get(stat)
                if line is None or act is None or xgb is None:
                    continue

                l5    = hist[stat].tail(5).mean()
                l10   = hist[stat].tail(10).mean()
                base  = 0.6 * l5 + 0.4 * l10
                blend = 0.5 * xgb + 0.5 * base

                for method, pred in (("xgb", xgb), ("baseline", base), ("blend", blend)):
                    abserr[method][0] += abs(float(act) - pred)
                    abserr[method][1] += 1
                    if pred == line:                                   # push
                        continue
                    if early_exit and pred > line and act < line:      # voided over
                        continue
                    tally[method][stat][1] += 1
                    tally[method][stat][0] += int((pred > line) == (float(act) > line))

    def cell(c: int, t: int) -> str:
        return f"{100 * c / t:5.1f}% ({c}/{t})" if t else "     n/a"

    print(f"\nBenchmark - {args.league or 'ALL'}   ({len(players)} players, {skipped} skipped)\n")
    print(f"{'method':9} {'PTS':>16} {'AST':>16} {'REB':>16} {'OVERALL':>18} {'MAE':>7}")
    for m in METHODS:
        oc = ot = 0
        cells = []
        for s in STATS:
            c, t = tally[m][s]
            oc += c
            ot += t
            cells.append(cell(c, t))
        mae = abserr[m][0] / abserr[m][1] if abserr[m][1] else float("nan")
        print(f"{m:9} {cells[0]:>16} {cells[1]:>16} {cells[2]:>16} {cell(oc, ot):>18} {mae:7.2f}")
    print("\nIf 'xgb' isn't clearly ahead of 'baseline', the model isn't earning its complexity.")


if __name__ == "__main__":
    main()
