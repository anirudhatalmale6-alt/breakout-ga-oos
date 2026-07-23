#!/usr/bin/env python3
"""
Genetic-Algorithm Optimizer + Out-Of-Sample Backtest
=====================================================
Wraps the EXACT v2 breakout engine (Trailing Stop Previous Close v2) so any
parameter set the GA finds transfers 1:1 to the sim you already trust.

What it does
------------
  1. Loads your bar data (same CSV format as v2: Date,Time,Open,High,Low,Last,
     Volume,NumberOfTrades,BidVolume,AskVolume).
  2. Splits chronologically into IN-SAMPLE (default first 70%) and
     OUT-OF-SAMPLE (last 30%), snapped to a day boundary so no single trade
     straddles the cut.
  3. Runs a genetic algorithm on the IN-SAMPLE slice ONLY, evolving:
        - entry trigger ticks      (v2 hard-coded value = 3)
        - exit trigger ticks       (v2 hard-coded value = 4)
        - exit offset ticks        (v2: fill was trigger-1 tick => offset = 1)
        - (optional) session start/end minutes
  4. Takes the winning genome and runs it FRESH on the untouched OOS slice.
     The IS-vs-OOS gap is your overfitting tell.
  5. Prints an IS-vs-OOS metrics table, plots a combined equity curve with the
     split marked, and writes the OOS trade list to CSV.

Fill model
----------
  FILL_MODE = 'stop'  -> exactly your v2 fill: min(Open, trigger-offset). This is
                         pessimistic (stop-style) and is the faithful default.
  FILL_MODE = 'limit' -> LE model: fill capped at the limit (never worse than
                         trigger-offset), only "misses" on a straight blow-through
                         where Open gaps past it (that bar becomes a market exit at
                         Open). Use this to see the world your LE .cpp exit lives in.

Run
---
  python breakout_ga_oos.py "path/to/YourData.txt"
  (or edit BAR_FILE below)
"""
import csv, sys, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ===================== GLOBAL CONFIGURATION =====================
TICK                = 0.01        # 0.01 for stocks/ETFs; set to your instrument's tick
POINT_VALUE         = 1.0         # $ value of a 1.00 point move (per 1 unit)
TRADE_SIZE          = 100
COMMISSION_PER_UNIT = 0.0035      # per side
INITIAL_EQUITY      = 10_000

# Fixed session (used when session genes are OFF)
START_TIME_SEC = 8 * 3600 + 30 * 60   # 08:30
END_TIME_SEC   = 15 * 3600            # 15:00

BAR_FILE  = r"C:/Users/Administrator/Desktop/Stock Data/CL P&F 15-5.txt"
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))   # writes next to this script

# ---- Out-of-sample split ----
IS_FRACTION = 0.70                # first 70% = in-sample (optimize), last 30% = OOS (test)

# ---- Fill model ----
FILL_MODE = 'stop'                # 'stop' (faithful to v2) or 'limit' (LE model)

# ---- What the GA is allowed to tune ----
# (low, high) inclusive integer-tick bounds. Set OPTIMIZE_SESSION=True to also
# evolve the session window.
GENE_BOUNDS = {
    'entry_ticks':        (1, 12),
    'exit_trigger_ticks': (1, 12),
    'exit_offset_ticks':  (0, 6),    # 0 = limit sits exactly at trigger
}
OPTIMIZE_SESSION = False
SESSION_BOUNDS = {
    'start_min': (8 * 60,  10 * 60),   # minutes-of-day
    'end_min':   (13 * 60, 15 * 60),
}

# ---- GA hyper-parameters ----
POP_SIZE        = 40
GENERATIONS     = 25
TOURNAMENT_K    = 3
CROSSOVER_RATE  = 0.7
MUTATION_RATE   = 0.25            # per-gene chance to mutate
ELITE_COUNT     = 2
MIN_TRADES      = 30             # genomes with fewer IS trades are penalised
RANDOM_SEED     = 42

# ---- Fitness metric: 'sharpe' | 'net' | 'profit_factor' | 'calmar' ----
FITNESS_METRIC  = 'sharpe'
# ===============================================================


# ----------------------------- data ----------------------------
def parse_time_secs(ts):
    ts = ts.strip().split('.')[0]
    p = ts.split(':')
    return int(p[0]) * 3600 + int(p[1]) * 60 + (int(p[2]) if len(p) > 2 else 0)


def date_to_int(ds):
    ds = ds.strip()
    p = ds.split('-') if '-' in ds else ds.split('/')
    if int(p[0]) > 1000:
        return int(p[0]) * 10000 + int(p[1]) * 100 + int(p[2])
    return int(p[2]) * 10000 + int(p[0]) * 100 + int(p[1])


def round_tick(v):
    return round(v / TICK) * TICK


def load_bars(fn):
    print(f"Loading bar data from {fn} ...")
    dates, times = [], []
    O, H, L, C, V, BidV, AskV = [], [], [], [], [], [], []
    with open(fn, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for r in reader:
            r = {k.strip(): v.strip() for k, v in r.items()}
            dates.append(r['Date'])
            times.append(r['Time'])
            O.append(float(r['Open']))
            H.append(float(r['High']))
            L.append(float(r['Low']))
            C.append(float(r.get('Last', r.get('Close', '0'))))
            V.append(float(r.get('Volume', '0')))
            BidV.append(float(r.get('BidVolume', '0')))
            AskV.append(float(r.get('AskVolume', '0')))
    n = len(O)
    print(f"  Loaded {n} bars, {dates[0]} -> {dates[-1]}")
    return {
        'dates': dates, 'times': times,
        'O': np.array(O), 'H': np.array(H), 'L': np.array(L), 'C': np.array(C),
        'V': np.array(V), 'BidV': np.array(BidV), 'AskV': np.array(AskV),
        'date_int': np.array([date_to_int(d) for d in dates]),
        'time_sec': np.array([parse_time_secs(t) for t in times]),
    }


def slice_data(d, lo, hi):
    """Return a shallow view of bars [lo:hi]."""
    return {
        'dates': d['dates'][lo:hi], 'times': d['times'][lo:hi],
        'O': d['O'][lo:hi], 'H': d['H'][lo:hi], 'L': d['L'][lo:hi], 'C': d['C'][lo:hi],
        'V': d['V'][lo:hi], 'BidV': d['BidV'][lo:hi], 'AskV': d['AskV'][lo:hi],
        'date_int': d['date_int'][lo:hi], 'time_sec': d['time_sec'][lo:hi],
    }


def split_index(d, frac):
    """Bar index of the IS/OOS cut, snapped forward to the next new-day boundary."""
    raw = int(len(d['C']) * frac)
    di = d['date_int']
    cut = raw
    while cut < len(di) and di[cut] == di[raw]:
        cut += 1
    return cut


# ------------------------- the engine --------------------------
# This is your v2 run_backtest, with the 3 hard-coded tick numbers (and the
# session window) lifted out into `params`. Fill logic is byte-identical to v2
# under FILL_MODE='stop'.
def run_backtest(d, params):
    C, O, L = d['C'], d['O'], d['L']
    dates, times = d['dates'], d['times']
    time_secs = d['time_sec']
    n = len(C)

    entry_t   = params['entry_ticks']
    exit_trig = params['exit_trigger_ticks']
    exit_off  = params['exit_offset_ticks']
    start_sec = params.get('start_sec', START_TIME_SEC)
    end_sec   = params.get('end_sec', END_TIME_SEC)

    time_ok = (time_secs >= start_sec) & (time_secs < end_sec)

    pos = 0
    equity = INITIAL_EQUITY
    entry_idx = 0
    entry_price = 0.0
    trades = []
    equity_curve = [equity, equity]

    for i in range(2, n):
        current_time = time_secs[i]

        if pos == 0:
            if time_ok[i] and H_cross(d, i, entry_t):
                entry_idx = i
                entry_price = round_tick(C[i-1] + entry_t * TICK)
                pos = 1
            equity_curve.append(equity)
        else:
            is_eod_forced = current_time >= end_sec
            trigger_level = C[i-1] - exit_trig * TICK
            limit_level   = C[i-1] - (exit_trig + exit_off) * TICK
            is_exit_hit = L[i] <= trigger_level

            exit_type = None
            if is_eod_forced:
                exit_price = round_tick(C[i])
                exit_type = 'EOD_Forced'
                pos = 0
            elif is_exit_hit:
                if FILL_MODE == 'limit':
                    # LE / stop-limit model: fill where a stop would (min of Open and
                    # the trigger), but FLOORED at the limit -> you never fill worse
                    # than trigger-offset. That floor is the slippage cap your .cpp
                    # LE exit buys you. (A straight gap below the limit rests unfilled
                    # in live trading and the .cpp time-backstop markets you out; here
                    # it is capped at the limit, so treat OOS 'limit' net as the
                    # optimistic bound and 'stop' net as the pessimistic bound.)
                    exit_price = round_tick(max(min(O[i], trigger_level), limit_level))
                else:
                    # v2 faithful (stop-style, pessimistic): always the worse price
                    exit_price = round_tick(min(O[i], limit_level))
                exit_type = 'Exit_Trigger'
                pos = 0

            if pos == 0:
                _commit(trades, dates, times, entry_idx, i, entry_price, exit_price, exit_type)
                equity += trades[-1]['pnl']
            equity_curve.append(equity)

    if pos == 1:  # close open inventory at data end
        exit_price = round_tick(C[-1])
        _commit(trades, dates, times, entry_idx, n-1, entry_price, exit_price, 'Data_End')
        equity += trades[-1]['pnl']
        equity_curve[-1] = equity

    return trades, np.array(equity_curve)


def H_cross(d, i, entry_t):
    # v2 entry: previous-bar setup + this bar's high crosses target
    return d['H'][i] > d['C'][i-1] + entry_t * TICK and d['L'][i] != d['H'][i-1]


def _commit(trades, dates, times, entry_idx, i, entry_price, exit_price, exit_type):
    gross = (exit_price - entry_price) * POINT_VALUE * TRADE_SIZE
    comm  = 2 * (COMMISSION_PER_UNIT * TRADE_SIZE)
    net   = gross - comm
    trades.append({
        'entry_date': dates[entry_idx], 'entry_time': times[entry_idx],
        'exit_date': dates[i], 'exit_time': times[i],
        'entry_price': entry_price, 'exit_price': exit_price,
        'gross_pnl': gross, 'commission': comm, 'pnl': net,
        'points': exit_price - entry_price,
        'pnl_per_share': net / TRADE_SIZE,
        'exit_type': exit_type,
    })


# ------------------------- metrics -----------------------------
def compute_metrics(trades, equity_curve):
    if not trades:
        return {'trades': 0, 'net': 0.0, 'profit_factor': 0.0, 'sharpe': 0.0,
                'calmar': 0.0, 'max_dd': 0.0, 'win_pct': 0.0, 'expectancy': 0.0,
                'final_equity': float(equity_curve[-1]) if len(equity_curve) else INITIAL_EQUITY}
    pnls = np.array([t['pnl'] for t in trades])
    winners = pnls[pnls > 0]
    losers  = pnls[pnls <= 0]

    daily = {}
    for t in trades:
        daily[t['exit_date']] = daily.get(t['exit_date'], 0.0) + t['pnl']
    dv = list(daily.values())
    if len(dv) > 1 and np.std(dv, ddof=1) != 0:
        sharpe = (np.mean(dv) / np.std(dv, ddof=1)) * np.sqrt(252)
    else:
        sharpe = 0.0

    peak = np.maximum.accumulate(equity_curve)
    max_dd = float((equity_curve - peak).min())
    net = float(pnls.sum())
    pf = float(abs(winners.sum() / losers.sum())) if losers.sum() != 0 else float('inf')
    calmar = net / abs(max_dd) if max_dd != 0 else (float('inf') if net > 0 else 0.0)

    return {
        'trades': len(trades),
        'net': net,
        'profit_factor': pf,
        'sharpe': float(sharpe),
        'calmar': float(calmar),
        'max_dd': max_dd,
        'win_pct': 100.0 * len(winners) / len(trades),
        'expectancy': float(pnls.mean()),
        'final_equity': float(equity_curve[-1]),
    }


def fitness(metrics):
    """Higher = better. Genomes below MIN_TRADES are pushed to the bottom."""
    if metrics['trades'] < MIN_TRADES:
        return -1e12 + metrics['trades']       # still rank by trade count so GA climbs toward validity
    m = FITNESS_METRIC
    if m == 'net':
        return metrics['net']
    if m == 'profit_factor':
        pf = metrics['profit_factor']
        return 1e6 if pf == float('inf') else pf
    if m == 'calmar':
        c = metrics['calmar']
        return 1e6 if c == float('inf') else c
    # default: sharpe
    return metrics['sharpe']


# --------------------------- GA --------------------------------
def gene_spec():
    spec = dict(GENE_BOUNDS)
    if OPTIMIZE_SESSION:
        spec['start_min'] = SESSION_BOUNDS['start_min']
        spec['end_min']   = SESSION_BOUNDS['end_min']
    return spec


def random_genome(rng, spec):
    return {k: int(rng.integers(lo, hi + 1)) for k, (lo, hi) in spec.items()}


def genome_to_params(g):
    p = {
        'entry_ticks':        g['entry_ticks'],
        'exit_trigger_ticks': g['exit_trigger_ticks'],
        'exit_offset_ticks':  g['exit_offset_ticks'],
    }
    if 'start_min' in g:
        p['start_sec'] = g['start_min'] * 60
        p['end_sec']   = g['end_min'] * 60
    return p


def evaluate(g, is_data, cache):
    key = tuple(sorted(g.items()))
    if key in cache:
        return cache[key]
    trades, eq = run_backtest(is_data, genome_to_params(g))
    m = compute_metrics(trades, eq)
    f = fitness(m)
    cache[key] = (f, m)
    return f, m


def tournament(pop, scores, rng, k):
    idx = rng.integers(0, len(pop), size=k)
    best = idx[0]
    for j in idx[1:]:
        if scores[j] > scores[best]:
            best = j
    return dict(pop[best])


def crossover(a, b, rng):
    if rng.random() > CROSSOVER_RATE:
        return dict(a)
    child = {}
    for k in a:
        child[k] = a[k] if rng.random() < 0.5 else b[k]
    return child


def mutate(g, rng, spec):
    for k, (lo, hi) in spec.items():
        if rng.random() < MUTATION_RATE:
            # gaussian step, clamped
            step = int(round(rng.normal(0, max(1, (hi - lo) * 0.15))))
            g[k] = int(min(hi, max(lo, g[k] + step)))
    # keep session window sane if evolved
    if 'start_min' in g and 'end_min' in g and g['end_min'] <= g['start_min'] + 30:
        g['end_min'] = min(SESSION_BOUNDS['end_min'][1], g['start_min'] + 30)
    return g


def run_ga(is_data):
    rng = np.random.default_rng(RANDOM_SEED)
    spec = gene_spec()
    cache = {}
    pop = [random_genome(rng, spec) for _ in range(POP_SIZE)]

    best_g, best_f, best_m = None, -np.inf, None
    print(f"\nRunning GA: pop={POP_SIZE}, generations={GENERATIONS}, "
          f"fitness='{FITNESS_METRIC}', fill='{FILL_MODE}'")
    print(f"{'Gen':>4} {'BestFit':>12} {'Trades':>7} {'Net$':>12} {'PF':>6} {'Sharpe':>7} {'MaxDD$':>12}")

    for gen in range(GENERATIONS):
        scored = [evaluate(g, is_data, cache) for g in pop]
        scores = [s[0] for s in scored]
        order = np.argsort(scores)[::-1]

        # track global best
        top = order[0]
        if scores[top] > best_f:
            best_f, best_m, best_g = scores[top], scored[top][1], dict(pop[top])

        print(f"{gen:>4} {best_f:>12.3f} {best_m['trades']:>7} {best_m['net']:>12.2f} "
              f"{best_m['profit_factor']:>6.2f} {best_m['sharpe']:>7.2f} {best_m['max_dd']:>12.2f}")

        # next generation: elitism + tournament breeding
        new_pop = [dict(pop[order[e]]) for e in range(ELITE_COUNT)]
        while len(new_pop) < POP_SIZE:
            pa = tournament(pop, scores, rng, TOURNAMENT_K)
            pb = tournament(pop, scores, rng, TOURNAMENT_K)
            child = mutate(crossover(pa, pb, rng), rng, spec)
            new_pop.append(child)
        pop = new_pop

    return best_g, best_m


# ------------------------- reporting ---------------------------
def metrics_table(title, m):
    lines = [
        f"  {'Trades':<16}{m['trades']}",
        f"  {'Net P&L':<16}${m['net']:,.2f}",
        f"  {'Win %':<16}{m['win_pct']:.1f}%",
        f"  {'Profit factor':<16}{m['profit_factor']:.2f}",
        f"  {'Sharpe':<16}{m['sharpe']:.2f}",
        f"  {'Calmar':<16}{m['calmar']:.2f}",
        f"  {'Max DD':<16}${m['max_dd']:,.2f}",
        f"  {'Expectancy':<16}${m['expectancy']:,.2f}/trade",
        f"  {'Final equity':<16}${m['final_equity']:,.2f}",
    ]
    return f"{title}\n" + "\n".join(lines)


def side_by_side(is_m, oos_m):
    rows = [
        ('Trades',        f"{is_m['trades']}",              f"{oos_m['trades']}"),
        ('Net P&L',       f"${is_m['net']:,.0f}",           f"${oos_m['net']:,.0f}"),
        ('Win %',         f"{is_m['win_pct']:.1f}%",        f"{oos_m['win_pct']:.1f}%"),
        ('Profit factor', f"{is_m['profit_factor']:.2f}",   f"{oos_m['profit_factor']:.2f}"),
        ('Sharpe',        f"{is_m['sharpe']:.2f}",          f"{oos_m['sharpe']:.2f}"),
        ('Calmar',        f"{is_m['calmar']:.2f}",          f"{oos_m['calmar']:.2f}"),
        ('Max DD',        f"${is_m['max_dd']:,.0f}",        f"${oos_m['max_dd']:,.0f}"),
        ('Expectancy',    f"${is_m['expectancy']:,.2f}",    f"${oos_m['expectancy']:,.2f}"),
    ]
    out = [f"  {'Metric':<16}{'IN-SAMPLE':>16}{'OUT-OF-SAMPLE':>18}",
           f"  {'-'*16}{'-'*16:>16}{'-'*18:>18}"]
    for name, a, b in rows:
        out.append(f"  {name:<16}{a:>16}{b:>18}")
    return "\n".join(out)


def plot_combined(is_eq, oos_eq, split_bar, out_path):
    fig, ax = plt.subplots(figsize=(13, 6))
    x_is  = np.arange(len(is_eq))
    x_oos = np.arange(len(is_eq), len(is_eq) + len(oos_eq))
    ax.plot(x_is,  is_eq,  color='navy',  lw=0.9, label='In-sample (optimized)')
    ax.plot(x_oos, oos_eq, color='darkorange', lw=0.9, label='Out-of-sample (test)')
    ax.axvline(len(is_eq), color='gray', ls='--', alpha=0.7)
    ax.axhline(INITIAL_EQUITY, color='gray', ls=':', alpha=0.4)
    ax.set_title('Equity Curve — IS optimize / OOS test (best GA genome)')
    ax.set_xlabel('Bar'); ax.set_ylabel('Equity ($)')
    ax.legend(loc='upper left'); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()
    print(f"\nEquity curve saved: {out_path}")


def save_trades(trades, out_path):
    if not trades:
        return
    with open(out_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(trades[0].keys()))
        w.writeheader()
        for t in trades:
            w.writerow(t)
    print(f"OOS trade list saved: {out_path}")


# --------------------------- main ------------------------------
def main():
    bar_file = sys.argv[1] if len(sys.argv) > 1 else BAR_FILE
    if not os.path.exists(bar_file):
        print(f"Error: data file not found: {bar_file}")
        return

    d = load_bars(bar_file)
    cut = split_index(d, IS_FRACTION)
    is_data  = slice_data(d, 0, cut)
    oos_data = slice_data(d, cut, len(d['C']))
    print(f"Split at bar {cut}: IS={len(is_data['C'])} bars "
          f"({is_data['dates'][0]}->{is_data['dates'][-1]}), "
          f"OOS={len(oos_data['C'])} bars "
          f"({oos_data['dates'][0]}->{oos_data['dates'][-1]})")

    # ---- optimize on IS ----
    best_g, best_is_m = run_ga(is_data)

    # ---- re-run best genome on IS (full metrics) and on untouched OOS ----
    is_trades,  is_eq  = run_backtest(is_data,  genome_to_params(best_g))
    oos_trades, oos_eq = run_backtest(oos_data, genome_to_params(best_g))
    is_m  = compute_metrics(is_trades,  is_eq)
    oos_m = compute_metrics(oos_trades, oos_eq)

    print("\n" + "=" * 62)
    print("BEST GENOME (optimized on in-sample only)")
    print("=" * 62)
    print(f"  entry_ticks          = {best_g['entry_ticks']}")
    print(f"  exit_trigger_ticks   = {best_g['exit_trigger_ticks']}")
    print(f"  exit_offset_ticks    = {best_g['exit_offset_ticks']}   "
          f"(limit fill = trigger - {best_g['exit_offset_ticks']} ticks)")
    if 'start_min' in best_g:
        print(f"  session              = {best_g['start_min']//60:02d}:{best_g['start_min']%60:02d}"
              f" -> {best_g['end_min']//60:02d}:{best_g['end_min']%60:02d}")

    print("\n" + "=" * 62)
    print("IN-SAMPLE  vs  OUT-OF-SAMPLE   (the gap is your overfit tell)")
    print("=" * 62)
    print(side_by_side(is_m, oos_m))

    # degradation flag
    print("\n" + "-" * 62)
    if oos_m['trades'] == 0:
        print("WARNING: best genome produced ZERO trades out-of-sample.")
    else:
        deg = "" if is_m['net'] == 0 else f"{100*(1 - oos_m['net']/is_m['net']):.0f}% lower net"
        verdict = "HOLDS UP" if (oos_m['net'] > 0 and oos_m['sharpe'] > 0) else "DID NOT HOLD"
        print(f"OOS verdict: {verdict}"
              + (f"  (OOS {deg} than IS on a like-for-like slice)" if deg else ""))
    print("-" * 62)

    plot_combined(is_eq, oos_eq, len(is_eq),
                  os.path.join(OUTPUT_DIR, 'ga_oos_equity.png'))
    save_trades(oos_trades, os.path.join(OUTPUT_DIR, 'ga_oos_trades.csv'))


if __name__ == '__main__':
    main()
