# Breakout GA Optimizer + Out-Of-Sample Backtest

Genetic-algorithm parameter search wrapped around your **v2 breakout engine**, with a
proper in-sample / out-of-sample split so you can see whether a parameter set holds up
or was just curve-fit.

It calls your exact v2 backtest (same fills, commission, `round_tick`), so whatever the
GA finds transfers 1:1 to the sim you already trust.

## Run

```bash
python breakout_ga_oos.py "path/to/YourData.txt"
```

CSV format is the same one v2 reads: `Date, Time, Open, High, Low, Last, Volume, NumberOfTrades, BidVolume, AskVolume`.

## What it does

1. Loads your bars.
2. Splits chronologically: first 70% = **in-sample** (optimize), last 30% = **out-of-sample** (test). The cut is snapped to a day boundary so no trade straddles it.
3. Runs a GA on the in-sample slice only, evolving the parameters.
4. Runs the single best genome **fresh on the untouched out-of-sample slice**.
5. Prints an IS-vs-OOS table (the gap is your overfitting tell), plots a combined equity curve with the split marked, and writes the OOS trade list to CSV.

## Genes it tunes (all in `GENE_BOUNDS`)

| gene | v2 hard-coded value | meaning |
|------|--------------------|---------|
| `entry_ticks` | 3 | breakout distance above prev close |
| `exit_trigger_ticks` | 4 | how far below prev close the exit arms |
| `exit_offset_ticks` | 1 | gap between trigger and the limit fill (0 = limit sits exactly at the trigger) |

Set `OPTIMIZE_SESSION = True` to also evolve the session start/end.

## The two knobs that matter most

```python
FITNESS_METRIC = 'sharpe'   # 'sharpe' | 'net' | 'profit_factor' | 'calmar'
FILL_MODE      = 'stop'     # 'stop' = faithful to v2 (pessimistic)
                            # 'limit' = LE model, fill floored at the limit (slippage capped)
```

Run it once with `FILL_MODE='stop'` (your current worst-case reality) and once with
`'limit'` (the world your LE .cpp exit lives in). The net gap between the two is exactly
what the limit exit is buying you.

`TICK`, `POINT_VALUE`, `TRADE_SIZE`, `COMMISSION_PER_UNIT`, session times, `IS_FRACTION`,
and all GA sizes (`POP_SIZE`, `GENERATIONS`, etc.) are at the top of the file.

## Reading the output

- **IS holds, OOS holds** -> the parameters generalise; trade them.
- **IS great, OOS falls apart** -> curve-fit; widen bounds, lengthen data, or use a more
  robust fitness (Sharpe/Calmar over raw net).
- **OOS zero trades** -> the genome is too tight for the test window.
