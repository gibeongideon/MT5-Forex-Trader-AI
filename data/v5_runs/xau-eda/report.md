# XAUUSD H4 Pattern EDA

- bars: 18407  range: 2015-01-01 20:00:00 → 2026-07-07 08:00:00
- engine trades: 1014
- overall 6-bar trend-reversal base rate: 0.017
- trend run-length: {'n_runs': 1648, 'mean_bars': 11.17, 'median_bars': 1, 'p90_bars': 1, 'max_bars': 4157}
- forecast autocorr: {'acf_lag1': 0.999, 'acf_lag6': 0.991, 'acf_lag24': 0.953, 'half_life_bars': None, 'max_lag': 60}
- |return| autocorr (vol clustering): {'acf_lag1': 0.191, 'acf_lag6': 0.226, 'acf_lag24': 0.145, 'half_life_bars': 1, 'max_lag': 60}

## Highest reversal-rate states (top 10)

- conf=low vol=lo session=london_ny n=517 reversal=0.052
- conf=low vol=lo session=asia n=1284 reversal=0.048
- conf=low vol=lo session=london n=1198 reversal=0.047
- conf=low vol=lo session=ny n=726 reversal=0.047
- conf=low vol=mid session=london n=685 reversal=0.038
- conf=low vol=mid session=london_ny n=406 reversal=0.032
- conf=low vol=mid session=asia n=635 reversal=0.028
- conf=low vol=mid session=ny n=416 reversal=0.026
- conf=low vol=hi session=asia n=444 reversal=0.023
- conf=low vol=hi session=london_ny n=284 reversal=0.021

## Trade win-rate map (top 10 by win rate)

- conf=high session=asia n=84 win=0.488 meanR=0.275
- conf=low session=asia n=58 win=0.483 meanR=0.17
- conf=med session=asia n=31 win=0.452 meanR=0.539
- conf=high session=ny n=117 win=0.427 meanR=0.259
- conf=med session=london_ny n=90 win=0.422 meanR=-0.023
- conf=med session=ny n=62 win=0.419 meanR=-0.021
- conf=high session=london_ny n=120 win=0.392 meanR=-0.009
- conf=low session=ny n=67 win=0.388 meanR=-0.021
- conf=low session=london n=81 win=0.37 meanR=-0.026
- conf=high session=london n=129 win=0.357 meanR=-0.031