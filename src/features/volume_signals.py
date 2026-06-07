"""
Volume anomaly features for Phase 22-A signal experiment.

Hypothesis: tick_volume is present in OHLCV but has never been used beyond ATR
computation. Volume spikes indicate institutional activity and may carry
directional information orthogonal to the 31 base features + enc8 latent.

Three features derived from tick_volume:
    vol_ratio      — current bar volume / 20-bar rolling average (>1 = spike)
    vol_zscore     — z-score: (vol - mean20) / std20
    vol_fast_slow  — 5-bar avg / 20-bar avg (momentum of volume)

All features are backward-looking (no lookahead). FeaturePipeline shifts
everything by 1 bar before it enters the matrix, so these are safe.

Usage (via extra_spec in a comparison script):
    from src.features.volume_signals import volume_signals

    VOLUME_SPEC = [
        (("vol_ratio", "vol_zscore", "vol_fast_slow"), volume_signals, {}),
    ]
    import src.features.feature_pipeline as _fp_module
    _fp_module._OHLCV_FUNS |= {volume_signals}
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def volume_signals(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Compute volume anomaly features from the tick_volume column.

    Parameters
    ----------
    df : full OHLCV DataFrame with a 'tick_volume' column

    Returns
    -------
    vol_ratio      : current / rolling-20-mean  (1.0 = average, >2 = large spike)
    vol_zscore     : (current - mean20) / std20  (signed, ~N(0,1))
    vol_fast_slow  : rolling-5-mean / rolling-20-mean  (fast/slow ratio)
    """
    if "tick_volume" not in df.columns:
        n = len(df)
        zeros = pd.Series(0.0, index=df.index)
        return zeros, zeros, zeros

    vol = df["tick_volume"].astype(float)

    vol_ma20 = vol.rolling(20).mean()
    vol_std20 = vol.rolling(20).std()
    vol_ma5  = vol.rolling(5).mean()

    vol_ratio     = vol / vol_ma20.replace(0.0, np.nan)
    vol_zscore    = (vol - vol_ma20) / vol_std20.replace(0.0, np.nan)
    vol_fast_slow = vol_ma5 / vol_ma20.replace(0.0, np.nan)

    return vol_ratio, vol_zscore, vol_fast_slow
