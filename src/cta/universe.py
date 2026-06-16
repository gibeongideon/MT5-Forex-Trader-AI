"""CTA universe — alias → Yahoo ticker, asset class, and per-unit-turnover cost (bps).

Daily data from Yahoo Finance (free, ~2000→present). Cost is a conservative fixed
bps-per-unit-traded (daily CTA cost is small vs daily moves; no bid-ask in daily data).
The download writes a `spread` column = cost_bps/1e4 * close (price units), so with
pip=1.0 the pnl cost reduces to turnover * cost_bps/1e4 — transparent and unit-checked.
"""
UNIVERSE = {
    # FX majors (USD)
    "EURUSD": dict(ticker="EURUSD=X", asset_class="FX_USD",   cost_bps=1.0),
    "GBPUSD": dict(ticker="GBPUSD=X", asset_class="FX_USD",   cost_bps=1.0),
    "USDJPY": dict(ticker="JPY=X",    asset_class="FX_USD",   cost_bps=1.0),
    "AUDUSD": dict(ticker="AUDUSD=X", asset_class="FX_USD",   cost_bps=1.5),
    "USDCHF": dict(ticker="CHF=X",    asset_class="FX_USD",   cost_bps=1.5),
    "USDCAD": dict(ticker="CAD=X",    asset_class="FX_USD",   cost_bps=1.5),
    "NZDUSD": dict(ticker="NZDUSD=X", asset_class="FX_USD",   cost_bps=2.0),
    # FX crosses
    "EURGBP": dict(ticker="EURGBP=X", asset_class="FX_CROSS", cost_bps=2.0),
    "EURJPY": dict(ticker="EURJPY=X", asset_class="FX_CROSS", cost_bps=2.0),
    "GBPJPY": dict(ticker="GBPJPY=X", asset_class="FX_CROSS", cost_bps=2.5),
    "AUDJPY": dict(ticker="AUDJPY=X", asset_class="FX_CROSS", cost_bps=2.5),
    "EURCHF": dict(ticker="EURCHF=X", asset_class="FX_CROSS", cost_bps=2.0),
    "EURAUD": dict(ticker="EURAUD=X", asset_class="FX_CROSS", cost_bps=2.5),
    # Metals (futures)
    "GOLD":   dict(ticker="GC=F",     asset_class="METAL",    cost_bps=2.0),
    "SILVER": dict(ticker="SI=F",     asset_class="METAL",    cost_bps=3.0),
    # Energy (futures)
    "WTI":    dict(ticker="CL=F",     asset_class="ENERGY",   cost_bps=3.0),
    "BRENT":  dict(ticker="BZ=F",     asset_class="ENERGY",   cost_bps=3.0),
    # Equity indices
    "SPX":    dict(ticker="^GSPC",    asset_class="EQ_INDEX", cost_bps=1.0),
    "NDX":    dict(ticker="^NDX",     asset_class="EQ_INDEX", cost_bps=1.5),
    "DJI":    dict(ticker="^DJI",     asset_class="EQ_INDEX", cost_bps=1.5),
    "DAX":    dict(ticker="^GDAXI",   asset_class="EQ_INDEX", cost_bps=2.0),
    "FTSE":   dict(ticker="^FTSE",    asset_class="EQ_INDEX", cost_bps=2.0),
    "NIKKEI": dict(ticker="^N225",    asset_class="EQ_INDEX", cost_bps=2.0),
    "STOXX":  dict(ticker="^STOXX50E", asset_class="EQ_INDEX", cost_bps=2.0),
    "ASX":    dict(ticker="^AXJO",    asset_class="EQ_INDEX", cost_bps=2.5),
    # --- Lever #2 expansion: rates / more commodities / ags / crypto / crosses ---
    # Rates futures (new diversifying class — bonds trend well)
    "UST10Y": dict(ticker="ZN=F", asset_class="RATES",  cost_bps=1.0),
    "UST30Y": dict(ticker="ZB=F", asset_class="RATES",  cost_bps=1.5),
    "UST5Y":  dict(ticker="ZF=F", asset_class="RATES",  cost_bps=1.0),
    "UST2Y":  dict(ticker="ZT=F", asset_class="RATES",  cost_bps=1.0),
    # More metals
    "COPPER": dict(ticker="HG=F", asset_class="METAL",  cost_bps=3.0),
    "PLAT":   dict(ticker="PL=F", asset_class="METAL",  cost_bps=4.0),
    "PALL":   dict(ticker="PA=F", asset_class="METAL",  cost_bps=5.0),
    # More energy
    "NATGAS": dict(ticker="NG=F", asset_class="ENERGY", cost_bps=4.0),
    "HEATOIL":dict(ticker="HO=F", asset_class="ENERGY", cost_bps=4.0),
    "GASOIL": dict(ticker="RB=F", asset_class="ENERGY", cost_bps=4.0),
    # Agriculture (new class)
    "CORN":   dict(ticker="ZC=F", asset_class="AG",     cost_bps=4.0),
    "WHEAT":  dict(ticker="ZW=F", asset_class="AG",     cost_bps=4.0),
    "SOY":    dict(ticker="ZS=F", asset_class="AG",     cost_bps=4.0),
    "COFFEE": dict(ticker="KC=F", asset_class="AG",     cost_bps=5.0),
    "SUGAR":  dict(ticker="SB=F", asset_class="AG",     cost_bps=5.0),
    "COTTON": dict(ticker="CT=F", asset_class="AG",     cost_bps=5.0),
    # Crypto (new class; short history)
    "BTC":    dict(ticker="BTC-USD", asset_class="CRYPTO", cost_bps=8.0),
    "ETH":    dict(ticker="ETH-USD", asset_class="CRYPTO", cost_bps=10.0),
    # More FX crosses
    "NZDJPY": dict(ticker="NZDJPY=X", asset_class="FX_CROSS", cost_bps=3.0),
    "CADJPY": dict(ticker="CADJPY=X", asset_class="FX_CROSS", cost_bps=3.0),
    "AUDNZD": dict(ticker="AUDNZD=X", asset_class="FX_CROSS", cost_bps=3.0),
    "GBPCHF": dict(ticker="GBPCHF=X", asset_class="FX_CROSS", cost_bps=3.0),
    "GBPAUD": dict(ticker="GBPAUD=X", asset_class="FX_CROSS", cost_bps=3.0),
}
ASSET_CLASSES = sorted({v["asset_class"] for v in UNIVERSE.values()})

# FX pair → (base, quote) currency for the carry sleeve: carry = sign(rate_base - rate_quote)
FX_PAIRS = {
    "EURUSD": ("EUR", "USD"), "GBPUSD": ("GBP", "USD"), "USDJPY": ("USD", "JPY"),
    "AUDUSD": ("AUD", "USD"), "USDCHF": ("USD", "CHF"), "USDCAD": ("USD", "CAD"),
    "NZDUSD": ("NZD", "USD"), "EURGBP": ("EUR", "GBP"), "EURJPY": ("EUR", "JPY"),
    "GBPJPY": ("GBP", "JPY"), "AUDJPY": ("AUD", "JPY"), "EURCHF": ("EUR", "CHF"),
    "EURAUD": ("EUR", "AUD"),
    "NZDJPY": ("NZD", "JPY"), "CADJPY": ("CAD", "JPY"), "AUDNZD": ("AUD", "NZD"),
    "GBPCHF": ("GBP", "CHF"), "GBPAUD": ("GBP", "AUD"),
}
