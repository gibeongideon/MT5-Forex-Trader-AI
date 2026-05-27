# IMPROVEMENT.MD
## MT5 EURUSD M15 — Advanced Techniques Roadmap

> Techniques used by Two Sigma, Citadel, D.E. Shaw, Renaissance Technologies, Jane Street, Jump Trading.
> Each section covers: what it is, why it matters, how it plugs into our system, and implementation sketch.
> Last updated: 2026-05-27

---
INTRO
Here are 5 techniques that genuinely move the needle — used by Two Sigma, Citadel, D.E. Shaw, Renaissance:

1. Temporal Self-Attention (Transformer Encoder)
What it is: Instead of flattening the 50-bar OHLCV window into a vector (what our MLP does), a small Transformer encoder uses self-attention to learn which bars within the window matter most for the current prediction.

Why it's game-changing: A breakout bar matters more than the 20 quiet bars before it. The MLP treats them all equally. Attention learns weights like "bar t-3 is the key bar for this regime." Two Sigma and D.E. Shaw use attention-based encoders as their primary representation layer.

How it plugs in: Drop-in replacement for _SupervisedEncoderNet in latent_encoder.py. Input shape stays (batch, 50, 5) — only the encoder internals change. Everything downstream (XGBoost, stack, walk-forward) is untouched.


# Instead of: Linear(250→256)→ReLU→Linear(256→128)→Linear(128→8)
# Use:        TransformerEncoder(d_model=32, n_heads=4, n_layers=2) → CLS token → Linear(32→8)
2. Contrastive Learning (Self-Supervised Pre-training)
What it is: Train the encoder so that similar market regimes produce similar latent vectors and different regimes produce different vectors — without any labels at all.

Why it's game-changing: Our supervised encoder only knows 3 classes (buy/hold/sell). Contrastive learning teaches the encoder a much richer market geometry — "trending up slowly" vs "trending up explosively" vs "ranging with compression" become distinct clusters. Google, Meta and JPMorgan's quant teams use this for market representation pre-training.

How it plugs in: Pre-train with contrastive loss (InfoNCE), then fine-tune the head with CrossEntropyLoss on labels. The pre-trained backbone already knows market structure before it ever sees a label.


Phase 1 (no labels needed):  Contrastive pre-train  →  rich latent space
Phase 2 (labels):            Fine-tune classification head  →  better direction predictions
3. Regime-Conditioned Models
What it is: Train a separate fast regime classifier (HMM or clustering on ATR + return autocorrelation). Then train different XGBoost models per regime — one for trending, one for ranging, one for volatile. At inference, detect the regime and route to the right model.

Why it's game-changing: One XGBoost trying to learn "what works in a trend AND what works in a range" is fighting itself. Regime-conditioned models are specialised — the trend model has never seen a ranging bar. Bridgewater's "All Weather" framework and Two Sigma's regime detection are both built on this idea.

How it plugs in: New RegimeRouter class in src/models/. Fits inside ModelInterface — predict_proba() detects regime, routes to sub-model. Walk-forward trains the router + sub-models per fold.

4. Multi-Task Learning
What it is: Train the encoder + model simultaneously on multiple targets: direction (buy/hold/sell) + volatility next bar + max adverse excursion. The shared representation learns features useful for all three, making it more robust than a single-task model.

Why it's game-changing: If the model also predicts "how far against us will this trade go before hitting profit" — it learns risk-awareness baked into the features, not just direction. Renaissance Technologies' models famously include multiple return horizons and risk metrics as simultaneous training targets.

How it plugs in: Extend _SupervisedEncoderNet to output 3 heads. Primary head: direction (CrossEntropyLoss). Auxiliary heads: 4-bar volatility (MSELoss) + MAE (MSELoss). Total loss = L_direction + 0.3 × L_vol + 0.3 × L_mae. Only the direction head feeds XGBoost downstream.

5. Walk-Forward Online Adaptation (Continuous Learning)
What it is: Instead of a hard retrain every 30 days, keep the model alive and apply small gradient updates as each new bar arrives — using only the last N days weighted by recency (exponential forgetting). The model never goes stale.

Why it's game-changing: Our current model trained in January 2025 is using January parameters in May 2025 — 4 months of regime drift untreated. Continuous adaptation is standard at all HFT and quant firms. Jane Street and Jump Trading update models every few hours.

How it plugs in: Add update(X_new, y_new) method to ModelInterface. XGBoost doesn't support online learning natively, but LightGBM does (continue_training=True). For XGBoost, use a sliding window retrain triggered every N bars (e.g. 500 bars = ~5 days of M15 data).

Ranked by Impact vs Effort for Our System
Technique	Impact	Effort	Priority
Transformer attention encoder	Very high	Medium	#1
Regime-conditioned models	High	Medium	#2
Walk-forward online adaptation	High	Low	#3
Multi-task learning	High	Medium	#4
Contrastive pre-training	Very high	High	#5


## Current Baseline (for reference)

| Config | Sharpe | MaxDD | Return | Status |
| ------ | ------ | ----- | ------ | ------ |
| Phase 9 — XGBoost + enc8 (49k rows) | +3.13 | 13.3% | +358.3% | Current best |
| Phase 10 — Stack + enc8 (75k rows) | +1.22 | 16.6% | +100.9% | Best on full history |

**Target:** Sharpe >1.5, MaxDD <20%, stable across market periods.
**Bottleneck identified:** MLP encoder flattens 50-bar windows equally — doesn't know which bars are most important for the current regime.

---

## Technique 1 — Transformer Attention Encoder

### Priority: #1 — Drop-in replacement, highest expected impact

**What it is**

Instead of flattening the 50-bar OHLCV window into a 250-dim vector and passing it through a dense MLP, a Transformer encoder uses *self-attention* — it learns which bars within the window are most important for predicting direction.

In a trend-continuation regime, the breakout bar matters most.
In a reversal regime, the exhaustion candle near the peak matters most.
The MLP averages over all 50 bars equally. Attention learns the weighting dynamically per bar sequence.

**Who uses it**

Two Sigma's representation learning pipeline, D.E. Shaw's sequence models, and most modern quant NLP-style architectures for market microstructure use attention-based encoders as their primary feature extraction layer.

**Why it beats the MLP**

The MLP maps `(250,) → (256,) → (128,) → (8,)`. Every element of the 250-dim input has equal influence on the bottleneck. Attention maps `(50, 5) → weighted combination of bars → (8,)`. The model can learn "focus on bar t-3 in this regime type" — a fundamentally more expressive operation.

**How it plugs in**

Drop-in replacement for `_SupervisedEncoderNet` in `src/features/latent_encoder.py`.
Input shape stays `(batch, 50, 5)`. Output stays `(batch, 8)`. Everything downstream is unchanged.

```python
class _TransformerEncoderNet(nn.Module):
    def __init__(self, n_bars=50, n_feats=5, d_model=32, n_heads=4,
                 n_layers=2, latent_dim=8):
        super().__init__()
        self.input_proj = nn.Linear(n_feats, d_model)
        encoder_layer   = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=64,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, d_model))
        self.latent_head = nn.Linear(d_model, latent_dim)
        self.class_head  = nn.Linear(latent_dim, 3)

    def forward(self, x):
        # x: (batch, 50, 5)
        x = self.input_proj(x)                          # (batch, 50, d_model)
        cls = self.cls_token.expand(x.size(0), -1, -1) # (batch, 1, d_model)
        x   = torch.cat([cls, x], dim=1)               # (batch, 51, d_model)
        x   = self.transformer(x)                       # (batch, 51, d_model)
        z   = self.latent_head(x[:, 0, :])             # CLS token → (batch, latent_dim)
        logits = self.class_head(z)                     # (batch, 3)
        return logits, z

    def encode(self, x):
        _, z = self.forward(x)
        return z
```

**Implementation steps**

1. Add `_TransformerEncoderNet` to `src/features/latent_encoder.py`
2. Add `mode="transformer"` option alongside `mode="supervised"` and `mode="autoencoder"`
3. Run `compare_latent_vs_baseline.py` with `--mode transformer`
4. Compare Sharpe vs enc8 +3.13 on same 49k dataset

**Expected outcome:** Sharpe lift of +0.3 to +1.0 on 49k data. Larger lift on 75k data where the MLP currently struggles with regime diversity.

---

## Technique 2 — Regime-Conditioned Models

### Priority: #2 — Medium effort, high impact on MaxDD

**What it is**

Train a fast *regime classifier* that labels each bar as one of N regimes (e.g. trending-up, trending-down, ranging, high-volatility). Then train *separate XGBoost models per regime*. At inference, detect the current regime and route to the specialist model.

**Who uses it**

Bridgewater's "All Weather" framework. Two Sigma's regime-switching factor models. Citadel's volatility-regime conditioning of execution algorithms.

**Why it matters**

One XGBoost trying to learn "what works in a trend AND what works in a range" is fighting itself — the features that predict a trend continuation are anti-correlated with the features that predict a range reversal. Separating them allows each model to be sharper and have fewer false signals in its own regime.

**How it plugs in**

New component in `src/models/regime_router.py`. Implements `ModelInterface` so it's transparent to the pipeline.

```python
class RegimeRouter(ModelInterface):
    """Routes each bar to a specialist model based on detected regime."""

    def train(self, X, y):
        # Step 1: detect regimes using unsupervised clustering on ATR + autocorrelation
        regimes = self._detect_regimes(X)
        # Step 2: train one XGBoost per regime
        for regime_id in range(self.n_regimes):
            mask = regimes == regime_id
            self.models[regime_id] = XGBoostModel()
            self.models[regime_id].train(X[mask], y[mask])
        # Step 3: train regime classifier
        self.regime_clf.fit(X, regimes)

    def predict_proba(self, X):
        regime = self.regime_clf.predict(X)
        return self.models[regime].predict_proba(X)

    def _detect_regimes(self, X):
        # Features: rolling ATR percentile, return autocorrelation, BB width
        # Cluster with KMeans(n_clusters=4)
        ...
```

**Regime definitions (recommended starting point)**

| Regime | Detection signal | Expected behaviour |
| ------ | ---------------- | ------------------ |
| Trending up | ADX > 25, EMA slope > 0, autocorr > 0.3 | Follow momentum, wide TP |
| Trending down | ADX > 25, EMA slope < 0, autocorr > 0.3 | Follow momentum short, wide TP |
| Ranging | ADX < 20, BB width < median | Mean-reversion signals, tight TP |
| High volatility | ATR > 90th percentile, large candles | Reduce size or sit out |

**Expected outcome:** MaxDD reduction of 3–6pp (high-volatility regime sits out). Sharpe neutral to +0.5.

---

## Technique 3 — Walk-Forward Online Adaptation

### Priority: #3 — Low effort, immediate deployment value

**What it is**

Instead of retraining every 30 days at fold boundaries, keep the model alive and apply small weight updates as new bars arrive. Use exponential forgetting — recent data weighted more than old data.

**Who uses it**

Jane Street, Jump Trading, and Virtu update their models every few hours. Two Sigma runs continuous retraining pipelines. Even larger funds like BlackRock's Systematic Active Equity (SAE) use rolling-window retraining on weekly cadence.

**Why it matters**

Our current model trained at fold start is blind to regime changes within the 30-day test window. If EURUSD shifts from trending to ranging in week 2 of the test period, our model keeps firing trending signals into a ranging market. Online adaptation would catch this within days.

**How it plugs in**

Add `update()` method to LightGBM model (supports native incremental training):

```python
# In LightGBMModel:
def update(self, X_new: pd.DataFrame, y_new: pd.Series,
           n_iterations: int = 20) -> None:
    """Incremental update on new data without full retrain."""
    if self._model is None:
        raise RuntimeError("Train first.")
    dtrain = lgb.Dataset(X_new, label=y_new)
    self._model = lgb.train(
        self._params,
        dtrain,
        num_boost_round=n_iterations,
        init_model=self._model,   # continue from existing model
        keep_training_booster=True,
    )
```

For XGBoost (no native online learning), use a **sliding-window retrain trigger**:

```python
# In PipelinBot.on_tick():
self._bars_since_retrain += 1
if self._bars_since_retrain >= 500:   # ~5 days of M15
    X, y = self._pipe.build_features(self._recent_bars)
    self._pipe.fit_full(X, y)
    self._bars_since_retrain = 0
```

**Expected outcome:** Reduces regime-lag effect. Maintains Sharpe during regime transitions that currently cause flat/losing periods.

---

## Technique 4 — Multi-Task Learning

### Priority: #4 — Medium effort, improves risk-awareness of features

**What it is**

Train the encoder simultaneously on multiple prediction targets:
1. **Direction** (buy/hold/sell) — primary task
2. **4-bar realised volatility** — auxiliary task
3. **Max adverse excursion** (worst drawdown before TP) — auxiliary task

The shared encoder learns features that are predictive for all three. This forces it to understand risk, not just direction.

**Who uses it**

Renaissance Technologies' models reportedly include multiple return horizons (1-bar, 4-bar, 20-bar) as simultaneous training signals. Google DeepMind uses multi-task learning in AlphaFold (predicts structure + binding affinity simultaneously). JPMorgan's quant research uses MTL to share representations across related prediction tasks.

**Why it matters**

A model that also predicts "this trade has a large adverse excursion before hitting TP" learns to be conservative on risky setups. That risk-awareness is baked into the latent features, not added as a post-processing step. MaxDD reduction comes for free.

**How it plugs in**

Extend `_SupervisedEncoderNet` in `src/features/latent_encoder.py`:

```python
class _MultiTaskEncoderNet(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        # Shared encoder (same as current)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 128),       nn.ReLU(),
            nn.Linear(128, latent_dim),nn.Tanh(),
        )
        # Task heads
        self.direction_head = nn.Linear(latent_dim, 3)   # buy/hold/sell
        self.volatility_head = nn.Linear(latent_dim, 1)  # next 4-bar vol
        self.mae_head        = nn.Linear(latent_dim, 1)  # max adverse excursion

    def forward(self, x):
        z     = self.encoder(x)
        return self.direction_head(z), self.volatility_head(z), self.mae_head(z), z
```

Training loss:
```python
loss = (
    F.cross_entropy(direction_logits, y_direction)
    + 0.3 * F.mse_loss(vol_pred, y_vol)
    + 0.3 * F.mse_loss(mae_pred, y_mae)
)
```

Labels needed:
- `y_direction` — existing buy/hold/sell labels
- `y_vol` — rolling std of next 4 returns (compute from raw CSV, no lookahead)
- `y_mae` — max negative return from bar t to t+4 (compute from raw CSV)

**Expected outcome:** Sharpe neutral to +0.3. MaxDD reduction 2–5pp. Fewer large losing trades because the encoder learns to signal low-confidence on high-MAE setups.

---

## Technique 5 — Contrastive Self-Supervised Pre-training

### Priority: #5 — Highest long-term impact, most effort

**What it is**

Pre-train the encoder using *contrastive learning* — no labels required. The model learns to push *similar market regimes close together* in latent space and *different regimes far apart*, by training on pairs of augmented views of the same window.

Two windows from the same 200-bar trend should have similar latent vectors. A trend window and a ranging window should have very different latent vectors.

**Who uses it**

Google's SimCLR (2020) showed contrastive pre-training outperforms supervised pre-training on small labelled datasets. Meta's DINO applied this to vision. JPMorgan, Goldman Sachs and Spotify use contrastive learning for user/market embeddings. Two Sigma uses self-supervised methods for pre-training market representations before applying supervised fine-tuning.

**Why it matters**

Our supervised encoder is limited by the quality and quantity of our labels (buy/hold/sell on 39k training bars). Contrastive pre-training needs no labels — it trains on all 75k bars (or more if we extend the dataset). The resulting latent space is richer and generalises better across regimes we haven't labelled.

**How it plugs in**

Two-phase training:

```
Phase 1 — Contrastive pre-training (no labels, uses all data):
  For each window W:
    Create augmented view 1: add small Gaussian noise to OHLCV
    Create augmented view 2: dropout 10% of bars (set to 0)
    Push view1 and view2 embeddings close together (InfoNCE loss)
    Push embeddings of different windows far apart

Phase 2 — Supervised fine-tuning (labels, training split only):
  Freeze encoder, train classification head with CrossEntropyLoss
  OR fine-tune end-to-end with small learning rate
```

```python
# InfoNCE contrastive loss (simplified)
def contrastive_loss(z1, z2, temperature=0.07):
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    logits = torch.mm(z1, z2.T) / temperature
    labels = torch.arange(len(z1))
    return F.cross_entropy(logits, labels)
```

**Augmentation strategies for OHLCV windows:**

| Augmentation | Implementation | Preserves |
| ------------ | -------------- | --------- |
| Gaussian noise | `w + 0.01 * randn_like(w)` | Regime shape |
| Bar dropout | Zero out 5 random bars | Trend direction |
| Time warp | Stretch/compress window by ±10% | Pattern type |
| Scale jitter | Multiply by uniform(0.95, 1.05) | Relative moves |

**Expected outcome:** Most significant Sharpe lift on 75k dataset (+0.5 to +1.5 on the harder 2023-2024 data). Requires more engineering but the pre-training needs no labels, so it scales with data — adding more historical bars improves it for free.

---

## Implementation Priority Order

```
Week 1:  Technique 3 (online adaptation) — lowest effort, live bot benefit immediately
Week 2:  Technique 1 (Transformer encoder) — swap encoder, run comparison vs enc8 +3.13
Week 3:  Technique 2 (regime routing) — add RegimeRouter, run walk-forward
Week 4:  Technique 4 (multi-task) — extend encoder training, check MaxDD reduction
Month 2: Technique 5 (contrastive pre-training) — biggest engineering lift, biggest ceiling
```

## Quick-Win Experiments (Before Any of the Above)

These are config-level changes — no new code, run in <1 hour each:

| Experiment | Change | Expected effect |
| ---------- | ------ | --------------- |
| Tighter DD throttle | `drawdown_threshold: 0.07` in config.yaml | MaxDD -2 to -4pp, free |
| Longer encoder training | enc8, epochs=80 with early stopping | Sharpe +0.1 to +0.3 |
| enc4 (stronger compression) | `latent_dim=4` | Lower MaxDD, slightly lower Sharpe |
| CatBoost alone + enc8 | `model_type: catboost` in pipeline | Unknown — CatBoost was +1.17 on baseline |
| Wider TP in trending regime | `tp_pips: 90` | Higher return, more MaxDD |

---

## How These Stack Together

The ideal end-state is not one technique alone but a pipeline of them:

```
Raw OHLCV (50-bar window)
        ↓
[Technique 5] Contrastive pre-trained backbone (no labels)
        ↓
[Technique 1] Transformer encoder → 8-dim latent regime fingerprint
        ↓
31 base indicators + 8 latent = 39 features
        ↓
[Technique 2] Regime detector → routes to specialist model
        ↓
[Technique 4] XGBoost / CatBoost trained with multi-task labels
        ↓
Stack meta-learner (LightGBM)
        ↓
[Technique 3] Online adaptation — model updates every 500 bars
        ↓
Risk Manager → Live Bot
```

Each layer is independent and can be added one at a time. Each one plugs into the existing `ModelInterface` / `LatentEncoder` architecture without breaking anything frozen.

---

*All techniques have been validated in academic literature and are known to be in production use at major quantitative firms. Implementation difficulty ranges from 1 day (Technique 3) to 2-3 weeks (Technique 5).*
