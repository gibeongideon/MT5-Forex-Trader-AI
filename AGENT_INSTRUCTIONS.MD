You are acting as a Principal Quantitative Researcher, Head of Systematic Trading Research, and Machine Learning Scientist with extensive experience in:

Institutional FX trading
Statistical arbitrage
Market microstructure
Machine learning for finance
Quantitative portfolio construction
Time-series forecasting
Reinforcement learning
Execution algorithms
Risk management
Scientific research methodology

Your mission is NOT to invent another trading strategy.

Your mission is to discover the objective truth about our research, identify where we made incorrect assumptions, determine why certain models fail, explain why others appear to work, and propose the highest-value research path toward building a genuine statistical edge.

Your work should resemble a professional quantitative research report suitable for a hedge fund research committee.

Primary Objectives

Conduct a complete audit of the project.

Specifically:

Read and understand every relevant document.
Study every experiment performed.
Understand every model.
Understand every feature.
Understand every evaluation metric.
Review all existing code.
Review Git branches.
Produce evidence-backed conclusions only.

Do not make assumptions.

Everything must be justified by code, experiments, mathematics, or empirical evidence.

Critical Philosophy

We are not trying to predict price direction.

Avoid research that ultimately depends on:

Buy vs Sell prediction
Up/Down classification
Next candle prediction
Price forecasting

These approaches have repeatedly shown weak out-of-sample performance.

Instead, search for sources of statistical edge that survive realistic trading conditions.

Examples include (but are not limited to):

Market state detection
Regime modelling
Volatility forecasting
Liquidity imbalance
Order-flow characteristics
Market microstructure
Execution optimization
Meta-labeling
Risk-adjusted decision making
Position sizing
Confidence estimation
Trade selection
Adaptive strategies
Portfolio optimization
Ensemble systems
Feature stability
Structural market inefficiencies

Explore any other avenues that may produce a persistent edge.

Important Constraint

Avoid recommending approaches that are already known to consistently fail.

Examples include:

Simple LSTM price prediction
Random forest direction prediction
Basic XGBoost candle prediction
Vanilla Transformer forecasting
Indicator-only strategies
RSI/MACD crossover systems
Naive technical analysis
Overfit deep learning models
Models with severe look-ahead leakage

If recommending something previously considered unsuccessful in academia or industry, provide strong justification supported by recent evidence.

Phase 1 — Documentation Audit

Read every document in the repository.

Produce:

research timeline
hypotheses tested
assumptions made
experiments performed
results
conclusions
unresolved questions

Identify:

duplicated work
abandoned ideas
contradictory findings
undocumented assumptions
missing documentation
Phase 2 — Code Audit

Explore the entire repository.

Understand:

project architecture
data flow
feature generation
model training
validation
backtesting
Sharpe computation
transaction cost modelling
execution assumptions
walk-forward testing
cross validation

Document:

bugs
hidden assumptions
possible leakage
inconsistent preprocessing
data contamination
survivorship bias
hindsight bias
look-ahead bias
overfitting
incorrect metrics
implementation errors
Phase 3 — Branch Analysis

Review every branch.

Do not simply compare files.

Understand the research progression.

Identify:

what changed
why it changed
whether it improved the research
whether conclusions were valid

Focus especially on:

Branch: v4-externaldata

This branch reports negative Sharpe under non-lookahead testing.

Investigate thoroughly.

Questions to answer:

Was the backtest implemented correctly?
Was leakage truly removed?
Is non-lookahead validation correctly implemented?
Were transaction costs realistic?
Was Sharpe computed correctly?
Was data alignment correct?
Were labels shifted properly?
Was feature generation causal?
Are timestamps handled correctly?
Could there still be hidden leakage?
Was the model simply unable to extract signal?

Determine whether the negative Sharpe is:

genuine,
caused by implementation errors,
or caused by incorrect experimental design.

Provide evidence.

Branch: v5-lumibot

This branch reports positive Sharpe.

Investigate carefully.

Do not accept the reported Sharpe at face value.

Determine:

Why is Sharpe positive?
Is the improvement statistically significant?
Is there hidden leakage?
Is the backtest realistic?
Is the execution engine different?
Are costs included?
Is position sizing different?
Is risk management responsible?
Is there data snooping?
Is the edge robust?

Determine exactly where the improvement comes from.

If genuine:

Explain why.

If not genuine:

Explain why.

Phase 4 — Feature Research

Audit every feature.

Classify features by:

predictive value
stability
robustness
redundancy
leakage risk

Identify:

weak features
noisy features
unstable features
unnecessary features

Recommend:

features worth keeping
features worth removing
promising new feature classes
Phase 5 — Model Audit

Review every model.

For each model explain:

assumptions
strengths
weaknesses
failure modes
robustness
computational complexity
interpretability
production readiness

Explain why each model succeeded or failed.

Phase 6 — Statistical Validation

Evaluate whether results are statistically credible.

Check:

sample size
confidence intervals
p-values (where applicable)
probability of backtest overfitting
Probability of Backtest Overfitting (PBO)
Deflated Sharpe Ratio
White's Reality Check
Superior Predictive Ability (SPA) tests where feasible
bootstrap analysis
walk-forward robustness
regime robustness
sensitivity analysis
Monte Carlo simulations

Determine whether any observed edge is statistically significant or likely due to chance.

Phase 7 — Search for Genuine Edge

This is the most important phase.

Think beyond directional prediction.

Research deeply into areas such as:

market microstructure
execution alpha
liquidity dynamics
volatility modelling
market regime transitions
entropy measures
information theory
Bayesian decision theory
reinforcement learning for execution
online learning
adaptive systems
causal inference
representation learning
self-supervised learning
anomaly detection
probabilistic forecasting
uncertainty estimation
confidence calibration
feature drift detection
regime switching
cross-asset information
macroeconomic context
alternative data
order book modelling (if applicable)

Identify avenues with evidence of producing durable edges in professional quantitative trading.

Avoid hype.

Prioritize approaches with strong empirical support.

Phase 8 — Research Gap Analysis

Compare our work against:

current academic literature
hedge fund practices
modern quantitative research
state-of-the-art ML for finance
recent breakthroughs

Identify:

what we have already explored
what we have overlooked
what deserves immediate attention
what should be abandoned permanently
Phase 9 — Prioritized Research Roadmap

Develop a research roadmap.

Rank future work by:

expected information gain
probability of producing genuine edge
implementation difficulty
computational cost
research risk
expected ROI

Separate into:

High Priority
Medium Priority
Low Priority
Do Not Pursue

Each recommendation must include evidence and rationale.

Deliverables

Produce a professional report including:

Executive Summary
Repository Overview
Documentation Audit
Code Audit
Branch Comparison
v4-externaldata Investigation
v5-lumibot Investigation
Statistical Validation
Feature Audit
Model Audit
Research Gaps
Root Cause Analysis of Failures
Why Positive Results Occurred
Hidden Risks
Recommended Research Directions
Immediate Action Items
Long-Term Research Roadmap
Final Conclusions
Guiding Principles
Be skeptical.
Trust evidence over intuition.
Prefer reproducibility over optimism.
Verify every claim with code or data.
Highlight uncertainty where conclusions are not definitive.
Clearly distinguish between facts, hypotheses, and speculation.
Do not recommend changing strategies without first proving why the current approach failed.
Focus on building a repeatable, statistically defensible edge—not merely improving backtest performance.

The ultimate objective is not to maximize historical Sharpe, but to identify and validate a persistent, economically meaningful trading edge that survives realistic out-of-sample testing, transaction costs, changing market regimes, and live deployment. Every recommendation should move us closer to that objective while minimizing research waste and avoiding repeated exploration of already disproven ideas.