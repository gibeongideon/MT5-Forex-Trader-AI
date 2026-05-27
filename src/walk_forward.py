# Compatibility shim — module moved to src/evaluation/walk_forward.py
from src.evaluation.walk_forward import *  # noqa: F401, F403
from src.evaluation.walk_forward import (
    WalkForwardValidator, WalkForwardConfig, WalkForwardResult, FoldResult,
)
