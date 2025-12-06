from dataclasses import dataclass
from typing import Final, List


class GameConfig:
    """Configuration parameters for the game"""

    HEIGHT: Final[int] = 6
    WIDTH: Final[int] = 7
    SEARCH_ORDER: Final[List[int]] = [3, 2, 4, 1, 5, 0, 6]  # Center-out search


@dataclass
class SearchConfig:
    """Configuration parameters for the search"""

    # Iterative deepening settings
    MAX_DEPTH: Final[int] = 30  # Maximum search depth
    TIME_LIMIT: Final[float] = 3.0  # Seconds per move
    MIN_DEPTH: Final[int] = 4  # Always search at least this deep

    # Phase-based evaluation settings
    MODEL_ONLY_PHASE: Final[int] = 20  # Pure model evaluation (fast) for most of game
    HYBRID_PHASE_END: Final[int] = 28  # Hybrid phase for late-mid game
    MODEL_WEIGHT: Final[float] = 0.6  # Weight for model in hybrid phase
    NUM_ROLLOUTS: Final[int] = 8  # Fewer rollouts = deeper search

    # Legacy fixed depth (used as fallback)
    NEGAMAX_DEPTH: Final[int] = 6
