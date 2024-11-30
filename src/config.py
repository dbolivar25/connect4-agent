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

    NEGAMAX_DEPTH: Final[int] = 3  # Fixed 3-ply search depth
    MODEL_ONLY_PHASE: Final[int] = 9  # Pure model evaluation until move 9
    HYBRID_PHASE_END: Final[int] = 12  # Switch to pure rollouts after move 12
    MODEL_WEIGHT: Final[float] = 0.7  # Weight given to model evaluation in hybrid phase
    NUM_ROLLOUTS: Final[int] = 75  # Number of rollouts per position
