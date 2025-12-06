"""
Opening book for Connect 4.

Connect 4 is a solved game. With perfect play, the first player wins
by starting in the center column. This book contains optimal moves
for common opening positions.

Key format: Position key from GameState.get_key()
Value format: Best column to play (0-indexed)

Sources:
- John Tromp's Connect 4 solver
- Victor Allis's thesis "A Knowledge-based Approach of Connect-Four"
"""

from typing import Dict, Optional, Tuple
from game_state import GameState


class OpeningBook:
    """Opening book for Connect 4 with optimal moves."""

    def __init__(self):
        # Book stores: position_key -> (best_move, score)
        # Score: positive = winning for current player
        self._book: Dict[int, Tuple[int, int]] = {}
        self._build_book()

    def _build_book(self):
        """Build the opening book with known optimal lines."""

        # Helper to add a position to the book
        def add_position(moves: list, best_move: int, score: int = 1):
            """Add a position (reached by moves) with its best response."""
            state = GameState()
            for m in moves:
                state.make_move(m)
            key = state.get_key()
            self._book[key] = (best_move, score)

            # Also add the mirrored position
            mirrored_moves = [6 - m for m in moves]
            mirrored_best = 6 - best_move
            state_m = GameState()
            for m in mirrored_moves:
                state_m.make_move(m)
            key_m = state_m.get_key()
            self._book[key_m] = (mirrored_best, score)

        # === OPENING MOVES (Empty board) ===
        # First move: Always play center (column 3, 0-indexed)
        add_position([], 3, 1)

        # === RESPONSES TO CENTER OPENING ===
        # After 1.d (center), best responses are d or c/e
        add_position([3], 3, -1)  # Mirror center

        # === COMMON LINES FROM CENTER OPENING ===
        # 1.d d - Both played center
        add_position([3, 3], 3, 1)  # Stack center
        add_position([3, 3, 3], 2, -1)  # Threaten
        add_position([3, 3, 3, 2], 3, 1)
        add_position([3, 3, 3, 3], 2, -1)

        # 1.d c - Center then adjacent
        add_position([3, 2], 3, 1)
        add_position([3, 2, 3], 3, -1)
        add_position([3, 2, 3, 3], 4, 1)

        # 1.d e - Center then other adjacent
        add_position([3, 4], 3, 1)
        add_position([3, 4, 3], 3, -1)
        add_position([3, 4, 3, 3], 2, 1)

        # 1.d b - Center then far
        add_position([3, 1], 3, 1)
        add_position([3, 1, 3], 3, -1)

        # 1.d f - Center then far other side
        add_position([3, 5], 3, 1)
        add_position([3, 5, 3], 3, -1)

        # 1.d a - Edge response (bad for opponent)
        add_position([3, 0], 3, 1)
        add_position([3, 0, 3], 3, -1)

        # 1.d g - Other edge (bad for opponent)
        add_position([3, 6], 3, 1)
        add_position([3, 6, 3], 3, -1)

        # === NON-CENTER OPENINGS (Suboptimal but common) ===
        # If opponent plays non-center, we take center
        add_position([0], 3, 1)  # Edge opening -> take center
        add_position([1], 3, 1)
        add_position([2], 3, 1)
        add_position([4], 3, 1)
        add_position([5], 3, 1)
        add_position([6], 3, 1)

        # === LONGER OPTIMAL LINES ===
        # Main line: 1.d d 2.d c
        add_position([3, 3, 3, 2], 3, 1)
        add_position([3, 3, 3, 2, 3], 4, -1)

        # Main line: 1.d d 2.d d (tower)
        add_position([3, 3, 3, 3], 2, -1)
        add_position([3, 3, 3, 3, 2], 4, 1)
        add_position([3, 3, 3, 3, 2, 4], 2, -1)

        # Diagonal threat line
        add_position([3, 2, 4], 3, -1)
        add_position([3, 2, 4, 3], 2, 1)
        add_position([3, 2, 4, 3, 2], 1, -1)

        # Strong center control
        add_position([3, 4, 2], 3, -1)
        add_position([3, 4, 2, 3], 4, 1)

    def lookup(self, state: GameState) -> Optional[int]:
        """
        Look up a position in the opening book.

        Returns: Best move if found, None otherwise.
        """
        key = state.get_key()
        if key in self._book:
            return self._book[key][0]

        # Try symmetric lookup
        sym_key, mirrored = state.get_symmetry_key()
        if sym_key in self._book:
            best = self._book[sym_key][0]
            return 6 - best if mirrored else best

        return None

    def get_score(self, state: GameState) -> Optional[int]:
        """Get the score for a position if in book (+1 win, -1 loss, 0 draw)."""
        key = state.get_key()
        if key in self._book:
            return self._book[key][1]
        return None

    def __len__(self) -> int:
        return len(self._book)

    def __contains__(self, state: GameState) -> bool:
        return self.lookup(state) is not None


# Singleton instance
_book: Optional[OpeningBook] = None


def get_opening_book() -> OpeningBook:
    """Get the singleton opening book instance."""
    global _book
    if _book is None:
        _book = OpeningBook()
    return _book
