from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from config import GameConfig


class GameState:
    """Represents the current state of the Connect 4 game"""

    def __init__(
        self,
        board: Optional[NDArray[np.int8]] = None,
        height_map: Optional[NDArray[np.int8]] = None,
        current_player: int = 1,
        last_move: Optional[Tuple[int, int]] = None,
        ply_count: int = 0,
    ):
        """Initialize game state"""
        self.board = (
            board
            if board is not None
            else np.zeros((GameConfig.HEIGHT, GameConfig.WIDTH), dtype=np.int8)
        )
        self.current_player = current_player
        self.last_move = last_move
        self.ply_count = ply_count

        # Initialize height map
        if height_map is not None:
            self.height_map = height_map
        else:
            self._init_height_map()

    def _init_height_map(self):
        """Initialize or update height map"""
        self.height_map = np.array(
            [
                next(
                    (
                        row - 1
                        for row in range(GameConfig.HEIGHT)
                        if self.board[row][col] != 0
                    ),
                    GameConfig.HEIGHT - 1,
                )
                for col in range(GameConfig.WIDTH)
            ]
        )

    def clone(self) -> "GameState":
        """Create a deep copy of the current state"""
        return GameState(
            board=self.board.copy(),
            height_map=self.height_map.copy(),
            current_player=self.current_player,
            last_move=self.last_move,
            ply_count=self.ply_count,
        )

    def get_valid_moves(self) -> List[int]:
        """Get list of valid moves in preferred order"""
        return [col for col in GameConfig.SEARCH_ORDER if self.height_map[col] >= 0]

    def make_move(self, col: int) -> bool:
        """Make a move in the specified column"""
        if self.height_map[col] < 0:
            return False

        row = self.height_map[col]
        self.board[row][col] = self.current_player
        self.last_move = (row, col)
        self.height_map[col] -= 1
        self.current_player = -self.current_player
        self.ply_count += 1
        return True

    def check_win(self) -> Optional[int]:
        """
        Check if the last move resulted in a win
        Returns: 1 for win, -1 for loss, 0 for draw, None for ongoing
        """
        if self.last_move is None:
            return None

        row, col = self.last_move
        player = -self.current_player  # Check previous player's move

        # Horizontal check
        for c in range(max(0, col - 3), min(col + 1, GameConfig.WIDTH - 3)):
            if np.all(self.board[row, c : c + 4] == player):
                return player

        # Vertical check
        if row <= 2:
            if np.all(self.board[row : row + 4, col] == player):
                return player

        # Diagonal checks
        # Check both positive and negative diagonals centered at the last move
        for dr, dc in [(1, 1), (1, -1)]:  # Positive and negative slopes
            # Check all possible 4-in-a-row containing the last move
            for offset in range(-3, 1):
                r, c = row - offset * dr, col - offset * dc
                if (
                    0 <= r <= GameConfig.HEIGHT - 4 * dr
                    and 0 <= c <= GameConfig.WIDTH - 4 * abs(dc)
                ):
                    diagonal = [self.board[r + i * dr][c + i * dc] for i in range(4)]
                    if np.all(np.array(diagonal) == player):
                        return player

        # Check for draw
        if not self.get_valid_moves():
            return 0

        return None
