from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from config import GameConfig


class GameState:
    """
    Connect 4 game state using bitboard representation for efficiency.

    Bitboard layout (column-major, 7 bits per column):
        Column:  0  1  2  3  4  5  6
                 .  .  .  .  .  .  .   <- guard bits (overflow detection)
                 5 12 19 26 33 40 47   <- row 5 (top)
                 4 11 18 25 32 39 46   <- row 4
                 3 10 17 24 31 38 45   <- row 3
                 2  9 16 23 30 37 44   <- row 2
                 1  8 15 22 29 36 43   <- row 1
                 0  7 14 21 28 35 42   <- row 0 (bottom)

    Two 64-bit integers represent the full state:
        - mask: positions of ALL pieces
        - position: positions of CURRENT player's pieces
        - opponent pieces = mask XOR position
    """

    # Precomputed constants for bitboard operations
    _BOTTOM_ROW = 0b0000001_0000001_0000001_0000001_0000001_0000001_0000001
    _BOARD_MASK = 0b0111111_0111111_0111111_0111111_0111111_0111111_0111111

    # Column base positions (bottom of each column)
    _COL_BASES = [0, 7, 14, 21, 28, 35, 42]

    def __init__(
        self,
        position: int = 0,
        mask: int = 0,
        current_player: int = 1,
        last_move: Optional[Tuple[int, int]] = None,
        ply_count: int = 0,
        # Legacy support for numpy board initialization
        board: Optional[NDArray[np.int8]] = None,
        height_map: Optional[NDArray[np.int8]] = None,
    ):
        """Initialize game state with bitboards or legacy numpy array."""
        self.current_player = current_player
        self.last_move = last_move
        self.ply_count = ply_count
        self._board_cache: Optional[NDArray[np.int8]] = None

        if board is not None:
            # Convert numpy board to bitboards (legacy support)
            self._position, self._mask = self._numpy_to_bitboards(board, current_player)
        else:
            self._position = position
            self._mask = mask

    def _numpy_to_bitboards(self, board: NDArray[np.int8], current_player: int) -> Tuple[int, int]:
        """Convert numpy board to bitboard representation."""
        mask = 0
        position = 0

        for col in range(7):
            for row in range(6):
                bit_pos = col * 7 + row
                cell = board[5 - row, col]  # numpy is row 0 at top
                if cell != 0:
                    mask |= (1 << bit_pos)
                    if cell == current_player:
                        position |= (1 << bit_pos)

        return position, mask

    @property
    def board(self) -> NDArray[np.int8]:
        """
        Get numpy board representation (for ML compatibility).
        Cached for efficiency when accessed multiple times.
        """
        if self._board_cache is not None:
            return self._board_cache

        board = np.zeros((6, 7), dtype=np.int8)
        opponent = self._mask ^ self._position

        for col in range(7):
            for row in range(6):
                bit_pos = col * 7 + row
                numpy_row = 5 - row  # Convert to numpy indexing (row 0 at top)
                if self._position & (1 << bit_pos):
                    board[numpy_row, col] = self.current_player
                elif opponent & (1 << bit_pos):
                    board[numpy_row, col] = -self.current_player

        self._board_cache = board
        return board

    @property
    def height_map(self) -> NDArray[np.int8]:
        """Get height map for compatibility (computed from bitboards)."""
        heights = np.zeros(7, dtype=np.int8)
        for col in range(7):
            col_mask = 0b0111111 << (col * 7)
            col_pieces = self._mask & col_mask
            height = 0
            for row in range(6):
                if col_pieces & (1 << (col * 7 + row)):
                    height = row + 1
            heights[col] = 5 - height  # Convert to "next available row" format
        return heights

    def clone(self) -> "GameState":
        """Create a copy of the current state (very fast with bitboards)."""
        new_state = GameState.__new__(GameState)
        new_state._position = self._position
        new_state._mask = self._mask
        new_state.current_player = self.current_player
        new_state.last_move = self.last_move
        new_state.ply_count = self.ply_count
        new_state._board_cache = None  # Don't copy cache
        return new_state

    def get_valid_moves(self) -> List[int]:
        """Get list of valid moves in preferred order (center-out)."""
        valid = []
        for col in GameConfig.SEARCH_ORDER:
            # Check if top of column is empty (bit 5 of column)
            top_bit = 1 << (col * 7 + 5)
            if not (self._mask & top_bit):
                valid.append(col)
        return valid

    def can_play(self, col: int) -> bool:
        """Check if a column is playable."""
        top_bit = 1 << (col * 7 + 5)
        return not (self._mask & top_bit)

    def make_move(self, col: int) -> bool:
        """Make a move in the specified column."""
        if not self.can_play(col):
            return False

        # Find the lowest empty position in the column
        col_base = col * 7
        for row in range(6):
            bit_pos = col_base + row
            if not (self._mask & (1 << bit_pos)):
                # Place piece here
                self._position |= (1 << bit_pos)
                self._mask |= (1 << bit_pos)
                self.last_move = (5 - row, col)  # numpy-style coordinates
                break

        # Switch players: swap position with opponent
        self._position = self._mask ^ self._position
        self.current_player = -self.current_player
        self.ply_count += 1
        self._board_cache = None  # Invalidate cache
        return True

    def check_win(self) -> Optional[int]:
        """
        Check if the last move resulted in a win.
        Uses ultra-fast bitboard win detection.
        Returns: 1/-1 for player win, 0 for draw, None for ongoing.
        """
        if self.last_move is None:
            return None

        # Check the previous player (who just moved)
        # Their pieces are: mask XOR position (since we already swapped)
        opponent_pieces = self._mask ^ self._position

        if self._is_winning_position(opponent_pieces):
            return -self.current_player  # Previous player won

        # Check for draw (board full)
        if self._mask == self._BOARD_MASK:
            return 0

        return None

    def _is_winning_position(self, position: int) -> bool:
        """
        Ultra-fast 4-in-a-row detection using bitboard operations.
        Only 12 bitwise operations total.
        """
        # Horizontal (shift by 7 = one column)
        m = position & (position >> 7)
        if m & (m >> 14):
            return True

        # Vertical (shift by 1 = one row)
        m = position & (position >> 1)
        if m & (m >> 2):
            return True

        # Diagonal / (shift by 6)
        m = position & (position >> 6)
        if m & (m >> 12):
            return True

        # Diagonal \ (shift by 8)
        m = position & (position >> 8)
        if m & (m >> 16):
            return True

        return False

    def has_winning_move(self) -> Optional[int]:
        """Check if current player has an immediate winning move. Returns column or None."""
        for col in range(7):
            if not self.can_play(col):
                continue

            # Find where piece would land
            col_base = col * 7
            for row in range(6):
                bit_pos = col_base + row
                if not (self._mask & (1 << bit_pos)):
                    # Simulate the move
                    new_position = self._position | (1 << bit_pos)
                    if self._is_winning_position(new_position):
                        return col
                    break
        return None

    def get_key(self) -> int:
        """Get a unique key for this position (for transposition table)."""
        # Combine position and mask into a single unique key
        # Position alone isn't unique because it depends on current_player perspective
        return self._position + self._mask + (self._mask << 1)

    def get_symmetry_key(self) -> Tuple[int, int]:
        """Get canonical key considering left-right symmetry."""
        # Mirror the position horizontally
        def mirror(bb: int) -> int:
            result = 0
            for col in range(7):
                col_bits = (bb >> (col * 7)) & 0b0111111
                result |= col_bits << ((6 - col) * 7)
            return result

        key1 = self.get_key()
        key2 = mirror(self._position) + mirror(self._mask) + (mirror(self._mask) << 1)

        # Return canonical (smaller) key and whether we mirrored
        if key1 <= key2:
            return key1, False
        return key2, True
