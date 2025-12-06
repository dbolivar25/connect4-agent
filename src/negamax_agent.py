import random
import time
from typing import Dict, List, Optional, Tuple

from config import SearchConfig
from game_state import GameState
from logger import logger
from opening_book import get_opening_book


# Transposition table entry types
TT_EXACT = 0
TT_LOWER = 1  # Alpha cutoff (score is lower bound)
TT_UPPER = 2  # Beta cutoff (score is upper bound)

# Special scores
SCORE_WIN = 1000
SCORE_LOSS = -1000


class TranspositionTable:
    """Hash table for storing previously evaluated positions."""

    def __init__(self, max_size: int = 2_000_000):
        self.max_size = max_size
        self.table: Dict[int, Tuple[float, int, int, Optional[int]]] = {}
        self.hits = 0
        self.stores = 0

    def get(self, key: int, depth: int, alpha: float, beta: float) -> Optional[Tuple[float, Optional[int]]]:
        """Look up position. Returns (score, best_move) if usable."""
        entry = self.table.get(key)
        if entry is None:
            return None

        stored_score, stored_depth, flag, best_move = entry

        if stored_depth < depth:
            return None

        self.hits += 1

        if flag == TT_EXACT:
            return stored_score, best_move
        elif flag == TT_LOWER and stored_score >= beta:
            return stored_score, best_move
        elif flag == TT_UPPER and stored_score <= alpha:
            return stored_score, best_move

        return None

    def get_best_move(self, key: int) -> Optional[int]:
        """Get just the best move for move ordering."""
        entry = self.table.get(key)
        return entry[3] if entry else None

    def store(self, key: int, score: float, depth: int, flag: int, best_move: Optional[int]):
        """Store a position evaluation."""
        if len(self.table) >= self.max_size:
            keys_to_remove = list(self.table.keys())[:self.max_size // 10]
            for k in keys_to_remove:
                del self.table[k]

        self.table[key] = (score, depth, flag, best_move)
        self.stores += 1

    def clear(self):
        self.table.clear()
        self.hits = 0
        self.stores = 0


class NegamaxAgent:
    """Negamax agent with ML-guided move ordering and fast heuristic evaluation."""

    _COLUMN_WEIGHTS = [0, 1, 2, 3, 2, 1, 0]  # Center preference

    def __init__(self, classifier, time_limit: float = SearchConfig.TIME_LIMIT):
        self.classifier = classifier
        self.time_limit = time_limit
        self.tt = TranspositionTable()
        self.rng = random.Random()
        self.opening_book = get_opening_book()

        # Search state
        self.start_time = 0.0
        self.nodes_searched = 0
        self.search_aborted = False

        # ML-ordered moves for root (computed once per get_best_move call)
        self._root_move_order: Optional[List[int]] = None

    def _time_remaining(self) -> float:
        """Get remaining search time."""
        return self.time_limit - (time.time() - self.start_time)

    def _should_abort(self) -> bool:
        """Check if we should abort the search."""
        if self.search_aborted:
            return True
        if self._time_remaining() < 0.05:  # 50ms buffer
            self.search_aborted = True
            return True
        return False

    def _fast_heuristic(self, state: GameState) -> float:
        """
        Ultra-fast heuristic evaluation (microseconds).
        Returns score from current player's perspective.
        """
        score = 0.0

        # Center control - pieces in center columns are better
        # _position stores P1's pieces, (_mask ^ _position) stores P2's pieces
        for col in range(7):
            col_base = col * 7
            col_weight = self._COLUMN_WEIGHTS[col]
            for row in range(6):
                bit_pos = col_base + row
                if state._mask & (1 << bit_pos):
                    weight = col_weight + row * 0.1
                    if state._position & (1 << bit_pos):
                        score += weight  # P1's piece
                    else:
                        score -= weight  # P2's piece

        # Threat detection
        # Current player's position for threat checking
        current_pos = state._position if state.current_player == 1 else (state._mask ^ state._position)
        opponent_pos = (state._mask ^ state._position) if state.current_player == 1 else state._position

        for col in range(7):
            col_base = col * 7
            for row in range(6):
                bit_pos = col_base + row
                if not (state._mask & (1 << bit_pos)):
                    # Current player's threat
                    test_pos = current_pos | (1 << bit_pos)
                    if state._is_winning_position(test_pos):
                        score += 20
                    # Opponent's threat
                    test_pos = opponent_pos | (1 << bit_pos)
                    if state._is_winning_position(test_pos):
                        score -= 20
                    break

        # Convert to current player's perspective
        # score is currently from P1's perspective (positive = good for P1)
        normalized = score / 50.0
        return normalized if state.current_player == 1 else -normalized

    def _ml_order_root_moves(self, state: GameState, moves: List[int]) -> List[int]:
        """
        Use ML model to order moves at root (one-time cost).
        Returns moves sorted by ML evaluation (best first).
        """
        if len(moves) <= 1:
            return moves

        # Evaluate each child position with ML
        move_scores = []
        boards = []
        valid_moves = []

        for move in moves:
            child = state.clone()
            child.make_move(move)
            boards.append(child.board)
            valid_moves.append(move)

        # Batch ML evaluation
        results = self.classifier.batch_analyze(boards)

        for move, (win_prob, loss_prob, draw_prob) in zip(valid_moves, results):
            # From parent's perspective: we want child positions where opponent is worse
            # Higher loss_prob for opponent = better for us
            # Also consider draw as partial success
            score = loss_prob + 0.5 * draw_prob - win_prob
            move_scores.append((move, score))

        # Sort by score (highest first)
        move_scores.sort(key=lambda x: x[1], reverse=True)

        # If all scores are the same (ML overconfident), use center-out ordering
        if len(set(round(s, 2) for _, s in move_scores)) == 1:
            logger.info("ML ordering: all scores equal, using center-out")
            return [m for m in [3, 2, 4, 1, 5, 0, 6] if m in moves]

        logger.info(f"ML move ordering: {[(m+1, f'{s:.2f}') for m, s in move_scores]}")

        return [m for m, _ in move_scores]

    def _order_moves(self, state: GameState, moves: List[int], tt_move: Optional[int], is_root: bool = False) -> List[int]:
        """Order moves for better alpha-beta pruning."""
        if not moves:
            return moves

        # At root, use pre-computed ML ordering
        if is_root and self._root_move_order is not None:
            ordered = []
            # TT move still goes first if available
            if tt_move is not None and tt_move in moves:
                ordered.append(tt_move)
            # Then ML-ordered moves
            for m in self._root_move_order:
                if m in moves and m not in ordered:
                    ordered.append(m)
            return ordered

        ordered = []

        # 1. TT move first (from previous iteration)
        if tt_move is not None and tt_move in moves:
            ordered.append(tt_move)

        # 2. Winning moves
        winning = state.has_winning_move()
        if winning is not None and winning in moves and winning not in ordered:
            ordered.insert(0, winning)  # Wins go first

        # 3. Blocking moves
        for col in moves:
            if col in ordered:
                continue
            col_base = col * 7
            for row in range(6):
                bit_pos = col_base + row
                if not (state._mask & (1 << bit_pos)):
                    opp_position = (state._mask ^ state._position) | (1 << bit_pos)
                    if state._is_winning_position(opp_position):
                        ordered.append(col)
                    break

        # 4. Rest in center-out order
        for col in [3, 2, 4, 1, 5, 0, 6]:
            if col in moves and col not in ordered:
                ordered.append(col)

        return ordered

    def _negamax(
        self, state: GameState, depth: int, alpha: float, beta: float, is_root: bool = False
    ) -> Tuple[float, Optional[int]]:
        """Negamax with alpha-beta, TT, and fast heuristic."""
        self.nodes_searched += 1

        # Time check (every 2000 nodes for speed)
        if self.nodes_searched % 2000 == 0 and self._should_abort():
            return 0, None

        alpha_orig = alpha

        # Terminal check
        result = state.check_win()
        if result is not None:
            if result == 0:
                return 0, None
            score = (SCORE_WIN - state.ply_count) * result * state.current_player
            return score, None

        moves = state.get_valid_moves()
        if not moves:
            return 0, None

        # TT lookup
        tt_key = state.get_key()
        tt_result = self.tt.get(tt_key, depth, alpha, beta)
        if tt_result is not None:
            return tt_result

        # Leaf evaluation with FAST heuristic
        if depth <= 0:
            return self._fast_heuristic(state), None

        # Get TT move for ordering
        tt_move = self.tt.get_best_move(tt_key)
        ordered_moves = self._order_moves(state, moves, tt_move, is_root)

        best_score = float("-inf")
        best_move = ordered_moves[0]

        for move in ordered_moves:
            if self._should_abort():
                break

            next_state = state.clone()
            next_state.make_move(move)
            score, _ = self._negamax(next_state, depth - 1, -beta, -alpha, is_root=False)
            score = -score

            if score > best_score:
                best_score = score
                best_move = move

            alpha = max(alpha, score)
            if alpha >= beta:
                break

        # Store in TT
        if not self.search_aborted:
            if best_score <= alpha_orig:
                flag = TT_UPPER
            elif best_score >= beta:
                flag = TT_LOWER
            else:
                flag = TT_EXACT
            self.tt.store(tt_key, best_score, depth, flag, best_move)

        return best_score, best_move

    def _iterative_deepening(self, state: GameState) -> Tuple[int, float, int]:
        """
        Iterative deepening search with ML-guided root move ordering.
        Returns: (best_move, best_score, depth_reached)
        """
        valid_moves = state.get_valid_moves()

        # Use ML to order root moves ONCE at the start
        logger.info("Computing ML move ordering at root...")
        self._root_move_order = self._ml_order_root_moves(state, valid_moves)

        best_move = self._root_move_order[0] if self._root_move_order else valid_moves[0]
        best_score = float("-inf")
        depth_reached = 0

        for depth in range(1, SearchConfig.MAX_DEPTH + 1):
            if self._should_abort() and depth > SearchConfig.MIN_DEPTH:
                break

            self.nodes_searched = 0
            score, move = self._negamax(
                state, depth, float("-inf"), float("inf"), is_root=True
            )

            # Only update if search completed or we have a valid result
            if not self.search_aborted and move is not None:
                best_move = move
                best_score = score
                depth_reached = depth

                logger.info(f"Depth {depth}: move={move+1}, score={score:.3f}, nodes={self.nodes_searched}")

                # Early exit on proven win/loss
                if abs(score) > SCORE_WIN - 50:
                    logger.info(f"Found forced {'win' if score > 0 else 'loss'}")
                    break

            # Check time for next iteration
            elapsed = time.time() - self.start_time
            if elapsed > self.time_limit * 0.6:  # Don't start if >60% time used
                break

        return best_move, best_score, depth_reached

    def get_best_move(self, state: GameState) -> int:
        """Find best move using opening book + ML-guided iterative deepening."""
        self.start_time = time.time()
        self.search_aborted = False
        self._root_move_order = None

        valid_moves = state.get_valid_moves()
        if not valid_moves:
            raise ValueError("No valid moves")

        # Check opening book first
        book_move = self.opening_book.lookup(state)
        if book_move is not None and book_move in valid_moves:
            logger.info(f"Opening book: column {book_move + 1}")
            return book_move

        # Quick check for immediate win
        winning_move = state.has_winning_move()
        if winning_move is not None:
            logger.info(f"Found winning move: {winning_move + 1}")
            return winning_move

        # Quick check for immediate block
        for move in valid_moves:
            col_base = move * 7
            for row in range(6):
                bit_pos = col_base + row
                if not (state._mask & (1 << bit_pos)):
                    opp_position = (state._mask ^ state._position) | (1 << bit_pos)
                    if state._is_winning_position(opp_position):
                        logger.info(f"Found blocking move: {move + 1}")
                        return move
                    break

        # Iterative deepening search with ML-guided ordering
        logger.info(f"\nIterative deepening (time limit: {self.time_limit}s)...")
        logger.info(f"Game phase: Move {state.ply_count + 1}")

        best_move, best_score, depth = self._iterative_deepening(state)

        elapsed = time.time() - self.start_time
        logger.info(f"Search complete: depth={depth}, score={best_score:.3f}, time={elapsed:.2f}s")
        logger.info(f"TT: {self.tt.hits} hits, {self.tt.stores} stores")
        logger.info(f"Selected: column {best_move + 1}")

        return best_move

    def clear_cache(self):
        """Clear transposition table."""
        self.tt.clear()
        self._root_move_order = None
