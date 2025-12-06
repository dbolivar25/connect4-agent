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
    """Negamax agent with iterative deepening, opening book, and ML evaluation."""

    _COLUMN_WEIGHTS = [1, 2, 4, 5, 4, 2, 1]

    def __init__(self, classifier, time_limit: float = SearchConfig.TIME_LIMIT):
        self.classifier = classifier
        self.time_limit = time_limit
        self.ml_cache: Dict[int, Tuple[float, float, float]] = {}
        self.tt = TranspositionTable()
        self.rng = random.Random()
        self.opening_book = get_opening_book()

        # Search state
        self.start_time = 0.0
        self.nodes_searched = 0
        self.search_aborted = False

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

    def _get_model_evaluation(self, state: GameState) -> float:
        """Get ML model evaluation normalized to [-1, 1] range."""
        key = state.get_key()

        if key not in self.ml_cache:
            analysis = self.classifier.analyze_position(state.board)
            self.ml_cache[key] = (
                analysis["win_probability"],
                analysis["loss_probability"],
                analysis["draw_probability"],
            )

        win_prob, loss_prob, _ = self.ml_cache[key]
        score = win_prob - loss_prob
        return score if state.current_player == 1 else -score

    def _batch_prefetch_children(self, state: GameState, moves: List[int]) -> None:
        """Pre-evaluate all child positions using batched ML inference."""
        uncached_states = []
        uncached_keys = []

        for move in moves:
            child = state.clone()
            child.make_move(move)
            key = child.get_key()
            if key not in self.ml_cache:
                uncached_states.append(child)
                uncached_keys.append(key)

        if not uncached_states:
            return

        # Batch evaluate all uncached positions
        boards = [s.board for s in uncached_states]
        results = self.classifier.batch_analyze(boards)

        # Store in cache
        for key, (win_prob, loss_prob, draw_prob) in zip(uncached_keys, results):
            self.ml_cache[key] = (win_prob, loss_prob, draw_prob)

    def _smart_playout(self, state: GameState) -> float:
        """Play out with smart heuristics: take wins, block threats, prefer center."""
        current = state.clone()
        starting_player = current.current_player

        while True:
            result = current.check_win()
            if result is not None:
                return result * starting_player

            moves = current.get_valid_moves()
            if not moves:
                return 0

            # Take immediate win
            winning_move = current.has_winning_move()
            if winning_move is not None:
                current.make_move(winning_move)
                continue

            # Block immediate threat
            blocking_move = None
            for col in moves:
                col_base = col * 7
                for row in range(6):
                    bit_pos = col_base + row
                    if not (current._mask & (1 << bit_pos)):
                        opp_position = (current._mask ^ current._position) | (1 << bit_pos)
                        if current._is_winning_position(opp_position):
                            blocking_move = col
                        break
                if blocking_move is not None:
                    break

            if blocking_move is not None:
                current.make_move(blocking_move)
                continue

            # Weighted random (prefer center)
            weights = [self._COLUMN_WEIGHTS[m] for m in moves]
            total = sum(weights)
            r = self.rng.random() * total
            cumulative = 0
            for i, m in enumerate(moves):
                cumulative += weights[i]
                if r <= cumulative:
                    current.make_move(m)
                    break

        return 0

    def _evaluate_position(self, state: GameState) -> float:
        """Evaluate position based on game phase."""
        if state.ply_count < SearchConfig.MODEL_ONLY_PHASE:
            return self._get_model_evaluation(state)

        elif state.ply_count <= SearchConfig.HYBRID_PHASE_END:
            model_score = self._get_model_evaluation(state)
            rollout_sum = sum(
                self._smart_playout(state.clone())
                for _ in range(SearchConfig.NUM_ROLLOUTS)
            )
            rollout_score = rollout_sum / SearchConfig.NUM_ROLLOUTS
            return float(
                SearchConfig.MODEL_WEIGHT * model_score
                + (1 - SearchConfig.MODEL_WEIGHT) * rollout_score
            )

        else:
            rollout_sum = sum(
                self._smart_playout(state.clone())
                for _ in range(SearchConfig.NUM_ROLLOUTS)
            )
            return rollout_sum / SearchConfig.NUM_ROLLOUTS

    def _order_moves(self, state: GameState, moves: List[int], tt_move: Optional[int]) -> List[int]:
        """Order moves for better alpha-beta pruning."""
        if not moves:
            return moves

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
        for col in moves:
            if col not in ordered:
                ordered.append(col)

        return ordered

    def _negamax(
        self, state: GameState, depth: int, alpha: float, beta: float
    ) -> Tuple[float, Optional[int]]:
        """Negamax with alpha-beta, TT, and iterative deepening support."""
        self.nodes_searched += 1

        # Time check (every 1000 nodes)
        if self.nodes_searched % 1000 == 0 and self._should_abort():
            return 0, None

        alpha_orig = alpha

        # Terminal check
        result = state.check_win()
        if result is not None:
            # Return high score adjusted by depth (prefer faster wins)
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

        # Leaf evaluation
        if depth <= 0:
            return self._evaluate_position(state), None

        # Get TT move for ordering
        tt_move = self.tt.get_best_move(tt_key)
        ordered_moves = self._order_moves(state, moves, tt_move)

        # Batch prefetch children ML evaluations at shallow depths
        if depth <= 2 and state.ply_count < SearchConfig.MODEL_ONLY_PHASE:
            self._batch_prefetch_children(state, ordered_moves)

        best_score = float("-inf")
        best_move = ordered_moves[0]

        for move in ordered_moves:
            if self._should_abort():
                break

            next_state = state.clone()
            next_state.make_move(move)
            score, _ = self._negamax(next_state, depth - 1, -beta, -alpha)
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
        Iterative deepening search.
        Returns: (best_move, best_score, depth_reached)
        """
        valid_moves = state.get_valid_moves()
        best_move = valid_moves[0]
        best_score = float("-inf")
        depth_reached = 0

        for depth in range(1, SearchConfig.MAX_DEPTH + 1):
            if self._should_abort() and depth > SearchConfig.MIN_DEPTH:
                break

            self.nodes_searched = 0
            score, move = self._negamax(
                state, depth, float("-inf"), float("inf")
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
        """Find best move using opening book + iterative deepening."""
        self.start_time = time.time()
        self.search_aborted = False

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

        # Iterative deepening search
        logger.info(f"\nIterative deepening (time limit: {self.time_limit}s)...")
        logger.info(f"Game phase: Move {state.ply_count + 1}")

        best_move, best_score, depth = self._iterative_deepening(state)

        elapsed = time.time() - self.start_time
        logger.info(f"Search complete: depth={depth}, score={best_score:.3f}, time={elapsed:.2f}s")
        logger.info(f"TT: {self.tt.hits} hits, {self.tt.stores} stores")
        logger.info(f"Selected: column {best_move + 1}")

        return best_move

    def clear_cache(self):
        """Clear ML cache and transposition table."""
        self.ml_cache.clear()
        self.tt.clear()
