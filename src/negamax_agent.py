import random
import time
from typing import Dict, Optional, Tuple

import numpy as np

from config import SearchConfig
from game_state import GameState
from logger import logger


# Transposition table entry types
TT_EXACT = 0
TT_LOWER = 1  # Alpha cutoff (score is lower bound)
TT_UPPER = 2  # Beta cutoff (score is upper bound)


class TranspositionTable:
    """Hash table for storing previously evaluated positions."""

    def __init__(self, max_size: int = 1_000_000):
        self.max_size = max_size
        self.table: Dict[int, Tuple[float, int, int, Optional[int]]] = {}
        self.hits = 0
        self.stores = 0

    def get(self, key: int, depth: int, alpha: float, beta: float) -> Optional[Tuple[float, Optional[int]]]:
        """
        Look up position in table.
        Returns (score, best_move) if usable, None otherwise.
        """
        entry = self.table.get(key)
        if entry is None:
            return None

        stored_score, stored_depth, flag, best_move = entry

        # Only use if searched at least as deep
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

    def store(self, key: int, score: float, depth: int, flag: int, best_move: Optional[int]):
        """Store a position evaluation."""
        # Simple replacement scheme - always replace
        if len(self.table) >= self.max_size:
            # Remove ~10% of entries when full
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
    """Negamax-based game-playing agent with ML position evaluation."""

    # Weights for center-biased random playouts
    _COLUMN_WEIGHTS = [1, 2, 4, 5, 4, 2, 1]

    def __init__(self, classifier):
        self.classifier = classifier
        self.start_time = 0.0
        self.ml_cache: Dict[int, Tuple[float, float, float]] = {}
        self.tt = TranspositionTable()
        self.rng = random.Random()

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

    def _smart_playout(self, state: GameState) -> float:
        """
        Play out the position to a terminal state with smart heuristics:
        1. Always take immediate wins
        2. Always block immediate opponent wins
        3. Prefer center columns

        Returns result from the perspective of the player who started the playout.
        """
        current = state.clone()
        starting_player = current.current_player

        while True:
            result = current.check_win()
            if result is not None:
                # Return from starting player's perspective
                return result * starting_player

            moves = current.get_valid_moves()
            if not moves:
                return 0  # Draw

            # Check for immediate win
            winning_move = current.has_winning_move()
            if winning_move is not None:
                current.make_move(winning_move)
                continue

            # Check for immediate block needed
            # Temporarily check opponent's winning moves
            blocking_move = None
            for col in moves:
                col_base = col * 7
                for row in range(6):
                    bit_pos = col_base + row
                    if not (current._mask & (1 << bit_pos)):
                        # Simulate opponent playing here
                        opp_position = (current._mask ^ current._position) | (1 << bit_pos)
                        if current._is_winning_position(opp_position):
                            blocking_move = col
                        break
                if blocking_move is not None:
                    break

            if blocking_move is not None:
                current.make_move(blocking_move)
                continue

            # Weighted random selection (prefer center)
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
            # Early game: Use pure model evaluation
            return self._get_model_evaluation(state)

        elif state.ply_count <= SearchConfig.HYBRID_PHASE_END:
            # Mid game: Use weighted combination of model and rollouts
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
            # Late game: Use pure rollouts
            rollout_sum = sum(
                self._smart_playout(state.clone())
                for _ in range(SearchConfig.NUM_ROLLOUTS)
            )
            return rollout_sum / SearchConfig.NUM_ROLLOUTS

    def _negamax(
        self, state: GameState, depth: int, alpha: float, beta: float
    ) -> Tuple[float, Optional[int]]:
        """Negamax with alpha-beta pruning, transposition table, and position evaluation."""
        alpha_orig = alpha

        # Check terminal states
        result = state.check_win()
        if result is not None:
            return result * state.current_player, None

        moves = state.get_valid_moves()
        if not moves:
            return 0, None  # Draw

        # Transposition table lookup
        tt_key = state.get_key()
        tt_result = self.tt.get(tt_key, depth, alpha, beta)
        if tt_result is not None:
            return tt_result

        if depth <= 0:
            return self._evaluate_position(state), None

        # Move ordering: try TT best move first if available
        entry = self.tt.table.get(tt_key)
        if entry is not None and entry[3] is not None:
            tt_move = entry[3]
            if tt_move in moves:
                moves.remove(tt_move)
                moves.insert(0, tt_move)

        best_score = float("-inf")
        best_move = moves[0]

        for move in moves:
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

        # Store in transposition table
        if best_score <= alpha_orig:
            flag = TT_UPPER
        elif best_score >= beta:
            flag = TT_LOWER
        else:
            flag = TT_EXACT

        self.tt.store(tt_key, best_score, depth, flag, best_move)

        return best_score, best_move

    def get_best_move(self, state: GameState) -> int:
        """Find the best move using Negamax search with improvements."""
        self.start_time = time.time()
        valid_moves = state.get_valid_moves()

        if not valid_moves:
            raise ValueError("No valid moves available")

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
                    # Simulate opponent playing here
                    opp_position = (state._mask ^ state._position) | (1 << bit_pos)
                    if state._is_winning_position(opp_position):
                        logger.info(f"Found blocking move: {move + 1}")
                        return move
                    break

        # Negamax search
        logger.info("\nStarting Negamax search...")
        logger.info(f"Game phase: Move {state.ply_count + 1}")

        best_score, best_move = self._negamax(
            state, SearchConfig.NEGAMAX_DEPTH, float("-inf"), float("inf")
        )

        if best_move is None:
            best_move = valid_moves[0]

        logger.info(f"TT stats: {self.tt.hits} hits, {self.tt.stores} stores")
        logger.info(f"Selected move {best_move + 1} with score {best_score:.3f}")

        return best_move

    def clear_cache(self):
        """Clear ML evaluation cache and transposition table."""
        self.ml_cache.clear()
        self.tt.clear()
