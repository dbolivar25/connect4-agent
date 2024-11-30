import time
from typing import Dict, Tuple

import numpy as np
from numpy.typing import NDArray

from config import SearchConfig
from game_state import GameState
from logger import logger


class NegamaxAgent:
    """Negamax-based game-playing agent with ML position evaluation"""

    def __init__(self, classifier, config: SearchConfig = SearchConfig()):
        self.classifier = classifier
        self.config = config
        self.start_time = 0.0
        self.ml_cache: Dict[str, Tuple[float, float, float]] = {}

    def _board_to_key(self, board: NDArray[np.int8]) -> str:
        """Convert board state to string key for caching"""
        return board.tobytes().hex()

    def _get_model_evaluation(self, state: GameState) -> float:
        """Get ML model evaluation normalized to [-1, 1] range"""
        key = self._board_to_key(state.board)

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

    def _random_playout(self, state: GameState) -> float:
        """Play out the position randomly to a terminal state"""
        current = state.clone()
        while True:
            result = current.check_win()
            if result is not None:
                return result * current.current_player

            moves = current.get_valid_moves()
            move = moves[np.random.randint(len(moves))]
            current.make_move(move)

    def _evaluate_position(self, state: GameState) -> float:
        """Evaluate position based on game phase"""
        if state.ply_count < SearchConfig.MODEL_ONLY_PHASE:
            # Early game: Use pure model evaluation
            return self._get_model_evaluation(state)

        elif state.ply_count <= SearchConfig.HYBRID_PHASE_END:
            # Mid game: Use weighted combination of model and rollouts
            model_score = self._get_model_evaluation(state)
            rollout_scores = [
                self._random_playout(state.clone())
                for _ in range(self.config.NUM_ROLLOUTS)
            ]
            rollout_score = np.mean(rollout_scores)

            return float(
                self.config.MODEL_WEIGHT * model_score
                + (1 - self.config.MODEL_WEIGHT) * rollout_score
            )

        else:
            # Late game: Use pure rollouts
            rollout_scores = [
                self._random_playout(state.clone())
                for _ in range(self.config.NUM_ROLLOUTS)
            ]
            return float(np.mean(rollout_scores))

    def _negamax(
        self, state: GameState, depth: int, alpha: float, beta: float
    ) -> float:
        """Negamax with alpha-beta pruning and position evaluation"""
        # Check terminal states
        result = state.check_win()
        if result is not None:
            return result * state.current_player

        if depth <= 0:
            return self._evaluate_position(state)

        moves = state.get_valid_moves()
        if not moves:
            return 0  # Draw

        value = float("-inf")
        for move in moves:
            next_state = state.clone()
            next_state.make_move(move)
            score = -self._negamax(next_state, depth - 1, -beta, -alpha)
            value = max(value, score)
            alpha = max(alpha, score)
            if alpha >= beta:
                break

        return value

    def get_best_move(self, state: GameState) -> int:
        """Find the best move using 3-ply Negamax search"""
        self.start_time = time.time()
        valid_moves = state.get_valid_moves()

        # Quick checks for winning/blocking moves
        for move in valid_moves:
            # Check for win
            test_state = state.clone()
            test_state.make_move(move)
            if test_state.check_win() == -test_state.current_player:
                logger.info(f"Found winning move: {move + 1}")
                return move

            # Check for block
            test_state = state.clone()
            test_state.current_player = -state.current_player
            test_state.make_move(move)
            if test_state.check_win() == -test_state.current_player:
                logger.info(f"Found blocking move: {move + 1}")
                return move

        # Negamax search
        best_move = valid_moves[0]
        best_score = float("-inf")
        alpha = float("-inf")
        beta = float("inf")

        logger.info("\nStarting 3-ply Negamax search...")
        logger.info(f"Game phase: Move {state.ply_count + 1}")

        for move in valid_moves:
            next_state = state.clone()
            next_state.make_move(move)
            score = -self._negamax(
                next_state, SearchConfig.NEGAMAX_DEPTH - 1, -beta, -alpha
            )
            logger.info(f"Move {move + 1}: score = {score:.3f}")

            if score > best_score:
                best_score = score
                best_move = move
            alpha = max(alpha, score)

        logger.info(f"\nSelected move {best_move + 1}")
        return best_move

    def clear_cache(self):
        """Clear ML evaluation cache"""
        self.ml_cache.clear()
