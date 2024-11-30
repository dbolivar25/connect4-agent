import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Final
import logging
import time
from numpy.typing import NDArray

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Game constants
HEIGHT: Final[int] = 6
WIDTH: Final[int] = 7
SEARCH_ORDER: Final[List[int]] = [3, 2, 4, 1, 5, 0, 6]  # Center-out search
SYMBOLS: Final[Dict[int, str]] = {0: "⚪", 1: "🔴", -1: "🔵"}

# Search parameters
NEGAMAX_DEPTH: Final[int] = 3  # Fixed 3-ply search depth
MODEL_ONLY_PHASE: Final[int] = 9  # Pure model evaluation until move 9
HYBRID_PHASE_END: Final[int] = 12  # Switch to pure rollouts after move 12
NUM_ROLLOUTS: Final[int] = 75  # Number of rollouts per position


@dataclass
class SearchConfig:
    """Configuration parameters for the search"""

    model_weight: float = 0.7  # Weight given to model evaluation in hybrid phase
    num_rollouts: int = NUM_ROLLOUTS  # Number of rollouts per position


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
            board if board is not None else np.zeros((HEIGHT, WIDTH), dtype=np.int8)
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
                    (row - 1 for row in range(HEIGHT) if self.board[row][col] != 0),
                    HEIGHT - 1,
                )
                for col in range(WIDTH)
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
        return [col for col in SEARCH_ORDER if self.height_map[col] >= 0]

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
        for c in range(max(0, col - 3), min(col + 1, WIDTH - 3)):
            if np.all(self.board[row, c : c + 4] == player):
                return player

        # Vertical check
        if row <= 2:
            if np.all(self.board[row : row + 4, col] == player):
                return player

        # Diagonal checks
        for offset in range(-3, 1):
            # Positive slope
            if 0 <= row + offset < HEIGHT - 3 and 0 <= col + offset < WIDTH - 3:
                if np.all(
                    np.array(
                        [
                            self.board[row + offset + i][col + offset + i]
                            for i in range(4)
                        ]
                    )
                    == player
                ):
                    return player

            # Negative slope
            if 3 <= row + offset < HEIGHT and 0 <= col + offset < WIDTH - 3:
                if np.all(
                    np.array(
                        [
                            self.board[row + offset - i][col + offset + i]
                            for i in range(4)
                        ]
                    )
                    == player
                ):
                    return player

        # Check for draw
        if not self.get_valid_moves():
            return 0

        return None


class Agent:
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
        if state.ply_count < MODEL_ONLY_PHASE:
            # Early game: Use pure model evaluation
            return self._get_model_evaluation(state)

        elif state.ply_count <= HYBRID_PHASE_END:
            # Mid game: Use weighted combination of model and rollouts
            model_score = self._get_model_evaluation(state)
            rollout_scores = [
                self._random_playout(state.clone())
                for _ in range(self.config.num_rollouts)
            ]
            rollout_score = np.mean(rollout_scores)

            return float(
                self.config.model_weight * model_score
                + (1 - self.config.model_weight) * rollout_score
            )

        else:
            # Late game: Use pure rollouts
            rollout_scores = [
                self._random_playout(state.clone())
                for _ in range(self.config.num_rollouts)
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
            score = -self._negamax(next_state, NEGAMAX_DEPTH - 1, -beta, -alpha)
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


def play_game(player1: Agent, player2: Agent):
    """Play a complete game between two agents"""
    state = GameState()

    while True:
        display_board(state)
        current_player = player1 if state.current_player == 1 else player2
        player_symbol = SYMBOLS[state.current_player]

        move = current_player.get_best_move(state)
        print(
            f"\nPlayer {state.current_player} {player_symbol} chooses column {move + 1}"
        )

        state.make_move(move)
        result = state.check_win()

        if result is not None:
            display_board(state)
            if result == 0:
                print("\nIt's a draw!")
            else:
                winner = f"Player {result} {SYMBOLS[result]}"
                print(f"\n{winner} wins!")
            break


def display_board(state: GameState):
    """Display the current board state"""
    print("\n")
    for row in state.board:
        print("|", end="")
        for cell in row:
            print(f"{SYMBOLS[cell]}", end="")
        print("|")
    print("\n")


def get_human_move(state: GameState) -> int:
    """Get move from human player"""
    valid_moves = state.get_valid_moves()
    while True:
        try:
            move = int(input("\nEnter column (1-7): ")) - 1
            if move in valid_moves:
                return move
            print("Invalid move! Please try again.")
        except ValueError:
            print("Please enter a valid number!")


def play_game_with_mode(
    mode: str,
    player1: Agent,
    player2: Optional[Agent] = None,
    human_starts: bool = True,
):
    """Play a game based on selected mode"""
    state = GameState()

    # For human vs agent games, determine if human is player 1 or 2
    def is_human_turn(current_player):
        return (
            human_starts
            and current_player == 1
            or not human_starts
            and current_player == -1
            if mode == "human_vs_agent"
            else False
        )

    # Display who goes first
    if mode == "human_vs_agent":
        print(f"\n{'You' if human_starts else 'AI'} will play first as {SYMBOLS[1]}")
    else:
        print(f"\nPlayer 1 {SYMBOLS[1]} will play first")

    while True:
        display_board(state)
        player_symbol = SYMBOLS[state.current_player]

        # Determine current player's move
        if mode == "agent_vs_agent":
            current_agent = player1 if state.current_player == 1 else player2
            move = current_agent.get_best_move(state)  # pyright: ignore
            print(
                f"\nPlayer {state.current_player} {player_symbol} chooses column {move + 1}"
            )

        elif mode == "human_vs_agent":
            if is_human_turn(state.current_player):
                move = get_human_move(state)
                print(f"\nYou {player_symbol} chose column {move + 1}")
            else:
                move = player1.get_best_move(state)
                print(f"\nAI {player_symbol} chooses column {move + 1}")

        # Make the move
        state.make_move(move)  # pyright: ignore
        result = state.check_win()

        # Check for game end
        if result is not None:
            display_board(state)
            if result == 0:
                print("\nIt's a draw!")
            else:
                # Determine winner message based on game mode and who played as which player
                if mode == "human_vs_agent":
                    is_human_winner = (result == 1 and human_starts) or (
                        result == -1 and not human_starts
                    )
                    winner = "You" if is_human_winner else "AI"
                else:
                    winner = f"Player {result} {SYMBOLS[result]}"
                print(f"\n{winner} wins!")
            break


def main():
    """Main function with game mode selection and random player order"""
    from position_classifier import PositionClassifier
    import os
    import random

    # Load the trained classifier
    classifier = PositionClassifier()
    model_path = "models/connect4_analyzer_final.pkl"

    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        classifier.load_model(model_path)
    else:
        print("Training new classifier model...")
        classifier.train(save_path=model_path, tune_params=False, balance_data=True)

    # Create agents
    config = SearchConfig(model_weight=0.7, num_rollouts=NUM_ROLLOUTS)
    agent1 = Agent(classifier=classifier, config=config)
    agent2 = Agent(classifier=classifier, config=config)

    # Game mode selection
    print("\nWelcome to Connect 4!")
    print("1. Human vs AI")
    print("2. AI vs AI")

    while True:
        try:
            choice = int(input("\nSelect game mode (1 or 2): "))
            if choice in [1, 2]:
                break
            print("Invalid choice! Please select 1 or 2.")
        except ValueError:
            print("Please enter a valid number!")

    # Randomly determine who goes first
    human_starts = random.choice([True, False])

    # Play game based on selected mode
    if choice == 1:
        play_game_with_mode("human_vs_agent", agent1, human_starts=human_starts)
    else:
        play_game_with_mode("agent_vs_agent", agent1, agent2)

    # Ask to play again
    while True:
        play_again = input("\nPlay again? (y/n): ").lower()
        if play_again in ["y", "n"]:
            if play_again == "y":
                print("\n" + "=" * 50 + "\n")
                main()  # Restart the game
            break
        print("Please enter 'y' or 'n'!")


if __name__ == "__main__":
    main()
