import numpy as np
import time
from typing import Dict, List, Tuple, Optional
from numpy.typing import NDArray
import logging
import os

from position_classifier import PositionClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

HEIGHT = 6
WIDTH = 7
SEARCH_ORDER = [3, 2, 4, 1, 5, 0, 6]  # Center-out search order
EXPLORATION_CONSTANT = np.sqrt(2)  # Standard UCT constant
SYMBOLS = {0: "⚪", 1: "🔴", -1: "🔵"}  # Updated symbols to match classifier notation


WIN_POSITIONS = (
    [
        # Horizontal
        [(row, col + i) for i in range(4)]
        for row in range(HEIGHT)
        for col in range(WIDTH - 3)
    ]
    + [
        # Vertical
        [(row + i, col) for i in range(4)]
        for row in range(HEIGHT - 3)
        for col in range(WIDTH)
    ]
    + [
        # Diagonal (positive slope)
        [(row + i, col + i) for i in range(4)]
        for row in range(HEIGHT - 3)
        for col in range(WIDTH - 3)
    ]
    + [
        # Diagonal (negative slope)
        [(row - i, col + i) for i in range(4)]
        for row in range(3, HEIGHT)
        for col in range(WIDTH - 3)
    ]
)


class GameState:
    """Enhanced GameState class compatible with both MCTS and Classifier"""

    def __init__(
        self,
        board: Optional[NDArray[np.int8]] = None,
        height_map: Optional[NDArray[np.int8]] = None,
        current_player: int = 1,
        last_move: Optional[Tuple[int, int]] = None,
    ):
        self.board = (
            board if board is not None else np.zeros((HEIGHT, WIDTH), dtype=np.int8)
        )
        self.current_player = current_player
        self.last_move = last_move
        self.height_map = (
            height_map
            if board is not None and height_map is not None
            else np.array([HEIGHT - 1] * WIDTH)
        )

        # Initialize height map if board is provided and its height map isnt
        if board is not None and height_map is None:
            self.height_map = np.array(
                [
                    min(
                        [row - 1 for row in range(HEIGHT) if board[row][col] != 0]
                        or [HEIGHT - 1]
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
        self.current_player = self.get_opponent(self.current_player)
        return True

    def get_opponent(self, player: int) -> int:
        """Get the opponent's player number"""
        return -1 if player == 1 else 1

    def is_win_move(self, player: int) -> bool:
        """Check if the last move resulted in a win"""
        if self.last_move is None:
            return False

        row, col = self.last_move

        # Check all possible winning lines containing the last move
        for win_pos in WIN_POSITIONS:
            if (row, col) in win_pos:
                if all(self.board[r][c] == player for r, c in win_pos):
                    return True
        return False

    def get_winning_threat_moves(self, player: int) -> List[int]:
        """Get list of moves that would result in an immediate win"""
        threats = []
        for col in range(WIDTH):
            if self.height_map[col] >= 0:
                row = self.height_map[col]
                # Make move
                self.board[row][col] = player
                if self.is_win_move(player):
                    threats.append(col)
                # Undo move
                self.board[row][col] = 0
        return threats


class Node:
    """Enhanced MCTS node with ML evaluation integration"""

    def __init__(
        self,
        board: GameState,
        classifier: PositionClassifier,
        parent: Optional["Node"] = None,
        move: Optional[int] = None,
    ):
        self.board = board
        self.classifier = classifier
        self.parent = parent
        self.move = move
        self.children: Dict[int, "Node"] = {}
        self.visits = 0
        self.wins = 0
        self.untried_moves = self.board.get_valid_moves()

        # Cache the ML evaluation
        self._cached_evaluation = None

    def add_child(self, move: int, board: GameState) -> "Node":
        """Create a new child node"""
        child = Node(board, self.classifier, self, move)
        self.children[move] = child
        return child

    def update(self, result: float):
        """Update node statistics"""
        self.visits += 1
        self.wins += result

    def get_ucb_score(self, exploration_weight: float = EXPLORATION_CONSTANT) -> float:
        """Calculate UCB score with ML evaluation influence"""
        if self.visits == 0:
            return float("inf")

        # Basic UCB formula
        exploitation = self.wins / self.visits
        exploration = exploration_weight * np.sqrt(
            np.log(self.parent.visits) / self.visits
        )

        # Add ML evaluation influence
        if self._cached_evaluation is None:
            analysis = self.classifier.analyze_position(self.board.board)
            self._cached_evaluation = (
                analysis["win_probability"] - analysis["loss_probability"]
            )

        ml_influence = 0.3 * self._cached_evaluation  # Adjust weight as needed

        return exploitation + exploration + ml_influence


class Agent:
    """Enhanced agent combining MCTS, MiniMax, and ML board evaluation"""

    def __init__(
        self,
        classifier: PositionClassifier,
        min_simulations: int = 2000,
        max_time: float = 5.0,
        rollout_depth: int = 3,
    ):
        self.classifier = classifier
        self.min_simulations = min_simulations
        self.max_time = max_time
        self.rollout_depth = rollout_depth
        self.start_time = 0
        self.position_cache: Dict[str, float] = {}

    def board_to_cache_key(self, board: NDArray[np.int8]) -> str:
        """Convert board state to string for caching"""
        return ",".join(map(str, board.flatten()))

    def get_cached_evaluation(self, board: NDArray[np.int8]) -> Optional[float]:
        """Retrieve cached evaluation if available"""
        key = self.board_to_cache_key(board)
        return self.position_cache.get(key)

    def cache_evaluation(self, board: NDArray[np.int8], value: float):
        """Store evaluation in cache"""
        key = self.board_to_cache_key(board)
        self.position_cache[key] = value

    def select(self, node: Node) -> Node:
        """Select most promising node for expansion"""
        while (
            node.untried_moves is not None
            and len(node.untried_moves) == 0
            and len(node.children) > 0
        ):
            children_scores = {
                move: child.get_ucb_score() for move, child in node.children.items()
            }
            best_move = max(children_scores.items(), key=lambda x: x[1])[0]
            node = node.children[best_move]
        return node

    def expand(self, node: Node) -> Optional[Node]:
        """Expand selected node"""
        if not node.untried_moves:
            return None

        move = node.untried_moves.pop(0)
        new_board = node.board.clone()
        new_board.make_move(move)
        return node.add_child(move, new_board)

    def minimax(
        self,
        board: GameState,
        depth: int,
        alpha: float,
        beta: float,
        maximizing: bool,
        use_cache: bool = True,
    ) -> Tuple[int, float]:
        """Enhanced MiniMax with ML evaluation integration"""
        if time.time() - self.start_time > self.max_time * 0.95:
            return -1, 0.0  # Emergency timeout

        # Terminal state check
        if board.is_win_move(board.get_opponent(board.current_player)):
            return -1, -1.0 if maximizing else 1.0

        if depth == 0:
            # Use cached evaluation if available
            if use_cache:
                cached_value = self.get_cached_evaluation(board.board)
                if cached_value is not None:
                    return -1, cached_value

            # Get ML evaluation
            analysis = self.classifier.analyze_position(board.board)
            evaluation = analysis["win_probability"] - analysis["loss_probability"]

            if use_cache:
                self.cache_evaluation(board.board, evaluation)

            return -1, evaluation

        valid_moves = board.get_valid_moves()
        if not valid_moves:
            return -1, 0.0

        best_move = valid_moves[0]
        if maximizing:
            max_eval = float("-inf")
            for move in valid_moves:
                new_board = board.clone()
                new_board.make_move(move)
                _, eval_score = self.minimax(
                    new_board, depth - 1, alpha, beta, False, use_cache
                )
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            return best_move, max_eval
        else:
            min_eval = float("inf")
            for move in valid_moves:
                new_board = board.clone()
                new_board.make_move(move)
                _, eval_score = self.minimax(
                    new_board, depth - 1, alpha, beta, True, use_cache
                )
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = move
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            return best_move, min_eval

    def simulate(self, board: GameState) -> float:
        """Enhanced simulation with MiniMax rollout and ML evaluation"""
        if time.time() - self.start_time > self.max_time * 0.95:
            return 0.5  # Emergency timeout

        if board.is_win_move(board.get_opponent(board.current_player)):
            return 0.0

        # Get ML evaluation of current position
        analysis = self.classifier.analyze_position(board.board)
        initial_eval = analysis["win_probability"] - analysis["loss_probability"]

        # If position is clearly winning/losing, trust ML evaluation
        if abs(initial_eval) > 0.8:
            return 1.0 if initial_eval > 0 else 0.0

        # Do shallow MiniMax rollout
        _, rollout_eval = self.minimax(
            board,
            self.rollout_depth,
            float("-inf"),
            float("inf"),
            True,
            use_cache=True,
        )

        # Combine ML evaluation with rollout result
        combined_eval = 0.7 * rollout_eval + 0.3 * initial_eval
        return 1.0 if combined_eval > 0 else 0.0

    def backpropagate(self, node: Optional[Node], result: float):
        """Update statistics back up the tree"""
        while node is not None:
            node.update(result)
            node = node.parent
            result = 1 - result

    def get_move(self, board: GameState) -> int:
        """Get best move using hybrid approach"""
        self.start_time = time.time()
        valid_moves = board.get_valid_moves()

        # First priority: Find winning moves
        for move in valid_moves:
            test_board = board.clone()
            test_board.make_move(move)
            if test_board.is_win_move(
                test_board.get_opponent(test_board.current_player)
            ):
                logger.info(f"Found immediate winning move: {move + 1}")
                return move

        # Second priority: Block opponent wins
        opponent = board.get_opponent(board.current_player)
        for move in valid_moves:
            test_board = board.clone()
            test_board.current_player = opponent
            test_board.make_move(move)
            if test_board.is_win_move(opponent):
                logger.info(f"Found blocking move against opponent win: {move + 1}")
                return move

        # MCTS search with ML integration
        root = Node(board, self.classifier)
        simulation_count = 0

        # First phase: Complete minimum required simulations
        while simulation_count < self.min_simulations:
            node = self.select(root)
            child = self.expand(node)

            if child is not None:
                result = self.simulate(child.board)
                self.backpropagate(child, result)
            else:
                result = self.simulate(node.board)
                self.backpropagate(node, result)

            simulation_count += 1

        # Second phase: Continue until timeout
        while True:
            elapsed_time = time.time() - self.start_time
            if elapsed_time >= self.max_time * 0.95:
                break

            node = self.select(root)
            child = self.expand(node)

            if child is not None:
                result = self.simulate(child.board)
                self.backpropagate(child, result)
            else:
                result = self.simulate(node.board)
                self.backpropagate(node, result)

            simulation_count += 1

        # Log statistics
        elapsed_time = time.time() - self.start_time
        logger.info(
            f"\nCompleted {simulation_count} simulations in {elapsed_time:.3f}s"
        )

        logger.info("\nMove analysis:")
        move_stats = []
        for move in board.get_valid_moves():
            if move in root.children:
                child = root.children[move]
                win_rate = child.wins / max(1, child.visits)
                logger.info(
                    f"Move {move + 1}: "
                    f"Visits={child.visits}, "
                    f"Win Rate={win_rate:.3f}"
                )
                move_stats.append((move, child.visits, win_rate))

        # Select best move, considering both win rate and visit count
        best_move = max(move_stats, key=lambda x: (x[2], x[1]))[0]
        logger.info(f"\nSelected move {best_move + 1}")
        return best_move


def play_game(player1: Agent, player2: Agent):
    """Play a game between two agents"""
    board = GameState()
    game_over = False

    while not game_over:
        display_board(board)
        current_player = player1 if board.current_player == 1 else player2
        player_symbol = SYMBOLS[board.current_player]
        col = current_player.get_move(board)
        print(
            f"\nPlayer {board.current_player} {player_symbol} chooses column {col + 1}"
        )

        board.make_move(col)
        if board.is_win_move(board.get_opponent(board.current_player)):
            display_board(board)
            winner = (
                f"Player {board.get_opponent(board.current_player)} "
                + f"{SYMBOLS[board.get_opponent(board.current_player)]}"
            )
            print(f"\n{winner} wins!")
            game_over = True
        elif not board.get_valid_moves():
            display_board(board)
            print("\nIt's a draw!")
            game_over = True


def display_board(board: GameState):
    """Display the current board state"""
    print("\n")
    for row in board.board:
        print("|", end="")
        for cell in row:
            print(f"{SYMBOLS[cell]}", end="")
        print("|")
    print("-" * (WIDTH * 2 + 2))
    print(" ", end="")
    for i in range(WIDTH):
        print(f"{i + 1}", end=" ")
    print("\n")


def main():
    """Example usage of the hybrid agent"""
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
    player1 = Agent(
        classifier=classifier,
        min_simulations=500,
        max_time=5.0,
        rollout_depth=2,
    )

    player2 = Agent(
        classifier=classifier,
        min_simulations=500,
        max_time=5.0,
        rollout_depth=2,
    )

    # Play a game
    play_game(player1, player2)


if __name__ == "__main__":
    main()
