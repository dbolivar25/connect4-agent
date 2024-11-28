import numpy as np
import time
from typing import Dict, Tuple, List, Optional
from numpy.typing import NDArray
import random
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

HEIGHT = 6
WIDTH = 7
SEARCH_ORDER = [3, 2, 4, 1, 5, 0, 6]  # Center-out search order
EXPLORATION_CONSTANT = 1.41  # Standard UCT constant
SYMBOLS = {0: "⚪", 1: "🔴", 2: "🔵"}


# Precompute all possible winning positions
def generate_win_positions():
    positions = []
    # Horizontal
    for row in range(HEIGHT):
        for col in range(WIDTH - 3):
            positions.append([(row, col + i) for i in range(4)])
    # Vertical
    for row in range(HEIGHT - 3):
        for col in range(WIDTH):
            positions.append([(row + i, col) for i in range(4)])
    # Diagonal (positive slope)
    for row in range(HEIGHT - 3):
        for col in range(WIDTH - 3):
            positions.append([(row + i, col + i) for i in range(4)])
    # Diagonal (negative slope)
    for row in range(3, HEIGHT):
        for col in range(WIDTH - 3):
            positions.append([(row - i, col + i) for i in range(4)])
    return positions


WIN_POSITIONS = generate_win_positions()


class GameState:
    def __init__(
        self,
        board: Optional[NDArray[np.int8]] = None,
        starting_player: int = 1,
        last_move: Optional[Tuple[int, int]] = None,
    ):
        self.board = (
            board if board is not None else np.zeros((HEIGHT, WIDTH), dtype=np.int8)
        )
        self.current_player = starting_player
        self.last_move = last_move
        self._height_map = np.array(
            [HEIGHT - 1] * WIDTH
        )  # Track lowest empty row for each column

        # Update height map for existing pieces
        if board is not None:
            for col in range(WIDTH):
                for row in range(HEIGHT):
                    if board[row][col] != 0:
                        self._height_map[col] = row - 1

    def clone(self):
        new_state = GameState(self.board.copy(), self.current_player, self.last_move)
        new_state._height_map = self._height_map.copy()
        return new_state

    def get_valid_moves(self) -> List[int]:
        return [col for col in SEARCH_ORDER if self._height_map[col] >= 0]

    def make_move(self, col: int) -> bool:
        if self._height_map[col] < 0:
            return False

        row = self._height_map[col]
        self.board[row][col] = self.current_player
        self.last_move = (row, col)
        self._height_map[col] -= 1
        self.current_player = self.get_opponent(self.current_player)
        return True

    def get_opponent(self, player: int) -> int:
        return 3 - player

    def is_win_move(self, player: int) -> bool:
        if self.last_move is None:
            return False

        row, col = self.last_move

        # Only check lines that could contain the last move
        for win_pos in WIN_POSITIONS:
            if (row, col) in win_pos:
                if all(self.board[r][c] == player for r, c in win_pos):
                    return True
        return False

    def evaluate_position(self) -> float:
        """Enhanced tactical evaluation with better threat recognition"""
        if self.is_win_move(self.get_opponent(self.current_player)):
            return -1.0  # Loss

        opponent = self.get_opponent(self.current_player)
        score = 0.0

        # Track vertical threats specifically
        vertical_threats = {
            "our_uncontested": 0,
            "opp_uncontested": 0,
            "our_contested": 0,
            "opp_contested": 0,
        }

        for win_pos in WIN_POSITIONS:
            our_count = opp_count = empty_count = 0
            empty_positions = []
            is_vertical = all(
                win_pos[i][1] == win_pos[0][1] for i in range(len(win_pos))
            )

            for r, c in win_pos:
                cell = self.board[r][c]
                if cell == self.current_player:
                    our_count += 1
                elif cell == opponent:
                    opp_count += 1
                else:
                    empty_count += 1
                    empty_positions.append((r, c))

            # Check if positions are immediately playable
            playable_empty = sum(
                1 for r, c in empty_positions if self._height_map[c] == r
            )

            # Vertical threats are especially dangerous - detect and prioritize them early
            if is_vertical:
                if opp_count == 0 and our_count >= 2 and playable_empty > 0:
                    vertical_threats["our_uncontested"] += 1
                    score += (
                        0.4 * our_count
                    )  # Increase score for each piece in vertical line
                elif our_count == 0 and opp_count >= 2 and playable_empty > 0:
                    vertical_threats["opp_uncontested"] += 1
                    score -= (
                        0.5 * opp_count
                    )  # Penalize more for opponent vertical threats

            # Enhanced threat detection
            if opp_count == 0:
                if our_count == 3 and playable_empty == 1:
                    score += 0.9  # Near win
                elif our_count == 2:
                    if playable_empty == 2:
                        score += 0.4  # Strong developing threat
                        # Extra bonus for connected pieces
                        if any(
                            abs(win_pos[i][0] - win_pos[i + 1][0]) == 1
                            and win_pos[i][1] == win_pos[i + 1][1]
                            for i in range(len(win_pos) - 1)
                        ):
                            score += 0.2
            elif our_count == 0:
                if opp_count == 3 and playable_empty == 1:
                    score -= 0.95  # Must block immediately
                elif opp_count == 2:
                    if playable_empty == 2:
                        score -= 0.5  # Opponent developing threat
                        # Extra penalty for connected opponent pieces
                        if any(
                            abs(win_pos[i][0] - win_pos[i + 1][0]) == 1
                            and win_pos[i][1] == win_pos[i + 1][1]
                            for i in range(len(win_pos) - 1)
                        ):
                            score -= 0.3

        # Additional penalties for uncontested vertical threats
        if vertical_threats["opp_uncontested"] > 0:
            score -= 0.4 * vertical_threats["opp_uncontested"]

        # Center control bonus
        center_column = 3
        center_count = sum(
            1
            for row in range(HEIGHT)
            if self.board[row][center_column] == self.current_player
        )
        score += 0.1 * center_count

        return np.clip(score, -0.9, 0.9)


class Node:
    def __init__(
        self,
        board: GameState,
        parent: Optional["Node"] = None,
        move: Optional[int] = None,
    ):
        self.board = board
        self.parent = parent
        self.move = move
        self.children: Dict[int, "Node"] = {}
        self.visits = 0
        self.wins = 0
        self.untried_moves = self.board.get_valid_moves()

    def add_child(self, move: int, board: GameState) -> "Node":
        child = Node(board, self, move)
        self.children[move] = child
        return child

    def update(self, result: float):
        self.visits += 1
        self.wins += result

    def get_ucb_score(self) -> float:
        if self.visits == 0:
            return float("inf")

        exploitation = self.wins / self.visits
        exploration = EXPLORATION_CONSTANT * np.sqrt(
            np.log(self.parent.visits) / self.visits
        )
        return exploitation + exploration


class Agent:
    def __init__(
        self,
        minimax_depth: int = 8,
        min_simulations: int = 2000,
        max_time: float = 5.0,
        fast_win_check: bool = True,
    ):
        self.minimax_depth = minimax_depth
        self.min_simulations = min_simulations
        self.max_time = max_time
        self.start_time = 0
        self.fast_win_check = fast_win_check
        self.position_cache = {}  # Add position cache

    def select(self, node: Node) -> Node:
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
        if not node.untried_moves:
            return None

        move = node.untried_moves.pop(0)
        new_board = node.board.clone()
        new_board.make_move(move)
        return node.add_child(move, new_board)

    def minimax(
        self, board: GameState, depth: int, alpha: float, beta: float, maximizing: bool
    ) -> Tuple[int, float]:
        if time.time() - self.start_time > self.max_time * 0.95:
            return -1, 0.0  # Emergency timeout

        is_terminal = board.is_win_move(board.get_opponent(board.current_player))
        if is_terminal:
            return -1, -1.0

        if depth == 0:
            return -1, board.evaluate_position()

        valid_moves = board.get_valid_moves()
        if not valid_moves:
            return -1, 0.0

        best_move = valid_moves[0]
        if maximizing:
            max_eval = float("-inf")
            for move in valid_moves:
                new_board = board.clone()
                new_board.make_move(move)
                _, eval_score = self.minimax(new_board, depth - 1, alpha, beta, False)
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
                _, eval_score = self.minimax(new_board, depth - 1, alpha, beta, True)
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = move
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            return best_move, min_eval

    def simulate(self, board: GameState) -> float:
        """Enhanced simulation with better tactical awareness"""
        if time.time() - self.start_time > self.max_time * 0.95:
            return 0.5  # Emergency timeout

        if board.is_win_move(board.get_opponent(board.current_player)):
            return 0.0

        # Evaluate current position
        score = board.evaluate_position()

        # If position is critical, do deeper search
        if abs(score) > 0.6:
            _, eval_score = self.minimax(
                board, self.minimax_depth, float("-inf"), float("inf"), True
            )
            return 1.0 if eval_score > 0 else 0.0

        # Tactical playout
        current_board = board.clone()
        while True:
            valid_moves = current_board.get_valid_moves()
            if not valid_moves:
                return 0.5

            # Evaluate all possible moves
            move_scores = []
            for move in valid_moves:
                test_board = current_board.clone()
                test_board.make_move(move)
                score = test_board.evaluate_position()
                move_scores.append((move, score))

            # Choose best move with high probability
            move_scores.sort(key=lambda x: x[1], reverse=True)
            if random.random() < 0.9:  # 90% chance to pick one of the top moves
                move = move_scores[0][0]
            else:
                move = random.choice(valid_moves)

            current_board.make_move(move)
            if current_board.is_win_move(
                current_board.get_opponent(current_board.current_player)
            ):
                return (
                    1.0 if current_board.current_player != board.current_player else 0.0
                )

    def backpropagate(self, node: Optional[Node], result: float):
        while node is not None:
            node.update(result)
            node = node.parent
            result = 1 - result

    def get_move(self, board: GameState) -> int:
        """Enhanced move selection with more aggressive win detection"""
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

        # MCTS search
        root = Node(board)
        simulation_count = 0
        strong_move_found = False

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
        if strong_move_found:
            logger.info("Stopped early due to finding a strong move")
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
    board = GameState()
    game_over = False
    while not game_over:
        # print("\033[H\033[J")  # Clear screen
        display_board(board)
        current_player = player1 if board.current_player == 1 else player2
        player_symbol = SYMBOLS[board.current_player]
        col = current_player.get_move(board)
        print(
            f"\nPlayer {board.current_player} {player_symbol} chooses column {col + 1}"
        )
        board.make_move(col)
        if board.is_win_move(board.get_opponent(board.current_player)):
            # print("\033[H\033[J")
            display_board(board)
            winner = (
                f"Player {board.get_opponent(board.current_player)} "
                + f"{SYMBOLS[board.get_opponent(board.current_player)]}"
            )
            print(f"\n{winner} wins!")
            game_over = True
        elif not board.get_valid_moves():
            # print("\033[H\033[J")
            display_board(board)
            print("\nIt's a draw!")
            game_over = True


def display_board(board: GameState):
    print("\n")
    for row in board.board:
        print("|", end="")
        for cell in row:
            print(f"{SYMBOLS[cell]}", end="")
        print("|")
    print("")


def main():
    player1 = Agent(minimax_depth=8, min_simulations=6000, max_time=6.0)
    player2 = Agent(minimax_depth=8, min_simulations=6000, max_time=6.0)
    play_game(player1, player2)


if __name__ == "__main__":
    main()
