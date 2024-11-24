from sys import stderr
from typing import List, Optional
import random
import time
from math import sqrt, log
import numpy as np
from numpy.typing import NDArray

HEIGHT = 6
WIDTH = 7

C = sqrt(2)


class GameState:
    def __init__(
        self,
        board: NDArray[np.int8],
        starting_player: int = 1,
    ):
        self.board = board
        self.current_player = starting_player

    def clone(self):
        return GameState(self.board.copy(), self.current_player)

    def get_valid_moves(self) -> List[int]:
        return [col for col in range(7) if self.board[0][col] == 0]

    def make_move(self, column: int) -> bool:
        if column not in self.get_valid_moves():
            return False

        for row in range(5, -1, -1):
            if self.board[row][column] == 0:
                self.board[row][column] = self.current_player
                self.current_player *= -1
                return True
        return False

    def check_winner(self) -> Optional[int]:
        # Horizontal
        windows = np.lib.stride_tricks.sliding_window_view(self.board, 4, axis=1)
        sums = np.sum(windows, axis=2)  # Changed from axis=3 to axis=2
        if np.any(np.abs(sums) == 4):
            return int(np.sign(sums[np.abs(sums) == 4][0]))

        # Vertical
        windows = np.lib.stride_tricks.sliding_window_view(self.board, 4, axis=0)
        sums = np.sum(windows, axis=2)  # This one was correct
        if np.any(np.abs(sums) == 4):
            return int(np.sign(sums[np.abs(sums) == 4][0]))

        # Diagonal (positive slope)
        diags = []
        for i in range(3):
            for j in range(4):
                diags.append(np.diagonal(self.board[i : i + 4, j : j + 4]))
        diag_sums = np.sum(diags, axis=1)
        if np.any(np.abs(diag_sums) == 4):
            return int(np.sign(diag_sums[np.abs(diag_sums) == 4][0]))

        # Diagonal (negative slope)
        flipped_board = np.flipud(self.board)
        diags = []
        for i in range(3):
            for j in range(4):
                diags.append(np.diagonal(flipped_board[i : i + 4, j : j + 4]))
        diag_sums = np.sum(diags, axis=1)
        if np.any(np.abs(diag_sums) == 4):
            return int(np.sign(diag_sums[np.abs(diag_sums) == 4][0]))

        return None


class MCTSNode:
    def __init__(self, state: GameState, parent=None, parent_action=None):
        self.state = state
        self.parent = parent
        self.parent_action = parent_action
        self.children = []
        self.score = 0.0
        self.visits = 0
        self.untried_actions = state.get_valid_moves()

    def uct_select_child(self) -> "MCTSNode":
        return max(
            self.children,
            key=lambda c: (
                c.score / c.visits + C * sqrt(log(self.visits) / c.visits)
                if c.visits != 0
                else float("inf")
            ),
        )

    def add_child(self, action: int, state: GameState) -> "MCTSNode":
        child = MCTSNode(state, self, action)
        self.untried_actions.remove(action)
        self.children.append(child)
        return child

    def update(self, result_score: float):
        self.visits += 1
        self.score += result_score


class Agent:
    def __init__(self, max_time=3.0):
        self.max_time = max_time

    def get_move(self, state: GameState) -> int:
        return self._check_threats(state) or self._mcts_search(state)

    def _check_threats(self, state: GameState) -> Optional[int]:
        # Check for our winning move first
        for move in state.get_valid_moves():
            test_state = state.clone()
            test_state.make_move(move)
            if test_state.check_winner() == state.current_player:
                return move

        # Check for opponent's winning move that we must block
        opponent = -state.current_player
        for move in state.get_valid_moves():
            test_state = state.clone()
            test_state.current_player = opponent  # Temporarily switch player
            test_state.make_move(move)
            if test_state.check_winner() == opponent:
                return move

        return None

    def _mcts_search(self, state: GameState) -> int:
        root = MCTSNode(state)
        end_time = time.time() + self.max_time

        while time.time() < end_time:
            node = root

            # Selection
            while node.untried_actions == [] and node.children != []:
                node = node.uct_select_child()

            # Expansion
            if node.untried_actions != []:
                action = random.choice(node.untried_actions)
                state = node.state.clone()
                state.make_move(action)
                node = node.add_child(action, state)

            # Simulation
            state = node.state.clone()
            while not state.get_valid_moves() == []:
                winner = state.check_winner()
                if winner is not None:
                    break
                valid_moves = state.get_valid_moves()
                if not valid_moves:
                    break
                state.make_move(random.choice(valid_moves))

            # Backpropagation
            while node is not None:
                winner = state.check_winner()
                if winner is None:
                    result = 0.5
                else:
                    result = 1.0 if winner == root.state.current_player else 0.0
                node.update(result)
                node = node.parent

        return max(root.children, key=lambda c: c.visits).parent_action


def main():
    state = GameState(np.zeros((HEIGHT, WIDTH), dtype=np.int8))
    agent = Agent()

    while True:
        print("\nCurrent board:")
        for row in state.board:
            print(
                " ".join(map(lambda x: "O" if x == 1 else "X" if x == -1 else ".", row))
            )

        if state.current_player == 1:
            try:
                move = int(input("\nEnter your move (0-6): "))
                if not state.make_move(move):
                    print("Invalid move! Try again.")
                    continue
            except ValueError:
                print("Invalid input! Please enter a number between 0 and 6.")
                continue
        else:
            move = agent.get_move(state)
            state.make_move(move)
            print(f"\nAI plays column {move}")

        winner = state.check_winner()
        if winner is not None:
            print("\nFinal board:")
            for row in state.board:
                print(
                    " ".join(
                        map(lambda x: "O" if x == 1 else "X" if x == -1 else ".", row)
                    )
                )
            print(f"\n{'Human' if winner == 1 else 'AI'} wins!")
            break

        if not state.get_valid_moves():
            print("\nGame over - It's a draw!")
            break


def play_agent_vs_agent(
    num_games: int = 1, display_moves: bool = True, max_time: float = 1.0
):
    agent1 = Agent(max_time)
    agent2 = Agent(max_time)
    stats = {"agent1_wins": 0, "agent2_wins": 0, "draws": 0}

    def display_board(board):
        print("\nCurrent board:")
        for row in board:
            print(
                " ".join(map(lambda x: "O" if x == 1 else "X" if x == -1 else ".", row))
            )
        print()  # Empty line for readability

    for game in range(num_games):
        state = GameState(np.zeros((HEIGHT, WIDTH), dtype=np.int8))

        moves_played = 0

        print(f"\nGame {game + 1}/{num_games}")
        if display_moves:
            display_board(state.board)

        while True:
            # Get current agent
            current_agent = agent1 if state.current_player == 1 else agent2
            agent_name = "Agent 1" if state.current_player == 1 else "Agent 2"

            # Get and make move
            move = current_agent.get_move(state)
            state.make_move(move)
            moves_played += 1

            if display_moves:
                print(f"{agent_name} plays column {move}")
                display_board(state.board)

            # Check for winner
            winner = state.check_winner()
            if winner is not None:
                if not display_moves:
                    display_board(state.board)
                if winner == 1:
                    stats["agent1_wins"] += 1
                    print(f"Agent 1 wins in {moves_played} moves!")
                    print(f"Agent 1 wins in {moves_played} moves!", file=stderr)
                else:
                    stats["agent2_wins"] += 1
                    print(f"Agent 2 wins in {moves_played} moves!")
                    print(f"Agent 2 wins in {moves_played} moves!", file=stderr)
                break

            # Check for draw
            if not state.get_valid_moves():
                if not display_moves:
                    display_board(state.board)
                stats["draws"] += 1
                print(f"Game is a draw after {moves_played} moves!")
                print(f"Game is a draw after {moves_played} moves!", file=stderr)
                break

    # Print final statistics
    print("\nFinal Statistics:")
    print(f"Total games: {num_games}")
    print(
        f"Agent 1 wins: {stats['agent1_wins']} ({stats['agent1_wins']/num_games*100:.1f}%)"
    )
    print(
        f"Agent 2 wins: {stats['agent2_wins']} ({stats['agent2_wins']/num_games*100:.1f}%)"
    )
    print(f"Draws: {stats['draws']} ({stats['draws']/num_games*100:.1f}%)")


if __name__ == "__main__":
    NUM_GAMES = 10  # Number of games to play
    DISPLAY_MOVES = True  # Whether to display each move
    MAX_TIME = 3.0  # Seconds per move

    play_agent_vs_agent(NUM_GAMES, DISPLAY_MOVES, MAX_TIME)


# if __name__ == "__main__":
#     main()
