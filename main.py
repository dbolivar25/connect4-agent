from typing import List, Tuple, Optional
import random
import time
from math import sqrt, log
import numpy as np
from numpy.typing import NDArray


HEIGHT = 6
WIDTH = 7
C = sqrt(2)


class State:
    def __init__(
        self,
        board: NDArray[np.int8] = np.zeros((HEIGHT, WIDTH), dtype=np.int8),
        starting_player: int = 1,
    ):
        self.board = board
        self.current_player = starting_player

    def clone(self):
        return State(self.board.copy(), self.current_player)

    def get_valid_moves(self) -> List[int]:
        return [col for col in range(WIDTH) if self.board[0][col] == 0]

    def make_move(self, column: int) -> bool:
        if column not in self.get_valid_moves():
            return False

        # Find the lowest empty row in the column
        for row in range(5, -1, -1):
            if self.board[row][column] == 0:
                self.board[row][column] = self.current_player
                self.current_player *= -1  # Switch player
                return True
        return False

    def check_winner(self) -> Optional[int]:
        # Check horizontal, vertical, and diagonal wins
        # Returns 1 for player 1 win, -1 for player 2 win, None for no winner

        # Horizontal
        for row in range(6):
            for col in range(4):
                window = self.board[row, col : col + 4]
                if abs(sum(window)) == 4:
                    return np.sign(sum(window))

        # Vertical
        for row in range(3):
            for col in range(7):
                window = self.board[row : row + 4, col]
                if abs(sum(window)) == 4:
                    return np.sign(sum(window))

        # Diagonal (positive slope)
        for row in range(3):
            for col in range(4):
                window = [self.board[row + i][col + i] for i in range(4)]
                if abs(sum(window)) == 4:
                    return np.sign(sum(window))

        # Diagonal (negative slope)
        for row in range(3, 6):
            for col in range(4):
                window = [self.board[row - i][col + i] for i in range(4)]
                if abs(sum(window)) == 4:
                    return np.sign(sum(window))

        return None


class MCTSNode:
    def __init__(self, state: State, parent=None, parent_action=None):
        self.state = state
        self.parent = parent
        self.parent_action = parent_action
        self.children = []
        self.wins = 0
        self.visits = 0
        self.untried_actions = state.get_valid_moves()

    def uct_select_child(self) -> "MCTSNode":
        # UCT = wins/visits + C * sqrt(ln(parent_visits)/visits)
        return max(
            self.children,
            key=lambda c: c.wins / c.visits + C * sqrt(log(self.visits) / c.visits),
        )

    def add_child(self, action: int, state: State) -> "MCTSNode":
        child = MCTSNode(state, self, action)
        self.untried_actions.remove(action)
        self.children.append(child)
        return child

    def update(self, result: int):
        self.visits += 1
        self.wins += result


class Agent:
    def __init__(self, max_time=3.0):
        self.max_time = max_time
        # self.knowledge_base = KnowledgeBase()
        # self.csp_solver = CSPSolver()

    def get_move(self, state: State) -> int:
        return self._mcts_search(state)

    def _mcts_search(self, state: State) -> int:
        root = MCTSNode(state)
        end_time = time.time() + self.max_time

        # Run MCTS iterations
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
                state.make_move(random.choice(valid_moves))

            # Backpropagation
            while node is not None:
                winner = state.check_winner()
                if winner is None:
                    result = 0.5
                else:
                    result = 1 if winner == root.state.current_player else 0
                node.update(result)
                node = node.parent

        # Return the move of the most visited child
        return max(root.children, key=lambda c: c.visits).parent_action


def main():
    # Initialize game
    state = State()
    agent = Agent()

    # Main game loop
    while True:
        # Print current board
        print("\nCurrent board:")
        for row in state.board:
            print(
                " ".join(map(lambda x: "O" if x == 1 else "X" if x == -1 else ".", row))
            )

        # Get moves
        if state.current_player == 1:
            # Human player
            try:
                move = int(input("\nEnter your move (0-6): "))
                if not state.make_move(move):
                    print("Invalid move! Try again.")
                    continue
            except ValueError:
                print("Invalid input! Please enter a number between 0 and 6.")
                continue
        else:
            # AI player
            move = agent.get_move(state)
            state.make_move(move)
            print(f"\nAI plays column {move}")

        # Check for game end
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


if __name__ == "__main__":
    main()
