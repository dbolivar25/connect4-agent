import math
import random
import time
from typing import Dict, Optional


from game_state import GameState


class MCTSNode:
    """Node in the MCTS tree"""

    def __init__(
        self,
        state: GameState,
        parent: Optional["MCTSNode"] = None,
        move: Optional[int] = None,
    ):
        self.state = state
        self.parent = parent
        self.move = move  # Move that led to this state
        self.wins = 0
        self.visits = 0
        self.children: Dict[int, MCTSNode] = {}  # Map moves to child nodes
        self.untried_moves = state.get_valid_moves()

    def add_child(self, move: int, state: GameState) -> "MCTSNode":
        """Create a child node with the given move and state"""
        child = MCTSNode(state=state, parent=self, move=move)
        self.untried_moves.remove(move)
        self.children[move] = child
        return child

    def update(self, result: float):
        """Update node statistics"""
        self.visits += 1
        self.wins += result

    def get_uct_score(self, exploration_constant: float) -> float:
        """Calculate UCT score for node selection"""
        if self.visits == 0:
            return float("inf")

        exploitation = self.wins / self.visits
        exploration = (
            exploration_constant * math.sqrt(math.log(self.parent.visits) / self.visits)  # type: ignore
            if self.parent
            else 0
        )

        return exploitation + exploration


class MCTSAgent:
    """Basic MCTS agent for Connect 4"""

    def __init__(
        self,
        simulation_time: float = 1.0,
        exploration_constant: float = math.sqrt(2),
        seed: Optional[int] = None,
    ):
        """
        Initialize MCTS agent

        Args:
            simulation_time: Time limit for MCTS simulations in seconds
            exploration_constant: UCT exploration parameter
            seed: Random seed for reproducibility
        """
        self.simulation_time = simulation_time
        self.exploration_constant = exploration_constant
        self.rng = random.Random(seed) if seed is not None else random.Random()

    def get_best_move(self, state: GameState) -> int:
        """Find best move using MCTS within time limit"""
        root = MCTSNode(state=state.clone())
        end_time = time.time() + self.simulation_time

        # Run MCTS iterations until time limit
        while time.time() < end_time:
            node = self._select(root)
            if node.state.check_win() is None and node.untried_moves:
                node = self._expand(node)
            result = self._simulate(node.state)
            self._backpropagate(node, result)

        # Choose move with most visits
        best_move = max(
            root.children.items(),
            key=lambda x: (x[1].visits, self.rng.random()),  # Random tiebreak
        )[0]

        return best_move

    def _select(self, node: MCTSNode) -> MCTSNode:
        """Select a node to expand using UCT"""
        while node.state.check_win() is None and not node.untried_moves:
            if not node.children:
                return node

            node = max(
                node.children.values(),
                key=lambda n: n.get_uct_score(self.exploration_constant),
            )

        return node

    def _expand(self, node: MCTSNode) -> MCTSNode:
        """Expand selected node with a random untried move"""
        move = self.rng.choice(node.untried_moves)
        new_state = node.state.clone()
        new_state.make_move(move)
        return node.add_child(move, new_state)

    def _simulate(self, state: GameState) -> float:
        """Simulate random playout from given state"""
        current = state.clone()

        while True:
            result = current.check_win()
            if result is not None:
                # Convert to result relative to original player
                return result * state.current_player

            valid_moves = current.get_valid_moves()
            move = self.rng.choice(valid_moves)
            current.make_move(move)

    def _backpropagate(self, node: MCTSNode, result: float):
        """Backpropagate simulation result up the tree"""
        while node:
            node.update(result)
            node = node.parent  # type: ignore
            result = -result  # Flip result for opponent

    def clear_cache(self):
        """
        Dummy method to match Agent interface
        MCTS doesn't use persistent caching
        """
        pass


def test_mcts_agent():
    """Simple test function for MCTS agent"""
    # Create a game state
    state = GameState()

    # Create MCTS agent with short simulation time
    agent = MCTSAgent(simulation_time=0.1, seed=42)

    # Play a few moves
    moves = []
    for _ in range(5):
        move = agent.get_best_move(state)
        moves.append(move)
        state.make_move(move)

    return moves


if __name__ == "__main__":
    # Run test and print results
    test_moves = test_mcts_agent()
    print(f"Test moves sequence: {[m+1 for m in test_moves]}")  # 1-based columns

    # Test reproducibility
    moves1 = test_mcts_agent()
    moves2 = test_mcts_agent()
    assert moves1 == moves2, "MCTS with same seed should produce same moves"
    print("Reproducibility test passed!")
