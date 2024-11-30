import random
from typing import Optional

from game_state import GameState


class RandomAgent:
    """A simple agent that makes random moves for Connect 4"""

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the random agent

        Args:
            seed: Optional random seed for reproducibility
        """
        self.rng = random.Random(seed) if seed is not None else random.Random()

    def get_best_move(self, state: GameState) -> int:
        """
        Choose a random valid move from the current game state

        Args:
            state: Current game state

        Returns:
            Column index for the chosen move (0-based)
        """
        valid_moves = state.get_valid_moves()
        return self.rng.choice(valid_moves)


def test_random_agent():
    """Simple test function to verify random agent behavior"""
    # Create a game state
    state = GameState()

    # Create two random agents with different seeds
    agent1 = RandomAgent(seed=42)
    agent2 = RandomAgent(seed=123)

    # Play a few moves
    moves = []
    for _ in range(5):
        current_agent = agent1 if state.current_player == 1 else agent2
        move = current_agent.get_best_move(state)
        moves.append(move)
        state.make_move(move)

    return moves


if __name__ == "__main__":
    # Run test and print results
    test_moves = test_random_agent()
    print(f"Test moves sequence: {[m+1 for m in test_moves]}")  # 1-based column numbers

    # Additional test for reproducibility
    moves1 = test_random_agent()
    moves2 = test_random_agent()
    assert moves1 == moves2, "Random agent with same seed should produce same moves"
    print("Reproducibility test passed!")
