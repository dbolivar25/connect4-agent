"""Quick benchmark to verify improvements."""
import time
import numpy as np
from game_state import GameState
from position_classifier import PositionClassifier
from negamax_agent import NegamaxAgent
from random_agent import RandomAgent
from mcts_agent import MCTSAgent


def benchmark_game_state():
    """Benchmark core game state operations."""
    print("=" * 50)
    print("GAME STATE PERFORMANCE")
    print("=" * 50)

    # Setup a mid-game position
    state = GameState()
    for col in [3, 2, 4, 1, 3, 2, 4]:
        state.make_move(col)

    # Benchmark cloning
    n = 100000
    start = time.time()
    for _ in range(n):
        _ = state.clone()
    clone_time = time.time() - start
    print(f"Clone:     {n/clone_time:,.0f} ops/sec")

    # Benchmark win check
    start = time.time()
    for _ in range(n):
        _ = state.check_win()
    check_time = time.time() - start
    print(f"Win check: {n/check_time:,.0f} ops/sec")

    # Benchmark move generation
    start = time.time()
    for _ in range(n):
        _ = state.get_valid_moves()
    move_time = time.time() - start
    print(f"Get moves: {n/move_time:,.0f} ops/sec")

    # Benchmark has_winning_move
    start = time.time()
    for _ in range(n):
        _ = state.has_winning_move()
    win_move_time = time.time() - start
    print(f"Win move:  {n/win_move_time:,.0f} ops/sec")


def play_games(agent1, agent2, n_games=10, name1="Agent1", name2="Agent2"):
    """Play n games and return stats."""
    wins = {1: 0, -1: 0, 0: 0}
    move_times = []
    game_lengths = []

    for game in range(n_games):
        state = GameState()
        game_move_times = []

        while True:
            current = agent1 if state.current_player == 1 else agent2

            start = time.time()
            move = current.get_best_move(state)
            game_move_times.append(time.time() - start)

            state.make_move(move)
            result = state.check_win()

            if result is not None:
                wins[result] += 1
                game_lengths.append(state.ply_count)
                move_times.extend(game_move_times)
                break

    return {
        "wins_p1": wins[1],
        "wins_p2": wins[-1],
        "draws": wins[0],
        "avg_length": np.mean(game_lengths),
        "avg_move_time": np.mean(move_times),
        "total_time": sum(move_times)
    }


def main():
    print("\n" + "=" * 50)
    print("CONNECT 4 BOT BENCHMARK")
    print("=" * 50 + "\n")

    # Benchmark game state
    benchmark_game_state()

    # Load classifier
    print("\nLoading ML model...")
    classifier = PositionClassifier()
    classifier.load_model("models/connect4_analyzer_final.pkl")

    # Create agents
    negamax = NegamaxAgent(classifier)
    random_agent = RandomAgent(seed=42)
    mcts = MCTSAgent(simulation_time=1.0, seed=42)  # 1 second per move

    # Test 1: Negamax vs Random
    print("\n" + "=" * 50)
    print("NEGAMAX vs RANDOM (10 games)")
    print("=" * 50)

    stats = play_games(negamax, random_agent, n_games=10,
                       name1="Negamax", name2="Random")

    print(f"Negamax wins: {stats['wins_p1']}/10")
    print(f"Random wins:  {stats['wins_p2']}/10")
    print(f"Draws:        {stats['draws']}/10")
    print(f"Avg game:     {stats['avg_length']:.1f} moves")
    print(f"Avg move:     {stats['avg_move_time']*1000:.1f}ms")
    print(f"Total time:   {stats['total_time']:.1f}s")

    # Test 2: Negamax vs MCTS (fewer games since MCTS is slow)
    print("\n" + "=" * 50)
    print("NEGAMAX vs MCTS (5 games, MCTS has 1s/move)")
    print("=" * 50)

    stats = play_games(negamax, mcts, n_games=5,
                       name1="Negamax", name2="MCTS")

    print(f"Negamax wins: {stats['wins_p1']}/5")
    print(f"MCTS wins:    {stats['wins_p2']}/5")
    print(f"Draws:        {stats['draws']}/5")
    print(f"Avg game:     {stats['avg_length']:.1f} moves")
    print(f"Avg move:     {stats['avg_move_time']*1000:.1f}ms")

    # Test 3: Negamax self-play
    print("\n" + "=" * 50)
    print("NEGAMAX vs NEGAMAX (5 games)")
    print("=" * 50)

    negamax2 = NegamaxAgent(classifier)
    stats = play_games(negamax, negamax2, n_games=5,
                       name1="Negamax1", name2="Negamax2")

    print(f"First player wins:  {stats['wins_p1']}/5")
    print(f"Second player wins: {stats['wins_p2']}/5")
    print(f"Draws:              {stats['draws']}/5")
    print(f"Avg game:           {stats['avg_length']:.1f} moves")
    print(f"Avg move:           {stats['avg_move_time']*1000:.1f}ms")

    print("\n" + "=" * 50)
    print("BENCHMARK COMPLETE")
    print("=" * 50)


if __name__ == "__main__":
    main()
