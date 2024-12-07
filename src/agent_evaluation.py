import os
import time
import numpy as np
from datetime import datetime
from typing import Any, Dict, Final, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from negamax_agent import NegamaxAgent
from random_agent import RandomAgent
from mcts_agent import MCTSAgent
from game_state import GameState
from position_classifier import PositionClassifier

SYMBOLS: Final[Dict[int, str]] = {0: "⚪", 1: "🔴", -1: "🔵"}
NUM_GAMES: Final[int] = 20
MCTS_SIMULATION_TIME: Final[float] = 150.0  # seconds
MAX_WORKERS: Final[int] = 14  # Number of CPU cores


class GameStats:
    """Enhanced class to track game statistics"""

    def __init__(self):
        self.wins_p1: int = 0
        self.wins_p2: int = 0
        self.draws: int = 0
        self.game_lengths: List[int] = []
        self.total_time: float = 0.0
        self.move_times_p1: List[float] = []
        self.move_times_p2: List[float] = []
        self.first_win_move: Optional[int] = None

    def merge(self, other: "GameStats"):
        """Merge another GameStats instance into this one"""
        self.wins_p1 += other.wins_p1
        self.wins_p2 += other.wins_p2
        self.draws += other.draws
        self.game_lengths.extend(other.game_lengths)
        self.total_time += other.total_time
        self.move_times_p1.extend(other.move_times_p1)
        self.move_times_p2.extend(other.move_times_p2)
        if other.first_win_move is not None:
            if (
                self.first_win_move is None
                or other.first_win_move < self.first_win_move
            ):
                self.first_win_move = other.first_win_move

    @property
    def total_games(self) -> int:
        return self.wins_p1 + self.wins_p2 + self.draws

    @property
    def avg_game_length(self) -> float:
        return float(np.mean(self.game_lengths) if self.game_lengths else 0)

    @property
    def std_game_length(self) -> float:
        return float(np.std(self.game_lengths) if self.game_lengths else 0)

    @property
    def min_game_length(self) -> int:
        return min(self.game_lengths) if self.game_lengths else 0

    @property
    def max_game_length(self) -> int:
        return max(self.game_lengths) if self.game_lengths else 0

    @property
    def avg_game_time(self) -> float:
        return self.total_time / self.total_games if self.total_games > 0 else 0

    @property
    def avg_move_time_p1(self) -> float:
        return float(np.mean(self.move_times_p1) if self.move_times_p1 else 0)

    @property
    def avg_move_time_p2(self) -> float:
        return float(np.mean(self.move_times_p2) if self.move_times_p2 else 0)

    @property
    def std_move_time_p1(self) -> float:
        return float(np.std(self.move_times_p1) if self.move_times_p1 else 0)

    @property
    def std_move_time_p2(self) -> float:
        return float(np.std(self.move_times_p2) if self.move_times_p2 else 0)


def play_test_game(
    player1: Any, player2: Any, game_id: int = 0
) -> Tuple[int, int, float, List[float], List[float]]:
    """Play a single test game between two agents"""
    state = GameState()
    start_time = time.time()
    p1_move_times = []
    p2_move_times = []

    while True:
        current_agent = player1 if state.current_player == 1 else player2
        move_start = time.time()
        move = current_agent.get_best_move(state)
        move_time = time.time() - move_start

        if state.current_player == 1:
            p1_move_times.append(move_time)
        else:
            p2_move_times.append(move_time)

        state.make_move(move)
        result = state.check_win()

        if result is not None:
            game_time = time.time() - start_time
            return result, state.ply_count, game_time, p1_move_times, p2_move_times


def process_game_result(
    result: Tuple[int, int, float, List[float], List[float]],
) -> GameStats:
    """Process a single game result into GameStats"""
    stats = GameStats()
    game_result, moves, game_time, p1_times, p2_times = result

    stats.game_lengths.append(moves)
    stats.total_time += game_time
    stats.move_times_p1.extend(p1_times)
    stats.move_times_p2.extend(p2_times)

    if game_result != 0:
        stats.first_win_move = moves

    if game_result == 1:
        stats.wins_p1 += 1
    elif game_result == -1:
        stats.wins_p2 += 1
    else:
        stats.draws += 1

    return stats


def run_test_series(
    player1: Any,
    player2: Any,
    p1_name: str,
    p2_name: str,
    num_games: int = NUM_GAMES,
) -> GameStats:
    """Run a series of test games between two agents in parallel"""
    print(f"\nRunning {num_games} games: {p1_name} vs {p2_name}")

    combined_stats = GameStats()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all games
        future_to_game = {
            executor.submit(play_test_game, player1, player2, i): i
            for i in range(num_games)
        }

        # Process results as they complete
        with tqdm(total=num_games, desc="Games", unit="game") as pbar:
            for future in as_completed(future_to_game):
                try:
                    result = future.result()
                    game_stats = process_game_result(result)
                    combined_stats.merge(game_stats)
                    pbar.update(1)
                except Exception as e:
                    print(f"Game {future_to_game[future]} generated an exception: {e}")

    return combined_stats


def format_stats_markdown(stats: GameStats, p1_name: str, p2_name: str) -> str:
    """Format enhanced game statistics as markdown"""
    win_rate_p1 = (stats.wins_p1 / stats.total_games) * 100
    win_rate_p2 = (stats.wins_p2 / stats.total_games) * 100
    draw_rate = (stats.draws / stats.total_games) * 100

    return f"""### {p1_name} vs {p2_name}

#### Game Outcomes
- Total Games: {stats.total_games}
- {p1_name} Wins: {stats.wins_p1} ({win_rate_p1:.1f}%)
- {p2_name} Wins: {stats.wins_p2} ({win_rate_p2:.1f}%)
- Draws: {stats.draws} ({draw_rate:.1f}%)

#### Game Length Statistics
- Average Length: {stats.avg_game_length:.1f} moves
- Standard Deviation: {stats.std_game_length:.1f} moves
- Minimum Length: {stats.min_game_length} moves
- Maximum Length: {stats.max_game_length} moves
- Earliest Win: {stats.first_win_move if stats.first_win_move is not None else 'N/A'} moves

#### Timing Statistics
- Average Game Time: {stats.avg_game_time:.2f} seconds
- {p1_name} Move Times:
  - Average: {stats.avg_move_time_p1:.3f} seconds
  - Std Dev: {stats.std_move_time_p1:.3f} seconds
- {p2_name} Move Times:
  - Average: {stats.avg_move_time_p2:.3f} seconds
  - Std Dev: {stats.std_move_time_p2:.3f} seconds
- Total Series Time: {stats.total_time:.1f} seconds

"""


def main():
    """Run comprehensive agent testing"""
    print("Starting comprehensive Connect 4 agent testing...")
    start_time = time.time()

    # Load classifier
    classifier = PositionClassifier()
    model_path = "models/connect4_analyzer_final.pkl"

    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        classifier.load_model(model_path)
    else:
        print("Training new classifier model...")
        classifier.train(save_path=model_path, tune_params=False, balance_data=True)

    # Create agents
    random_agent = RandomAgent(seed=42)
    mcts_agent = MCTSAgent(simulation_time=MCTS_SIMULATION_TIME, seed=42)
    negamax_agent = NegamaxAgent(classifier=classifier)

    # Run test series
    results = []

    # Random vs Negamax
    stats_random = run_test_series(
        random_agent, negamax_agent, "Random Agent", "Negamax Agent"
    )
    results.append((stats_random, "Random Agent", "Negamax Agent"))

    # MCTS vs Negamax
    stats_mcts = run_test_series(
        mcts_agent, negamax_agent, "MCTS Agent", "Negamax Agent"
    )
    results.append((stats_mcts, "MCTS Agent", "Negamax Agent"))

    # Negamax vs Negamax
    stats_negamax = run_test_series(
        NegamaxAgent(classifier=classifier),
        NegamaxAgent(classifier=classifier),
        "Negamax Agent 1",
        "Negamax Agent 2",
    )
    results.append((stats_negamax, "Negamax Agent 1", "Negamax Agent 2"))

    # Generate markdown report
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    total_time = time.time() - start_time

    report = f"""# Connect 4 Agent Testing Results
Test run completed on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Test Configuration
- Games per matchup: {NUM_GAMES}
- MCTS simulation time: {MCTS_SIMULATION_TIME} seconds
- Parallel games: {MAX_WORKERS}
- Random seed: 42
- Total test time: {total_time:.1f} seconds

## Results

"""

    for stats, p1_name, p2_name in results:
        report += format_stats_markdown(stats, p1_name, p2_name)

    # Save report
    os.makedirs("results", exist_ok=True)
    filename = f"results/test_results_{timestamp}.md"
    with open(filename, "w") as f:
        f.write(report)

    print(f"\nTesting complete! Results saved to {filename}")
    print(f"Total time: {total_time:.1f} seconds")


if __name__ == "__main__":
    main()
