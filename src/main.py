import os
import random
from typing import Any, Dict, Final, Optional

from negamax_agent import NegamaxAgent
from random_agent import RandomAgent
from mcts_agent import MCTSAgent
from game_state import GameState
from position_classifier import PositionClassifier

SYMBOLS: Final[Dict[int, str]] = {0: "⚪", 1: "🔴", -1: "🔵"}


def play_game(player1: NegamaxAgent, player2: NegamaxAgent):
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
    player1: Any,
    player2: Optional[Any] = None,
    human_starts: bool = True,
    player1_name: str = "Player 1",
    player2_name: str = "Player 2",
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
        print(f"\n{player1_name} {SYMBOLS[1]} will play first")

    while True:
        display_board(state)
        player_symbol = SYMBOLS[state.current_player]

        # Determine current player's move
        if mode == "agent_vs_agent":
            current_agent = player1 if state.current_player == 1 else player2
            current_name = player1_name if state.current_player == 1 else player2_name
            move = current_agent.get_best_move(state)  # type: ignore
            print(f"\n{current_name} {player_symbol} chooses column {move + 1}")

        elif mode == "human_vs_agent":
            if is_human_turn(state.current_player):
                move = get_human_move(state)
                print(f"\nYou {player_symbol} chose column {move + 1}")
            else:
                move = player1.get_best_move(state)
                print(f"\nAI {player_symbol} chooses column {move + 1}")

        # Make the move
        state.make_move(move)  # type: ignore
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
                    winner_name = player1_name if result == 1 else player2_name
                    winner = f"{winner_name} ({SYMBOLS[result]})"
                print(f"\n{winner} wins!")
            break


def main():
    """Main function with game mode selection and random player order"""

    # Load the trained classifier
    classifier = PositionClassifier()
    model_path = "models/connect4_analyzer_final.pkl"

    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        classifier.load_model(model_path)
    else:
        print("Training new classifier model...")
        classifier.train(save_path=model_path, tune_params=False, balance_data=True)

    # Game mode selection
    print("\nWelcome to Connect 4!")
    print("1. Human vs Negamax")
    print("2. Random vs Negamax")
    print("3. MCTS vs Negamax")
    print("4. Negamax vs Negamax")

    while True:
        try:
            choice = int(input("\nSelect game mode (1-4): "))
            if choice in [1, 2, 3, 4]:
                break
            print("Invalid choice! Please select 1-4.")
        except ValueError:
            print("Please enter a valid number!")

    # Randomly determine who goes first
    human_starts = random.choice([True, False])

    # Play game based on selected mode
    if choice == 1:
        play_game_with_mode(
            "human_vs_agent",
            NegamaxAgent(classifier=classifier),
            human_starts=human_starts,
        )
    elif choice == 2:
        play_game_with_mode(
            "agent_vs_agent",
            RandomAgent(seed=42),
            NegamaxAgent(classifier=classifier),
            player1_name="Random Agent",
            player2_name="Negamax Agent",
        )
    elif choice == 3:
        play_game_with_mode(
            "agent_vs_agent",
            MCTSAgent(simulation_time=3.0, seed=42),
            NegamaxAgent(classifier=classifier),
            player1_name="MCTS Agent",
            player2_name="Negamax Agent",
        )
    else:  # choice == 4
        play_game_with_mode(
            "agent_vs_agent",
            NegamaxAgent(classifier=classifier),
            NegamaxAgent(classifier=classifier),
            player1_name="Negamax 1",
            player2_name="Negamax 2",
        )

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
