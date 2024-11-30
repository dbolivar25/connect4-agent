import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_validate, RandomizedSearchCV
from sklearn.metrics import classification_report
from sklearn.utils import resample
from sklearn.inspection import permutation_importance
from scipy.stats import uniform, randint
import pickle
import os
from typing import Tuple, Dict, Union, List, Optional
from pathlib import Path
import requests
from tqdm import tqdm
import subprocess

BEST_MAX_ITER = 490
BEST_LEARNING_RATE = 0.34
BEST_MAX_DEPTH = 7
BEST_MIN_SAMPLES_LEAF = 42
BEST_MAX_LEAF_NODES = 58
BEST_L2 = 3.11
BEST_MAX_FEATURES = 1.00
RANDOM_STATE = 229
BEST_VALIDATION_FRACTION = 0.2


class PositionResultDataset:
    """Handles downloading and processing the Connect 4 dataset"""

    DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/connect-4/connect-4.data.Z"

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.compressed_path = self.data_dir / "connect-4.data.Z"
        self.data_path = self.data_dir / "connect-4.data"

    def download_dataset(self) -> None:
        """Download and decompress the Connect 4 dataset if not already present"""
        if self.data_path.exists():
            print("Dataset already downloaded and decompressed.")
            return

        print("Downloading Connect 4 dataset...")
        try:
            response = requests.get(self.DATASET_URL, stream=True)
            response.raise_for_status()
            total_size = int(response.headers.get("content-length", 0))

            with (
                open(self.compressed_path, "wb") as f,
                tqdm(
                    total=total_size, unit="iB", unit_scale=True, unit_divisor=1024
                ) as pbar,
            ):
                for data in response.iter_content(chunk_size=1024):
                    size = f.write(data)
                    pbar.update(size)

            print("Decompressing dataset...")
            try:
                subprocess.run(
                    ["uncompress", "-f", str(self.compressed_path)], check=True
                )
            except (subprocess.CalledProcessError, FileNotFoundError):
                print("uncompress failed, trying gunzip...")
                subprocess.run(["gunzip", "-f", str(self.compressed_path)], check=True)

        except Exception as e:
            print(f"Error processing dataset: {e}")
            if self.compressed_path.exists():
                self.compressed_path.unlink()
            if self.data_path.exists():
                self.data_path.unlink()
            raise

    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and process the Connect 4 dataset

        Returns:
            Tuple of (X, y) where X is the board positions and y is the outcomes
        """
        if not self.data_path.exists():
            self.download_dataset()

        data = []
        labels = []

        print("Processing dataset...")
        with open(self.data_path, "r", encoding="ascii") as f:
            for line in tqdm(f.readlines()):
                if not line.strip():
                    continue

                items = line.strip().split(",")
                if len(items) != 43:  # 42 positions + 1 label
                    continue

                # Convert board position strings to integers
                features = [
                    0 if x == "b" else 1 if x == "x" else -1 for x in items[:-1]
                ]

                # Reshape into 6x7 board and flip to match input orientation
                board = np.array(features).reshape(6, 7, order="F")
                board = np.flipud(board)
                # Flatten back to 1D array in the correct orientation
                features = board.flatten()

                # Convert outcome to integer
                label = 1 if items[-1] == "win" else 0 if items[-1] == "draw" else -1

                data.append(features)
                labels.append(label)

        if not data:
            raise ValueError("No valid data was loaded from the dataset")

        print(f"Loaded {len(data)} game positions")
        return np.array(data), np.array(labels)


class PositionClassifier:
    """Gradient Boosting Classifier for Connect 4 positions"""

    def __init__(
        self,
        learning_rate: float = BEST_LEARNING_RATE,
        max_iter: int = BEST_MAX_ITER,
        max_leaf_nodes: int = BEST_MAX_LEAF_NODES,
        max_depth: int = BEST_MAX_DEPTH,
        min_samples_leaf: int = BEST_MIN_SAMPLES_LEAF,
        l2_regularization: float = BEST_L2,
        max_features: float = BEST_MAX_FEATURES,
        random_state: int = RANDOM_STATE,
        validation_fraction: float = BEST_VALIDATION_FRACTION,
    ):
        """
        Initialize the classifier with default hyperparameters
        """

        self.LINE_COORDINATES = [
            [(row + i * dr, col + i * dc) for i in range(4)]
            for row in range(6)
            for col in range(7)
            for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]
            if (0 <= row + 3 * dr < 6) and (0 <= col + 3 * dc < 7)
        ]

        self.clf = HistGradientBoostingClassifier(
            loss="log_loss",
            early_stopping="auto",
            n_iter_no_change=10,
            learning_rate=learning_rate,
            max_iter=max_iter,
            max_leaf_nodes=max_leaf_nodes,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            l2_regularization=l2_regularization,
            max_features=max_features,
            validation_fraction=validation_fraction,
            random_state=random_state,
        )
        self.dataset = PositionResultDataset()
        self.best_params = None
        self.feature_names = None

    def _board_to_feature_vector(self, board: np.ndarray) -> np.ndarray:
        """Convert a 6x7 board to a 1D feature vector"""
        assert board.shape == (6, 7), f"Expected board shape (6, 7), got {board.shape}"
        return board.flatten(order="F")  # Column-major order

    def _add_engineered_features(self, board_features: np.ndarray) -> np.ndarray:
        """Add optimized engineered features focusing on strategic game patterns"""
        # Ensure 2D shape
        if len(board_features.shape) == 1:
            board_features = board_features.reshape(1, -1)
        n_samples = board_features.shape[0]

        # Reshape each sample into a board
        boards = board_features.reshape(n_samples, 6, 7)
        additional_features = []
        feature_names = []

        for board_idx, board in enumerate(boards):
            sample_features = []

            # 2. Threat Analysis
            # Count immediate winning threats
            p1_threats = self._count_winning_threats(board, 1)
            p2_threats = self._count_winning_threats(board, -1)
            # Count potential threats (two-in-a-row with two spaces)
            p1_potential = self._count_potential_threats(board, 1)
            p2_potential = self._count_potential_threats(board, -1)

            sample_features.extend(
                [
                    p1_threats,
                    p2_threats,
                    p1_potential,
                    p2_potential,
                ]
            )
            if board_idx == 0:
                feature_names.extend(
                    [
                        "p1_winning_threats",
                        "p2_winning_threats",
                        "p1_potential_threats",
                        "p2_potential_threats",
                    ]
                )

            # 4. Pattern-Based Features
            # Diagonal patterns
            p1_diag_strength = self._evaluate_diagonal_strength(board, 1)
            p2_diag_strength = self._evaluate_diagonal_strength(board, -1)

            # Piece clustering
            p1_zone_control = self._evaluate_zone_control(board, 1)
            p2_zone_control = self._evaluate_zone_control(board, -1)

            # Control of key positions (corners and adjacent to center)
            p1_key_pos = self._evaluate_key_positions(board, 1)
            p2_key_pos = self._evaluate_key_positions(board, -1)

            sample_features.extend(
                [
                    p1_diag_strength,
                    p2_diag_strength,
                    p1_zone_control,
                    p2_zone_control,
                    p1_key_pos,
                    p2_key_pos,
                ]
            )
            if board_idx == 0:
                feature_names.extend(
                    [
                        "p1_diagonal_strength",
                        "p2_diagonal_strength",
                        "p1_zone_control",
                        "p2_zone_control",
                        "p1_key_positions",
                        "p2_key_positions",
                    ]
                )

            additional_features.append(sample_features)

        if self.feature_names is None:
            # Original position names
            orig_names = [f"pos_r{i}c{j}" for i in range(6) for j in range(7)]
            self.feature_names = orig_names + feature_names

        # Convert to numpy array and combine with original features
        additional_features = np.array(additional_features)
        return np.hstack([board_features, additional_features])

    def _check_win(self, board: np.ndarray, player: int) -> bool:
        """Optimized win check using numpy operations"""
        # Horizontal using numpy
        for r in range(6):
            for c in range(4):
                if np.all(board[r, c : c + 4] == player):
                    return True

        # Vertical using strided array operation
        for c in range(7):
            for r in range(3):
                if np.all(board[r : r + 4, c] == player):
                    return True

        # Pre-compute all diagonals using numpy instead of list comprehension
        for r in range(3):
            for c in range(4):
                # Down-right diagonal
                if np.all(np.diag(board[r : r + 4, c : c + 4]) == player):
                    return True
                # Down-left diagonal (flip matrix horizontally first)
                if np.all(np.diag(np.fliplr(board[r : r + 4, c : c + 4])) == player):
                    return True

        return False

    def _count_winning_threats(self, board: np.ndarray, player: int) -> int:
        """Count number of positions that would result in an immediate win"""
        threats = 0
        # Try each column
        for col in range(7):
            # Find first empty row
            for row in range(5, -1, -1):
                if board[row, col] == 0:
                    # Make move
                    board[row, col] = player
                    if self._check_win(board, player):
                        threats += 1
                    # Undo move
                    board[row, col] = 0
                    break
        return threats

    def _count_potential_threats(self, board: np.ndarray, player: int) -> int:
        potential = 0
        for line_coords in self.LINE_COORDINATES:
            line = [board[r, c] for r, c in line_coords]
            if line.count(player) == 2 and line.count(0) == 2:
                potential += 1
        return potential

    def _evaluate_diagonal_strength(self, board: np.ndarray, player: int) -> float:
        """Simplified diagonal strength focusing on winning patterns"""
        strength = 0
        for start_col in [0, 1, 2, 3]:  # Only need to check starting positions
            for row in range(3):
                # Check both diagonal directions from each starting point
                for dc in [1, -1]:
                    if 0 <= start_col + 3 * dc < 7:  # Valid diagonal
                        diagonal = [
                            board[row + i][start_col + i * dc] for i in range(4)
                        ]
                        player_pieces = sum(1 for x in diagonal if x == player)
                        empty_spaces = sum(1 for x in diagonal if x == 0)
                        # Weight more heavily if closer to winning
                        if empty_spaces > 0:  # Only count if playable
                            strength += (player_pieces * empty_spaces) / (
                                4 - player_pieces
                            )
        return min(strength / 24.0, 1.0)  # Normalize

    def _evaluate_zone_control(self, board: np.ndarray, player: int) -> float:
        """Vectorized zone control evaluation"""
        control = 0
        for row in range(5):
            for col in range(6):
                zone = board[row : row + 2, col : col + 2]
                # Use numpy sum instead of individual counts
                if not np.any(zone == -player):  # No opponent pieces
                    control += np.sum(zone == player) / 4.0
        return min(control / 15.0, 1.0)

    def _evaluate_key_positions(self, board: np.ndarray, player: int) -> float:
        """Dynamic key position evaluation based on game state"""
        density = np.sum(board != 0) / board.size
        early_game = density < 0.3
        mid_game = 0.3 <= density < 0.6

        if early_game:
            key_pos = [(5, 3), (5, 2), (5, 4)]  # Focus on bottom center control
        elif mid_game:
            key_pos = [
                (4, 2),
                (4, 3),
                (4, 4),
                (3, 2),
                (3, 3),
                (3, 4),
            ]  # Focus on middle positions
        else:
            key_pos = [
                (2, 2),
                (2, 3),
                (2, 4),
                (1, 2),
                (1, 3),
                (1, 4),
            ]  # Focus on top positions

        score = sum(2.0 if board[r, c] == player else 0.0 for r, c in key_pos)
        return score / (2.0 * len(key_pos))

    def _balance_dataset(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Balance dataset using combination of up/down sampling"""
        X_df = pd.DataFrame(X)
        classes = np.unique(y)

        # Find the middle class size
        class_sizes = [sum(y == c) for c in classes]
        target_size = sorted(class_sizes)[1]  # Use middle class size as target

        balanced_dfs = []
        balanced_ys = []

        for c in classes:
            idx = y == c
            if sum(idx) > target_size:
                # Downsample majority class
                X_balanced, y_balanced = resample(  # pyright: ignore
                    X_df[idx], y[idx], n_samples=target_size, random_state=RANDOM_STATE
                )
            else:
                # Upsample minority class
                X_balanced, y_balanced = resample(  # pyright: ignore
                    X_df[idx], y[idx], n_samples=target_size, random_state=RANDOM_STATE
                )
            balanced_dfs.append(X_balanced)
            balanced_ys.append(y_balanced)

        return (
            np.vstack([df.to_numpy() for df in balanced_dfs]),
            np.hstack(balanced_ys),
        )

    def tune_hyperparameters(self, X: np.ndarray, y: np.ndarray, n_iter: int = 60):
        """Find optimal hyperparameters using RandomizedSearchCV"""
        param_dist = {
            "learning_rate": uniform(0.2, 0.5),
            "max_iter": randint(200, 550),
            "max_leaf_nodes": randint(10, 60),
            "max_depth": randint(2, 8),
            "min_samples_leaf": randint(20, 60),
            "l2_regularization": uniform(0.2, 3.5),
            # "max_features": uniform(0.6, 0.3),
        }

        random_search = RandomizedSearchCV(
            estimator=self.clf,
            param_distributions=param_dist,
            n_iter=n_iter,
            scoring=["accuracy", "f1_macro"],
            n_jobs=-1,
            verbose=2,
            random_state=RANDOM_STATE,
            refit="accuracy",  # pyright: ignore
            return_train_score=True,
        )

        print("\nPerforming hyperparameter tuning...")
        random_search.fit(X, y)

        print("\nBest parameters:", random_search.best_params_)

        # Create a new classifier with the best parameters
        self.clf = random_search.best_estimator_

    def cross_validate(
        self, X: np.ndarray, y: np.ndarray, n_folds: int = 5
    ) -> Dict[str, List[float]]:
        """Perform cross-validation to get reliable performance metrics"""
        scoring = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]

        print(f"\nPerforming {n_folds}-fold cross-validation...")
        cv_results = cross_validate(
            self.clf,
            X,
            y,
            cv=n_folds,
            scoring=scoring,
            return_train_score=True,
            n_jobs=-1,
        )

        # Print detailed cross-validation results
        for metric in scoring:
            train_scores = cv_results[f"train_{metric}"]
            val_scores = cv_results[f"test_{metric}"]
            print(f"\n{metric}:")
            print(
                f"Train: {train_scores.mean():.4f} (+/- {train_scores.std() * 2:.4f})"
            )
            print(f"Val:   {val_scores.mean():.4f} (+/- {val_scores.std() * 2:.4f})")

        return cv_results

    def train(
        self,
        save_path: Optional[str] = None,
        tune_params: bool = False,
        balance_data: bool = True,
    ) -> None:
        """Train the classifier with optional hyperparameter tuning and data balancing"""
        print("Loading dataset...")
        X, y = self.dataset.load_data()

        # Add engineered features
        print("Adding engineered features...")
        X = self._add_engineered_features(X)

        if balance_data:
            print("\nBalancing dataset...")
            X, y = self._balance_dataset(X, y)

        if tune_params:
            self.tune_hyperparameters(X, y)

        # Split data into train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
        )

        print(f"\nTraining classifier on {len(X_train)} examples...")
        self.clf.fit(X_train, y_train)

        # Calculate metrics
        train_accuracy = self.clf.score(X_train, y_train)
        val_accuracy = self.clf.score(X_val, y_val)
        y_pred = self.clf.predict(X_val)

        print(f"\nTraining accuracy: {train_accuracy:.4f}")
        print(f"Validation accuracy: {val_accuracy:.4f}")
        print("\nClassification Report:")
        print(
            classification_report(y_val, y_pred, target_names=["Loss", "Draw", "Win"])
        )

        # Save model if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "wb") as f:
                pickle.dump(self.clf, f)
            print(f"\nModel saved to {save_path}")

    def analyze_feature_importance(self, X, y, n_repeats=10):
        """
        Analyze feature importance using permutation importance

        Args:
            X: Feature matrix
            y: Target values
            n_repeats: Number of times to repeat permutation
        """
        # Calculate permutation importance
        result = permutation_importance(
            self.clf, X, y, n_repeats=n_repeats, random_state=RANDOM_STATE, n_jobs=-1
        )

        # Create DataFrame with importance scores
        importance_df = pd.DataFrame(
            {
                "Feature": self.feature_names,
                "Importance_Mean": result.importances_mean,  # pyright: ignore
                "Importance_Std": result.importances_std,  # pyright: ignore
            }
        )

        # Sort by importance
        importance_df = importance_df.sort_values("Importance_Mean", ascending=False)

        # Calculate feature groups importance
        feature_groups = {
            "Terminal State": ["is_p1_win", "is_p2_win"],
            "Threats": [
                "p1_winning_threats",
                "p2_winning_threats",
                "p1_potential_threats",
                "p2_potential_threats",
                "p1_blocked_threats",
                "p2_blocked_threats",
            ],
            "Distance": [
                "p1_manhattan_to_center",
                "p1_diagonal_to_center",
                "p2_manhattan_to_center",
                "p2_diagonal_to_center",
            ],
            "Board State": ["height_std_norm", "playable_cols_ratio", "board_density"],
            "Patterns": [
                "p1_diagonal_strength",
                "p2_diagonal_strength",
                "p1_zone_control",
                "p2_zone_control",
                "p1_key_positions",
                "p2_key_positions",
            ],
            "Raw Board": [f"pos_r{i}c{j}" for i in range(6) for j in range(7)],
        }

        group_importance = {}
        for group, features in feature_groups.items():
            group_importance[group] = {
                "mean": importance_df[importance_df["Feature"].isin(features)][
                    "Importance_Mean"
                ].sum(),
                "top_features": importance_df[importance_df["Feature"].isin(features)]
                .head(3)["Feature"]
                .tolist(),
            }

        return importance_df, group_importance

    def load_model(self, model_path: str) -> None:
        """Load a previously trained model"""
        with open(model_path, "rb") as f:
            self.clf = pickle.load(f)

    def predict_board(self, board: np.ndarray) -> Tuple[str, np.ndarray, float]:
        """
        Predict the outcome for a given board state

        Args:
            board: np.ndarray of shape (6, 7) with dtype np.int8
                  0 for empty, 1 for player 1, -1 for player 2

        Returns:
            Tuple containing:
                - predicted outcome ('Win', 'Draw', or 'Loss')
                - array of probabilities for each outcome
                - confidence score
        """
        features = self._board_to_feature_vector(board)
        features = self._add_engineered_features(features)

        prediction = self.clf.predict(features)[0]
        probabilities = self.clf.predict_proba(features)[0]

        outcome_map = {-1: "Loss", 0: "Draw", 1: "Win"}
        predicted_outcome = outcome_map[prediction]
        confidence = np.max(probabilities)

        return predicted_outcome, probabilities, confidence

    def analyze_position(
        self, board: np.ndarray
    ) -> Dict[str, Union[str, float, Dict[str, float]]]:
        """
        Provide detailed analysis of a board position

        Args:
            board: np.ndarray of shape (6, 7) with dtype np.int8

        Returns:
            Dictionary containing analysis results including:
            - predicted outcome
            - confidence
            - win/draw/loss probabilities
        """
        features = self._board_to_feature_vector(board)
        features = self._add_engineered_features(features)

        outcome, probs, conf = self.predict_board(board)

        return {
            "predicted_outcome": outcome,
            "confidence": conf,
            "win_probability": probs[2],
            "draw_probability": probs[1],
            "loss_probability": probs[0],
        }


def main():
    """Example usage of the enhanced classifier"""

    boards = [
        np.array(
            [
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, -1, -1, -1, 0, 0],
                [0, 0, 1, 1, 1, 0, 0],
            ]
        ),
        np.array(
            [
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 1, 1, 0, 0, 0],
                [0, 0, -1, -1, -1, 0, 1],
            ]
        ),
        np.array(
            [
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, -1, 0, 0, 0],
                [0, 1, -1, 1, 0, 0, 0],
                [1, -1, 1, -1, 1, 0, 0],
                [-1, 1, -1, 1, -1, -1, 1],
            ]
        ),
        np.array(
            [
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, -1, 0, 0],
                [0, 1, -1, 1, 1, -1, 0],
                [1, -1, 1, -1, -1, 1, -1],
                [-1, 1, -1, 1, 1, -1, 1],
            ]
        ),
        np.array(
            [
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, -1, 0, 0],
                [0, 0, 1, -1, 1, -1, 0],
                [0, 1, -1, 1, -1, 1, 0],
                [1, -1, 1, -1, 1, -1, 1],
                [-1, 1, -1, 1, -1, 1, -1],
            ]
        ),
        np.array(
            [
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, -1, 0, 0, 0],
                [0, 1, -1, 1, -1, 0, 0],
                [1, -1, 1, -1, 1, -1, 0],
                [-1, 1, -1, 1, -1, 1, 1],
                [1, -1, 1, -1, 1, -1, -1],
            ]
        ),
    ]

    # Initialize classifier
    classifier = PositionClassifier()

    X, y = classifier.dataset.load_data()
    X = classifier._add_engineered_features(X)

    model_path = "../models/connect4_analyzer_v16.pkl"
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        classifier.load_model(model_path)
    else:
        print("Training new model...")
        classifier.train(save_path=model_path, tune_params=False, balance_data=True)
        classifier.cross_validate(X, y)

    _, X_val, _, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # Run feature importance analysis
    print("\nAnalyzing feature importance...")
    importance_df, group_importance = classifier.analyze_feature_importance(
        X_val, y_val
    )

    print("\nTop 20 Most Important Features:")
    print(importance_df.head(20).to_string())

    print("\nFeature Group Importance:")
    for group, stats in sorted(
        group_importance.items(), key=lambda x: x[1]["mean"], reverse=True
    ):
        print(f"\n{group}:")
        print(f"Total Importance: {stats['mean']:.4f}")
        print(f"Top Features: {', '.join(stats['top_features'])}")

    # Get analysis
    for board in boards:
        analysis = classifier.analyze_position(board)

        print("\nBoard Analysis:")
        print(f"Predicted outcome: {analysis['predicted_outcome']}")
        print(f"Confidence: {analysis['confidence']:.4f}")
        print(f"Win probability: {analysis['win_probability']:.2f}")
        print(f"Draw probability: {analysis['draw_probability']:.2f}")
        print(f"Loss probability: {analysis['loss_probability']:.2f}")


if __name__ == "__main__":
    main()
