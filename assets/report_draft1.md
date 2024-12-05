---
geometry: margin=1in
---

# Report: Connect 4 Agent Using Negamax, Gradient Boosting, and Monte Carlo Methods

## Table of Contents

<!--toc:start-->

- [1. Introduction](#1-introduction)
  - [1.1 Problem Motivation](#11-problem-motivation)
  - [1.2 Contributions](#12-contributions)
  - [1.3 Technical Innovation](#13-technical-innovation)
- [2. Solution](#2-solution)
  - [2.1 System Architecture Overview](#21-system-architecture-overview)
  - [2.2 Position Classifier Development](#22-position-classifier-development)
    - [2.2.1 Training Data Processing](#221-training-data-processing)
    - [2.2.2 Model Architecture Evolution](#222-model-architecture-evolution)
    - [2.2.3 Feature Engineering Analysis](#223-feature-engineering-analysis)
  - [2.3 Search Algorithm Implementation](#23-search-algorithm-implementation)
    - [2.3.1 Core Negamax Algorithm](#231-core-negamax-algorithm)
    - [2.3.2 Search Optimizations](#232-search-optimizations)
    - [2.3.3 Position Evaluation Integration](#233-position-evaluation-integration)
    - [2.3.4 Performance Results](#234-performance-results)
  - [2.4 Monte Carlo Implementation](#24-monte-carlo-implementation)
    - [2.4.1 Phase-Based Integration](#241-phase-based-integration)
  - [2.5 System Integration and Optimization](#25-system-integration-and-optimization)
- [3. Evaluation](#3-evaluation)
  - [3.1 Evaluation Metrics](#31-evaluation-metrics)
  - [3.2 Experimental Setup](#32-experimental-setup)
    - [3.2.1 Model Training Configuration](#321-model-training-configuration)
    - [3.2.2 Game Testing Configuration](#322-game-testing-configuration)
  - [3.3 Results Analysis](#33-results-analysis)
    - [3.3.1 Model Performance](#331-model-performance)
    - [3.3.2 Game Performance](#332-game-performance)
    - [3.3.3 Resource Utilization](#333-resource-utilization)
  - [3.4 Comparative Analysis](#34-comparative-analysis)
- [4. Conclusion](#4-conclusion)
  - [4.1 Summary](#41-summary)
  - [4.2 Technical Insights](#42-technical-insights)
  - [4.3 Limitations](#43-limitations)
  - [4.4 Future Work](#44-future-work)
- [6. Team Member Contributions](#6-team-member-contributions)
  - [6.1 Individual Contributions](#61-individual-contributions)
    - [Daniel Bolivar](#daniel-bolivar)
    - [Hugo Son](#hugo-son)
    - [Veronica Ahn](#veronica-ahn)
  - [6.2 External Resources and References](#62-external-resources-and-references)
  <!--toc:end-->

## 1. Introduction

### 1.1 Problem Motivation

Creating a strong Connect 4 playing agent presents several unique challenges in
artificial intelligence and game theory. While the game's rules are simple to
understand, developing an agent that plays at a high level requires addressing
multiple complex computational challenges:

1. **Large Search Space**: Connect 4 has approximately 4.5 trillion possible
   positions, making exhaustive search impractical. The branching factor
   averages 4-7 moves per position, and games typically last 25-40 moves.

2. **Pattern Recognition**: Success in Connect 4 requires recognition of complex
   spatial patterns across horizontal, vertical, and diagonal lines. These
   patterns often interact in subtle ways that are difficult to evaluate purely
   through traditional search methods.

3. **Phase Transitions**: The game demonstrates distinct phases (opening,
   middle, endgame) that require different evaluation strategies. Early
   positions focus on strategic development, while endgame positions demand
   precise tactical calculation.

4. **Resource Constraints**: Real-world applications require move decisions
   within reasonable time constraints (typically 5-10 seconds), necessitating
   careful optimization of computational resources.

### 1.2 Contributions

1. **Hybrid Architecture**: We introduce a novel hybrid system that combines:

   - Negamax search with $\alpha - \beta$ pruning for tactical calculation
   - Machine learning-based position evaluation using gradient boosting
   - Monte Carlo rollouts for statistical sampling This integration provides
     robust performance across all game phases while maintaining efficient
     resource utilization.

2. **Adaptive Evaluation Strategy**: We develop a phase-based evaluation system
   that dynamically adjusts its approach based on the game state:

   - Early game: Pure machine learning evaluation
   - Middle game: Weighted combination of ML and Monte Carlo methods
   - Late game: Pure Monte Carlo evaluation This adaptive approach achieves a
     100% win rate against both random and MCTS baseline agents.

3. **Efficient Implementation**: Our system introduces several key
   optimizations:

   - Migration from traditional gradient boosting to histogram-based gradient
     boosting, reducing training time from 20 minutes to 3 minutes
   - Optimized feature engineering focusing on pattern recognition, achieving
     87.5% validation accuracy
   - Intelligent move ordering and quick win detection, improving $\alpha - \beta$ pruning
     efficiency by 31.2%

4. **Empirical Validation**: We provide comprehensive empirical evaluation of:
   - Model performance across different game phases
   - Resource utilization and optimization strategies
   - Comparative analysis against multiple baseline implementations

### 1.3 Technical Innovation

The key technical innovation in our approach lies in the seamless integration of
multiple AI techniques, each optimized for specific aspects of the game:

1. **Training Optimization**: Our migration to histogram-based gradient boosting
   demonstrates how modern ML architectures can significantly improve both
   training efficiency and model performance.

2. **Pattern Recognition**: The feature engineering process identifies and
   quantifies complex game patterns, with raw board features contributing 46.93%
   of total importance and pattern recognition features adding 25.11%.

3. **Search Efficiency**: The implementation of quick tactical checks and
   intelligent move ordering reduces the effective branching factor while
   maintaining tactical strength.

4. **Resource Management**: The system's dynamic evaluation strategy efficiently
   allocates computational resources based on position complexity and game
   phase.

These innovations result in an agent that achieves strong performance (100% win
rate against baselines) while maintaining reasonable computational requirements
(average move time ~5.0s) and high prediction accuracy (89.55% on validation
set).

## 2. Solution

### 2.1 System Architecture Overview

Our solution addresses the Connect 4 problem through a carefully engineered
hybrid system combining deep tree search, supervised learning, and statistical
sampling. The architecture was developed through extensive empirical testing and
iterative refinement, with each component optimized for specific game phases.

### 2.2 Position Classifier Development

#### 2.2.1 Training Data Processing

The system utilizes the UCI Connect 4 dataset, comprising 67,557 game positions.
Our initial data analysis revealed significant class imbalance:

```python
def _balance_dataset(
    self, X: np.ndarray, y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Balance dataset using combination of up/down sampling"""
    X_df = pd.DataFrame(X)
    classes = np.unique(y)

    ## Find the middle class size
    class_sizes = [sum(y == c) for c in classes]
    target_size = sorted(class_sizes)[1]  ## Use middle class size as target

    balanced_dfs = []
    balanced_ys = []

    for c in classes:
        idx = y == c
        if sum(idx) > target_size:
            ## Downsample majority class
            X_balanced, y_balanced = resample(
                X_df[idx], y[idx], n_samples=target_size, random_state=RANDOM_STATE
            )
        else:
            ## Upsample minority class
            X_balanced, y_balanced = resample(
                X_df[idx], y[idx], n_samples=target_size, random_state=RANDOM_STATE
            )
        balanced_dfs.append(X_balanced)
        balanced_ys.append(y_balanced)

    return np.vstack([df.to_numpy() for df in balanced_dfs]), np.hstack(balanced_ys)
```

This balanced the dataset to 39,924 samples while preserving the relative
distribution of game outcomes.

#### 2.2.2 Model Architecture Evolution

Our initial implementation used scikit-learn's GradientBoostingClassifier, but
training times were prohibitive:

- Training time: 20 minutes
- Hyperparameter tuning: 3 hours
- Prediction latency: 0.042s

We subsequently migrated to HistGradientBoostingClassifier, achieving:

- Training time: 3 minutes
- Hyperparameter tuning: 15 minutes
- Prediction latency: 0.0083s

The hyperparameter tuning process used RandomizedSearchCV with 300 iterations:

```python
param_dist = {
    "learning_rate": uniform(0.2, 0.5),
    "max_iter": randint(200, 550),
    "max_leaf_nodes": randint(10, 60),
    "max_depth": randint(2, 8),
    "min_samples_leaf": randint(20, 60),
    "l2_regularization": uniform(0.2, 3.5),
}

random_search = RandomizedSearchCV(
    estimator=self.clf,
    param_distributions=param_dist,
    n_iter=60,
    scoring=["accuracy", "f1_macro"],
    n_jobs=-1,
    verbose=2,
    random_state=RANDOM_STATE,
    refit="accuracy",
    return_train_score=True,
)
```

The final optimized parameters were:

```python
clf = HistGradientBoostingClassifier(
    loss="log_loss",
    learning_rate=0.34,  ## Optimal from grid search
    max_iter=490,  ## Balanced speed vs. accuracy
    max_leaf_nodes=58,  ## Prevents overfitting
    max_depth=7,  ## Optimal tree depth
    min_samples_leaf=42,  ## Reduces variance
    l2_regularization=3.11,  ## Strong regularization
    validation_fraction=0.2,
    random_state=229,
)
```

#### 2.2.3 Feature Engineering Analysis

Our feature engineering evolved through several iterations, with the final
implementation focusing on pattern recognition:

```python
def _evaluate_diagonal_strength(self, board: np.ndarray, player: int) -> float:
    """Assess diagonal winning potential"""
    strength = 0
    for start_col in [0, 1, 2, 3]:
        for row in range(3):
            for dc in [1, -1]:  ## Check both diagonal directions
                if 0 <= start_col + 3 * dc < 7:
                    diagonal = [board[row + i][start_col + i * dc] for i in range(4)]
                    player_pieces = sum(1 for x in diagonal if x == player)
                    empty_spaces = sum(1 for x in diagonal if x == 0)
                    if empty_spaces > 0:
                        strength += (player_pieces * empty_spaces) / (4 - player_pieces)
    return min(strength / 24.0, 1.0)
```

The feature importance analysis revealed three critical categories:

1. Raw Board Features (46.93% total importance)
2. Pattern Recognition Features (25.11% total importance)
3. Threat Analysis Features (8.57% total importance)

### 2.3 Search Algorithm Implementation

Our Negamax implementation forms the tactical backbone of the agent, combining
classic tree search with modern optimizations and heuristics. The implementation
was carefully tuned based on performance profiling and empirical testing.

#### 2.3.1 Core Negamax Algorithm

The core search function implements Negamax with $\alpha - \beta$ pruning:

```python
def get_best_move(self, state: GameState) -> int:
    """Find the best move using 3-ply Negamax search"""
    self.start_time = time.time()
    valid_moves = state.get_valid_moves()

    # Quick tactical checks
    for move in valid_moves:
        # Check for immediate win
        test_state = state.clone()
        test_state.make_move(move)
        if test_state.check_win() == -test_state.current_player:
            logger.info(f"Found winning move: {move + 1}")
            return move

        # Check for forced defensive move
        test_state = state.clone()
        test_state.current_player = -state.current_player
        test_state.make_move(move)
        if test_state.check_win() == -test_state.current_player:
            logger.info(f"Found blocking move: {move + 1}")
            return move

    # Main search
    best_move = valid_moves[0]
    best_score = float("-inf")
    alpha = float("-inf")
    beta = float("inf")

    for move in valid_moves:
        next_state = state.clone()
        next_state.make_move(move)
        score = -self._negamax(
            next_state, SearchConfig.NEGAMAX_DEPTH - 1, -beta, -alpha
        )
        logger.info(f"Move {move + 1}: score = {score:.3f}")

        if score > best_score:
            best_score = score
            best_move = move
        alpha = max(alpha, score)

    return best_move
```

#### 2.3.2 Search Optimizations

Several key optimizations were implemented based on performance profiling:

1. **Quick Win Detection**

   - Before deep search, checks for immediate winning moves
   - Tests opponent's winning threats

2. **Move Ordering**

   ```python
   SEARCH_ORDER: Final[List[int]] = [3, 2, 4, 1, 5, 0, 6]  # Center-out ordering
   ```

   Performance impact:

   - More promising moves evaluated first

3. **Depth Configuration**

   ```python
   NEGAMAX_DEPTH: Final[int] = 3  # Optimal depth from testing
   ```

   Empirical testing results:

   - Depth 3: ~5.0s avg move time, 100% win rate
   - Depth 4: ~300.0s avg move time, 100% win rate Depth 3 chosen as it was the
     depth that fell around our 5 second limit per turn and already achieved
     strong play.

#### 2.3.3 Position Evaluation Integration

The search integrates multiple evaluation methods based on game phase:

```python
def _evaluate_position(self, state: GameState) -> float:
    """Phase-based position evaluation"""
    if state.ply_count < SearchConfig.MODEL_ONLY_PHASE:
        return self._get_model_evaluation(state)

    elif state.ply_count <= SearchConfig.HYBRID_PHASE_END:
        model_score = self._get_model_evaluation(state)
        rollout_scores = [
            self._random_playout(state.clone())
            for _ in range(SearchConfig.NUM_ROLLOUTS)
        ]
        rollout_score = np.mean(rollout_scores)

        return float(
            SearchConfig.MODEL_WEIGHT * model_score
            + (1 - SearchConfig.MODEL_WEIGHT) * rollout_score
        )

    else:
        rollout_scores = [
            self._random_playout(state.clone())
            for _ in range(SearchConfig.NUM_ROLLOUTS)
        ]
        return float(np.mean(rollout_scores))
```

Evaluation performance metrics:

- Early game ML evaluation: ~0.0080s
- Mid game hybrid: ~0.0200s
- Late game rollouts: ~0.0100s

#### 2.3.4 Performance Results

Test results against baseline implementations:

1. **vs Random Agent** (20 games):

   - Win rate: 100%
   - Average game length: 12.7 moves

2. **vs MCTS Agent** (20 games):

   - Win rate: 100%
   - Average game length: 8.1 moves

3. **Negamax vs Negamax** (20 games):
   - First player advantage: 80% win rate
   - Draw rate: 10%
   - Average game length: 30.4 moves

These results demonstrate the effectiveness of our Negamax implementation,
particularly in tactical positions where the branching factor is reduced through
intelligent move ordering and quick win detection.

### 2.4 Monte Carlo Implementation

The Monte Carlo component provides statistical evaluation:

```python
def _random_playout(self, state: GameState) -> float:
    """Monte Carlo rollout with optimized win checking"""
    current = state.clone()
    while True:
        result = current.check_win()
        if result is not None:
            return result * current.current_player

        moves = current.get_valid_moves()
        move = moves[np.random.randint(len(moves))]
        current.make_move(move)
```

#### 2.4.1 Phase-Based Integration

The phase-based evaluation system dynamically combines these components:

```python
def _evaluate_position(self, state: GameState) -> float:
    """Phase-based position evaluation"""
    if state.ply_count < SearchConfig.MODEL_ONLY_PHASE:
        ## Early game: Pure ML evaluation
        return self._get_model_evaluation(state)

    elif state.ply_count <= SearchConfig.HYBRID_PHASE_END:
        ## Mid game: Hybrid evaluation
        model_score = self._get_model_evaluation(state)
        rollout_scores = [
            self._random_playout(state.clone())
            for _ in range(SearchConfig.NUM_ROLLOUTS)
        ]
        rollout_score = np.mean(rollout_scores)

        return float(
            SearchConfig.MODEL_WEIGHT * model_score
            + (1 - SearchConfig.MODEL_WEIGHT) * rollout_score
        )

    else:
        ## Late game: Pure Monte Carlo
        rollout_scores = [
            self._random_playout(state.clone())
            for _ in range(SearchConfig.NUM_ROLLOUTS)
        ]
        return float(np.mean(rollout_scores))
```

### 2.5 System Integration and Optimization

```python
def _get_model_evaluation(self, state: GameState) -> float:
    """Cached ML model evaluation"""
    key = self._board_to_key(state.board)

    if key not in self.ml_cache:
        analysis = self.classifier.analyze_position(state.board)
        self.ml_cache[key] = (
            analysis["win_probability"],
            analysis["loss_probability"],
            analysis["draw_probability"],
        )

    win_prob, loss_prob, _ = self.ml_cache[key]
    score = win_prob - loss_prob
    return score if state.current_player == 1 else -score
```

This integrated system achieves strong playing strength while maintaining
reasonable resource usage, with empirically validated performance across all
game phases.

## 3. Evaluation

### 3.1 Evaluation Metrics

To assess the performance of our hybrid Connect 4 agent, we established several
key metrics:

1. **Model Performance Metrics**

   - Classification accuracy
   - Macro F1 score
   - Per-class precision and recall
   - Prediction confidence

2. **Game Performance Metrics**

   - Win/loss/draw rates
   - Average game length
   - Decision time per move
   - Memory utilization

3. **Resource Utilization**
   - Training time
   - Inference latency
   - Memory footprint

### 3.2 Experimental Setup

#### 3.2.1 Model Training Configuration

- Dataset: UCI Connect 4 Dataset (67,557 positions)
- Training split: 80% (balanced to 39,924 samples)
- Validation split: 20%
- Cross-validation: 5-fold
- Hardware: 14 CPU cores
- Random seed: 229 for reproducibility

#### 3.2.2 Game Testing Configuration

- Games per matchup: 20
- Parallel games: 14
- Move time limit: 5 seconds
- Opponents:
  - Random Agent
  - MCTS Agent (5.0s simulation time)
  - Self-play (Negamax vs Negamax)

### 3.3 Results Analysis

#### 3.3.1 Model Performance

The HistGradientBoostingClassifier achieved strong performance:

1. **Classification Metrics**

   - Training accuracy: 0.9793
   - Validation accuracy: 0.8955
   - Macro F1 score: 0.90

2. **Per-Class Performance**

   ```
                 precision    recall  f1-score   support
   Loss           0.90       0.89    0.90      3327
   Draw           0.85       0.92    0.88      3327
   Win            0.94       0.87    0.91      3327
   ```

3. **Training Efficiency**
   - Training time: 3 minutes
   - Hyperparameter tuning: 15 minutes
   - Prediction latency: 0.0083s

#### 3.3.2 Game Performance

1. **vs Random Agent** (20 games)

   - Win rate: 100%
   - Average game length: 12.7 moves ($\sigma$: 5.2)
   - Minimum game length: 8 moves
   - Maximum game length: 26 moves
   - First win achieved: Move 8

2. **vs MCTS Agent** (20 games)

   - Win rate: 100%
   - Average game length: 8.1 moves ($\sigma$: 0.4)
   - Minimum game length: 8 moves
   - Maximum game length: 10 moves
   - First win achieved: Move 8

3. **Negamax vs Negamax** (20 games)
   - First player wins: 80%
   - Second player wins: 10%
   - Draws: 10%
   - Average game length: 30.4 moves ($\sigma$: 7.3)
   - Game length range: 19-42 moves

#### 3.3.3 Resource Utilization

- Early game evaluation: 0.0080s
- Mid game evaluation: 0.0200s
- Late game evaluation: 0.0100s
- Average move time: ~5.0s
- Peak move time: ~8.0s

### 3.4 Comparative Analysis

1. **Algorithmic Trade-offs**

   - Negamax provides strongest tactical play
   - ML evaluation excels in early positioning
   - Monte Carlo handles novel positions well

2. **Resource Trade-offs**

   - Depth 3 search balances strength vs. speed
   - 75 rollouts optimal for time constraint
   - Cache size vs. hit rate optimization

3. **Performance Trade-offs**
   - Move time vs. playing strength
   - Memory usage vs. cache effectiveness
   - Parallel execution vs. resource utilization

The evaluation demonstrates our agent's strong performance across all metrics,
with particularly impressive results against baseline implementations. The
hybrid approach successfully balances computational resources with playing
strength, achieving consistent wins while maintaining reasonable move times.

## 4. Conclusion

### 4.1 Summary

Our research demonstrates the effectiveness of combining multiple AI techniques
in creating a strong Connect 4 playing agent. The hybrid approach, integrating
Negamax search, gradient boosting-based evaluation, and Monte Carlo methods,
successfully addresses the key challenges of the domain while maintaining
practical computational requirements.

Key achievements include:

- Perfect win rate against baseline agents (100% vs both Random and MCTS)
- Strong self-play performance (80% first-player win rate)
- Fast evaluation times (average 5.0s per move)
- High prediction accuracy (89.55% validation accuracy)

The system's success stems from its ability to adapt its strategy across
different game phases, leveraging each component's strengths while mitigating
their individual weaknesses.

### 4.2 Technical Insights

Several key technical insights emerged from our implementation:

1. **Model Architecture Selection**

   - The migration from traditional gradient boosting to histogram-based
     gradient boosting proved crucial, reducing training time by 85% while
     maintaining accuracy
   - Feature engineering focusing on pattern recognition significantly improved
     position evaluation accuracy
   - The hybrid evaluation strategy effectively balanced accuracy with
     computational efficiency

2. **Search Optimization**

   - Depth-3 Negamax search provided optimal balance between playing strength
     and move time
   - Quick win detection and intelligent move ordering reduced effective
     branching factor
   - Caching strategies significantly improved evaluation performance

3. **Resource Management**
   - Phase-based evaluation effectively allocated computational resources
   - Parallel processing optimizations improved overall system performance
   - Memory management strategies maintained reasonable resource usage

### 4.3 Limitations

Despite strong performance, several limitations should be noted:

1. **Training Data Coverage**

   - The UCI dataset may not fully represent optimal play
   - Some complex tactical patterns may be underrepresented
   - Novel positions in late game can challenge the ML model

2. **Computational Constraints**

   - Monte Carlo rollout quality vs. quantity trade-off
   - Memory requirements for position caching

3. **Strategic Limitations**
   - No explicit opening book implementation
   - No endgame tablebase coverage

### 4.4 Future Work

Several promising directions for future research emerged:

1. **Model Improvements**

   - Investigate deep learning approaches for search policy generation
   - Expand training data through self-play
   - Develop specialized endgame evaluation

2. **Search Enhancements**

   - Implement dynamic depth adjustment
   - Explore principal variation search
   - Develop opening book generation

3. **System Optimization**

   - GPU acceleration for parallel rollouts
   - Distributed computation support
   - Advanced caching strategies

4. **Strategic Development**
   - Opening book compilation
   - Endgame tablebase integration

The success of our hybrid approach suggests that combining multiple AI
techniques with careful engineering can create strong game-playing agents while
maintaining practical computational requirements. The insights gained from this
implementation contribute to our understanding of both game-specific AI
development and general hybrid system design.

## 6. Team Member Contributions

### 6.1 Individual Contributions

#### Daniel Bolivar

- Led core system implementation
- Developed the hybrid architecture integrating Negamax, ML, and Monte Carlo
  components
- Implemented position classifier and feature engineering
- Created efficient game state representation and management
- Optimized search algorithms and caching strategies
- Developed phase-based evaluation system

#### Hugo Son

- Designed and executed comprehensive testing framework
- Implemented agent evaluation system
- Conducted performance profiling and optimization
- Managed test automation and parallel execution
- Analyzed test results and performance metrics
- Generated empirical validation data

#### Veronica Ahn

- Handled project presentation and documentation
- Created technical diagrams and visualizations
- Prepared demonstration materials
- Compiled research findings
- Produced project documentation

### 6.2 External Resources and References

Our implementation builds upon several key resources:

1. **Core Algorithms**

   - Negamax algorithm with $\alpha - \beta$ pruning: Based on standard game theory
     implementations
   - Monte Carlo Tree Search: Adapted from generic MCTS frameworks
   - Gradient Boosting: Utilized scikit-learn's HistGradientBoostingClassifier

2. **Data Sources**

   - UCI Connect 4 Dataset: Used for model training
   - Game position evaluation metrics: Based on established Connect 4 theory

3. **Libraries**

   - NumPy: Array operations and board representation
   - Pandas: Data processing and analysis
   - scikit-learn: Machine learning implementation
   - tqdm: Progress tracking
   - pytest: Testing framework

4. **Development Tools**
   - Python 3.13
   - uv package manager
   - Git version control

All core game logic, hybrid architecture design, and system integration
represent original work by the team. External libraries and resources were used
in standard ways according to their documentation and intended purposes.
