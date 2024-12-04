# Connect 4 AI Agent Using Negamax, Gradient Boosting, and Monte Carlo Methods

## Title & Team Members

- Project: Connect 4 AI Agent
- Team: [Veronica Ahn, Daniel Bolivar, Hugo Son]

## Introduction

### Problem

- Creating a strong Connect 4 playing agent that can:
  - Make intelligent moves in all game phases
  - Adapt its strategy based on the game state
  - Play effectively against both human and AI opponents
  - Handle the large branching factor and complex position evaluation of Connect 4

## Solution

### AI Techniques Used

1. Negamax Search with $\alpha - \beta$ Pruning
2. Machine Learning Position Evaluation using Gradient Boosting
3. Monte Carlo Rollouts
4. Hybrid Evaluation System

### Implementation Details

#### Phase-Based Evaluation Strategy

1. **Early Game (moves 1-9)**

   - Pure machine learning evaluation
   - Focus on strategic positioning and pattern recognition
   - Model has full confidence due to training data coverage

2. **Mid Game (moves 10-12)**

   - Hybrid evaluation combining:
     - ML position evaluation (70% weight)
     - Monte Carlo rollouts (30% weight)

3. **Late Game (moves 13+)**
   - Pure Monte Carlo rollouts
   - 75 rollouts per position evaluation

#### Technical Implementation

##### Negamax Search with $\alpha - \beta$ Pruning

###### Implementation Details

```python
def _negamax(self, state: GameState, depth: int, alpha: float, beta: float) -> float:
    # Check terminal states
    result = state.check_win()
    if result is not None:
        return result * state.current_player

    if depth <= 0:
        return self._evaluate_position(state)

    value = float("-inf")
    for move in state.get_valid_moves():
        next_state = state.clone()
        next_state.make_move(move)
        score = -self._negamax(next_state, depth - 1, -beta, -alpha)
        value = max(value, score)
        alpha = max(alpha, score)
        if alpha >= beta:
            break

    return value
```

###### Key Features

1. **Fixed 3-ply Search Depth**

   - Configured in SearchConfig
   - Balances depth vs computation time

2. **Quick Win Detection**

   - Checks for immediate winning moves
   - Checks for necessary blocking moves
   - Prioritizes urgent tactical play

3. **Move Ordering**
   - Center-out search pattern: [3, 2, 4, 1, 5, 0, 6]
   - Optimizes $\alpha - \beta$ pruning efficiency
   - Focuses on strategically important moves first

##### Machine Learning Position Evaluation

###### Gradient Boosting Implementation

```python
clf = HistGradientBoostingClassifier(
    loss="log_loss",
    learning_rate=0.34,
    max_iter=490,
    max_leaf_nodes=58,
    max_depth=7,
    min_samples_leaf=42,
    l2_regularization=3.11,
    validation_fraction=0.2,
    random_state=229
)
```

###### Feature Engineering

1. **Pattern Recognition**

   ```python
   def _evaluate_diagonal_strength(self, board: np.ndarray, player: int) -> float:
       strength = 0
       for start_col in [0, 1, 2, 3]:
           for row in range(3):
               for dc in [1, -1]:
                   if 0 <= start_col + 3 * dc < 7:
                       diagonal = [board[row + i][start_col + i * dc]
                                 for i in range(4)]
                       player_pieces = sum(1 for x in diagonal if x == player)
                       empty_spaces = sum(1 for x in diagonal if x == 0)
                       if empty_spaces > 0:
                           strength += (player_pieces * empty_spaces) /
                                     (4 - player_pieces)
       return min(strength / 24.0, 1.0)
   ```

2. **Position Control Analysis**
   - Raw board position features
   - Threat analysis calculations
   - Piece clustering evaluation
   - Strategic position control

##### Monte Carlo Rollouts

###### Implementation

```python
def _random_playout(self, state: GameState) -> float:
    current = state.clone()
    while True:
        result = current.check_win()
        if result is not None:
            return result * current.current_player

        moves = current.get_valid_moves()
        move = moves[np.random.randint(len(moves))]
        current.make_move(move)
```

###### Integration with Search

1. **Late Game Focus**

   - Pure rollouts after move 12
   - 75 rollouts per position
   - Statistical position evaluation

2. **Hybrid Phase**

   ```python
   if state.ply_count <= SearchConfig.HYBRID_PHASE_END:
       model_score = self._get_model_evaluation(state)
       rollout_scores = [self._random_playout(state.clone())
                        for _ in range(self.config.NUM_ROLLOUTS)]
       rollout_score = np.mean(rollout_scores)

       return float(
           self.config.MODEL_WEIGHT * model_score +
           (1 - self.config.MODEL_WEIGHT) * rollout_score
       )
   ```

### Why These Techniques?

1. **Negamax with $\alpha - \beta$ Pruning**

   - Efficient for tactical calculations
   - Proven effectiveness in game tree search
   - Quick identification of immediate threats/wins

2. **Gradient Boosting Classification**

   - Strong performance on pattern recognition
   - Effective with engineered Connect 4 features
   - Good generalization from training data

3. **Monte Carlo Rollouts**
   - Handle novel positions not in training data
   - Provide statistical confidence in evaluations
   - Strong in endgame situations

## Evaluation

### Metrics

1. Model Performance

   - Classification accuracy
   - Prediction confidence
   - Feature importance analysis

2. Search Performance
   - Position evaluation accuracy
   - Decision time
   - Memory usage

### Experiment Setup

- Training Data: UCI Connect 4 Dataset
- Model: Histogram-based Gradient Boosting Classifier
- Parameters:
  ```python
  NEGAMAX_DEPTH = 3
  MODEL_ONLY_PHASE = 9
  HYBRID_PHASE_END = 12
  MODEL_WEIGHT = 0.7
  NUM_ROLLOUTS = 75
  ```

### Results and Analysis

1. **Machine Learning Model**

   - ~85% prediction accuracy on validation set
   - Model trained on UCI Connect 4 Dataset

2. **Hybrid System Performance**
   - Early game: Pure ML evaluation
   - Mid game: ML + Monte Carlo hybrid
   - Late game: Pure Monte Carlo rollouts

## Conclusion

- Successfully implemented hybrid AI system for Connect 4
- Phase-based evaluation strategy adapts to game progression
- Combination of techniques provides robust gameplay
- System handles both tactical and strategic aspects effectively

## Demo

1. [Agent vs Agent Demo 1](https://asciinema.org/a/Hm669hw8VSazlWvaM6tknwvWZ)
2. [Agent vs Agent Demo 2](https://asciinema.org/a/uHWOATu8kjxBE5QAS6oJiZmf8)
