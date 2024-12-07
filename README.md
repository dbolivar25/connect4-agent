# Connect 4 Agent using Negamax, Gradient Boosting, and Monte Carlo Methods

<!--toc:start-->

- [Connect 4 Agent using Negamax, Gradient Boosting, and Monte Carlo Methods](#connect-4-agent-using-negamax-gradient-boosting-and-monte-carlo-methods)
  - [Demo](#demo)
  - [Features](#features)
  - [Technical Overview](#technical-overview)
    - [Core Components](#core-components)
  - [Installation & Development Setup](#installation-development-setup)
    - [Install with uv (recommended)](#install-with-uv-recommended)
    - [Install with pip](#install-with-pip)
  - [Usage](#usage)
    - [Running with uv (recommended)](#running-with-uv-recommended)
    - [Running with pip venv](#running-with-pip-venv)
    - [Playing the Game](#playing-the-game)
  - [Configuration](#configuration)
    - [Game Configuration](#game-configuration)
    - [Search Configuration](#search-configuration)
  - [Technical Details](#technical-details)
    - [Position Evaluation Strategy](#position-evaluation-strategy)
    - [Machine Learning Model](#machine-learning-model)
  - [Project Structure](#project-structure)
  - [Acknowledgments](#acknowledgments)
  <!--toc:end-->

## Demo

1. [Agent vs Agent Demo 1](https://asciinema.org/a/Hm669hw8VSazlWvaM6tknwvWZ)
2. [Agent vs Agent Demo 2](https://asciinema.org/a/uHWOATu8kjxBE5QAS6oJiZmf8)
3. [Evaluation Model Training](https://asciinema.org/a/P7yCVfh7JYdHdD0PYF3RSKAnx)

## Features

- Hybrid AI system combining:
  - Negamax search with alpha-beta pruning
  - Machine learning-based position evaluation
  - Monte Carlo rollouts
- Adaptive search strategy based on game phase
- Pre-trained position classifier using gradient boosting
- Interactive command-line interface
- Support for both AI vs AI and Human vs AI gameplay

## Technical Overview

### Core Components

1. **Position Classifier (`position_classifier.py`)**

   - Uses Gradient Boosting for position evaluation
   - Features engineered for Connect 4 pattern recognition
   - Trained on the UCI Connect 4 dataset
   - Evaluates winning probability for any given position

2. **Game Agent (`agent.py`)**

   - Implements Negamax search with alpha-beta pruning
   - Dynamic evaluation strategy:
     - Early game: Pure ML evaluation
     - Mid game: Hybrid ML + Monte Carlo rollouts
     - Late game: Pure Monte Carlo rollouts
   - Quick checks for immediate winning/blocking moves

3. **Game State (`game_state.py`)**

   - Efficient board representation using NumPy arrays
   - Fast win detection algorithms
   - Move validation and state management

4. **Main Game Loop (`main.py`)**
   - Handles game flow and player interaction
   - Supports multiple game modes
   - Provides visual board representation

## Installation & Development Setup

This project requires Python 3.13 or higher.

1. Clone the repository:

```bash
git clone git@github.com:dbolivar25/connect4-agent.git
cd connect4-agent
```

### Install with uv (recommended)

1. Create and activate a virtual environment:

```bash
uv venv
source .venv/bin/activate  # Unix
.venv\Scripts\activate     # Windows
```

2. Install dependencies:

```bash
uv sync
```

### Install with pip

1. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # Unix
.venv\Scripts\activate     # Windows
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Running with uv (recommended)

```bash
uv run src/main.py
```

### Running with pip venv

```bash
python src/main.py
```

The game will prompt you to select a mode:

1. Human vs AI
2. AI vs AI

In Human vs AI mode, the starting player is randomly determined.

### Playing the Game

- Columns are numbered 1-7 from left to right
- Enter the column number where you want to place your piece
- Player symbols:
  - Player 1: 🔴
  - Player 2: 🔵
  - Empty: ⚪

## Configuration

The game behavior can be customized through the `config.py` file:

### Game Configuration

```python
HEIGHT = 6          # Board height
WIDTH = 7           # Board width
SEARCH_ORDER       # Preferred move ordering for search
```

### Search Configuration

```python
NEGAMAX_DEPTH = 3        # Search depth
MODEL_ONLY_PHASE = 9     # Pure ML evaluation until move 9
HYBRID_PHASE_END = 12    # Switch to pure rollouts after move 12
MODEL_WEIGHT = 0.7       # ML weight in hybrid evaluation
NUM_ROLLOUTS = 75        # Number of Monte Carlo rollouts
```

## Technical Details

### Position Evaluation Strategy

The AI uses a phase-based evaluation strategy:

1. **Early Game (moves 1-8)**

   - Uses pure machine learning evaluation
   - Focus on strategic positioning and pattern recognition
   - Model has full confidence in this phase due to training data coverage

2. **Mid Game (moves 9-12)**

   - Hybrid evaluation with decaying ML weight:

     - ML position evaluation (starting at 70% weight and decreasing)
     - Monte Carlo rollouts (starting at 30% weight and increasing)

   - Model weight decreases as positions become less similar to training data

3. **Late Game (moves 13+)**
   - Pure Monte Carlo rollouts
   - 75 rollouts per position evaluation
   - Model is not used as positions are too far from training data

### Machine Learning Model

- **Algorithm**: Histogram-based Gradient Boosting Classifier
- **Features**:
  - Raw board position
  - Engineered features including:
    - Threat analysis
    - Pattern recognition
    - Piece clustering
    - Control of key positions
- **Training Data**: UCI Connect 4 Dataset
- **Performance**: ~90% prediction accuracy on validation set

## Project Structure

```bash
├── data/                # Position result dataset
├── models/              # Trained model storage
├── results/             # Agent evaluation results
└── src/                 # Implementation source code
```

## Acknowledgments

- UCI Machine Learning Repository for the Connect 4 dataset
