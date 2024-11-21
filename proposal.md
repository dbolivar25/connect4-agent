## Project Title

Connect Four AI: Integrating Knowledge-Based Systems with Monte Carlo Tree Search and Constraint Satisfaction

## Team members

Hugo Son, Veronica Ahn, and Daniel Bolivar

## Problem Formulation

### Problem Description

This project implements a Connect Four playing agent that combines modern search techniques with traditional knowledge-based systems. The system employs three complementary AI techniques (MCTS, Knowledge Base, and CSP) to create a strong and adaptable game-playing agent that can make both strategic and tactical decisions.

### Key Elements

- **Input**: Current game state (6x7 board representation)
- **Output**: Column selection (0-6) for next move
- **Rules**:
  - Standard Connect Four rules
  - Players alternate turns
  - Pieces fall to lowest available position
  - Win by connecting 4 pieces horizontally, vertically, or diagonally

### AI Techniques

1. **Monte Carlo Tree Search (MCTS)**

   - Used for general move selection
   - Explores move sequences and makes a decision upon deadline expiration
   - Domain restricted by CSP constraints

2. **Knowledge Base System**

   - Stores opening book moves
   - Pattern recognition for board evaluation

3. **Constraint Satisfaction Problem (CSP)**
   - MCTS exploration restriction
   - Move validation
   - Win detection
   - Threat detection

## Evaluation Plan

1. **Metrics**

   - Win rate against baseline agents
   - Average game length
   - Decision time per move

2. **Experiments**

   a. Component Analysis:

   - Compare the outputs from each component when presented with the same scenario.
   - Measure contribution of each component

   b. Performance Testing:

   - Against random player
   - Against pure MCTS player
   - Against human players

   c. Ablation Studies:

   - MCTS without knowledge base
   - Knowledge base without MCTS
   - System without CSP

3. **Success Criteria**
   - Win rate \> 80% against random player
   - Win rate \> 60% against pure MCTS
   - Decision time \< 3 seconds per move
