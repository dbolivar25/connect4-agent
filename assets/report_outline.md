# Report Outline: Connect 4 Agent Using Negamax, Gradient Boosting, and Monte Carlo Methods

## 1. Title & Team Members

- Project Title
- Team Members: [Veronica Ahn, Daniel Bolivar, Hugo Son]

## 1. Introduction

A. Problem Statement & Motivation

- Challenges in creating strong Connect 4 AI
- Computational complexity of game tree
- Difficulty of position evaluation
- Need for adaptive strategy across game phases

B. Contributions

1. Hybrid AI Architecture
2. Phase-based Evaluation Strategy
3. Novel Feature Engineering
4. Comprehensive Evaluation Framework

## 2. Solution

A. AI Techniques Implementation

1. Negamax Search with $\alpha - \beta$ Pruning

   - Algorithm implementation
   - Move ordering optimization
   - Depth configuration
   - Quick win detection

2. Machine Learning Position Evaluation

   - Gradient Boosting Classifier
   - Feature engineering
   - Training process
   - Model architecture

3. Monte Carlo Rollouts
   - Implementation details
   - Random playout strategy
   - Statistical evaluation

B. Phase-Based Hybrid System

1. Early Game (moves 1-9)

   - Pure ML evaluation
   - Pattern recognition focus

2. Mid Game (moves 10-12)

   - Hybrid evaluation
   - ML weight decay
   - Monte Carlo integration

3. Late Game (moves 13+)
   - Pure Monte Carlo rollouts
   - Statistical confidence

C. Implementation Details

- Pseudocode for key algorithms
- System architecture
- Integration approach
- Performance optimizations

## 3. Evaluation

A. Metrics

1. Win Rates
2. Game Length Statistics
3. Decision Time Analysis
4. Model Performance Metrics

B. Experiment Setup

1. Test Configuration

   - Games per matchup
   - Hardware/environment
   - Time constraints
   - Random seeds

2. Baseline Implementations
   - Random agent
   - Pure MCTS
   - Pure Negamax

C. Results Analysis

1. Performance Metrics
   - Win rates against different opponents
   - Average game length
   - Decision time statistics
2. Model Analysis

   - Classification accuracy
   - Feature importance
   - Cross-validation results

3. Comparative Analysis
   - Algorithm strengths/weaknesses
   - Performance trade-offs
   - Resource utilization

## 4. Conclusion

A. Summary

- Key findings
- System effectiveness
- Novel contributions

B. Limitations

1. Technical Constraints

   - Search depth limitations
   - Computation time
   - Model coverage

2. Strategic Limitations
   - Opening book absence
   - End-game tablebase
   - Pattern recognition gaps

C. Future Work

- Potential improvements
- Research directions
- Scaling opportunities

## 6. Team Member Contributions

A. Individual Contributions

- Specific responsibilities
- Implementation areas
- Research contributions

B. External Resources

- Referenced implementations
- Utilized libraries
- Academic sources
