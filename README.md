# Early Feedback in Neural Combinatorial Optimization for Offline Bin Packing

## Overview

This project investigates the impact of **Early Feedback** in Neural Combinatorial Optimization (NCO) for **Offline Bin Packing**, starting with the **1D Bin Packing Problem** and extending the research to **2D Bin Packing**.

The main objective is to evaluate whether providing intermediate feedback during the construction of a solution can improve:

* Solution quality
* Convergence speed
* Training stability
* Generalization
* Optimality gap

Three Actor-Critic reinforcement learning approaches are compared:

1. **Monte Carlo (MC) Actor-Critic** – learning from terminal rewards.
2. **Temporal Difference (TD) Actor-Critic** – using TD learning without Early Feedback.
3. **TD Actor-Critic with Early Feedback (TD-EF)** – using intermediate rewards during solution construction.

---

## Problem Description

The **Offline 1D Bin Packing Problem** consists of a set of items with different sizes and bins with a fixed capacity.

The objective is to assign every item to a bin while minimizing the total number of bins used.

The problem is formulated as a sequential decision-making process:

```text
State
  ↓
Actor selects an action
  ↓
Environment updates the packing
  ↓
Reward
  ↓
Next State
  ↓
...
```

The episode terminates when all items have been packed.

The research will subsequently be extended to the **Offline 2D Bin Packing Problem**, where both item dimensions and two-dimensional bin space must be considered.

---

## Methods

### Monte Carlo Actor-Critic

The Monte Carlo Actor-Critic method uses the final solution to calculate the learning signal.

The terminal reward is defined as:

```text
R_terminal = -NumBins + α · VolumeUtilization
```

The agent therefore receives the main reward after completing the packing episode.

This method serves as the terminal-reward baseline.

---

### TD Actor-Critic

The TD Actor-Critic method uses temporal-difference learning to update the Critic during the episode.

The TD target is:

```text
TD Target = r_t + γV(s_{t+1})
```

For terminal states, the future-value term is omitted.

Unlike the TD-EF method, the TD baseline does not introduce the proposed Early Feedback reward shaping.

---

### TD Actor-Critic with Early Feedback

The main method investigated in this project is **TD Actor-Critic with Early Feedback**.

Instead of relying only on the final solution quality, the agent receives feedback based on changes in the packing solution after each action.

The Early Feedback reward is defined as:

```text
r_t = -ΔNumBins_t + α · ΔVolumeUtilization_t
```

where:

* `ΔNumBins_t` represents the change in the number of bins.
* `ΔVolumeUtilization_t` represents the change in volume utilization.
* `α` controls the contribution of volume utilization.

This provides the agent with information about the quality of intermediate decisions before the complete solution is available.

---

## Network Architecture

The Actor is implemented using PyTorch.

The network architecture is:

```text
Input
  ↓
Linear(state_dim, 128)
  ↓
ReLU
  ↓
Linear(128, 64)
  ↓
ReLU
  ↓
Linear(64, 1)
```

The input state represents the current packing configuration, including information about remaining items and available bin capacities.

---

## Experimental Setup

Experiments are conducted on Offline 1D Bin Packing instances with the following problem sizes:

```text
60, 120, 249, 250, 500
```

The datasets contain different types of item sizes:

| Problem Size | Item Sizes          |
| ------------ | ------------------- |
| 60           | Integer             |
| 120          | Integer and Decimal |
| 249          | Decimal             |
| 250          | Integer             |
| 500          | Integer             |

More specifically:

* **60 items** – integer item sizes.
* **120 items (integer)** – integer item sizes.
* **120 items (decimal)** – decimal item sizes.
* **249 items** – contains only decimal item sizes.
* **250 items** – integer item sizes.
* **500 items** – integer item sizes.

Multiple random seeds are used for each experimental setting to evaluate the robustness and stability of the proposed methods.

The same experimental conditions are maintained across the compared approaches to ensure a fair comparison between:

1. Monte Carlo Actor-Critic
2. TD Actor-Critic
3. TD Actor-Critic with Early Feedback

---

## Dataset

The experiments use Offline 1D Bin Packing benchmark instances.

Each instance contains:

* A fixed bin capacity
* A set of item sizes
* A known or reference solution
* A corresponding optimal number of bins where available

The dataset is loaded through the project-specific data loader:

```text
ORLib1DBinPackingLoader
```

The environment used for the experiments is:

```text
Offline1DBinPackingEnv
```

---

## Environment

The custom environment is responsible for:

* Representing the current packing state
* Selecting and placing items
* Managing bin capacities
* Calculating rewards
* Detecting invalid actions
* Tracking the number of bins
* Calculating volume utilization
* Determining episode termination

The environment follows the standard reinforcement learning interaction:

```text
Agent → Action → Environment
                     ↓
               New State
                     ↓
                  Reward
                     ↓
                   Agent
```

---

## Evaluation Metrics

The methods are evaluated using several metrics.

### Number of Bins

The primary objective is to minimize the number of bins required to pack all items.

**Lower is better.**

### Volume Utilization

Volume utilization measures how efficiently the available bin capacity is used.

A higher value indicates better utilization of the available capacity.

**Higher is better.**

### Optimality Gap

The optimality gap measures the difference between the solution produced by the agent and the reference optimal solution.

It is calculated as:

```text
Optimality Gap =
(Agent Solution - Optimal Solution) / Optimal Solution
```

**Lower is better.**

A value of `0` indicates that the agent achieved the reference optimal number of bins.

### Convergence Speed

Convergence speed measures how quickly the model reaches stable solution quality during training.

This metric is used to investigate whether Early Feedback allows the agent to learn useful decisions earlier during training.

### Stability

Stability is evaluated by comparing performance across multiple random seeds.

Mean performance and variation between seeds are analyzed to determine the robustness of each method.

---

## Training and Validation

The training process evaluates the agent periodically using a fixed validation set.

Validation metrics include:

* Validation number of bins
* Validation volume utilization
* Validation optimality gap

Using a fixed validation set allows the performance of different training checkpoints to be compared consistently.

---

## Testing

After training, the final models are evaluated on a separate test set.

Test results are saved in CSV format for further statistical analysis.

The analysis includes comparisons between:

```text
MC
TD
TD + Early Feedback
```

across the different problem sizes and random seeds.

---

## Requirements

The project requires Python and the following main packages:

```text
Python 3.x
PyTorch
NumPy
Pandas
Matplotlib
```

A CUDA-compatible GPU is recommended for training larger instances.

---

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd ProjectNCO
```

Install the required Python packages:

```bash
pip install torch numpy pandas matplotlib
```

Additional dependencies may be required depending on the dataset and training configuration.

---

## Results and Analysis

Training and testing results are stored in CSV files.

The analysis pipeline is used to compare the performance of the different methods across:

* Problem sizes
* Random seeds
* Number of bins
* Volume utilization
* Optimality gap
* Convergence
* Stability

The results are subsequently used to generate tables and plots for the experimental evaluation.

---

## Reproducibility

Random seeds are explicitly controlled for:

```python
random
numpy
torch
```

When CUDA is available, CUDA random seeds are also configured.

This helps ensure that experiments can be reproduced and that differences between methods are not caused solely by random initialization.

Multiple seeds are used to provide a more reliable estimate of algorithm performance.

---

## Research Question

The central research question of this project is:

> **Does providing Early Feedback during solution construction improve the performance of Neural Combinatorial Optimization for Offline Bin Packing?**

The experiments investigate whether Early Feedback can provide improvements in:

* Solution quality
* Optimality gap
* Convergence speed
* Training stability
* Generalization

compared with terminal-reward and standard TD approaches.

The research begins with the **1D Bin Packing Problem** and will subsequently be extended to the more challenging **2D Bin Packing Problem**.

---

## Expected Contribution

The project evaluates whether intermediate feedback can improve reinforcement learning for combinatorial optimization problems where the quality of the final solution is only known after a complete sequence of decisions.

The main comparison is:

```text
MC
 │
 │ Terminal Reward
 ↓
TD
 │
 │ Temporal-Difference Learning
 ↓
TD + Early Feedback
 │
 │ Intermediate Reward
 ↓
Improved Learning Signal?
```

The experimental results are used to determine whether Early Feedback provides a consistent advantage across different bin-packing instance sizes and input types.

The subsequent 2D extension will investigate whether the observed effects remain consistent when moving from one-dimensional to two-dimensional packing.

---

## References

The project is based on research in Neural Combinatorial Optimization and Reinforcement Learning, including:

* Bengio et al. – *Machine Learning for Combinatorial Optimization*
* Cappart et al. – *Combinatorial Optimization and Reinforcement Learning*
* Vinyals et al. – *Pointer Networks*

The project combines concepts from:

* Neural Combinatorial Optimization
* Reinforcement Learning
* Actor-Critic methods
* Monte Carlo learning
* Temporal-Difference learning
* Reward shaping
* Bin Packing

---

## Author

**Graduation Project**

**Topic:** Early Feedback in Neural Combinatorial Optimization for Offline Bin Packing

**Current Focus:** Offline 1D Bin Packing

**Future Extension:** Offline 2D Bin Packing

**Methods:** MC Actor-Critic, TD Actor-Critic, TD Actor-Critic with Early Feedback
