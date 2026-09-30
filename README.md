A high-performance Python implementation of the **WeST (Weighted Strength Transfer)** algorithm for randomizing weighted undirected networks with integer weight, accelerated using **Numba** (`@njit`).

The algorithm performs a Markov Chain Monte Carlo (MCMC) edge-switching process that randomizes network topology while strictly **preserving the strength of each node**.

## Requirements

- Python >= 3.8
- `numpy`
- `numba`
## Quickstart

