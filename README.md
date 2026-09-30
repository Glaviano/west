A high-performance Python implementation of the **WeST (Weighted Strength Transfer)** algorithm for randomizing weighted undirected networks with integer weight, accelerated using **Numba** (`@njit`).

The algorithm performs a Markov Chain Monte Carlo (MCMC) edge-switching process that randomizes network topology while strictly **preserving the strength of each node**.

## Requirements

- Python >= 3.8
- `numpy`
- `numba`
## Quickstart

```python
import numpy as np
from west import from_adj_to_edgelist, WeST_optimized

# Symmetric weighted adjacency matrix
A = np.array([
    [0, 5, 2, 0],
    [5, 0, 0, 3],
    [2, 0, 0, 4],
    [0, 3, 4, 0]
], dtype=np.int64)

# Prepare edgelist buffer and run
# The changes are applied in place, so if you want to keep the original matrix A unchanged, you must create a copy:
# A_rand = A.copy()
edgelist, n_link = from_adj_to_edgelist(A)
A_rand, edgelist, n_link = WeST_optimized(A, edgelist, n_link, n_step=10_000)

```
