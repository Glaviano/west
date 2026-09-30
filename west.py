import numpy as np
from numba import njit


@njit
def from_adj_to_edgelist(A):
    """
    Convert an undirected adjacency matrix to a pre-allocated edge list.

    Parameters
    ----------
    A : np.ndarray
        2D square adjacency matrix representing an undirected graph.

    Returns
    -------
    edgelist : np.ndarray of shape (E_max, 2)
        Pre-allocated array containing the edges (node pairs).
        Unused positions are uninitialized.
    count : int
        The actual number of active edges found in the matrix.
    """
    n = A.shape[0]
    E_max = n * (n - 1) // 2  # Maximum possible edges in an undirected simple graph
    edgelist = np.empty((E_max, 2), dtype=np.int64)
    count = 0

    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j] != 0:
                edgelist[count, 0] = i
                edgelist[count, 1] = j
                count += 1

    return edgelist, count


@njit
def WeST(A, edgelist_array, n_link, n_step):
    """

    Performs Markov Chain Monte Carlo (MCMC) edge rewiring to randomize
    a weighted network while preserving vertex strengths,
    using the Metropolis-Hastings acceptance criterion.

    Parameters
    ----------
    A : np.ndarray
        Adjacency matrix of the graph. Modified IN-PLACE.
    edgelist_array : np.ndarray of shape (MAX_EDGES, 2)
        Pre-allocated array holding the current active edges.
        Must have sufficient capacity to accommodate new edges. Modified IN-PLACE.
    n_link : int
        Current number of active edges present in `edgelist_array`.
    n_step : int
        Number of MCMC rewiring iterations to perform.

    Returns
    -------
    A : np.ndarray
        Updated adjacency matrix.
    edgelist_array : np.ndarray
        Updated edge list buffer.
    n_link : int
        Updated count of active edges.

    Notes
    -----
    - Graph is assumed to be undirected and without self-loops.
    - Edges are removed in O(1) via swap-and-pop to avoid array restructuring.
    """
    for _ in range(n_step):
        # -------------------------------------------------------------
        # 1. Edge Selection: sample two distinct random edges
        # -------------------------------------------------------------
        idx1 = np.random.randint(n_link)
        idx2 = np.random.randint(n_link)
        while idx2 == idx1:
            idx2 = np.random.randint(n_link)

        a, b = edgelist_array[idx1]
        c, d = edgelist_array[idx2]

        w_ab = A[a, b]
        w_cd = A[c, d]

        # Prevent rewiring if the two edges share any endpoint
        if a == c or a == d or b == c or b == d:
            continue

        minimum = w_ab if w_ab < w_cd else w_cd

        # -------------------------------------------------------------
        # 2. Weight Perturbation: sample an integer weight to transfer
        # -------------------------------------------------------------
        w = np.random.randint(1, minimum + 1)

        new_n_link = n_link
        new_w_ab = w_ab - w
        new_w_cd = w_cd - w

        # Track potential edge removals
        if new_w_ab == 0:
            new_n_link -= 1
        if new_w_cd == 0:
            new_n_link -= 1

        # -------------------------------------------------------------
        # 3. Topology Pairing: randomly choose crossing orientation
        # -------------------------------------------------------------
        if np.random.randint(2) == 0:
            k, l = c, d
        else:
            k, l = d, c

        # -------------------------------------------------------------
        # 4. Target Weights & Acceptance Ratio Computation
        # -------------------------------------------------------------
        old_w_ak = A[a, k]
        old_w_bl = A[b, l]
        w_ak = old_w_ak + w
        w_bl = old_w_bl + w

        minimum2 = w_ak if w_ak < w_bl else w_bl

        # Track potential edge creations
        if old_w_ak == 0:
            new_n_link += 1
        if old_w_bl == 0:
            new_n_link += 1

        # Metropolis-Hastings acceptance probability:
        # P = [q(new -> old)] / [q(old -> new)]
        # Cast to float to prevent integer overflow with large graphs
        num = float(n_link * (n_link - 1) * minimum)
        den = float(new_n_link * (new_n_link - 1) * minimum2)
        P = num / den

        # -------------------------------------------------------------
        # 5. Metropolis Acceptance Step & State Update
        # -------------------------------------------------------------
        if np.random.rand() < P:
            # Update adjacency matrix (symmetrically for undirected graph)
            A[a, b] = new_w_ab
            A[b, a] = new_w_ab
            A[c, d] = new_w_cd
            A[d, c] = new_w_cd

            A[a, k] = w_ak
            A[k, a] = w_ak
            A[b, l] = w_bl
            A[l, b] = w_bl

            # O(1) Edge Removal via Swap-and-Pop:
            # Always remove the higher index first to avoid invalidating the lower index
            if new_w_ab == 0 and new_w_cd == 0:
                idx_max = idx1 if idx1 > idx2 else idx2
                idx_min = idx1 if idx1 < idx2 else idx2

                edgelist_array[idx_max] = edgelist_array[n_link - 1]
                n_link -= 1

                edgelist_array[idx_min] = edgelist_array[n_link - 1]
                n_link -= 1

            elif new_w_ab == 0:
                edgelist_array[idx1] = edgelist_array[n_link - 1]
                n_link -= 1

            elif new_w_cd == 0:
                edgelist_array[idx2] = edgelist_array[n_link - 1]
                n_link -= 1

            # Append newly created edges to the edge list
            if old_w_ak == 0:
                edgelist_array[n_link, 0] = a
                edgelist_array[n_link, 1] = k
                n_link += 1

            if old_w_bl == 0:
                edgelist_array[n_link, 0] = b
                edgelist_array[n_link, 1] = l
                n_link += 1

    return A, edgelist_array, n_link