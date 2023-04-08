from collections import deque
import numpy as np
import networkx as nx

def strong_collapse(g):
    """
    calculates persistent diagrams of a graph

    Strong Collapse for Persistence
    https://arxiv.org/pdf/1809.10945.pdf
    """
    mat = nx.incidence_matrix(g).toarray()

    row_q = deque()
    col_q = deque()
    [row_q.append(row) for row in mat]
    row_visited = [False] * mat.shape[0]
    col_visited = [False] * mat.shape[1]

    while len(row_q) != 0:
        while len(row_q) != 0:
            v = row_q.pop()
            nonzero_cols = np.where(v != 0)[0]
            if len(nonzero_cols) == 0:
                continue
            nonzero_col = mat[:, nonzero_cols[0]]
            nonzero_rows = mat[np.where(nonzero_col != 0)[0]]
            subset = np.where(np.all(nonzero_rows == v))[0]
            mat = np.delete(mat, (subset), axis=0)
            for i in nonzero_cols:
                if not col_visited[i]:
                    col_q.append(mat[:, i])
                    col_visited[i] = True

        while len(col_q) != 0:
            v = col_q.pop()
            nonzero_rows = np.where(v != 0)[0]
            if len(nonzero_rows) == 0:
                continue
            nonzero_row = mat[:, nonzero_rows[0]]
            nonzero_cols = mat[np.where(nonzero_row != 0)[0]]
            subset = np.where(np.all(nonzero_cols == v))[0]
            mat = np.delete(mat, (subset), axis=0)
            for i in nonzero_rows:
                if not row_visited[i]:
                    row_q.append(mat[:, i])
                    row_visited[i] = True

    return mat