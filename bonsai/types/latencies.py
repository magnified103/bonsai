import time

import numpy as np

from joblib import Parallel, delayed

class LatencyMatrix:
    """A class to represent a latency matrix

Attributes
----------
matrix : numpy.ndarray
    The latency matrix

Methods
-------
set_latency(node1, node2, latency)
    Sets the latency between node1 and node2 to latency
get_latency(node1, node2)
    Returns the latency between node1 and node2
get_matrix()
    Returns the latency matrix

    """

    def __init__(self, nodes, nodes_x_nodes=None, latencies=None):
        self.matrix = np.zeros((len(nodes), len(nodes)))
        self.node_dict = {node: i for i, node in enumerate(nodes)}
        if nodes_x_nodes is not None and latencies is not None:
            for i, (node1, node2) in enumerate(nodes_x_nodes):
                self.set_latency(node1, node2, latencies[i])
        elif nodes_x_nodes is None and latencies is not None:
            raise ValueError("nodes_x_nodes must be provided if latencies is provided")
        elif nodes_x_nodes is not None and latencies is None:
            raise ValueError("latencies must be provided if nodes_x_nodes is provided")

    def set_latency(self, node1, node2, latency):
        idx_node1, idx_node2 = self.idx_node(node1), self.idx_node(node2)
        self.matrix[idx_node1, idx_node2] = latency
        self.matrix[idx_node2, idx_node1] = latency

    def get_latency(self, node1_idx, node2_idx):
        return self.matrix[node1_idx, node2_idx]

    def get_matrix(self):
        return self.matrix

    def __str__(self):
        return str(self.matrix)

    def __repr__(self):
        return str(self.matrix)

    def idx_node(self, node):
        if isinstance(node, int):
            return node
        else:
            return self.node_dict[node]

    class NetworkNodePair:
        def __init__(self, node1, node2):
            self.node1 = node1
            self.node2 = node2

        def __hash__(self):
            return hash((self.node1, self.node2))

        def __eq__(self, other):
            return (self.node1, self.node2) == (other.node1, other.node2)
    def run_vectorized(self, nodes_x_nodes, latencies):
        pairs = [self.NetworkNodePair(node1, node2) for node1, node2 in nodes_x_nodes]
        def _set_latency(pair, latency):
            node1, node2 = pair.node1, pair.node2
            idx_node1, idx_node2 = self.idx_node(node1), self.idx_node(node2)
            self.matrix[idx_node1, idx_node2] = latency
            self.matrix[idx_node2, idx_node1] = latency
        vf = np.vectorize(_set_latency)
        vf(pairs, latencies)
