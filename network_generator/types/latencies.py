import numpy as np

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

    def __init__(self, nodes):
        self.matrix = np.zeros((nodes, nodes))

    def set_latency(self, node1, node2, latency):
        self.matrix[node1, node2] = latency
        self.matrix[node2, node1] = latency

    def get_latency(self, node1, node2):
        return self.matrix[node1, node2]

    def get_matrix(self):
        return self.matrix

