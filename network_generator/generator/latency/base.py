from ...types import LatencyMatrix


class LatencyGenerator:

    def __init__(self, estimator):
        self.estimator = estimator

    def generate(self, nodes, network_spec):
        latMat = LatencyMatrix(len(nodes))
        for i, node1 in enumerate(nodes):
            for j, node2 in enumerate(nodes[:i]):
                if i == j:
                    continue
                latMat.set_latency(i, j, self.estimator.estimate(node1, node2))


class LatencyEstimator:

    def __init__(self):
        pass

    def estimate(self, node1, node2):
        raise NotImplementedError("LatencyEstimator.estimate() must be implemented in a subclass.")
