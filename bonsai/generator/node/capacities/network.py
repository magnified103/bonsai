import numpy as np
from bonsai.generator.node.base import CapacityEstimator
from bonsai.types import NetworkCapacity


class NetworkCapacityEstimator(CapacityEstimator):

    def estimate(self, node):
        """Estimate the network capacity of a node.

        :param node: Node.
        :type node: MetaNode

        :return: Network capacity.
        :rtype: NetworkCapacity
        """
        raise NotImplementedError("NetworkCapacityEstimator.estimate() must be implemented in a subclass.")


class FixedNetworkCapacityEstimator(NetworkCapacityEstimator):
        def __init__(self, download, upload):
            self.network_capacity = NetworkCapacity(download, upload)

        def estimate(self, node):
            return self.network_capacity
