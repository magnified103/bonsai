from network_generator.generator.node.base import CapacityEstimator
from network_generator.types import ComputationCapacity


class ComputationCapacityEstimator(CapacityEstimator):
    """Base class for computation capacity estimators. """

    def estimate(self, node):
        """Estimate the computation capacity of a node.

        :param node: Node.
        :type node: MetaNode

        :return: Computation capacity.
        :rtype: ComputationCapacity
        """
        raise NotImplementedError("ComputationCapacityEstimator.estimate() must be implemented in a subclass.")


class FixedComputationCapacityEstimator(ComputationCapacityEstimator):
    """Fixed computation capacity estimator. """

    def __init__(self, cpu, memory):
        self.computation_capacity = ComputationCapacity(cpu, memory)

    def estimate(self, node):
        return self.computation_capacity
