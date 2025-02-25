from bonsai.generator.node.base import CapacityEstimator
from bonsai.types import StorageCapacity


class StorageCapacityEstimator(CapacityEstimator):
    """Base class for storage capacity estimators. """

    def estimate(self, node):
        """Estimate the storage capacity of a node.

        :param node: Node.
        :type node: MetaNode

        :return: Storage capacity.
        :rtype: StorageCapacity
         """
        raise NotImplementedError("StorageEstimator.estimate() must be implemented in a subclass.")


class FixedStorageCapacityEstimator(StorageCapacityEstimator):

    def __init__(self, bytes):
        self.storage_capacity = StorageCapacity(bytes)

    def estimate(self, node):
        return self.storage_capacity

