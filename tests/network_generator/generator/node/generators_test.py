import unittest

from matplotlib import pyplot as plt

import network_generator.generator.node.generators as generators
from network_generator.generator.node.capacities import FixedComputationCapacityEstimator, \
    FixedStorageCapacityEstimator, FixedNetworkCapacityEstimator
from network_generator.types import NetworkSpecification, NetworkNode
import network_generator.utils.location as utils_location


class GeneratorTestCase(unittest.TestCase):
    def test_UniformNodeGenerator(self):
        generator = generators.UniformNodeGenerator(FixedNetworkCapacityEstimator(100, 20),
                                                    FixedComputationCapacityEstimator(4, 2),
                                                    FixedStorageCapacityEstimator(1 * 1024 * 1024 * 1024)
                                                    )
        net_spec = NetworkSpecification(100, {}, {})
        nodes = generator.generate(net_spec)
        self.assertEqual(len(nodes), 100)
        utils_location.country_bounds.plot()
        for node in nodes:
            node: NetworkNode = node
            self.assertEqual(node.computation_capacity.cpu, 4)
            self.assertEqual(node.computation_capacity.memory, 2)
            self.assertEqual(node.storage_capacity.storage, 1 * 1024 * 1024 * 1024)
            self.assertEqual(node.network_capacity.download, 100)
            self.assertEqual(node.network_capacity.upload, 20)
            plt.plot(node.meta_node.longitude, node.meta_node.latitude, 'ro', markersize=1)
        plt.show()

    def test_UniformNodeGeneratorWithContinents(self):
        generator = generators.UniformNodeGenerator(FixedNetworkCapacityEstimator(100, 20),
                                                    FixedComputationCapacityEstimator(4, 2),
                                                    FixedStorageCapacityEstimator(1 * 1024 * 1024 * 1024)
                                                    )
        net_spec = NetworkSpecification(10, {'EU': 5, 'AS': 5}, {})
        nodes = generator.generate(net_spec)
        self.assertEqual(len(nodes), 10)
        utils_location.country_bounds.plot()
        for node in nodes:
            node: NetworkNode = node
            self.assertEqual(node.computation_capacity.cpu, 4)
            self.assertEqual(node.computation_capacity.memory, 2)
            self.assertEqual(node.storage_capacity.storage, 1 * 1024 * 1024 * 1024)
            self.assertEqual(node.network_capacity.download, 100)
            self.assertEqual(node.network_capacity.upload, 20)
            self.assertTrue(utils_location.get_continent(node.meta_node.country_code) == 'EU' or utils_location.get_continent(node.meta_node.country_code) == 'AS')
            plt.plot(node.meta_node.longitude, node.meta_node.latitude, 'ro', markersize=2)
        plt.show()



    def test_UniformNodeGeneratorWithCountries(self):
        generator = generators.UniformNodeGenerator(FixedNetworkCapacityEstimator(100, 20),
                                                    FixedComputationCapacityEstimator(4, 2),
                                                    FixedStorageCapacityEstimator(1 * 1024 * 1024 * 1024)
                                                    )
        net_spec = NetworkSpecification(10, {}, {'DE': 5, 'AT': 5})
        nodes = generator.generate(net_spec)
        self.assertEqual(len(nodes), 10)
        for node in nodes:
            node: NetworkNode = node
            self.assertEqual(node.computation_capacity.cpu, 4)
            self.assertEqual(node.computation_capacity.memory, 2)
            self.assertEqual(node.storage_capacity.storage, 1 * 1024 * 1024 * 1024)
            self.assertEqual(node.network_capacity.download, 100)
            self.assertEqual(node.network_capacity.upload, 20)
            self.assertTrue(node.meta_node.country_code == 'DE' or node.meta_node.country_code == 'AT')




if __name__ == '__main__':
    unittest.main()
