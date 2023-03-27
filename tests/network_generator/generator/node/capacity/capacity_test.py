import unittest
import network_generator.generator.node.capacities as capacities

class CapacityTestCase(unittest.TestCase):
    def test_FixedComputationCapacityEstimator(self):
        estimator = capacities.FixedComputationCapacityEstimator(4, 2)
        self.assertEqual(estimator.estimate(None).cpu, 4)
        self.assertEqual(estimator.estimate(None).memory, 2)

    def test_FixedStorageCapacityEstimator(self):
        estimator = capacities.FixedStorageCapacityEstimator(1*1024*1024*1024)
        self.assertEqual(estimator.estimate(None).storage, 1*1024*1024*1024)

    def test_FixedNetworkCapacityEstimator(self):
        estimator = capacities.FixedNetworkCapacityEstimator(100, 20)
        self.assertEqual(estimator.estimate(None).download, 100)
        self.assertEqual(estimator.estimate(None).upload, 20)



if __name__ == '__main__':
    unittest.main()
