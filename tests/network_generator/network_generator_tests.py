import unittest

from network_generator import NetworkGenerator

class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.network_generator = NetworkGenerator()

    def test_generate(self):
        self.network_generator.generate()
    def test_something(self):
        self.assertEqual(True, False)  # add assertion here


if __name__ == '__main__':
    unittest.main()
