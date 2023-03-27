import unittest
import network_generator.interpreter.interperters as interpreters

class DefaultInterpreterTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.interpreter = interpreters.DefaultInterpreter()

    def test_interpret(self):
        with self.assertRaises(ValueError) as context:
            self.interpreter.interpret({})
        self.assertTrue("Missing 'nodes' key in network specification." in str(context.exception))

        net_spec = self.interpreter.interpret({
            'nodes': 10
        })

        self.assertEqual(net_spec.nodes, 10)

        net_spec = self.interpreter.interpret({
            'nodes': 10,
            'continents': {}
        })

        self.assertEqual(net_spec.nodes, 10)
        self.assertEqual(len(net_spec.continents), 0)

        net_spec = self.interpreter.interpret({
            'nodes': 10,
            'continents': {
                'EU': 5,
                'AS': 5
            }
        })

        self.assertEqual(net_spec.nodes, 10)
        self.assertEqual(len(net_spec.continents), 2)
        self.assertEqual(net_spec.continents['EU'], 5)
        self.assertEqual(net_spec.continents['AS'], 5)

        net_spec = self.interpreter.interpret({
            'nodes': 10,
            'continents': {
                'EU': 5,
                'AS': 5
            },
            'countries': {
                'DE': 2,
                'FR': 2,
                'CN': 2,
                'JP': 2
            }
        })

        self.assertEqual(net_spec.nodes, 10)
        self.assertEqual(len(net_spec.countries), 4)
        self.assertEqual(net_spec.countries['DE'], 2)
        self.assertEqual(net_spec.countries['FR'], 2)
        self.assertEqual(net_spec.countries['CN'], 2)
        self.assertEqual(len(net_spec.continents), 2)
        self.assertEqual(net_spec.continents['EU'], 1)
        self.assertEqual(net_spec.continents['AS'], 1)

        with self.assertRaises(ValueError) as context:
            net_spec = self.interpreter.interpret({
                'nodes': 10,
                'continents': {
                    'EU': 5,
                    'AS': 5
                },
                'countries': {
                    'DE': 2,
                    'FR': 2,
                    'CN': 2,
                    'JP': 2,
                    'US': 2,
                    'CA': 2,
                    'MX': 2,
                    'BR': 2,
                },
            })
        self.assertTrue("Invalid specification:" in str(context.exception))










if __name__ == '__main__':
    unittest.main()
