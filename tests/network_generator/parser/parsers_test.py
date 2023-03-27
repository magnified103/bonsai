import unittest
import network_generator.parser.parsers as parsers


class ParsersTestCase(unittest.TestCase):
    def test_yaml_parser(self):
        yaml_parser = parsers.YAMLParser()

        config = '''
        nodes: 10
        continents:
            EU: 5
            AS: 5
        countries:
            DE: 2
            FR: 2
            CN: 2
            JP: 2
        '''

        parsed = yaml_parser.parse(config)
        self.assertEqual(parsed['nodes'], 10)
        self.assertEqual(parsed['continents']['EU'], 5)
        self.assertEqual(parsed['continents']['AS'], 5)
        self.assertEqual(parsed['countries']['DE'], 2)
        self.assertEqual(parsed['countries']['FR'], 2)
        self.assertEqual(parsed['countries']['CN'], 2)
        self.assertEqual(parsed['countries']['JP'], 2)

    def test_yaml_parser_from_file(self):
        yaml_parser = parsers.YAMLParser()

        config = '''
        nodes: 10
        continents:
            EU: 5
            AS: 5
        countries:
            DE: 2
            FR: 2
            CN: 2
            JP: 2
        '''

        with open('test.yaml', 'w') as f:
            f.write(config)

        parsed = yaml_parser.parse('test.yaml')
        self.assertEqual(parsed['nodes'], 10)
        self.assertEqual(parsed['continents']['EU'], 5)
        self.assertEqual(parsed['continents']['AS'], 5)
        self.assertEqual(parsed['countries']['DE'], 2)
        self.assertEqual(parsed['countries']['FR'], 2)
        self.assertEqual(parsed['countries']['CN'], 2)
        self.assertEqual(parsed['countries']['JP'], 2)


if __name__ == '__main__':
    unittest.main()
