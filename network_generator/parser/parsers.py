from .base import BaseParser
import yaml


class YAMLParser(BaseParser):
    """YAML parser.

Parse a YAML file into a dictionary.

----------------
Method
----------------
parse(network_specification)
    Parse network specification.
    :param network_specification: Network specification file.
    :type network_specification: str or file stream
    :return: Parsed network specification.
    :rtype: dict

----------------
:Example:

>>> from network_generator.parser.parsers import YAMLParser
>>> parser = YAMLParser()
>>> parser.parse('network_specification.yaml')
{'nodes': 10, 'continents': {'EU': 5, 'AS': 5}, 'countries': {'DE': 2, 'FR': 2, 'CN': 2, 'JP': 2}}
>>> parser.parse('''
... nodes: 10
... continents:
...     EU: 5
...     AS: 5
... countries:
...     DE: 2
...     FR: 2
...     CN: 2
...     JP: 2
... ''')
{'nodes': 10, 'continents': {'EU': 5, 'AS': 5}, 'countries': {'DE': 2, 'FR': 2, 'CN': 2, 'JP': 2}}
>>> with open('network_specification.yaml', 'r') as f:
...     parser.parse(f)
{'nodes': 10, 'continents': {'EU': 5, 'AS': 5}, 'countries': {'DE': 2, 'FR': 2, 'CN': 2, 'JP': 2}}
    """

    def parse(self, network_specification):
        """Parse network specification.

        :param network_specification: Network specification file.
        :type network_specification: str or file

        :return: Parsed network specification.
        :rtype: dict
        """
        if isinstance(network_specification, str) and\
                (network_specification.endswith('.yaml') or network_specification.endswith('.yml')):
            with open(network_specification, 'r') as f:
                return yaml.load(f, Loader=yaml.FullLoader)
        else:
            return yaml.load(network_specification, Loader=yaml.FullLoader)
