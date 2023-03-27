from .base import BaseInterpreter
from ..types import NetworkSpecification
from ..utils import get_continent


class DefaultInterpreter(BaseInterpreter):

    def interpret(self, network_specification):
        """Interpret a network specification.

        :param network_specification: Network specification file.
        :type network_specification: dict

        :return: Interpreted network specification.
        :rtype: NetworkSpecification
        """

        if 'nodes' not in network_specification:
            raise ValueError("Missing 'nodes' key in network specification.")
        nodes = network_specification['nodes']
        continents = network_specification.get('continents', {})
        countries = network_specification.get('countries', {})

        total_nodes = 0
        for country, number in countries.items():
            continent = get_continent(country)
            total_nodes += number
            if continent in continents:
                continents[continent] -= number
                if continents[continent] < 0:
                    print("Warning: Number of nodes in continent '{}' is less than number of nodes in countries, ignoring this.".format(continent))
                    continents[continent] = 0

        for continent, number in continents.items():
            total_nodes += number

        if total_nodes > nodes:
            raise ValueError("Invalid specification: Number of nodes in countries and continents '{}' is greater than total number of nodes '{}'.".format(total_nodes, nodes))

        return NetworkSpecification(nodes, continents, countries)