
class NetworkSpecification:

    """A class that represents the specification of a network.

:param nodes: Number of nodes.
:type nodes: int
:param continents: Number of nodes per continent.
:type continents: dict
:param countries: Number of nodes per country.
:type countries: dict

Attributes:
-----------
nodes (int): Number of nodes.
continents (dict): Number of nodes per continent.
countries (dict): Number of nodes per country.


----------------
Methods:
----------------
get_how_many_nodes()
    Returns the number of nodes.
get_nodes_per_continent()
    Returns a generator that yields the number of nodes per continent.
get_nodes_per_country()
    Returns a generator that yields the number of nodes per country.

----------------
:Example:

>>> network_spec = NetworkSpecification(10, {'EU': 5, 'AS': 5}, {'DE': 2, 'FR': 2, 'CN': 2, 'JP': 2})
>>> network_spec.get_how_many_nodes()
10
>>> network_spec.get_nodes_per_continent()
<generator object NetworkSpecification.get_nodes_per_continent at 0x7f8b8c0b9f68>
>>> network_spec.get_nodes_per_country()
<generator object NetworkSpecification.get_nodes_per_country at 0x7f8b8c0b9f68>
>>> list(network_spec.get_nodes_per_continent())
[('EU', 5), ('AS', 5)]
>>> list(network_spec.get_nodes_per_country())
[('DE', 2), ('FR', 2), ('CN', 2), ('JP', 2)]

    """

    def __init__(self, nodes, continents, countries):
        self.nodes = nodes
        self.continents = continents
        self.countries = countries

    def get_how_many_nodes(self):
        return self.nodes

    def get_nodes_per_continent(self):
        if self.continents is None:
            return
        for continent, number in self.continents.items():
            yield continent, number

    def get_nodes_per_country(self):
        if self.countries is None:
            return
        for country, number in self.countries.items():
            yield country, number

