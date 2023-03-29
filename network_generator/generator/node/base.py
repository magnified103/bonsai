from ...types import NetworkSpecification, NetworkNode, MetaNode
from ...utils import load_country_asn
from ...utils.loaders import load_asns


class NodeGenerator(object):
    """Base class for node generators.

    A node generator is responsible for generating nodes for a network.

    Parameters
    ----------
    None

    Methods
    -------
    generate(network_spec)
        Generate nodes.

    """

    def __init__(self, network_capacity_estimator, computational_capacity_estimator, storage_capacity_estimator):
        self.network_capacity_estimator = network_capacity_estimator
        self.computational_capacity_estimator = computational_capacity_estimator
        self.storage_capacity_estimator = storage_capacity_estimator

    def generate(self, network_spec):
        """Generate nodes.

        :param network_spec: Network specification.
        :type network_spec: NetworkSpecification

        :return: Nodes.
        :rtype: list
        """
        nodes = []
        idx = 0
        for country, number in network_spec.countries.items():
            n, idx = self.generate_nodes(idx, number, country)
            nodes.extend(n)

        for continent, number in network_spec.continents.items():
            n, idx = self.generate_nodes(idx, number, continent=continent)
            nodes.extend(n)

        if len(nodes) < network_spec.nodes:
            n, idx = self.generate_nodes(idx, network_spec.nodes - len(nodes))
            nodes.extend(n)

        return nodes

    def generate_nodes(self, idx, how_many, country=None, continent=None):
        """Generate nodes.

        :param idx: Index of the first node.
        :type idx: int
        :param how_many: Number of nodes to generate.
        :type how_many: int
        :param country: Country.
        :type country: str | None
        :param continent: Continent.
        :type continent: str | None

        :return: Nodes.
        :rtype: list

        :return: Index of the last node.
        :rtype: int

        """
        nodes = []
        if country is None:
            countries = self.generate_country(continent, how_many)
            countries_dict = {}
            for c in countries:
                if c in countries_dict:
                    countries_dict[c] += 1
                else:
                    countries_dict[c] = 1
            for c, n in countries_dict.items():
                ns, idx = self.generate_nodes(idx, n, c)
                nodes.extend(ns)
        else:
            longs, lats = self.generate_coordinates(country, how_many)
            asns = self.generate_asn(country, how_many)
            for i in range(how_many):
                idx += 1
                meta_node = MetaNode(idx, country, lats[i], longs[i], asns[i])
                network_capacity = self.network_capacity_estimator.estimate(meta_node)
                computational_capacity = self.computational_capacity_estimator.estimate(meta_node)
                storage_capacity = self.storage_capacity_estimator.estimate(meta_node)
                nodes.append(NetworkNode(meta_node, network_capacity, computational_capacity, storage_capacity))
        return nodes, idx

    def generate_node(self, i, country=None, continent=None):
        """Generate a node.

        :param i: Index of the node.
        :type i: int
        :param country: Country.
        :type country: str | None
        :param continent: Continent.
        :type continent: str | None

        :return: Node.
        :rtype: NetworkNode

        """
        if country is None:
            country = self.generate_country(continent)
        long, lat = self.generate_coordinates(country)
        asn = self.generate_asn(country)
        meta_node = MetaNode(i, country, lat, long, asn)

        network_capacity = self.network_capacity_estimator.estimate(meta_node)
        computational_capacity = self.computational_capacity_estimator.estimate(meta_node)
        storage_capacity = self.storage_capacity_estimator.estimate(meta_node)
        return NetworkNode(meta_node, network_capacity, computational_capacity, storage_capacity)

    def generate_coordinates(self, country, how_many=None):
        raise NotImplementedError("NodeGenerator.generate_long_lat() must be implemented in a subclass.")

    def generate_asn(self, country, how_many=None):
        raise NotImplementedError("NodeGenerator.generate_asn() must be implemented in a subclass.")

    def generate_country(self, continent=None, how_many=None):
        raise NotImplementedError("NodeGenerator.generate_country() must be implemented in a subclass.")

    pass


class CapacityEstimator(object):
    """Base class for capacity estimators.

    A capacity estimator is responsible for estimating the capacity of a link.

    Parameters
    ----------
    None

    Methods
    -------
    estimate(link)
        Estimate the capacity of a link.

    """

    def __init__(self):
        pass

    def estimate(self, node):
        """Estimate the capacity of a node.

        :param node: Node.
        :type node: MetaNode

        :return: Capacity.
        :rtype: Capacity

        """
        raise NotImplementedError("CapacityEstimator.estimate() must be implemented in a subclass.")
