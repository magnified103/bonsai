from ...types import NetworkSpecification, NetworkNode, MetaNode


class NodeGenerator:
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
        :rtype: list[NetworkNode]
        """
        nodes = []
        idx = 0
        for country, number in network_spec.countries.items():
            n, idx = self.generate_nodes(idx, number, country=country)
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
            countries = self.generate_countries(continent, how_many)
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
            coordinates = self.generate_coordinates(country, how_many)
            asns = self.generate_asns(country, how_many)
            for i in range(how_many):
                idx += 1
                long, lat = coordinates[i]
                meta_node = MetaNode(idx, country, lat, long, asns[i])
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
            country = self.generate_countries(continent)
        long, lat = self.generate_coordinates(country)
        asn = self.generate_asns(country)
        meta_node = MetaNode(i, country, lat, long, asn)

        network_capacity = self.network_capacity_estimator.estimate(meta_node)
        computational_capacity = self.computational_capacity_estimator.estimate(meta_node)
        storage_capacity = self.storage_capacity_estimator.estimate(meta_node)
        return NetworkNode(meta_node, network_capacity, computational_capacity, storage_capacity)

    def generate_coordinates(self, country, how_many=None):
        """
        Generate coordinates.

        :param country: Country of the coordinates.
        :type country: str
        :param how_many: Number of coordinates to generate.
        :type how_many: int | None
        :return: List of longitudes and latitudes.
        :rtype: list
        """
        raise NotImplementedError("NodeGenerator.generate_long_lat() must be implemented in a subclass.")

    def generate_asns(self, country, how_many=None):
        """
        Generate ASNs.

        :param country: Country of the ASNs.
        :type country: str
        :param how_many: Number of ASNs to generate.
        :type how_many: int | None
        :return: List of ASNs.
        :rtype: list
        """
        raise NotImplementedError("NodeGenerator.generate_asn() must be implemented in a subclass.")

    def generate_countries(self, continent=None, how_many=None):
        """
        Generate countries.

        :param continent: Countries continent.
        :type continent: str | None
        :param how_many: Number of countries to generate.
        :type how_many: int | None
        :return: List of countries.
        :rtype: list
        """
        raise NotImplementedError("NodeGenerator.generate_country() must be implemented in a subclass.")

    def config(self):
        """Return the configuration of the generator.

        :return: Configuration.
        :rtype: dict
        """
        raise NotImplementedError("NodeGenerator.config() must be implemented in a subclass.")




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

    def config(self):
        """Return the configuration of the estimator.

        :return: Configuration.
        :rtype: dict
        """
        raise NotImplementedError("CapacityEstimator.config() must be implemented in a subclass.")
