from ...types import NetworkSpecification, NetworkNode, MetaNode
from ...utils import load_country_asn


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
        for country, number in network_spec.countries.items():
            for i in range(number):
                nodes.append(self.generate_node(i, country))

        for continent, number in network_spec.continents.items():
            for i in range(number):
                nodes.append(self.generate_node(i, continent=continent))

        if len(nodes) < network_spec.nodes:
            for i in range(network_spec.nodes - len(nodes)):
                nodes.append(self.generate_node(i))

        return nodes

    def generate_node(self, i, country=None, continent=None):
        if country is None:
            country = self.generate_country(continent)
        long, lat = self.generate_long_lat(country)
        asn = self.generate_asn(country)
        meta_node = MetaNode(i, country, lat, long, asn)

        network_capacity = self.network_capacity_estimator.estimate(meta_node)
        computational_capacity = self.computational_capacity_estimator.estimate(meta_node)
        storage_capacity = self.storage_capacity_estimator.estimate(meta_node)
        return NetworkNode(meta_node, network_capacity, computational_capacity, storage_capacity)

    def generate_long_lat(self, country):
        raise NotImplementedError("NodeGenerator.generate_long_lat() must be implemented in a subclass.")

    def generate_asn(self, country):
        raise NotImplementedError("NodeGenerator.generate_asn() must be implemented in a subclass.")

    def generate_country(self, continent=None):
        raise NotImplementedError("NodeGenerator.generate_country() must be implemented in a subclass.")


class Distribution(object):
    """Base class for distributions.

    A distribution is responsible for generating values from a distribution.

    Parameters
    ----------
    None

    Methods
    -------
    next()
        Generate a value.

    """

    def __init__(self):
        pass

    def next(self):
        """Generate a value.

        :param args: Arguments.
        :type args: list

        :param kwargs: Keyword arguments.
        :type kwargs: dict

        :return: Value.
        :rtype: object

        """
        raise NotImplementedError("Distribution.next() must be implemented in a subclass.")


class AsnDistribution(Distribution):
    """ASN distribution.

    Generate ASNs from a distribution.

    Parameters
    ----------
    None

    Methods
    -------
    next(country)
        Generate an ASN.

    """

    def __init__(self, asn_file=None):
        asns = load_country_asn(asn_file)
        super().__init__()

    def next(self, country):
        """Generate an ASN.

        :param country: Country.
        :type country: str

        :return: ASN.
        :rtype: int

        """
        raise NotImplementedError("AsnDistribution.next() must be implemented in a subclass.")


class CountryDistribution(Distribution):
    """Country distribution.

    Generate countries from a distribution.

    Parameters
    ----------
    None

    Methods
    -------
    next()
        Generate a country.

    """

    def __init__(self, countries_file=None):
        super().__init__()

    def next(self, continent=None):
        """Generate a country.

        :param continent: Continent.
        :type continent: str

        :return: Country.
        :rtype: str

        """
        raise NotImplementedError("CountryDistribution.next() must be implemented in a subclass.")


class LocationDistribution(Distribution):
    """Location distribution.

    Generate locations from a distribution.

    Parameters
    ----------
    None

    Methods
    -------
    next(country)
        Generate a location.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def next(self, country):
        """Generate a location.

        :param country: Country.
        :type country: str

        :return: Location.
        :rtype: tuple

        """
        raise NotImplementedError("LocationDistribution.next() must be implemented in a subclass.")


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
