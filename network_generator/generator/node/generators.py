from .base import NodeGenerator
from ...types import NetworkSpecification, NetworkNode
from ...types import MetaNode

import numpy as np

from ...utils import get_random_long_lat, get_asns, get_countries


class UniformNodeGenerator(NodeGenerator):
    """Uniform node generator.

    Generate nodes with equal probability.
    """

    def generate_long_lat(self, country):
        return get_random_long_lat(country)

    def generate_asn(self, country):
        asns = get_asns(country)
        return np.random.choice(asns)

    def generate_country(self, continent=None):
        countries = get_countries(continent)
        return np.random.choice(countries)


class BiasedNodeGenerator(NodeGenerator):
    """BiasedNodeGenerator node generator.

    Generate nodes with the provided multinomial distributions.
    """

    def __init__(self, network_capacity_estimator, computational_capacity_estimator, storage_capacity_estimator,
                 long_lat_distribution, asn_distribution, country_distribution):
        super().__init__(network_capacity_estimator, computational_capacity_estimator, storage_capacity_estimator)
        self.long_lat_distribution = long_lat_distribution
        self.asn_distribution = asn_distribution
        self.country_distribution = country_distribution

    def generate_long_lat(self, country):
        return self.long_lat_distribution.next(country)

    def generate_asn(self, country):
        return self.asn_distribution.next(country)

    def generate_country(self, continent=None):
        return self.country_distribution.next(continent)

