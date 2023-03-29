from .base import NodeGenerator
from .distributions import *

from ...utils.loaders import load_coordinates, load_asns, load_countries, load_country_bounds


class ConfigurableNodeGenerator(NodeGenerator):
    """ConfigurableNodeGenerator node generator.

    Generate nodes with the provided distributions.
    """

    def __init__(self, network_capacity_estimator, computational_capacity_estimator, storage_capacity_estimator,
                 coordinates_distribution_factory, asn_distribution_factory, country_distribution_factory):
        super().__init__(network_capacity_estimator, computational_capacity_estimator, storage_capacity_estimator)
        self.coordinates_distribution_factory = coordinates_distribution_factory
        self.asn_distribution_factory = asn_distribution_factory
        self.country_distribution_factory = country_distribution_factory

    def generate_coordinates(self, country, how_many=None):
        return self.coordinates_distribution_factory.next(country, how_many)

    def generate_asn(self, country, how_many=None):
        return self.asn_distribution_factory.next(country, how_many)

    def generate_country(self, continent=None, how_many=None):
        return self.country_distribution_factory.next(continent, how_many)


class UniformNodeGenerator(ConfigurableNodeGenerator):
    """Uniform node generator.

    Generate nodes with equal probability.
    """

    def __init__(self, network_capacity_estimator, computational_capacity_estimator, storage_capacity_estimator,
                 asns_dataset=None, countries_dataset=None):

        country_bounds = load_country_bounds()
        coordinates_distribution_factory = UniformGeoDistributionFactory(country_bounds)
        if asns_dataset is None:
            asns_dataset = load_asns()

        asn_distribution_factory = UniformDistributionFactory(asns_dataset)

        if countries_dataset is None:
            countries_dataset = load_countries()
        country_distribution_factory = UniformDistributionFactory(countries_dataset)

        super().__init__(network_capacity_estimator, computational_capacity_estimator, storage_capacity_estimator,
                         coordinates_distribution_factory, asn_distribution_factory, country_distribution_factory)


class MultinomialNodeGenerator(ConfigurableNodeGenerator):
    """MultinomialNodeGenerator node generator.

    Generate nodes with the provided multinomial distributions.

    Parameters
    ----------

    network_capacity_estimator : NetworkCapacityEstimator
        Network capacity estimator.
    computational_capacity_estimator : ComputationalCapacityEstimator
        Computational capacity estimator.
    storage_capacity_estimator : StorageCapacityEstimator
        Storage capacity estimator.
    coordinates_dataset : Dataset
        Coordinates dataset.
    asns_dataset : Dataset
        ASNs dataset.
    countries_dataset : Dataset
        Countries dataset.

    """

    def __init__(self, network_capacity_estimator, computational_capacity_estimator, storage_capacity_estimator,
                 coordinates_dataset=None, asns_dataset=None, countries_dataset=None):
        country_bounds = load_country_bounds()
        if coordinates_dataset is None:
            coordinates_dataset = load_coordinates()
        if asns_dataset is None:
            asns_dataset = load_asns()
        if countries_dataset is None:
            countries_dataset = load_countries()

        coordinates_distribution_factory = MultinomialGeoDistributionFactory(coordinates_dataset, country_bounds)
        asn_distribution_factory = MultinomialDistributionFactory(asns_dataset)
        country_distribution_factory = MultinomialDistributionFactory(countries_dataset)

        super().__init__(network_capacity_estimator, computational_capacity_estimator, storage_capacity_estimator,
                         coordinates_distribution_factory, asn_distribution_factory, country_distribution_factory)
