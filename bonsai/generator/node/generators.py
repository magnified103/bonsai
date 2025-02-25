from .base import NodeGenerator
from .capacities import FixedNetworkCapacityEstimator, FixedComputationCapacityEstimator, FixedStorageCapacityEstimator
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
        coords = self.coordinates_distribution_factory.next(country, how_many)
        if type(coords) != list and type(coords) != np.ndarray:
            return [coords]
        return coords

    def generate_asns(self, country, how_many=None):
        asns = self.asn_distribution_factory.next(country, how_many)
        if type(asns) != list and type(asns) != np.ndarray:
            return [asns]
        return asns

    def generate_countries(self, continent=None, how_many=None):
        countries = self.country_distribution_factory.next(continent, how_many)
        if type(countries) != list and type(countries) != np.ndarray:
            return [countries]
        return countries

    def config(self):
        return {
            'network_capacity_estimator': self.network_capacity_estimator,
            'computational_capacity_estimator': self.computational_capacity_estimator,
            'storage_capacity_estimator': self.storage_capacity_estimator,
            'coordinates_distribution': self.coordinates_distribution_factory,
            'asn_distribution': self.asn_distribution_factory,
            'country_distribution': self.country_distribution_factory
        }


class UniformNodeGenerator(ConfigurableNodeGenerator):
    """Uniform node generator.

    Generate nodes with equal probability.
    """

    def __init__(self, network_capacity_estimator=None, computational_capacity_estimator=None, storage_capacity_estimator=None,
                 asns_dataset=None, countries_dataset=None):
        """
        Initialize the node generator.

        :param network_capacity_estimator:  (Default value = None)
        :type: network_capacity_estimator: NetworkCapacityEstimator
        :param computational_capacity_estimator:  (Default value = None)
        :type: computational_capacity_estimator: ComputationalCapacityEstimator
        :param storage_capacity_estimator: (Default value = None)
        :type: storage_capacity_estimator: StorageCapacityEstimator
        :param asns_dataset: (Default value = None)
        :type: asns_dataset: Dataset
        :param countries_dataset: (Default value = None)
        :type: countries_dataset: Dataset
        """
        if network_capacity_estimator is None:
            network_capacity_estimator = FixedNetworkCapacityEstimator(0, 0)
        if computational_capacity_estimator is None:
            computational_capacity_estimator = FixedComputationCapacityEstimator(0, 0)
        if storage_capacity_estimator is None:
            storage_capacity_estimator = FixedStorageCapacityEstimator(0)
        country_bounds = load_country_bounds()

        coordinates_distribution_factory = UniformGeoDistributionFactory(country_bounds)

        if asns_dataset is None:
            asns_dataset = load_asns()

        asn_distribution_factory = UniformDistributionFactory(asns_dataset, UniformDiscreteDistribution(asns_dataset.values()))

        if countries_dataset is None:
            countries_dataset = load_countries()
        country_distribution_factory = UniformDistributionFactory(countries_dataset, UniformDiscreteDistribution(countries_dataset.values()))

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

    def __init__(self, network_capacity_estimator=None, computational_capacity_estimator=None, storage_capacity_estimator=None,
                 coordinates_dataset=None, asns_dataset=None, countries_dataset=None):
        if network_capacity_estimator is None:
            network_capacity_estimator = FixedNetworkCapacityEstimator(0, 0)
        if computational_capacity_estimator is None:
            computational_capacity_estimator = FixedComputationCapacityEstimator(0, 0)
        if storage_capacity_estimator is None:
            storage_capacity_estimator = FixedStorageCapacityEstimator(0)

        country_bounds = load_country_bounds()
        if coordinates_dataset is None:
            coordinates_dataset = load_coordinates()
        if asns_dataset is None:
            asns_dataset = load_asns()
        if countries_dataset is None:
            countries_dataset = load_countries()

        coordinates_distribution_factory = MultinomialGeoDistributionFactory(coordinates_dataset, country_bounds)
        asn_distribution_factory = MultinomialDistributionFactory(asns_dataset, MultinomialDiscreteDistribution(asns_dataset.values(), asns_dataset.probabilities()))
        country_distribution_factory = MultinomialDistributionFactory(countries_dataset, MultinomialDiscreteDistribution(countries_dataset.values(), countries_dataset.probabilities()))

        super().__init__(network_capacity_estimator, computational_capacity_estimator, storage_capacity_estimator,
                         coordinates_distribution_factory, asn_distribution_factory, country_distribution_factory)

