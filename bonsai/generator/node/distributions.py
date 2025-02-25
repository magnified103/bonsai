import numpy as np
from pyproj import CRS, Transformer
from shapely import Point


class DistributionFactory:
    """ Distribution factory.

    Create distribution objects.

    Params
    ------
    base_distribution : Distribution
        Base distribution.

    """

    def __init__(self, base_distribution):
        self.distributions = {}
        self.base_distribution = base_distribution

    def next(self, key, how_many=None):
        """ Generate the next value.

        :param key: Key.
        :type key: any

        :param how_many: How many values to generate.
        :type how_many: int | None

        :return: Next values.
        :rtype: any
        """
        if key is None and self.base_distribution is not None:
            return self.base_distribution.next(how_many)
        elif key is None and self.base_distribution is None:
            raise ValueError("No base distribution provided, can't generate value for None key.")

        if key not in self.distributions:
            self.distributions[key] = self.create(key)
        return self.distributions[key].next(how_many)

    def create(self, key):
        """ Create a distribution.

        :param key: Key.
        :type key: any

        :return: Distribution.
        :rtype: Distribution
        """
        raise NotImplementedError("DistributionFactory.create() must be implemented in a subclass.")


class MultinomialDistributionFactory(DistributionFactory):
    """ Multinomial distribution factory.

    Create multinomial distributions.

    Params
    ------
    probabilities: dict
        Probabilities.
    values: dict
        Values.
    base_distribution: Distribution | None
        Base distribution.
    """

    def __init__(self, dataset, base_distribution=None):
        """ Initialize the factory.

        :param dataset: Dataset.
        :type dataset: Dataset

        :param base_distribution: Base distribution.
        :type base_distribution: Distribution | None
        """
        super().__init__(base_distribution)
        self.dataset = dataset

    def create(self, key):
        """ Create a distribution.

        :param key: Key.
        :type key: any

        :return: Distribution.
        :rtype: Distribution
        """
        probs = self.dataset.probabilities(key)
        values = self.dataset.values(key)
        return MultinomialDiscreteDistribution(values, probs)

    def __repr__(self):
        return "MultinomialDistributionFactory(dataset={})".format(self.dataset)

    def __str__(self):
        return self.__repr__()


class UniformDistributionFactory(DistributionFactory):
    """ Uniform distribution factory.

    Create uniform distributions.

    Params
    ------
    values: dict
        Values.
    base_distribution: Distribution | None
        Base distribution.

    """

    def __init__(self, dataset, base_distribution=None):
        """ Initialize the factory.

        :param dataset: Dataset.
        :type dataset: Dataset

        :param base_distribution: Base distribution.
        :type base_distribution: Distribution | None
        """
        super().__init__(base_distribution)
        self.dataset = dataset

    def create(self, key):
        """ Create a distribution.

        :param key: Key.
        :type key: any

        :return: Distribution.
        :rtype: Distribution
        """
        return UniformDiscreteDistribution(self.dataset.values(key))

    def __repr__(self):
        return "UniformDistributionFactory(dataset={})".format(self.dataset)

    def __str__(self):
        return self.__repr__()


class GeoDistributionFactory(DistributionFactory):
    """ GeoDistribution factory.

    Create distribution objects.

    Params
    ------
    polygons: dict
        Polygons.

    """

    def __init__(self, country_bounds):
        """
        Initialize the factory.

        :param country_bounds: Country bounds.
        :type country_bounds: geopandas.GeoDataFrame

        """
        super().__init__(None)
        self.country_bounds = country_bounds

    def create(self, key):
        """ Create a distribution.

        :param key: Key.
        :type key: any

        :return: Distribution.
        :rtype: GeoDistribution
        """
        raise NotImplementedError("DistributionFactory.create() must be implemented in a subclass.")


class UniformGeoDistributionFactory(GeoDistributionFactory):
    """ Uniform distribution factory.

    Create uniform distributions.
    """

    def create(self, key):
        """ Create a distribution.

        :param key: Key.
        :type key: any

        :return: Distribution.
        :rtype: GeoDistribution
        """
        return UniformGeoDistribution(self.country_bounds.loc[key].geometry)

    def __repr__(self):
        return "UniformDistributionFactory(country_bounds={})".format(self.country_bounds)

    def __str__(self):
        return self.__repr__()


class MultinomialGeoDistributionFactory(GeoDistributionFactory):
    """ Multinomial distribution factory.

    Create multinomial distributions.

    Params
    ------
    dataset: Dataset
        Dataset.
    polygons: dict
        Polygons.

    base_distribution: Distribution | None
        Base distribution.
    """

    def __init__(self, dataset, country_bounds, drift=50):
        """ Initialize the factory.

        :param dataset: Dataset.
        :type dataset: Dataset

        :param country_bounds: Country bounds.
        :type country_bounds: geopandas.GeoDataFrame

        :param drift: Drift.
        :type drift: int

        """
        super().__init__(country_bounds)
        self.dataset = dataset
        self.drift = drift

    def create(self, key):
        """ Create a distribution.

        :param key: Key.
        :type key: any

        :return: Distribution.
        :rtype: Distribution
        """
        probs = self.dataset.probabilities(key)
        values = self.dataset.values(key)
        dist = MultinomialDiscreteDistribution([i for i, _ in enumerate(values)], probs)
        return BiasedGeoDistribution(self.country_bounds.loc[key].geometry, values, dist, self.drift)

    def __repr__(self):
        return "MultinomialGeoDistributionFactory(dataset={}, country_bounds={})".format(self.dataset,
                                                                                         self.country_bounds)

    def __str__(self):
        return self.__repr__()


class Distribution(object):
    """Base class for distributions.

    A distribution is responsible for generating values from a distribution.

    Parameters
    ----------

    Methods
    -------
    next()
        Generate a value.

    """

    def __init__(self):
        pass

    def next(self, how_many=None):
        """Generate a value.

        :param how_many:
        :type how_many: int | None


        :return: Value.
        :rtype: Any

        """
        raise NotImplementedError("Distribution.next() must be implemented in a subclass.")


class MultinomialDiscreteDistribution(Distribution):
    """Multinomial discrete distribution.

    Generate values with the provided multinomial discrete distribution.
    """

    def __init__(self, values, probabilities):
        """Initialize the distribution.

        :param values: Values.
        :type values: list

        :param probabilities: Probabilities.
        :type probabilities: list | None
        """
        super().__init__()
        self.values = values
        self.probabilities = probabilities


    def next(self, how_many=None):
        """Generate the next value.

        :param how_many: How many values to generate.
        :type how_many: int | None

        :return: Next value.
        :rtype: Any
        """
        return np.random.choice(self.values, p=self.probabilities, size=how_many)

    def __repr__(self):
        return "MultinomialDistribution(probabilities={}, values={})".format(self.probabilities, self.values)

    def __str__(self):
        return "MultinomialDistribution(probabilities={}, values={})".format(self.probabilities, self.values)


class UniformDiscreteDistribution(MultinomialDiscreteDistribution):
    """Uniform discrete distribution.

    Generate values with equal probability.
    """

    def __init__(self, values):
        """Initialize the distribution.

        :param values: Values.
        :type values: list
        """
        super().__init__(values, None)

    def __repr__(self):
        return "UniformDistribution(values={})".format(self.values)

    def __str__(self):
        return "UniformDistribution(values={})".format(self.values)


class GeoDistribution(Distribution):
    """Geographic distribution.

    Generate geographic coordinates from a distribution.
    """

    def __init__(self, polygon):
        """Initialize the distribution.

        :param polygon: Polygon.
        :type polygon: shapely.geometry.Polygon

        """
        super().__init__()
        self.polygon = polygon

    def next(self, how_many=None):
        """Generate the next value.

        :param how_many: How many values to generate.
        :type how_many: int | None

        :return: Next value.
        :rtype: any
        """
        raise NotImplementedError("GeoDistribution.next() must be implemented in a subclass.")


class UniformGeoDistribution(GeoDistribution):
    """Uniform geographic distribution.

    Generate geographic coordinates with equal probability.
    """
    def next(self, how_many=None):
        """Generate the next value.

        :return: Next value.
        :rtype: any
        """
        import geopandas as gpd
        minx, miny, maxx, maxy = self.polygon.bounds
        if how_many is not None:
            points = []
            while len(points) < how_many:
                to_gen = how_many - len(points)
                xs = np.random.uniform(minx, maxx, to_gen)
                ys = np.random.uniform(miny, maxy, to_gen)
                pnts = [Point(x, y) for x, y in zip(xs, ys)]
                gdf_points = gpd.GeoDataFrame(geometry=pnts)
                gdf_polygon = gpd.GeoDataFrame(geometry=[self.polygon])
                gdf_points = gpd.sjoin(gdf_points, gdf_polygon, how="inner", predicate="within")
                points.extend([v.coords[0] for v in gdf_points.geometry.values])
            return points
        else:
            while True:
                pnt = Point(np.random.uniform(minx, maxx), np.random.uniform(miny, maxy))
                if self.polygon.contains(pnt):
                    return pnt.coords[0]

    def __repr__(self):
        return "UniformGeoDistribution(polygon={})".format(self.polygon)

    def __str__(self):
        return "UniformGeoDistribution(polygon={})".format(self.polygon)


class BiasedGeoDistribution(GeoDistribution):
    """Biased geographic distribution.

    Generate geographic coordinates with a bias towards the center of the polygon.
    """

    def __init__(self, polygon, values, bias, drift=50):
        """Initialize the distribution.

        :param polygon: Polygon.
        :type polygon: shapely.geometry.Polygon

        :param values: Values.
        :type values: list

        :param bias: Bias.
        :type bias: MultinomialDiscreteDistribution

        :param drift: Drift.
        :type drift: int
        """
        super().__init__(polygon)
        self.bias = bias
        self.drift = drift
        self.values = values

    def next(self, how_many=None):
        """Generate the next value.

        :return: Next value.
        :rtype: any
        """
        if how_many is not None:
            points = []
            for _ in range(how_many):
                while True:
                    pnt = self.gen_point()
                    if self.polygon.contains(pnt):
                        points.append(pnt.coords[0])
                        break
            return points
        else:
            while True:
                pnt = self.gen_point()
                if self.polygon.contains(pnt):
                    return pnt.coords[0]

    def gen_point(self):
        idx = self.bias.next()
        x, y = self.values[idx]
        if self.drift == 0:
            return Point(y, x)
        else:
            buffer = self.geodesic_point_buffer(x, y, self.drift)
            minx, miny, maxx, maxy = buffer.bounds
            pnt = Point(np.random.uniform(minx, maxx), np.random.uniform(miny, maxy))
            return pnt

    def __repr__(self):
        return "BiasedGeoDistribution(polygon={})".format(self.polygon)

    def __str__(self):
        return "BiasedGeoDistribution(polygon={})".format(self.polygon)

    def geodesic_point_buffer(self, lat, lon, km):
        """
        Create a buffer around a point with a given radius in kilometers.
        Taken from https://gis.stackexchange.com/questions/289044/creating-buffer-circle-x-kilometers-from-point-using-python

        :param lat: Latitude.
        :type lat: float

        :param lon: Longitude.
        :type lon: float

        :param km: Kilometers.
        :type km: float

        :return: Buffer.
        :rtype: shapely.geometry.Polygon

        """
        # Azimuthal equidistant projection
        from shapely.ops import transform
        aeqd_proj = CRS.from_proj4(
            f"+proj=aeqd +lat_0={lat} +lon_0={lon} +x_0=0 +y_0=0")
        tfmr = Transformer.from_proj(aeqd_proj, aeqd_proj.geodetic_crs)
        buf = Point(0, 0).buffer(km * 1000)  # distance in metres
        return transform(tfmr.transform, buf)
