from bonsai.types import Dataset


class AsnDataset(Dataset):
    """ASN dataset.

    Expected data format:
    {
        'country1': [ {
                'count': 10,
                'asn': 1234
                },
                ...
            ],
        'country2': [ {
                'count': 10,
                'asn': 1234
                },
                ...
            ],
        ...
    }
    """

    def properties(self):
        return ['countries', 'asns', 'probability']

    def get_property(self, property_name):
        if property_name == 'countries':
            return list(self.data.keys())
        elif property_name == 'asns':
            return [v['asn'] for v in self.data.values()]
        elif property_name == 'probability':
            return self.probabilities()
        else:
            raise ValueError('Invalid property name: {}'.format(property_name))

    def probabilities(self, by=None):
        if by is None:
            by = 'all'
        if by not in self.computed_probabilities:
            self.compute_probability(by)
        return self.computed_probabilities[by]

    def values(self, by=None):
        if by is None:
            return [v['asn'] for vv in self.data.values() for v in vv]
        elif by in self.get_property('countries'):
            return [v['asn'] for v in self.data[by]]
        else:
            raise ValueError('Invalid property: {}'.format(by))

    def compute_probability(self, by=None):
        if by is None or by == 'all':
            by = 'all'
            values = [v['count'] for vv in self.data.values() for v in vv]
            probs = [v / sum(values) for v in values]

        elif by in self.get_property('countries'):
            values = [v['count'] for v in self.data[by]]
            probs = [v / sum(values) for v in values]
        else:
            raise ValueError('Invalid property: {}'.format(by))

        self.computed_probabilities[by] = probs

    def __str__(self):
        return self.data.to_string()

    def __repr__(self):
        return self.data.to_string()


class CountryDataset(Dataset):
    """
    Country dataset.

    Expected data format:
    {
        'continent1': [
            {
            'country': 'country1',
            'count': 10
            },
            ...
        ],
        'continent2': [
            {
            'country': 'country1',
            'count': 10
            },
            ...
        ],
        ...
    }
    """

    def properties(self):
        return ['continents', 'countries', 'probability']

    def get_property(self, property_name):
        if property_name == 'continents':
            return list(self.data.keys())
        elif property_name == 'countries':
            return [v['country'] for v in self.data.values()]
        elif property_name == 'probability':
            return self.probabilities()
        else:
            raise ValueError('Invalid property name: {}'.format(property_name))

    def probabilities(self, by=None):
        if by is None:
            by = 'all'
        if by not in self.computed_probabilities:
            self.compute_probability(by)
        return self.computed_probabilities[by]

    def values(self, by=None):
        if by is None:
            return [v['country'] for vv in self.data.values() for v in vv]
        elif by in self.get_property('continents'):
            return [v['country'] for v in self.data[by]]
        else:
            raise ValueError('Invalid property: {}'.format(by))

    def compute_probability(self, by):
        if by is None or by == 'all':
            by = 'all'
            values = [v['count'] for vv in self.data.values() for v in vv]
            probs = [v / sum(values) for v in values]

        elif by in self.get_property('continents'):
            values = [v['count'] for v in self.data[by]]
            probs = [v / sum(values) for v in values]
        else:
            raise ValueError('Invalid property: {}'.format(by))

        self.computed_probabilities[by] = probs


class CoordinatesDataset(Dataset):
    """
    Coordinates dataset.

    Expected data format:
    {
        'country1': [
            {
            'latitude': 0.0,
            'longitude': 0.0
            'count': 10
            },
            ...
        },
        'country2': [
            {
            'latitude': 0.0,
            'longitude': 0.0
            'count': 10
            },
            ...
        ],
        ...
    }
    """

    def properties(self):
        return ['countries', 'coordinates', 'probability']

    def get_property(self, property_name):
        if property_name == 'countries':
            return list(self.data.keys())
        elif property_name == 'coordinates':
            return [(v['latitude'], v['longitude']) for vv in self.data.values() for v in vv]
        elif property_name == 'probability':
            return self.probabilities()
        else:
            raise ValueError('Invalid property name: {}'.format(property_name))

    def probabilities(self, by=None):
        if by is None:
            by = 'all'
        if by not in self.computed_probabilities:
            self.compute_probability(by)
        return self.computed_probabilities[by]

    def values(self, by=None):
        if by is None:
            return [(v['lat'], v['lon']) for vv in self.data.values() for v in vv]
        elif by in self.get_property('countries'):
            return [(v['lat'], v['lon']) for v in self.data[by]]
        else:
            raise ValueError('Invalid property: {}'.format(by))

    def compute_probability(self, by):
        if by is None or by == 'all':
            by = 'all'
            values = [v['count'] for vv in self.data.values() for v in vv]
            probs = [v / sum(values) for v in values]

        elif by in self.get_property('countries'):
            values = [v['count'] for v in self.data[by]]
            probs = [v / sum(values) for v in values]
        else:
            raise ValueError('Invalid property: {}'.format(by))

        self.computed_probabilities[by] = probs
