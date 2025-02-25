import json

import pandas as pd
import pkg_resources

from bonsai.datasets import AsnDataset, CoordinatesDataset, CountryDataset


def load_asns(file=None):
    """Load ASNs from a file.

:param file: ASN file.
:type file: str | None

:return: ASNs.
:rtype: Dataset
    """
    if file is None:
        file = pkg_resources.resource_filename(__name__, 'data/asns.json')
    with open(file, 'r') as f:
        country_asn = json.load(f)
    return AsnDataset(file, country_asn)


def load_coordinates(file=None):
    """ Load coordinates from a file.

:param file: File .
:type file: str | None

:return: Coordinates.
:rtype: Dataset
    """
    if file is None:
        file = pkg_resources.resource_filename(__name__, 'data/coords.json')
    with open(file, 'r') as f:
        coordinates = json.load(f)
    return CoordinatesDataset(file, coordinates)


def load_countries(file=None):
    """Load countries from a file.

:param file: File.
:type file: str | None

:return: Countries.
:rtype: Dataset
    """
    countries = {}
    if file is None:
        file = pkg_resources.resource_filename(__name__, 'data/countries.json')
    with open(file, 'r') as f:
        countries_json = json.load(f)
    return CountryDataset(file, countries_json)


def load_country_bounds(file=None):
    """Load country bounds from a file.

    :param file: File.
    :type file: str

    :return: Country bounds.
    """

    import geopandas as gpd
    if file is None:
        file = pkg_resources.resource_filename(__name__, 'ne_10m_admin_0_countries/ne_10m_admin_0_countries.shp')
    df = gpd.read_file(file)
    df.set_index('NAME', inplace=True)
    df.at['Somaliland', 'ISO_A2'] = 'SO'
    df.at['Kosovo', 'ISO_A2'] = 'XK'
    df.at['France', 'ISO_A2'] = 'FR'
    df.at['Norway', 'ISO_A2'] = 'NO'
    df.at['Taiwan', 'ISO_A2'] = 'TW'

    df = df[['ISO_A2', 'geometry']].replace({'ISO_A2': {'-99': pd.NA}}).dropna()
    return df.set_index('ISO_A2')
