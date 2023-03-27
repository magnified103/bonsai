import csv
import json

import haversine as haversine
import pkg_resources

import pandas as pd
import pycountry as pycountry
import pycountry_convert as pycountry_convert


country_bounds = None
country_asn = None
hops_graph = None


def __load_country_bounds__():
    global country_bounds
    if country_bounds is None:
        country_bounds = load_country_bounds(
            pkg_resources.resource_filename(__name__, 'ne_10m_admin_0_countries/ne_10m_admin_0_countries.shp'))


def load_country_bounds(file):
    import geopandas as gpd
    df = gpd.read_file(file)
    df.set_index('NAME', inplace=True)
    df.at['Somaliland', 'ISO_A2'] = 'SO'
    df.at['Kosovo', 'ISO_A2'] = 'XK'
    df.at['France', 'ISO_A2'] = 'FR'
    df.at['Norway', 'ISO_A2'] = 'NO'
    df.at['Taiwan', 'ISO_A2'] = 'TW'

    df = df[['ISO_A2', 'geometry']].replace({'ISO_A2': {'-99': pd.NA}}).dropna()
    return df.set_index('ISO_A2')


def __load_country_asn__():
    global country_asn
    if country_asn is None:
        country_asn = load_country_asn(pkg_resources.resource_filename(__name__, 'country_asn.json'))

    ##Need to filter out countries that are not in the graph
    __load_hops_graph__()
    src_country_asn = hops_graph[['source', 'source_asn']].drop_duplicates().rename(
        columns={'source': 'country', 'source_asn': 'asn'})
    dst_country_asn = hops_graph[['destination', 'dest_asn']].drop_duplicates().rename(
        columns={'destination': 'country', 'dest_asn': 'asn'})
    graph_country_asn = pd.merge(src_country_asn, dst_country_asn, how='inner', on=['country', 'asn']).groupby('country')
    for country, asns in country_asn.items():
        if country not in graph_country_asn.groups:
            country_asn[country] = []
            continue
        graph_asns = graph_country_asn.get_group(country)['asn'].unique()
        for asn in asns:
            if asn not in graph_asns:
                country_asn[country].remove(asn)



def load_country_asn(file):
    country_asn = {}
    with open(file, 'r') as f:
        country_asn_json = json.load(f)
    for entry in country_asn_json:
        country = entry['country']
        asns = entry['asns']
        country_asn[country] = asns
    return country_asn


def __load_hops_graph__():
    global hops_graph
    if hops_graph is None:
        hops_graph = load_direct_hops_graph(pkg_resources.resource_filename(__name__, 'graphTraces.csv'))


def load_direct_hops_graph(file):
    df = pd.read_csv(file, keep_default_na=False, na_values=[-1]).dropna()
    df = df[df['count'] > 4]
    return df


def get_continent(country):
    """Get continent of a country.

    :param country: Country.
    :type country: str

    :return: Continent of the country.
    :rtype: str
    """
    try:
        return pycountry_convert.country_alpha2_to_continent_code(country)
    except KeyError:
        return None


def country_in_continent(country, continent):
    """Check if a country is in a continent.

    :param country: Country.
    :type country: str
    :param continent: Continent.
    :type continent: str

    :return: True if the country is in the continent, False otherwise.
    :rtype: bool
    """
    try:
        return pycountry_convert.country_alpha2_to_continent_code(country) == continent
    except KeyError:
        return False


# TODO constraint countries to minimal set of countries

country_set = None


def get_all_countries():
    """Get all countries.

    :return: All countries from the ASNs file.
    :rtype: list
    """
    global country_set
    if country_set is not None:
        return country_set

    __load_country_asn__()
    __load_country_bounds__()
    __load_hops_graph__()

    src_country_asn = hops_graph[['source', 'source_asn']].drop_duplicates().rename(
        columns={'source': 'country', 'source_asn': 'asn'})
    dst_country_asn = hops_graph[['destination', 'dest_asn']].drop_duplicates().rename(
        columns={'destination': 'country', 'dest_asn': 'asn'})
    graph_country_asn = pd.merge(src_country_asn, dst_country_asn, how='inner', on=['country', 'asn'])

    country_set = set(country_asn.keys()).intersection(country_bounds.index).intersection(graph_country_asn['country'])
    return country_set


def get_countries(continent=None):
    """Get countries.

    :param continent: Continent.
    :type continent: str

    :return: Countries.
    :rtype: list
    """
    all = get_all_countries()
    if continent is None:
        return list(all.intersection([c.alpha_2 for c in pycountry.countries]))
    else:
        return list(
            all.intersection([c.alpha_2 for c in pycountry.countries if country_in_continent(c.alpha_2, continent)]))


import numpy as np
from shapely.geometry import Point


## Taken from https://www.matecdev.com/posts/random-points-in-polygon.html
def Random_Points_in_Polygon(polygon, number):
    """ Get random points in a polygon.

:param polygon: Polygon.
:type polygon: shapely.geometry.Polygon
:param number: Number of points.
:type number: int

:return: List of points.
:rtype: list
    """
    points = []
    minx, miny, maxx, maxy = polygon.bounds
    while len(points) < number:
        pnt = Point(np.random.uniform(minx, maxx), np.random.uniform(miny, maxy))
        if polygon.contains(pnt):
            points.append(pnt)
    return points


def get_random_long_lat(country):
    """Get latitude and longitude of a country.

    :param country: Country.
    :type country: str

    :return: Longitude and Latitude of the country.
    :rtype: tuple
    """
    global country_bounds
    if country_bounds is None:
        __load_country_bounds__()
    return Random_Points_in_Polygon(country_bounds.loc[country].geometry, 1)[0].coords[0]


def get_asns(country):
    """Get ASNs of a country.

    :param country: Country.
    :type country: str

    :return: ASNs of the country.
    :rtype: list
    """
    global country_asn
    if country_asn is None:
        __load_country_asn__()

    info = country_asn.get(country, None)
    asns = []
    if info is not None:
        for entry in info:
            asns.append(entry['asn'])
    return asns


def get_asns_by_popularity(country):
    """Get ASNs of a country.

    :param country: Country.
    :type country: str

    :return: ASNs of the country.
    :rtype: list
    """
    global country_asn
    if country_asn is None:
        country_asn = load_country_asn(pkg_resources.resource_filename(__name__, 'country_asn.json'))

    info = country_asn.get(country, None)
    info.sort(key=lambda x: x['count'], reverse=True)
    asns = []
    if info is not None:
        for entry in info:
            asns.append(entry['asn'])
    return asns


def compute_distance(src, dst):
    """
    Compute the haversine distance between two points.
    :param src: tuple of (lat, lon)
    :param dst: tuple of (lat, lon)
    :return: the distance in km
    """
    return haversine.haversine(src, dst)
