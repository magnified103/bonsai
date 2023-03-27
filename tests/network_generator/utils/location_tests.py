import unittest
import matplotlib.pyplot as plt

from network_generator.utils import *



class UtilsLocationTestCase(unittest.TestCase):
    def setUp(self):
        self.bounds = load_country_bounds('ne_10m_admin_0_countries/ne_10m_admin_0_countries.shp')
        self.country_asns = load_country_asn('country_asn.json')

    def test_load_country_bounds(self):
        self.assertLess(len(self.bounds), 249)

    def test_get_continent(self):
        self.assertEqual(get_continent('US'), 'NA')
        self.assertEqual(get_continent('FR'), 'EU')
        self.assertEqual(get_continent('CN'), 'AS')
        self.assertEqual(get_continent('AU'), 'OC')
        self.assertEqual(get_continent('BR'), 'SA')
        self.assertEqual(get_continent('ZA'), 'AF')

    def test_get_countries(self):
        self.assertLess(len(get_countries()), 250)

        self.assertGreater((len(get_countries('NA'))), 3)
        self.assertGreater((len(get_countries('SA'))), 6)
        self.assertGreater((len(get_countries('EU'))), 20)
        self.assertGreater((len(get_countries('AS'))), 5)
        self.assertGreater((len(get_countries('OC'))), 2)
        self.assertGreater((len(get_countries('AF'))), 6)

        for country in get_countries():
            if get_continent(country) is None:
                print(country)
            else:
                self.assertIn(country, get_countries(get_continent(country)))

    def test_get_random_lat_long(self):
        self.bounds.plot()
        for i in range(100):
            long, lat = get_random_long_lat('US')
            plt.plot(long, lat, 'bo', markersize=1)
        plt.show()
        # long, lat = get_random_long_lat('US')
        # plt.plot(long, lat, 'ro')
        # long, lat = get_random_long_lat('FR')
        # plt.plot(long, lat, 'yo')
        # long, lat = get_random_long_lat('CN')
        # plt.plot(long, lat, 'go')
        # plt.show()

    def test_get_asns(self):
        self.assertGreater(len(get_asns('US')), 0)
        self.assertGreater(len(get_asns('FR')), 0)
        self.assertGreater(len(get_asns('CN')), 0)
        self.assertGreater(len(get_asns('AU')), 0)
        self.assertGreater(len(get_asns('BR')), 0)
        self.assertGreater(len(get_asns('ZA')), 0)

    def is_more_popular(self, country, asn1, asn2):
        asn1_pop = -1
        asn2_pop = -1
        for asn in self.country_asns[country]:
            if asn['asn'] == asn1:
               asn1_pop = asn['count']
            if asn['asn'] == asn2:
                asn2_pop = asn['count']
        return asn1_pop > asn2_pop

    def test_get_asns_by_popularity(self):
        self.assertTrue(self.is_more_popular('US', get_asns_by_popularity('US')[0], get_asns_by_popularity('US')[1]))
        self.assertTrue(self.is_more_popular('FR', get_asns_by_popularity('FR')[0], get_asns_by_popularity('FR')[1]))
        self.assertTrue(self.is_more_popular('CN', get_asns_by_popularity('CN')[0], get_asns_by_popularity('CN')[1]))
        self.assertTrue(self.is_more_popular('AU', get_asns_by_popularity('AU')[0], get_asns_by_popularity('AU')[1]))
        self.assertTrue(self.is_more_popular('BR', get_asns_by_popularity('BR')[0], get_asns_by_popularity('BR')[1]))
        self.assertTrue(self.is_more_popular('ZA', get_asns_by_popularity('ZA')[0], get_asns_by_popularity('ZA')[1]))



if __name__ == '__main__':
    unittest.main()
