import csv

import pkg_resources

from bonsai.types import Dataset


class PingProperty:
    """ Ping property.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std


class RipeDataSet(Dataset):
    """ Ripe dataset.

    :param pings: Ripe pings dataset.
    :type pings: list[dict]
    :param granularity: Granularity of the dataset.
    :type granularity: str ('country' | 'country:asn')
    """

    def __init__(self, granularity='country'):
        pings = load_Ripe_data(granularity)
        self.ping_dataset = {}
        self.granularity = granularity
        for ping in pings:
            src = ping['src']
            dst = ping['dst']
            if src < dst:
                src_dst = "{}-{}".format(src, dst)
            else:
                src_dst = "{}-{}".format(dst, src)
            self.ping_dataset[src_dst] = PingProperty(float(ping['mean']), float(ping['std']))

    def get(self, src, dst):
        """ Get the mean and std latency between two nodes.

        :param src: The source node.
        :type src: NetworkNode
        :param dst: The destination node.
        :type dst: NetworkNode
        """
        if self.granularity == 'country':
            src = src.get_country()
            dst = dst.get_country()
            if dst < src:
                src, dst = dst, src
            src_dst = "{}-{}".format(src, dst)
        elif self.granularity == 'country:asn':
            src = "{}:{}".format(src.get_country(), src.get_asn())
            dst = "{}:{}".format(dst.get_country(), dst.get_asn())
            if dst < src:
                src, dst = dst, src
            src_dst = "{}-{}".format(src, dst)

        return self.ping_dataset[src_dst]


def load_Ripe_data(granularity):
    """Load the pings for the RipeAtlas dataset.
    :return: Pings.
    :rtype: list[dict]
    """
    if granularity == 'country':
        file = pkg_resources.resource_filename(__name__, "data/ripe-atlas-country.csv")
    elif granularity == 'country:asn':
        file = pkg_resources.resource_filename(__name__, "data/ripe-atlas-country-asn.csv")
    else:
        raise ValueError("Granularity not supported: {}".format(granularity))

    pings = []
    with open(file, "r") as f:
        reader = csv.DictReader(f, fieldnames=['src', 'dst', 'mean', 'std'])
        next(reader)
        for row in reader:
            pings.append(row)

    return pings
