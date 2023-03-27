import csv

from network_generator.types import Dataset, NetworkCapacity


class MLabDataSet(Dataset):
    """ M-Lab dataset.
    """

    def __init__(self, upload, download):

        self.network_capacities = {}
        for country in upload:
            self.network_capacities[country] = NetworkCapacity(upload[country], download[country])

    def get(self, src):
        """ Get the network capacity of a node.

:param src: The source node.
:type src: str

:return: The network capacity of the node.
:rtype: NetworkCapacity

        """
        return self.network_capacities[src]


def load_mlab_dataset():
    """Load the bandwidths for the M-Lab dataset.
    :return: MLab Dataset.
    :rtype: Dataset
    """
    uploads = {}
    with open("mlab-ndt-uploads.csv", "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            uploads[row["CountryCode"]] = float(row["MeanUpload"])

    downloads = {}
    with open("mlab-ndt-downloads.csv", "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            downloads[row["CountryCode"]] = float(row["MeanDownload"])

    return MLabDataSet(uploads, downloads)
