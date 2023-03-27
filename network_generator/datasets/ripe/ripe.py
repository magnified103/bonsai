import csv


def load_Ripe_data():
    """Load the bandwidths for the M-Lab dataset.
    :return: Bandwidths.
    :rtype: dict
    """
    bandwidths = {}
    with open("ripe_data.csv", "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            bandwidths[row["node"]] = float(row["bandwidth"])
    return bandwidths
