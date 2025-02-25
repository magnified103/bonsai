import csv
import ipaddress

import numpy as np
from tqdm import tqdm

from .base import BaseExporter
from ..types import LatencyMatrix, NetworkNode


class LinkExporter(BaseExporter):
    """ Simple exporter for TC. """

    def __init__(self):
        super().__init__()

    def export(self, nodes, nodes_x_nodes, latencies, *args):
        """Export a network of nodes and edges. """
        output = args[0]
        verbose = args[1] if len(args) > 1 else False
        if isinstance(nodes_x_nodes[0], NetworkNode):
            nodes_dict = {node: i for i, node in enumerate(nodes)}
        else:
            nodes_dict = {node.get_id(): i for i, node in enumerate(nodes)}
        with open(output, 'w') as f:
            csv_writer = csv.writer(f)
            for i, n in enumerate(nodes_x_nodes) if not verbose else tqdm(enumerate(nodes_x_nodes)):
                node1, node2 = n
                csv_writer.writerow([nodes_dict[node1], nodes_dict[node2], latencies[i]])


    def config(self):
        """Return the configuration of the exporter.

        :return: Configuration.
        :rtype: dict
        """
        return {
            "name": "Simple Link exporter",
            "description": "Simple exporter for link output",
        }
