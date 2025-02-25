import ipaddress

import numpy as np

from .base import BaseExporter
from ..types import LatencyMatrix


class SimpleMatrixExporter(BaseExporter):
    """ Simple exporter for TC. """

    def __init__(self):
        super().__init__()

    def export(self, nodes, nodes_x_nodes, latencies, *args):
        """Export a network of nodes and edges. """
        latencyMatrix = LatencyMatrix(nodes, nodes_x_nodes, latencies)
        mat = latencyMatrix.get_matrix()
        np.savetxt(args[0], mat, delimiter=" ", fmt="%0.2f")
        return latencyMatrix

    def config(self):
        """Return the configuration of the exporter.

        :return: Configuration.
        :rtype: dict
        """
        return {
            "name": "SimpleMatrxiExporter",
            "description": "Simple exporter for matrix output",
        }
