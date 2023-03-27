from ..types import NetworkSpecification
from ..types import MetaNode
from ..types import LatencyMatrix

from .node.base import NodeGenerator


class Generator:
    """Default generator.

This is the default generator. It is used when no other generator is specified.

Parameters:å
------------
node_generator : NodeGenerator
    The node generator to use.
latency_generator : LatencyGenerator
    The latency generator to use.


----------------
Method:
----------------

generate(self, network_spec)
    Generate a network of nodes and edges.

    Parameters
    ----------
    network_spec : NetworkSpecification

    Returns
    -------
    nodes : list
        A list of nodes.
    latencyMatrix : LatencyMatrix

    """

    def __init__(self, node_generator, latency_generator):
        self.node_generator = node_generator
        self.latency_generator = latency_generator

    def generate(self, network_spec):
        """Generate a network of nodes and edges. """
        nodes = self.node_generator.generate(network_spec)
        latency_matrix = self.latency_generator.generate(nodes, network_spec)
        return nodes, latency_matrix
