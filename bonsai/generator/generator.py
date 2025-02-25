from .latency.base import LatencyGenerator
from ..types import NetworkSpecification
from ..types import MetaNode
from ..types import LatencyMatrix

from .node.base import NodeGenerator


class Generator:
    """Default generator.

This is the default generator. It is used when no other generator is specified.

Parameters:
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
        """Initialize the generator.

        :param node_generator: The node generator to use.
        :type node_generator: NodeGenerator

        :param latency_generator: The latency generator to use.
        :type latency_generator: LatencyGenerator
        """
        self.node_generator = node_generator
        self.latency_generator = latency_generator

    def generate(self, network_spec):
        """Generate a network of nodes and edges.

        :param network_spec: Network specification.
        :type network_spec: NetworkSpecification

        :return: List of nodes, List of parings, list of latencies
        :rtype: list, list, list

        """
        nodes = self.node_generator.generate(network_spec)
        return self.latency_generator.generate(nodes)

    def config(self):
        return {
            'node_generator': self.node_generator.config(),
            'latency_generator': self.latency_generator.config()
        }

    def validate(self):
        """
        Validate the generator configuration
        Check if the latency generator can generate the latency matrix for the node generator
        """
        self.latency_generator.validate(self.node_generator)
