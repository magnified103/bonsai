from .generator.latency.base import LatencyGenerator
from .generator.latency.estimators import RandomLatencyEstimator
from .generator.node.capacities.computation import FixedComputationCapacityEstimator
from .generator.node.capacities.network import FixedNetworkCapacityEstimator
from .generator.node.capacities.storage import FixedStorageCapacityEstimator
from .generator.node.generators import UniformNodeGenerator
from .parser import YAMLParser
from .generator import Generator
from .exporter import DefaultExporter
from .interpreter import DefaultInterpreter


class NetworkGenerator:
    """
    Instantiate a NetworkGenerator object to generate a network of nodes and edges.


    """

    def __init__(self, parser=YAMLParser(),
                 interpreter=DefaultInterpreter(),
                 generator=Generator(
                     UniformNodeGenerator(FixedNetworkCapacityEstimator(0, 0),
                                          FixedComputationCapacityEstimator(0, 0),
                                          FixedStorageCapacityEstimator(0)),
                     LatencyGenerator(RandomLatencyEstimator(10, 100))),
                 exporter=DefaultExporter()):
        self.parser = parser
        self.interpreter = interpreter
        self.generator = generator
        self.exporter = exporter

    def generate(self, network_specification):
        """
        Generate a network model for the given specification

        :param network_specification: Network specification file.
        :type network_specification: str

        :returns nodes: the set of nodes generated
        :returns latencies: the latencies between the nodes
        :type (list[Node], LatencyMatrix(len(nodes),len(nodes]))
        """
        network_spec_dict = self.parser.parse(network_specification)
        network_spec = self.interpreter.interpret(network_spec_dict)
        return self.generator.generate(network_spec)

    def export(self, nodes, edges, *args):
        return self.exporter.export(nodes, edges, *args)
