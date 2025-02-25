from .exporter import DefaultExporter
from .generator import Generator
from .generator.latency.base import LatencyGenerator
from .generator.node.generators import UniformNodeGenerator
from .interpreter import DefaultInterpreter
from .parser import YAMLParser


class NetworkGenerator:
    """
    Instantiate a NetworkGenerator object to generate a network of nodes and edges.


    """

    def __init__(self, parser=None, interpreter=None, generator=None, exporter=None):
        self.parser = parser if parser is not None else YAMLParser()
        self.interpreter = interpreter if interpreter is not None else DefaultInterpreter()
        self.generator = generator if generator is not None else Generator(
            UniformNodeGenerator(),
            LatencyGenerator())
        self.exporter = exporter if exporter is not None else DefaultExporter()

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
        """
        Export the nodes and edges through the exporter component
        :param nodes: nodes
        :param edges: edges
        :param args: other exporter arguments
        :return: exporter return (str | file | obj)
        """
        return self.exporter.export(nodes, edges, *args)

    def config(self):
        """
        Return the configuration of the network generator
        :return: dict with the network generator config
        """
        return {
            'parser': self.parser.config(),
            'interpreter': self.interpreter.config(),
            'generator': self.generator.config(),
            'exporter': self.exporter.config()
        }


