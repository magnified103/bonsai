from .base import BaseExporter


class DefaultExporter(BaseExporter):
    """Default exporter. """

    def export(self, nodes, edges, latencies, *args):
        """Export a network of nodes and edges. """
        return nodes, edges, latencies

    def config(self):
        """Return the configuration of the exporter.

        :return: Configuration.
        :rtype: dict
        """
        return {
            "name": "DefaultExporter",
            "description": "Default exporter",
            "parameters": {}
        }
