
class BaseParser(object):
    """Base class for all parsers. """

    def __init__(self, *args, **kwargs):
        pass

    def parse(self, network_specification):
        """Parse network specification.

        :param network_specification: Network specification file.
        :type network_specification: str

        :return: Parsed network specification.
        :rtype: dict
        """
        raise NotImplementedError("Parser.parse_args() must be implemented in a subclass.")

    def config(self):
        """Return the configuration of the parser.

        :return: Configuration of the parser.
        :rtype: dict
        """
        raise NotImplementedError("Parser.config() must be implemented in a subclass.")