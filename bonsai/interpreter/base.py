from bonsai.types.network_specs import NetworkSpecification

class BaseInterpreter:

    def __init__(self):
        pass

    def interpret(self, network_specification):
        """Interpret a network specification.

        :param network_specification: Network specification file.
        :type network_specification: dict

        :return: Interpreted network specification.
        :rtype: NetworkSpecification
        """
        raise NotImplementedError("Interpreter.interpret() must be implemented in a subclass.")

    def config(self):
        """Return the configuration of the interpreter.

        :return: Configuration of the interpreter.
        :rtype: dict
        """
        raise NotImplementedError("Interpreter.config() must be implemented in a subclass.")