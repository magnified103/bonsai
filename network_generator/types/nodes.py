from network_generator.types import NetworkCapacity, ComputationCapacity, StorageCapacity


class BaseNode:
    """ BaseNode class
    Defines a node in the network.

    Attributes:
    id (int): The id of the node.


    """

    def __init__(self, node_id):
        self.id = node_id

    def __str__(self):
        return "Node id: {}".format(self.id)

    def __repr__(self):
        return "Node id: {}".format(self.id)

    def __eq__(self, other):
        return self.id == other.id

    def __hash__(self):
        return hash(self.id)

    def __lt__(self, other):
        return self.id < other.id

    def __le__(self, other):
        return self.id <= other.id

    def __gt__(self, other):
        return self.id > other.id

    def __ge__(self, other):
        return self.id >= other.id

    def __ne__(self, other):
        return self.id != other.id


class MetaNode(BaseNode):

    """ MetaNode class
    Defines a node in the network.

    Attributes:
    id (int): The id of the node.
    country_code (str): The country code of the node.
    latitude (float): The latitude of the node.
    longitude (float): The longitude of the node.
    asn (int): The ASN of the node.

    """

    def __init__(self, node_id, country_code, latitude, longitude, asn):
        super().__init__(node_id)
        self.country_code = country_code
        self.latitude = latitude
        self.longitude = longitude
        self.asn = asn




class NetworkNode:
    """ NetworkNode class
    Defines a node in for emulated network.

    Attributes:
    meta_node (MetaNode): The meta node of the network node, contains the node's meta information.
    network_capacity (NetworkCapacity): The network capacity of the network node.
    computation_capacity (ComputationCapacity): The computation capacity of the network node.
    storage_capacity (StorageCapacity): The storage capacity of the network node.


    """

    def __init__(self, meta_node, network_capacity, computation_capacity, storage_capacity):
        self.meta_node: MetaNode = meta_node
        self.network_capacity: NetworkCapacity = network_capacity
        self.computation_capacity: ComputationCapacity = computation_capacity
        self.storage_capacity: StorageCapacity = storage_capacity

    def get_id(self):
        return self.meta_node.id

    def __eq__(self, other):
        return self.meta_node.__eq__(other.meta_node)

    def __hash__(self):
        return self.meta_node.__hash__()

    def __lt__(self, other):
        return self.meta_node.__lt__(other.meta_node)

    def __le__(self, other):
        return self.meta_node.__le__(other.meta_node)

    def __gt__(self, other):
        return self.meta_node.__gt__(other.meta_node)

    def __ge__(self, other):
        return self.meta_node.__ge__(other.meta_node)

    def __ne__(self, other):
        return self.meta_node.__ne__(other.meta_node)
