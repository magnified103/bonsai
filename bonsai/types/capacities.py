class Capacity:
    """ Capacity class
        Defines a capacity of a node in the network.

        """

    def __init__(self):
        pass

    def __str__(self):
        return ""

    def __repr__(self):
        return ""

    def __eq__(self, other):
        return True

    def __hash__(self):
        return hash(0)

    def __lt__(self, other):
        return True

    def __le__(self, other):
        return True

    def __gt__(self, other):
        return True

    def __ge__(self, other):
        return True

    def __ne__(self, other):
        return True

    def __add__(self, other):
        return self


class NetworkCapacity(Capacity):
    """ NetworkCapacity class
    Defines a network capacity of a node in the network.

    Attributes:
    download (int): The download capacity in Mbps.
    upload (int): The upload capacity in Mbps.
    """

    def __init__(self, download, upload):
        self.download = download
        self.upload = upload

    def __str__(self):
        return "Download: {}, Upload: {}".format(self.download, self.upload)

    def __repr__(self):
        return "Download: {}, Upload: {}".format(self.download, self.upload)

    def __eq__(self, other):
        return self.download == other.download and self.upload == other.upload

    def __hash__(self):
        return hash((self.download, self.upload))

    def __lt__(self, other):
        return self.download < other.download and self.upload < other.upload

    def __le__(self, other):
        return self.download <= other.download and self.upload <= other.upload

    def __gt__(self, other):
        return self.download > other.download and self.upload > other.upload

    def __ge__(self, other):
        return self.download >= other.download and self.upload >= other.upload

    def __ne__(self, other):
        return self.download != other.download and self.upload != other.upload

    def __add__(self, other):
        return NetworkCapacity(self.download + other.download, self.upload + other.upload)


class ComputationCapacity(Capacity):
    """ ComputationCapacity class
    Defines a computation capacity of a node in the network.

    Attributes:
    cpu (int): The number of CPU cores.
    memory (int): The amount of memory in MB.

    """

    def __init__(self, cpu, memory):
        self.cpu = cpu
        self.memory = memory

    def __str__(self):
        return "CPU: {}, Memory: {}".format(self.cpu, self.memory)

    def __repr__(self):
        return "CPU: {}, Memory: {}".format(self.cpu, self.memory)

    def __eq__(self, other):
        return self.cpu == other.cpu and self.memory == other.memory

    def __hash__(self):
        return hash((self.cpu, self.memory))

    def __lt__(self, other):
        return self.cpu < other.cpu and self.memory < other.memory

    def __le__(self, other):
        return self.cpu <= other.cpu and self.memory <= other.memory

    def __gt__(self, other):
        return self.cpu > other.cpu and self.memory > other.memory

    def __ge__(self, other):
        return self.cpu >= other.cpu and self.memory >= other.memory

    def __ne__(self, other):
        return self.cpu != other.cpu and self.memory != other.memory

    def __add__(self, other):
        return ComputationCapacity(self.cpu + other.cpu, self.memory + other.memory)


class StorageCapacity(Capacity):
    """ StorageCapacity class
    Defines a storage capacity.

    Attributes:
    storage (int): The storage capacity in bytes.

    """

    def __init__(self, storage):
        self.storage = storage

    def __str__(self):
        return "Storage: {}".format(self.storage)

    def __repr__(self):
        return "Storage: {}".format(self.storage)

    def __eq__(self, other):
        return self.storage == other.storage

    def __hash__(self):
        return hash(self.storage)

    def __lt__(self, other):
        return self.storage < other.storage

    def __le__(self, other):
        return self.storage <= other.storage

    def __gt__(self, other):
        return self.storage > other.storage

    def __ge__(self, other):
        return self.storage >= other.storage

    def __ne__(self, other):
        return self.storage != other.storage

    def __add__(self, other):
        return StorageCapacity(self.storage + other.storage)
