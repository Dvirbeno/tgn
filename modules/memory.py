import torch
from torch import nn

from collections import defaultdict
from copy import deepcopy


class Memory(nn.Module):

    def __init__(self, n_nodes, memory_dimension, input_dimension, message_dimension=None,
                 device="cpu"):
        super(Memory, self).__init__()
        self.n_nodes = n_nodes
        self.memory_dimension = memory_dimension
        self.input_dimension = input_dimension
        self.message_dimension = message_dimension
        self.device = device

        self.__init_memory__()

    def __init_memory__(self):
        """
        Initializes the memory to all zeros. It should be called at the start of each epoch.
        """
        # Treat memory as parameter so that it is saved and loaded together with the model
        self.memory = nn.Parameter(torch.zeros((self.n_nodes, self.memory_dimension)).to(self.device),
                                   requires_grad=False)
        self.last_update = nn.Parameter(torch.zeros(self.n_nodes).to(self.device),
                                        requires_grad=False)
        self.last_match_id = nn.Parameter(torch.zeros(self.n_nodes).to(self.device),
                                          requires_grad=False)

        self.raw_messages_storage = defaultdict(list)

    def store_raw_messages(self, nodes, node_id_to_raw_message):
        for node in nodes:
            self.raw_messages_storage[node].extend(node_id_to_raw_message[node])

    def get_memory(self, node_ids):
        return self.memory[node_ids, :]

    def set_memory(self, node_ids, values):
        self.memory[node_ids, :] = values

    def get_last_update(self, node_ids):
        return self.last_update[node_ids]

    def backup_memory(self):
        messages_clone = {}
        for k, v in self.raw_messages_storage.items():
            messages_clone[k] = [(x[0].clone(), x[1].clone()) for x in v]

        return self.memory.data.clone(), self.last_update.data.clone(), messages_clone

    def restore_memory(self, memory_backup):
        self.memory.data, self.last_update.data = memory_backup[0].clone(), memory_backup[1].clone()

        self.raw_messages_storage = defaultdict(list)
        for k, v in memory_backup[2].items():
            self.raw_messages_storage[k] = [(x[0].clone(), x[1].clone()) for x in v]

    def detach_memory(self):
        self.memory.detach_()

        # Detach all stored messages
        for k, v in self.raw_messages_storage.items():
            new_node_messages = []
            for message in v:
                new_node_messages.append((message[0].detach(), message[1]))

            self.raw_messages_storage[k] = new_node_messages

    def clear_messages(self, nodes):
        for node in nodes:
            if node in self.raw_messages_storage:
                del self.raw_messages_storage[node]
