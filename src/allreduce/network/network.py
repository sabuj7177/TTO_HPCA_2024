import os
import numpy as np
import math
from abc import ABC, abstractmethod


class Network(ABC):
    def __init__(self, args):
        self.args = args
        self.nodes = args.nodes
        self.from_nodes = {}
        self.to_nodes = {}
        self.edges = []
        self.adjacency_matrix = np.zeros((self.nodes, self.nodes))
        self.node_to_switch = {}
        self.switch_to_switch = {}
        self.switch_to_switch_track = {}
        self.node_to_node = {}
        self.hiererchical_connection = {}
        self.switch_connections_to_node = {}
        self.node_connections_to_switch = {}
        self.links_usage = {}
        self.total_possible_links = 0
        self.distance_tracking = {}
        self.switch_to_node = {}
        self.priority = [0] * self.nodes # used for allocation sequence
        self.ring = None
        self.link_start_times = {}
        self.link_end_times = {}
        self.node_to_node_connectivity_switch = {}
        self.node_to_switch_usage = {}
        self.switch_to_node_usage = {}


    '''
    build_graph() - build the topology graph
    @filename: filename to generate topology dotfile, optional
    '''
    @abstractmethod
    def build_graph(self, filename=None):
        pass


    '''
    distance() - distance between two nodes
    @src: source node ID
    @dest: destination node ID
    '''
    @abstractmethod
    def distance(self, src, dest):
        pass


from kncube import KNCube

'''
construct_network() - construct a network
@args: argumetns of the top simulation

return: a network object
'''
def construct_network(args):
    args.nodes = args.num_hmcs
    network = None

    if args.booksim_network == 'mesh':
        network = KNCube(args, mesh=True)
    else:
        raise RuntimeError('Unknown network topology: ' + args.booksim_network)

    network.build_graph()
    if args.allreduce == 'dtree' and args.num_hmcs % 2 != 0:
        network.nodes = args.num_hmcs - 1
    else:
        network.nodes = args.num_hmcs

    return network
