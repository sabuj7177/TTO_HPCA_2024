import os
from abc import ABC, abstractmethod

import numpy as np


class Allreduce(ABC):
    def __init__(self, args, network):
        self.args = args
        self.network = network
        self.num_flows = self.network.nodes  # default number of flows
        self.trees = None
        self.trees_parent = None
        self.trees_children = None
        self.timesteps = None
        self.conflicts_in_tree = None
        self.per_message_latency_dict = None
        self.schedule_timestamp = None
        self.reduce_scatter_ordering_dict = {}
        self.all_gather_ordering_dict = {}
        self.reduce_scatter_time_track_dict = {}
        self.all_gather_time_track_dict = {}
        self.data_percentage_in_flows = None
        self.total_full_trees = None
        self.total_partial_trees = None
        self.full_trees = None
        self.partial_trees = None
        self.reduce_scatter_schedule = None
        self.all_gather_schedule = None

        # TODO: reduce-scatter and all-gather schedulues are merged into a unified
        # schedule, opcodes: {'Reduce', 'Gather', 'NOP'}
        self.collective_schedule = None

    '''
    compute_schedule() - computes spanning trees and schedule for the given network
    '''

    def compute_schedule(self, kary, alternate=True, sort=True, verbose=False):
        self.compute_trees(kary, alternate, sort, verbose)
        self.generate_schedule(verbose)

    def generate_ordering(self):
        reduce_scatter_ordering_dict = {}
        all_gather_ordering_dict = {}

        for flow in range(self.network.nodes):
            for edge in self.trees[flow]:
                child = edge[0]
                parent = edge[1]
                rs_timestep = self.timesteps - edge[2] - edge[3]
                ag_timestep = self.timesteps + edge[2]
                if (child, parent) not in reduce_scatter_ordering_dict.keys():
                    reduce_scatter_ordering_dict[child, parent] = []
                reduce_scatter_ordering_dict[child, parent].append((rs_timestep, flow))
                if (parent, child) not in all_gather_ordering_dict.keys():
                    all_gather_ordering_dict[parent, child] = []
                all_gather_ordering_dict[parent, child].append((ag_timestep, flow))

        for key in reduce_scatter_ordering_dict.keys():
            self.reduce_scatter_ordering_dict[key] = sorted(reduce_scatter_ordering_dict[key])
        for key in all_gather_ordering_dict.keys():
            self.all_gather_ordering_dict[key] = sorted(all_gather_ordering_dict[key])

    '''
    compute_trees() - computes allreduce spanning trees for the given network
    '''

    @abstractmethod
    def compute_trees(self, kary, alternate=False, sort=True, verbose=False):
        pass

    '''
    generate_schedule()
    @verbose: print the generated schedules

    desc - generate reduce_scatter_schedule and all_gather_schedule from trees
    '''

    @abstractmethod
    def generate_schedule(self, verbose=False):
        pass

    '''
    generate_trees_dotfile() - generate dotfile for computed trees
    @filename: name of dotfile
    '''

    def generate_trees_dotfile(self, filename):
        # color palette for ploting nodes of different tree levels
        colors = ['#ffffff', '#f7f4f9', '#e7e1ef', '#d4b9da', '#c994c7',
                  '#df65b0', '#e7298a', '#ce1256', '#980043', '#67001f']

        tree = 'digraph tree {\n'
        tree += '  rankdir = BT;\n'
        tree += '  subgraph {\n'

        # group nodes with same rank (same tree level/iteration)
        # and set up the map for node name and its rank in node_rank
        ranks = {}
        node_rank = {}
        for rank in range(self.timesteps + 1):
            ranks[rank] = []

        for root in range(self.network.nodes):
            minrank = self.timesteps
            for edge in self.trees[root]:
                child = '"{}-{}"'.format(root, edge[0])
                rank = edge[2] + 1
                ranks[rank].append(child)
                node_rank[child] = rank
                if edge[1] == root and rank - 1 < minrank:
                    minrank = rank - 1
            ranks[minrank].append('"{}-{}"'.format(root, root))
            node_rank['"{}-{}"'.format(root, root)] = minrank

        for root in range(self.network.nodes):
            tree += '    /* tree {} */\n'.format(root)
            for edge in self.trees[root]:
                child = '"{}-{}"'.format(root, edge[0])
                parent = '"{}-{}"'.format(root, edge[1])
                cycle = self.timesteps - edge[2]
                minlen = node_rank[child] - node_rank[parent]  # for strict separation of ranks
                tree += ''.join('    {} -> {} [ label="{}" minlen={} ];\n'.format(child, parent, cycle, minlen))

        tree += '    // note that rank is used in the subgraph\n'
        for rank in range(self.timesteps + 1):
            if ranks[rank]:
                level = '    {rank = same;'
                for node in ranks[rank]:
                    level += ' {};'.format(node)
                level += '}\n'
                tree += level

        tree += '    // node colors\n'
        style = '    {} [style="filled", fillcolor="{}"];\n'
        for rank in range(self.timesteps + 1):
            if ranks[rank]:
                tree += ''.join(style.format(node, colors[rank % len(colors)]) for node in ranks[rank])

        tree += '  } /* closing subgraph */\n'
        tree += '}\n'

        f = open(filename, 'w')
        f.write(tree)
        f.close()

    '''
    max_num_concurrent_flows() - compute the concurrent flows for an accelerator
    '''

    def max_num_concurrent_flows(self):
        max_concurrent_reduce_scatter = np.zeros(self.network.nodes, dtype=int)
        max_concurrent_reduce_scatter_timestep = np.zeros(self.network.nodes, dtype=int)
        for root in range(self.network.nodes):
            timesteps = len(self.reduce_scatter_schedule[root])
            for timestep in range(timesteps):
                if self.reduce_scatter_schedule[root][timestep] == None:
                    continue
                num_concurrent_reduce_scatter = len(self.reduce_scatter_schedule[root][timestep])
                if max_concurrent_reduce_scatter[root] < num_concurrent_reduce_scatter:
                    max_concurrent_reduce_scatter[root] = num_concurrent_reduce_scatter
                    max_concurrent_reduce_scatter_timestep[root] = timestep + 1

        max_concurrent_all_gather = np.zeros(self.network.nodes, dtype=int)
        max_concurrent_all_gather_timestep = np.zeros(self.network.nodes, dtype=int)
        for root in range(self.network.nodes):
            timesteps = len(self.all_gather_schedule[root])
            for timestep in range(timesteps):
                num_concurrent_all_gather = 0
                for flow, children_parent_dependency in self.all_gather_schedule[root][timestep].items():
                    num_concurrent_all_gather += len(children_parent_dependency[0])
                if max_concurrent_all_gather[root] < num_concurrent_all_gather:
                    max_concurrent_all_gather[root] = num_concurrent_all_gather
                    max_concurrent_all_gather_timestep[root] = timestep + 1

        for root in range(self.network.nodes):
            print('Tree {}:'.format(root))
            print('  - reduce-scatter schedules:')
            for timestep in range(len(self.reduce_scatter_schedule[root])):
                print('    step {}: {}'.format(timestep + 1, self.reduce_scatter_schedule[root][timestep]))
            print('  - all-gather schedules:')
            for timestep in range(len(self.all_gather_schedule[root])):
                print('    step {}: {}'.format(timestep + 1, self.all_gather_schedule[root][timestep]))
            print('  - max number of concurrent reduce-scatter is {} (at timestep {})'
                  ', and and all-gather communications is {} (at timestep {})'.format(
                max_concurrent_reduce_scatter[root],
                max_concurrent_reduce_scatter_timestep[root],
                max_concurrent_all_gather[root],
                max_concurrent_all_gather_timestep[root]))
    # end of max_num_concurrent_flows()


import sys

sys.path.append('{}/src/allreduce/network'.format(os.environ['SIMHOME']))

from network import construct_network
from ring_allreduce import RingAllreduce
from dtree_allreduce import DTreeAllreduce
from multitree_allreduce import MultiTreeAllreduce
from ring2d_n_allreduce import Ring2DnAllreduce
from mesh_allreduce_2d_overlap_1 import MeshAllreduce2DOverlap1
from ring_odd_bi_allreduce import RingOddBiAllreduce
from ring_odd_allreduce import RingOddAllreduce
from ring_bidirectional_allreduce import RingBiAllreduce

'''
construct_allreduce() - construct an allreduce schedule
@args: arguments of the top simulation

return: an allreduce object
'''


def construct_allreduce(args):
    args.nodes = args.num_hmcs
    network = construct_network(args)

    if args.allreduce == 'mesh_overlap_2d_1':
        allreduce = MeshAllreduce2DOverlap1(args, network)
    elif args.allreduce == 'multitree':
        allreduce = MultiTreeAllreduce(args, network)
    elif args.allreduce == 'ring':
        allreduce = RingAllreduce(args, network)
    elif args.allreduce == 'ring_bi':
        allreduce = RingBiAllreduce(args, network)
    elif args.allreduce == 'dtree':
        allreduce = DTreeAllreduce(args, network)
    elif args.allreduce == 'ring2dn':
        allreduce = Ring2DnAllreduce(args, network)
    elif args.allreduce == 'ring_odd':
        allreduce = RingOddAllreduce(args, network)
    elif args.allreduce == 'ring_odd_bi':
        allreduce = RingOddBiAllreduce(args, network)
    else:
        raise RuntimeError('Unknown homogeneous allreduce schedule: ' + args.allreduce)

    return allreduce
