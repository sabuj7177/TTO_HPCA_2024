import argparse
import copy
import sys
import os
import math
import numpy as np
from copy import deepcopy

sys.path.append('{}/src/allreduce/network'.format(os.environ['SIMHOME']))

from network import construct_network
from allreduce import Allreduce


class MeshAllreduce2DOverlap1(Allreduce):
    def __init__(self, args, network):
        super().__init__(args, network)
        self.number_of_nodes = int(math.sqrt(self.network.nodes))
        self.trees = None
        self.ring = []
        self.full_trees = None
        self.partial_trees = None
        self.rs_schedule = {}
        self.ag_schedule = {}
        self.rs2_final_dep = {}

    def get_lrtb(self, node, nodes_per_dim):
        col_idx = node % nodes_per_dim
        row_idx = math.floor(node / nodes_per_dim)
        if col_idx == 0:
            left = None
            right = node + 1
        elif col_idx == nodes_per_dim - 1:
            left = node - 1
            right = None
        else:
            left = node - 1
            right = node + 1

        if row_idx == 0:
            top = None
            bottom = node + nodes_per_dim
        elif row_idx == nodes_per_dim - 1:
            top = node - nodes_per_dim
            bottom = None
        else:
            top = node - nodes_per_dim
            bottom = node + nodes_per_dim
        return left, right, top, bottom

    def build_trees(self):
        total_nodes = self.network.nodes
        per_dim_nodes = int(math.sqrt(total_nodes))
        left_nodes = {}
        right_nodes = {}
        top_nodes = {}
        bottom_nodes = {}
        for node in range(total_nodes):
            left, right, top, bottom = self.get_lrtb(node, per_dim_nodes)
            left_nodes[node] = left
            right_nodes[node] = right
            top_nodes[node] = top
            bottom_nodes[node] = bottom

        tree = []
        time_tracker = {}
        node_to_consider = 0
        time_tracker[node_to_consider] = 0
        for_right = []
        for_right.append(node_to_consider)

        for i in range(per_dim_nodes - 1):
            timestep = time_tracker[node_to_consider]
            bottom_node = bottom_nodes[node_to_consider]
            tree.append((bottom_node, node_to_consider, timestep + 1))
            time_tracker[bottom_node] = timestep + 1
            for_right.append(bottom_node)
            node_to_consider = bottom_node
        for target_node in for_right:
            node_to_consider = target_node
            for i in range(per_dim_nodes - 1):
                timestep = time_tracker[node_to_consider]
                right_node = right_nodes[node_to_consider]
                tree.append((right_node, node_to_consider, timestep + 1))
                time_tracker[right_node] = timestep + 1
                node_to_consider = right_node
        zero_tree = copy.deepcopy(tree)
        print("Yoo")
        tree = []
        time_tracker = {}
        node_to_consider = self.args.num_hmcs - 1
        time_tracker[node_to_consider] = 0
        for_top = []
        for_top.append(node_to_consider)

        for i in range(per_dim_nodes - 1):
            timestep = time_tracker[node_to_consider]
            left_node = left_nodes[node_to_consider]
            tree.append((left_node, node_to_consider, timestep + 1))
            time_tracker[left_node] = timestep + 1
            for_top.append(left_node)
            node_to_consider = left_node
        for target_node in for_top:
            node_to_consider = target_node
            for i in range(per_dim_nodes - 1):
                timestep = time_tracker[node_to_consider]
                top_node = top_nodes[node_to_consider]
                tree.append((top_node, node_to_consider, timestep + 1))
                time_tracker[top_node] = timestep + 1
                node_to_consider = top_node
        last_tree = copy.deepcopy(tree)

        tree = []
        time_tracker = {}
        node_to_consider = per_dim_nodes - 1
        time_tracker[node_to_consider] = 0
        # for_top = []
        # for_top.append(node_to_consider)

        for i in range(per_dim_nodes - 1):
            node_to_consider_left = node_to_consider
            while left_nodes[node_to_consider_left] is not None:
                timestep = time_tracker[node_to_consider_left]
                left_node = left_nodes[node_to_consider_left]
                tree.append((left_node, node_to_consider_left, timestep + 1))
                time_tracker[left_node] = timestep + 1
                node_to_consider_left = left_node

            node_to_consider_bottom = node_to_consider
            while bottom_nodes[node_to_consider_bottom] is not None:
                timestep = time_tracker[node_to_consider_bottom]
                bottom_node = bottom_nodes[node_to_consider_bottom]
                tree.append((bottom_node, node_to_consider_bottom, timestep + 1))
                time_tracker[bottom_node] = timestep + 1
                node_to_consider_bottom = bottom_node

            if i < per_dim_nodes - 2:
                left_node = left_nodes[node_to_consider]
                bottom_node = bottom_nodes[left_node]
                timestep = time_tracker[left_node]
                tree.append((bottom_node, left_node, timestep + 1))
                time_tracker[bottom_node] = timestep + 1
                node_to_consider = bottom_node
        middle_tree = copy.deepcopy(tree)
        return zero_tree, middle_tree, last_tree

    '''
    compute_trees() - computes allreduce rings (special tree) for the given network
    @kary: not used, skip
    @alternate: not used, skip
    @sort: not used, skip
    @verbose: print detailed info of ring construction process
    '''

    def compute_trees(self, kary=None, alternate=True, sort=False, verbose=False):
        zero_tree, middle_tree, last_tree = self.build_trees()
        self.template_trees = {}
        self.template_trees[0] = sorted(zero_tree, key=lambda x: x[2])
        self.template_trees[self.number_of_nodes - 1] = sorted(middle_tree, key=lambda x: x[2])
        self.template_trees[self.args.num_hmcs - 1] = sorted(last_tree, key=lambda x: x[2])
        self.tree_roots = []
        self.tree_roots.append(0)
        self.tree_roots.append(self.number_of_nodes - 1)
        self.tree_roots.append(self.args.num_hmcs - 1)
        edge_dict = {}
        for i in range(self.args.num_hmcs):
            left, right, top, bottom = self.get_lrtb(i, self.number_of_nodes)
            if left is not None:
                edge_dict[(i, left)] = 0
            if right is not None:
                edge_dict[(i, right)] = 0
            if top is not None:
                edge_dict[(i, top)] = 0
            if bottom is not None:
                edge_dict[(i, bottom)] = 0

        self.edge_dict = edge_dict
        self.edge_dict_ag = copy.deepcopy(edge_dict)
        self.time_relative_links_last = {}
        for key in self.template_trees.keys():
            tree = self.template_trees[key]
            for edge in tree:
                time = edge[2] - 1
                if time not in self.time_relative_links_last.keys():
                    self.time_relative_links_last[time] = []
                self.time_relative_links_last[time].append((edge[0], edge[1], key))
        self.total_partial_trees = self.args.total_partial_trees

    # def compute_trees(self, kary=None, alternate=True, sort=False, verbose=False)

    def get_dependency(self, tree, source):
        dependencies = []
        for dep in self.trees_children[tree][source]:
            dependencies.append(dep)
        return dependencies

    def get_ag_dependency(self, tree, source):
        dependencies = []
        if self.trees_parent[tree][source] is not None:
            dependencies.append(self.trees_parent[tree][source])
        return dependencies

    def get_start_time(self, edge_dict, source, dest, dependencies):
        max_dep_time = 0
        for dep in dependencies:
            if edge_dict[(dep, source)] > max_dep_time:
                max_dep_time = edge_dict[(dep, source)]
        if max_dep_time > edge_dict[(source, dest)]:
            return max_dep_time
        else:
            return edge_dict[(source, dest)]

    def update_rs_final_dep(self, root, chunk_id):
        dependencies = self.get_dependency(root, root)
        if root not in self.rs2_final_dep.keys():
            self.rs2_final_dep[root] = []
        self.rs2_final_dep[root].append((chunk_id, dependencies))

    def add_reduce_scatter(self, chunk_id, total_message):
        total_multiplied = 1
        for key in sorted(self.time_relative_links_last.keys(), reverse=True):
            for edge in self.time_relative_links_last[key]:
                link = (edge[0], edge[1])
                if link not in self.rs_schedule.keys():
                    self.rs_schedule[link] = []
                dependencies = self.get_dependency(tree=edge[2], source=edge[0])
                source_ni = self.get_ni(edge[0], edge[1])
                target_ni = self.get_ni(edge[1], edge[0])
                tree = edge[2]
                if link not in self.edge_dict.keys():
                    self.edge_dict[link] = 0
                start_time = self.get_start_time(self.edge_dict, edge[0], edge[1], dependencies)
                self.rs_schedule[link].append((tree, chunk_id, dependencies, total_message, total_multiplied,
                                               start_time, start_time + total_multiplied - 1, 0, source_ni, target_ni))
                self.edge_dict[link] = start_time + total_multiplied
        for root in self.tree_roots:
            self.update_rs_final_dep(root, chunk_id)

    def add_all_gather(self, chunk_id, total_message):
        total_multiplied = 1
        for key in sorted(self.time_relative_links_last.keys()):
            for edge in self.time_relative_links_last[key]:
                link = (edge[1], edge[0])
                if link not in self.ag_schedule.keys():
                    self.ag_schedule[link] = []
                dependencies = self.get_ag_dependency(tree=edge[2], source=edge[1])
                source_ni = self.get_ni(edge[1], edge[0])
                target_ni = self.get_ni(edge[0], edge[1])
                tree = edge[2]
                if link not in self.edge_dict_ag.keys():
                    self.edge_dict_ag[link] = 0
                start_time = self.get_start_time(self.edge_dict_ag, edge[1], edge[0], dependencies)
                self.ag_schedule[link].append((tree, chunk_id, dependencies, total_message, total_multiplied,
                                               start_time, start_time + total_multiplied - 1, 0, source_ni, target_ni))
                self.edge_dict_ag[link] = self.edge_dict_ag[link] + total_multiplied

    def get_ni(self, source_node, target_node):
        if target_node == source_node - 1:
            return 0
        elif target_node == source_node + 1:
            return 1
        elif target_node == source_node - self.number_of_nodes:
            return 2
        elif target_node == source_node + self.number_of_nodes:
            return 3
        else:
            raise RuntimeError('Error: NI info is wrong')

    def check_per_link_timestep_ordering(self, per_link_schedule):
        current_max_end = per_link_schedule[0][5] - 1
        for schedule in per_link_schedule:
            start = schedule[5]
            end = schedule[6]
            if end < start:
                raise RuntimeError("End time is earlier than start time")
            if start < current_max_end:
                raise RuntimeError("Start time is earlier than current max end time")
            # if start - current_max_end != 1:
            #     raise RuntimeError("Difference between start time and current max end time is not 1")
            current_max_end = end

    def check_timestep_ordering(self):
        for link in self.rs_schedule.keys():
            self.check_per_link_timestep_ordering(self.rs_schedule[link])
        for link in self.ag_schedule.keys():
            self.check_per_link_timestep_ordering(self.ag_schedule[link])

    '''
    generate_schedule()
    @verbose: print the generated schedules

    desc - generate reduce_scatter_schedule and all_gather_schedule from ring,
           verified with generate_schedule in MultiTree
    '''

    def generate_schedule(self, verbose=False):
        # compute parent-children dependency
        self.trees_parent = {}
        self.trees_children = {}
        for root in self.tree_roots:
            self.trees_parent[root] = {}
            self.trees_parent[root][root] = None
            self.trees_children[root] = {}
            for node in range(self.args.num_hmcs):
                self.trees_children[root][node] = []
            for edge in self.template_trees[root]:
                child = edge[0]
                parent = edge[1]
                self.trees_parent[root][child] = parent
                self.trees_children[root][parent].append(child)

        for i in range(self.total_partial_trees):
            self.add_reduce_scatter(chunk_id=i, total_message=self.args.partial_tree_message)
        for i in range(self.total_partial_trees):
            self.add_all_gather(chunk_id=i, total_message=self.args.partial_tree_message)

        self.check_timestep_ordering()

        self.final_reduce_scatter_schedule = {}
        for i in range(self.args.num_hmcs):
            self.final_reduce_scatter_schedule[i] = {}
        for link in self.edge_dict.keys():
            self.final_reduce_scatter_schedule[link[0]][link[1]] = []
        for link in self.rs_schedule.keys():
            source = link[0]
            dest = link[1]
            for schedule in self.rs_schedule[link]:
                tree_id = schedule[0]
                chunk_id = schedule[1]
                dependencies = schedule[2]
                total_messages = schedule[3]
                order = schedule[7]
                source_ni = schedule[8]
                dest_ni = schedule[9]
                self.final_reduce_scatter_schedule[source][dest].append(
                    (tree_id, chunk_id, dependencies, total_messages, order, source_ni, dest_ni))
        self.reduce_scatter_schedule = self.final_reduce_scatter_schedule

        self.final_ag_schedule = {}
        for i in range(self.args.num_hmcs):
            self.final_ag_schedule[i] = {}
        for link in self.edge_dict.keys():
            self.final_ag_schedule[link[0]][link[1]] = []
        for link in self.ag_schedule.keys():
            source = link[0]
            dest = link[1]
            for schedule in self.ag_schedule[link]:
                tree_id = schedule[0]
                chunk_id = schedule[1]
                dependencies = schedule[2]
                total_messages = schedule[3]
                order = schedule[7]
                source_ni = schedule[8]
                dest_ni = schedule[9]
                # print("Source dest")
                # print(source)
                # print(dest)
                self.final_ag_schedule[source][dest].append(
                    (tree_id, chunk_id, dependencies, total_messages, order, source_ni, dest_ni))
        self.all_gather_schedule = self.final_ag_schedule

    # def generate_schedule(self, verbose=False)

    '''
    generate_ring_dotfile() - generate dotfile for computed rings
    @filename: name of dotfile
    '''

    def generate_ring_dotfile(self, filename):
        # color palette for ploting nodes of different tree levels
        colors = ['#f7f4f9', '#e7e1ef', '#d4b9da', '#c994c7', '#df65b0',
                  '#e7298a', '#ce1256', '#980043', '#67001f']

        ring = 'digraph ring {\n'
        ring += '  rankdir = BT;\n'
        ring += '  /* ring */\n'

        for tree_no in range(len(self.full_trees)):
            trees = self.full_trees[tree_no]
            for i in range(self.number_of_nodes):
                for edge in trees[i]:
                    ring += '  {} -> {} [ label="{}"];\n'.format(
                        '"' + str(tree_no) + "-" + str(i) + '-' + str(edge[0]) + '"',
                        '"' + str(tree_no) + "-" + str(i) + '-' + str(edge[1]) + '"', edge[2] + 1)
        # ring += '  {} -> {};\n'.format(self.ring[-1], self.ring[0])

        # ring += '  // note that rank is used in the subgraph\n'
        # ring += '  subgraph {\n'
        # ring += '    {rank = same; ' + str(self.ring[0]) + ';}\n'
        # for i in range(1, self.network.nodes // 2):
        #     ring += '    {rank = same; '
        #     ring += '{}; {};'.format(self.ring[i], self.ring[self.network.nodes - i])
        #     ring += '}\n'
        # ring += '    {rank = same; ' + str(self.ring[self.network.nodes // 2]) + ';}\n'

        # ring += '  } /* closing subgraph */\n'
        ring += '}\n'

        f = open(filename, 'w')
        f.write(ring)
        f.close()
    # def generate_ring_dotfile(self, filename)


def main(args):
    # network = construct_network(args)
    # network.to_nodes[1].clear() # test no solution case

    allreduce = MeshAllreduce(args, None)
    allreduce.compute_trees(verbose=args.verbose)
    # allreduce.generate_schedule(verbose=args.verbose)
    # allreduce.max_num_concurrent_flows()
    # if args.gendotfile:
    # allreduce.generate_ring_dotfile('ring.dot')
    #     allreduce.generate_trees_dotfile('ring_trees.dot')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--num-hmcs', default=6, type=int,
                        help='number of nodes, default is 16')
    parser.add_argument('--bigraph-m', default=8, type=int,
                        help='logical groups size (# sub-node per switch), default 8')
    parser.add_argument('--bigraph-n', default=2, type=int,
                        help='# switches, default 2')
    parser.add_argument('--gendotfile', default=False, action='store_true',
                        help='generate tree dotfiles, default is False')
    parser.add_argument('--verbose', default=False, action='store_true',
                        help='detailed print')
    parser.add_argument('--booksim-network', default='torus',
                        help='network topology (torus | mesh | dgx2), default is torus')
    parser.add_argument('--kary', default=2, type=int,
                        help='generay kary tree, default is 2 (binary)')
    parser.add_argument('--total-full-trees', default=3, type=int,
                        help='Total number of full trees in mesh')
    parser.add_argument('--total-partial-trees', default=2, type=int,
                        help='Total number of partial trees in mesh')

    args = parser.parse_args()

    main(args)
