#!/bin/python3.6
import json
import os

import matplotlib.pyplot as plt
from easypyplot import pdf

plt.rcParams['font.family'] = ['serif']
plt.rcParams['font.size'] = 20

def draw_graph(ax, schemes, names, folder_names, ldata, xlabels, total_nodes, folder_path, text_to_add):
    gbps = {}
    comm_cycles = {}

    # get the file names
    for s, name in enumerate(names):
        if name not in comm_cycles.keys():
            comm_cycles[name] = {}

            for d, data in enumerate(ldata):
                if data not in comm_cycles[name].keys():
                    comm_cycles[name][data] = []

                    if name == 'multitree' or name == 'ring_bi' or name == 'ring' or name == 'ring_odd' or name == 'ring_odd_bi' or name == 'dtree' or name == 'ring2dn':
                        filename = "%s/%s/bw_%d_%s_%d_mesh_200.json" % (folder_path, folder_names[s], data, name, total_nodes)
                    else:
                        filename = "%s/%s/bw_%d_%s_%d_mesh.json" % (folder_path, folder_names[s], data, name, total_nodes)
                    if os.path.exists(filename):
                        with open(filename, 'r') as json_file:
                            sim = json.load(json_file)
                            comm_cycles[name][data] = float(sim['results']['performance']['allreduce']['total'])
                    else:
                        comm_cycles[name][data] = 0

    for s, name in enumerate(names):
        if name not in gbps.keys():
            gbps[name] = []
            for d, data in enumerate(ldata):
                if comm_cycles[name][data] != 0:
                    # gbps[name][element].append( (2*(node-1)*(data/(1024*1024))) / (comm_cycles[name][element][data] / (10 ** 9) ))
                    gbps[name].append(((float(data * 4) / (1024 * 1024 * 1024))) / (comm_cycles[name][data] / (10 ** 9)))
                else:
                    gbps[name].append(0)
    
    colors = ['#70ad47', '#ed7d31', '#4472c4', '#0D1282', '#7A316F', '#31AA75', '#EC255A']
    makercolors = ['#e2f0d9', '#fbe5d6', '#dae3f3', '#F0DE36', '#7A316F', '#31AA75', '#EC255A']
    linestyles = ['-', '-', '-', '-', '-', '-', '-', '-']
    markers = ['D', 'X', 'o', '^', '*', 'v', 'p', 'h']
    ylimit = 35

    for s, scheme in enumerate(names):
        if scheme == 'ring_odd_bi':
            ax.plot(
                gbps[scheme],
                marker=markers[-1],
                markersize=14,
                markeredgecolor=colors[-1],
                markerfacecolor=makercolors[-1],
                markeredgewidth=3,
                color=colors[s],
                linestyle=linestyles[-1],
                linewidth=3,
                label=schemes[s],
            )
        else:
            ax.plot(
                gbps[scheme],
                marker=markers[s],
                markersize=14,
                markeredgecolor=colors[s],
                markerfacecolor=makercolors[s],
                markeredgewidth=3,
                color=colors[s],
                linestyle=linestyles[s],
                linewidth=3,
                label=schemes[s],
            )
        ax.set_xticks(range(len(ldata)))
        ax.set_xticklabels(xlabels)
        ax.yaxis.set_tick_params(labelsize=20)
        ax.set_ylim(0, ylimit)
        ax.set_xlabel('All-Reduce Data Size for ' + text_to_add, y=5)
        ax.set_ylabel('Bandwidth (GB/s)')
        ax.yaxis.grid(True, linestyle='--', color='black')
        ax.xaxis.set_label_coords(0.5, -0.09)
        hdls, lab = ax.get_legend_handles_labels()


def main():
    schemes_evens = ['Ring', 'Ring-2D', 'DTree', 'Multitree', 'RingBiEven', 'TTO']
    schemes_odds = ['Ring', 'Ring-2D', 'DTree', 'Multitree', 'RingBiOdd', 'TTO']
    even_names = ['ring', 'ring2dn', 'dtree', 'multitree', 'ring_bi', 'mesh_overlap_2d_1']
    odd_names = ['ring_odd', 'ring2dn', 'dtree', 'multitree', 'ring_odd_bi', 'mesh_overlap_2d_1']
    folder_names = ['ring', 'ring2dn', 'dtree', 'multitree', 'ring_bi', 'mesh_overlap_2d_1']

    ldata = [262144, 524288, 1048576, 2097152, 4194304, 8388608, 16777216, 33554432, 67108864, 134217728, 268435456]
    xlabels = ['', '2MB', '', '8MB', '', '32MB', '', '128MB', '', '512MB', '']
    
    plt.rcParams["figure.figsize"] = [30.00, 7.0]
    plt.rcParams["figure.autolayout"] = True
    figure, ax1 = plt.subplots(1, 4)
    folder_path = '{}/HPCA_2024_final/bandwidth'.format(os.environ['SIMHOME'])

    total_nodes = 16
    draw_graph(ax1[0], schemes_evens, even_names, folder_names, ldata, xlabels, total_nodes, folder_path, '4x4 Mesh')
    # ax1[0].set_title("4x4 Mesh", y=-0.25)

    total_nodes = 25
    draw_graph(ax1[1], schemes_odds, odd_names, folder_names, ldata, xlabels, total_nodes, folder_path, '5x5 Mesh')
    # ax1[1].set_title("5x5 Mesh", y=-0.25)

    total_nodes = 64
    draw_graph(ax1[2], schemes_evens, even_names, folder_names, ldata, xlabels, total_nodes, folder_path, '8x8 Mesh')
    # ax1[2].set_title("8x8 Mesh", y=-0.25)

    total_nodes = 81
    draw_graph(ax1[3], schemes_odds, odd_names, folder_names, ldata, xlabels, total_nodes, folder_path, '9x9 Mesh')
    # ax1[3].set_title("9x9 Mesh", y=-0.25)
    
    lines_labels = [ax1[0].get_legend_handles_labels()]
    lines_labels_2 = [ax1[1].get_legend_handles_labels()]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    lines.insert(5, lines_labels_2[0][0][4])
    labels.insert(5, lines_labels_2[0][1][4])
    figure.legend(lines, labels, loc='upper center', ncol=7, bbox_to_anchor=(0.5, 1.06))
    # figure.tight_layout()
    figure.savefig('bandwidth.pdf', bbox_inches='tight')


    # plt.show()

if __name__== "__main__":
    # if len(sys.argv) != 2:
    #     print('usage: ' + sys.argv[0] + ' folder_path')
    #     exit()
    main()
