#!/bin/python3.6
import json
import os

import matplotlib.pyplot as plt

# plt.rcParams['font.family'] = ['serif']
plt.rcParams['font.size'] = 20
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

def draw_graph(ax, schemes, names, ldata, xlabels, total_nodes, folder_path):
    gbps = {}
    comm_cycles = {}

    # get the file names
    for s, name in enumerate(names):
        if name not in comm_cycles.keys():
            comm_cycles[name] = {}

            for d, data in enumerate(ldata):
                if data not in comm_cycles[name].keys():
                    comm_cycles[name][data] = []

                    filename = "%s/chunk_33554432_%d_%s_%d_mesh.json" % (folder_path, data, name, total_nodes)
                    if os.path.exists(filename):
                        with open(filename, 'r') as json_file:
                            sim = json.load(json_file)
                            comm_cycles[name][data] = float(sim['results']['performance']['allreduce']['total'])
                    else:
                        print("File missing: " + str(filename))

    for s, name in enumerate(names):
        if name not in gbps.keys():
            gbps[name] = []
            for d, data in enumerate(ldata):
                if comm_cycles[name][data] != 0:
                    gbps[name].append(((float(33554432 * 4) / (1024 * 1024 * 1024))) / (comm_cycles[name][data] / (10 ** 9)))
                else:
                    gbps[name].append(0)
    colors = ['#31AA75']
    makercolors = ['#31AA75']
    linestyles = ['-', '-']
    markers = ['p']
    ylimit = 35

    for s, scheme in enumerate(names):
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
        ax.set_xlabel('Chunk Size', y=5)
        ax.set_ylabel('Bandwidth (GB/s)')
        ax.yaxis.grid(True, linestyle='--', color='black')
        ax.xaxis.set_label_coords(0.5, -0.09)
        plt.legend()


def main():
    schemes = ['TTO']
    names = ['mesh_overlap_2d_1']

    ldata = [3072, 6144, 12288, 24576, 49152, 98304, 196608, 393216, 786432, 1572864]
    xlabels = ['', '24KB', '', '96KB', '', '384KB', '', '1.5MB', '', '6MB']

    plt.rcParams["figure.figsize"] = [9.00, 5.0]
    plt.rcParams["figure.autolayout"] = True
    figure, ax1 = plt.subplots(1, 1)
    folder_path = '{}/HPCA_2024_final/chunk_utilization'.format(os.environ['SIMHOME'])

    total_nodes = 64
    draw_graph(ax1, schemes, names, ldata, xlabels, total_nodes, folder_path)
    figure.savefig('chunk.pdf', bbox_inches='tight')

if __name__== "__main__":
    # if len(sys.argv) != 2:
    #     print('usage: ' + sys.argv[0] + ' folder_path')
    #     exit()
    main()
