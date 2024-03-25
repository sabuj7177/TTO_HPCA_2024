import json
import os
from easypyplot import pdf
import numpy as np
from matplotlib import pyplot as plt

plt.rcParams["figure.figsize"] = [12.0, 5.0]
plt.rcParams["figure.autolayout"] = True
# plt.rcParams['font.family'] = ['serif']
plt.rcParams['font.size'] = 20
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

colors = ['#70ad47', '#ed7d31', '#0D1282', '#7A316F', '#31AA75', '#EC255A']
makercolors = ['#e2f0d9', '#fbe5d6', '#F0DE36', '#7A316F', '#31AA75', '#EC255A']
linestyles = ['-', '-', '-', '-', '-', '-', '-']
markers = ['D', 'X', '^', '*', 'v', 'p', 'h']

def plot_it_2d(ax, nodes, names, schemes, folder_names, xlim, ylim, xlabel, ylabel):
    # schemes = ['Unidirectional Ring', 'Bidirectional Ring', 'Multitree', 'Hierarchical overlap', '2D overlap']
    # folder_names = ['ring', 'ring_bi', 'multitree', 'mesh_fermat', 'mesh_overlap_2d_1']
    folder_path = '{}/HPCA_2024_final/scalability'.format(os.environ['SIMHOME'])

    algorithmic_scalability = {}

    cycles = np.zeros(
        (int(len(schemes)), int(len(nodes))), dtype=np.float64)

    for s, name in enumerate(names):
        for n, node in enumerate(nodes):
            if name == 'multitree' or name == 'ring_bi' or name == 'ring' or name == 'ring_odd' or name == 'ring_odd_bi' or name == 'ring2dn' or name == 'dtree':
                filename = "%s/%s/scalability_%s_%d_mesh_200.json" % (folder_path, folder_names[s], name, node)
            else:
                filename = "%s/%s/scalability_%s_%d_mesh.json" % (folder_path, folder_names[s], name, node)
            # print (filename)
            if os.path.exists(filename):
                with open(filename, 'r') as json_file:
                    sim = json.load(json_file)
                    cycles[s][n] = sim['results']['performance']['total']
            else:
                cycles[s][n] = 0
                print("File missing: " + str(filename))

        algorithmic_scalability[name] = [int(ele)/cycles[0][0] for ele in cycles[s]]

    for s, scheme in enumerate(names):
        if scheme == 'ring_odd_bi':
            ax.plot(
                nodes,
                algorithmic_scalability[scheme],
                marker=markers[-1],
                markersize=14,
                markeredgecolor=colors[-1],
                markerfacecolor=makercolors[-1],
                markeredgewidth=3,
                color=colors[s],
                linestyle=linestyles[-1],
                linewidth=3,
                label=schemes[s]
            )
        else:
            ax.plot(
                nodes,
                algorithmic_scalability[scheme],
                marker=markers[s],
                markersize=14,
                markeredgecolor=colors[s],
                markerfacecolor=makercolors[s],
                markeredgewidth=3,
                color=colors[s],
                linestyle=linestyles[s],
                linewidth=3,
                label=schemes[s]
                )
        ax.set_xticks(nodes)
        ax.set_xlim(0, xlim)
        ax.set_ylim(0, ylim)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.yaxis.grid(True, linestyle='--', color='black')
        hdls, lab = ax.get_legend_handles_labels()

if __name__ == '__main__':
    figure, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    schemes_evens = ['Ring', 'Ring-2D', 'Multitree', 'RingBiEven', 'TTO']
    schemes_odds = ['Ring', 'Ring-2D', 'Multitree', 'RingBiOdd', 'TTO']
    even_names = ['ring', 'ring2dn', 'multitree', 'ring_bi', 'mesh_overlap_2d_1']
    odd_names = ['ring_odd', 'ring2dn', 'multitree', 'ring_odd_bi', 'mesh_overlap_2d_1']
    folder_names = ['ring', 'ring2dn', 'multitree', 'ring_bi', 'mesh_overlap_2d_1']
    even_nodes = [16, 36, 64, 100, 144, 196, 256]
    odd_nodes = [9, 25, 49, 81, 121, 169, 225]
    plot_it_2d(ax1, even_nodes, even_names, schemes_evens, folder_names, 256, 26, 'Nodes in Even-sized Mesh', 'Normalized Runtime')
    plot_it_2d(ax2, odd_nodes, odd_names, schemes_odds, folder_names, 225, 26, 'Nodes in Odd-sized Mesh', '')
    figure.subplots_adjust(top=1, bottom=0.1, right=1)
    lines_labels = [ax1.get_legend_handles_labels()]
    lines_labels_2 = [ax2.get_legend_handles_labels()]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    lines.insert(4, lines_labels_2[0][0][3])
    labels.insert(4, lines_labels_2[0][1][3])
    figure.legend(lines, labels, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.15))
    figure.savefig('scalability.pdf', bbox_inches='tight')
