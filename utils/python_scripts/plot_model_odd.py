import math

import json
import os

import matplotlib.pyplot as plt
import numpy as np
from easypyplot import barchart, pdf
from easypyplot import format as fmt

plt.rcParams['font.family'] = ['serif']
plt.rcParams['font.size'] = 18


def add_line(ax, xpos, ypos):
    line = plt.Line2D(
        #[xpos, xpos], [ypos + linelen, ypos],
        [xpos, xpos],
        [0, ypos],
        transform=ax.transAxes,
        color='black',
        linewidth=1)
    line.set_clip_on(False)
    ax.add_line(line)
    # ax.legend("AR Speedup")

def draw_graph(ax, folder_path, names, total_nodes, schemes, folder_names):
    benchmarks = ['alexnet', 'AlphaGoZero', 'FasterRCNN', 'Googlenet', 'NCF_recommendation', 'Resnet152', 'Transformer']
    training_time = [4210571, 43763, 1064149, 928096, 1152562, 1832399, 2151722]
    entry_names = ['AllReduce', 'Forward+Back-Propagation']
    xlabels = ['AlexNet', 'AlphaGoZero', 'FasterRCNN', 'GoogLeNet', 'NCF', 'ResNet152', 'Transformer']
    group_names = []

    cycles = np.zeros(
        (int(len(schemes)), int(len(benchmarks))), dtype=float)
    norm_cycles = np.zeros(
        (int(len(schemes)), int(len(xlabels))), dtype=float)
    norm_allreduce_cycles = np.zeros(
        (int(len(schemes)), int(len(xlabels))), dtype=float)
    training_cycles = np.zeros((int(len(schemes)), int(len(benchmarks))), dtype=float)
    allreduce_cycles = np.zeros((int(len(schemes)), int(len(benchmarks))), dtype=float)
    cycles_breakdown = np.zeros((2, int(len(benchmarks) * len(schemes))), dtype=float)
    norm_cycles_breakdown = np.zeros((2, int(len(benchmarks) * len(schemes))), dtype=float)
    total_imagenet_data = 1281167

    for s, name in enumerate(names):
        for b, bench in enumerate(benchmarks):
            if name == 'multitree' or name == 'ring_bi' or name == 'ring' or name == 'ring_odd' or name == 'ring_odd_bi' or name == 'dtree' or name == 'ring2dn':
                filename = "%s/%s/%s_%s_%d_mesh_200.json" % (folder_path, folder_names[s], bench, name, total_nodes)
            else:
                filename = "%s/%s/%s_%s_%d_mesh.json" % (folder_path, folder_names[s], bench, name, total_nodes)

            if os.path.exists(filename):
                with open(filename, 'r') as json_file:
                    sim = json.load(json_file)
                    if name == 'mesh_overlap_2d_1':
                        total_iteration = math.ceil(total_imagenet_data / ((total_nodes - 1) * 16))
                    else:
                        total_iteration = math.ceil(total_imagenet_data / (total_nodes * 16))

                    initial_ar_cycles = sim['results']['performance']['allreduce']['total']
                    allreduce_cycles[s][b] = initial_ar_cycles * total_iteration
                    training_cycles[s][b] = training_time[b] * total_iteration
                    cycles[s][b] = training_cycles[s][b] + allreduce_cycles[s][b]

                    norm_cycles[s][b] = cycles[s][b] / cycles[0][b]
                    norm_allreduce_cycles[s][b] = allreduce_cycles[s][b] / allreduce_cycles[0][b]
                    cycles_breakdown[0][b * len(schemes) + s] = allreduce_cycles[s][b]
                    cycles_breakdown[1][b * len(schemes) + s] = training_cycles[s][b]

                    json_file.close()
            else:
                norm_cycles[s][b] = 1
                norm_allreduce_cycles[s][b] = 1
                cycles_breakdown[0][b * len(schemes) + s] = 1
                cycles_breakdown[1][b * len(schemes) + s] = 1

    speedup = 1.0 / norm_cycles
    allreduce_speedup = 1.0 / norm_allreduce_cycles
    speedup[np.isnan(speedup)] = 0
    allreduce_speedup[np.isnan(allreduce_speedup)] = 0

    for b, bench in enumerate(benchmarks):
        for s, name in enumerate(names):
            group_names.append(schemes[s])
            for e, entry in enumerate(entry_names):
                norm_cycles_breakdown[e][b * len(schemes) + s] = cycles_breakdown[e][b * len(schemes) + s] / cycles[0][
                    b]
    norm_cycles_breakdown[np.isnan(norm_cycles_breakdown)] = 0

    colors = ['#B2A4FF', '#BE5A83', '#A4D0A4', '#F94A29']
    xticks = []
    for i in range(0, len(benchmarks)):
        for j in range(0, len(schemes)):
            xticks.append(i * (len(schemes) + 1) + j)
    data = [list(i) for i in zip(*norm_cycles_breakdown)]
    data = np.array(data, dtype=np.float64)
    hdls = barchart.draw(
        ax,
        data,
        group_names=group_names,
        entry_names=entry_names,
        breakdown=True,
        xticks=xticks,
        width=0.8,
        colors=colors,
        legendloc='upper center',
        legendncol=len(entry_names),
        xticklabelfontsize=20,
        xticklabelrotation=90,
        log=False)

    for i in range(len(benchmarks)):
        xpos = []
        ypos = []
        for j in range(len(names)):
            ypos.append(speedup.T[i][j])
            xpos.append(i*len(names) + i + j)
            ax.plot(xpos, ypos, marker="o", linewidth=2, color='black', label='End-to-end Training Speedup')

    # ax.set_ylabel('Normalized Runtime Breakdown', fontsize=20)
    ax.set_ylabel('Normalized Runtime Breakdown')
    ax.yaxis.grid(True, linestyle='--')
    # hdls, lab = ax.get_legend_handles_labels()
    # ax.legend(
    #     hdls,
    #     entry_names,
    #     loc='upper center',
    #     bbox_to_anchor=(0.5, 1.18),
    #     ncol=len(entry_names),
    #     frameon=False,
    #     handletextpad=0.6,
    #     columnspacing=1)
    fmt.resize_ax_box(ax, hratio=0.95)
    ly = len(benchmarks)
    scale = 1. / ly
    ypos = -.5
    for pos in range(ly + 1):
        lxpos = (pos + 0.5) * scale
        if pos < ly:
            ax.text(
                lxpos, ypos, xlabels[pos], ha='center', transform=ax.transAxes)
        add_line(ax, pos * scale, ypos)
    temp_legend = ax.get_legend()
    ax.get_legend().remove()
    ax.tick_params(axis='both')
    ax.set_ylim(0, 4)
    return temp_legend
    # pdf.plot_teardown(pdfpage)


def main():
    plt.rcParams["figure.figsize"] = [14.00, 7.0]
    plt.rcParams["figure.autolayout"] = True
    figure, ax1 = plt.subplots(1, 1)
    # figure.subplots_adjust(top=1.3)

    folder_path = '{}/HPCA_2024_final/models'.format(os.environ['SIMHOME'])
    schemes_evens = ['Ring', 'Ring-2D', 'DTree', 'Multitree', 'RingBiEven', 'TTO']
    schemes_odds = ['Ring', 'Ring-2D', 'DTree', 'Multitree', 'RingBiOdd', 'TTO']
    even_names = ['ring', 'ring2dn', 'dtree', 'multitree', 'ring_bi', 'mesh_overlap_2d_1']
    odd_names = ['ring_odd', 'ring2dn', 'dtree', 'multitree', 'ring_odd_bi', 'mesh_overlap_2d_1']
    folder_names = ['ring', 'ring2dn', 'dtree', 'multitree', 'ring_bi', 'mesh_overlap_2d_1']
    legend = draw_graph(ax1, folder_path, odd_names, 81, schemes_odds, folder_names)

    # legend = draw_graph(ax1, folder_path, odd_names, 81, schemes_odds, folder_names)

    lines_labels = [ax1.get_legend_handles_labels()]
    labels = [] if legend is None else [str(x._text) for x in legend.texts]
    handles = [] if legend is None else legend.legendHandles
    handles.append(lines_labels[0][0][0])
    labels.append(lines_labels[0][1][0])
    figure.legend(handles, labels, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.06))
    figure.savefig('models_odd.pdf', bbox_inches='tight')



if __name__ == '__main__':

    # if len(sys.argv) != 2:
    #     print('usage: ' + sys.argv[0] + ' folder_path')
    #     exit()
    main()
