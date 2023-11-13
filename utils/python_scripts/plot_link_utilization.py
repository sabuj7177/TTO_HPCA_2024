import json
import os
import pickle

import matplotlib.pyplot as plt


class DrawLinkUtilization:

    def get_used_cycles(self, link_start_time, link_end_time, current_last):
        assert len(link_start_time) == len(link_end_time)
        used_cycles = []
        for i in range(len(link_start_time)):
            start_time = link_start_time[i]
            end_time = link_end_time[i]
            if current_last < start_time:
                current_last = start_time
            while current_last <= end_time:
                used_cycles.append(current_last)
                current_last += 1
        return set(used_cycles)

    def get_utilized_link_info(self, file_name, time_diff, max_time, total_links):
        save_object = pickle.load(open(file_name, 'rb'))
        link_start_time = save_object['link_start_time']
        link_end_time = save_object['link_end_time']
        time_series = {}
        new_time = time_diff
        required_times = []
        while new_time < max_time:
            time_series[new_time] = []
            required_times.append(new_time)
            new_time += time_diff
        per_cycle_usage = {}
        for i in range(max_time):
            per_cycle_usage[i] = 0
        print("Done initialization")
        for key in link_start_time.keys():
            if len(link_start_time[key]) > 0:
                starts = link_start_time[key]
                ends = link_end_time[key]
                current_last = 0
                for i in range(len(starts)):
                    start_time = starts[i]
                    end_time = ends[i]
                    if current_last < start_time:
                        current_last = start_time
                    while current_last <= end_time:
                        per_cycle_usage[current_last] += 1
                        current_last += 1
            else:
                raise RuntimeError('No start time of link')
        print("Done per cycle usage computation")
        utilization_percentage = []
        start_counter = 0
        for r_time in required_times:
            total_used = 0
            all_total_links = 0
            while start_counter < r_time:
                total_used += per_cycle_usage[start_counter]
                all_total_links += total_links
                start_counter += 1
            utilization_percentage.append(total_used / all_total_links)
        return utilization_percentage, required_times

    def compute_link_utilization(self, ax, schemes, names, data, total_nodes, folder_path, total_links, colors, y_label):
        time_diff = 10000
        ax.set_xlabel("Total cycles")
        ax.set_ylabel("Link utilization percentage")
        max_time = 0
        for s, name in enumerate(names):
            if name == 'multitree' or name == 'ring_bi' or name == 'ring' or name == 'ring_odd' or name == 'ring_odd_bi' or name == 'ring2dn':
                filename = "%s/bw_%d_%s_%d_mesh_200.json" % (folder_path, data, name, total_nodes)
            else:
                filename = "%s/bw_%d_%s_%d_mesh.json" % (folder_path, data, name, total_nodes)
            if os.path.exists(filename):
                with open(filename, 'r') as json_file:
                    sim = json.load(json_file)
                    if sim['results']['performance']['total'] > max_time:
                        max_time = sim['results']['performance']['total']

        for s, name in enumerate(names):
            if name == 'multitree' or name == 'ring_bi' or name == 'ring' or name == 'ring_odd' or name == 'ring_odd_bi' or name == 'ring2dn':
                filename = "%s/bw_%d_%s_%d_mesh_200.pkl" % (folder_path, data, name, total_nodes)
            else:
                filename = "%s/bw_%d_%s_%d_mesh.pkl" % (folder_path, data, name, total_nodes)
            print("Before utilization percentage")
            utilization_percentage, required_times = self.get_utilized_link_info(filename, time_diff, max_time, total_links)
            ax.plot(required_times, utilization_percentage, color=colors[s], label=schemes[s], linewidth=3)
        ax.set_xlabel("Number of Cycles")
        ax.set_ylabel(y_label)
        ax.yaxis.grid(True, linestyle='--', color='black')


def main():
    tree = DrawLinkUtilization()
    plt.rcParams["figure.figsize"] = [16.00, 5.0]
    plt.rcParams["figure.autolayout"] = True
    plt.rcParams['font.family'] = ['serif']
    plt.rcParams['font.size'] = 24
    figure, ax1 = plt.subplots(1, 3, sharex=True, sharey=True)
    dataSize = 67108864

    folder_path = '{}/HPCA_2024_final/utilization'.format(os.environ['SIMHOME'])
    schemes_odds = ['Ring', 'Ring-2D']
    odd_names = ['ring_odd', 'ring2dn']
    tree.compute_link_utilization(ax1[0], schemes_odds, odd_names, dataSize, 81, folder_path, 288, ['#70ad47', '#ed7d31'], "Link Utilization(%)")

    schemes_odds = ['Multitree', 'RingBiOdd']
    odd_names = ['multitree', 'ring_odd_bi']
    tree.compute_link_utilization(ax1[1], schemes_odds, odd_names, dataSize, 81, folder_path, 288, ['#4472c4', '#d95319'], '')

    schemes_odds = ['TTO']
    odd_names = ['mesh_overlap_2d_1']
    tree.compute_link_utilization(ax1[2], schemes_odds, odd_names, dataSize, 81, folder_path, 288, ['#31AA75'], '')
    lines_labels = [ax1[0].get_legend_handles_labels()]
    lines_labels_2 = [ax1[1].get_legend_handles_labels()]
    lines_labels_3 = [ax1[2].get_legend_handles_labels()]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    lines2, labels2 = [sum(lol, []) for lol in zip(*lines_labels_2)]
    lines3, labels3 = [sum(lol, []) for lol in zip(*lines_labels_3)]
    lines.extend(lines2)
    labels.extend(labels2)
    lines.extend(lines3)
    labels.extend(labels3)
    figure.legend(lines, labels, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.2))

    figure.savefig('link_utilization.pdf', bbox_inches='tight')


if __name__ == '__main__':
    main()
