import argparse
import configparser as cp
import os
import time
import sys
import logging
import json
import math
import pickle

sys.path.append('{}/src/SCALE-Sim'.format(os.environ['SIMHOME']))
sys.path.append('{}/src/booksim2/src'.format(os.environ['SIMHOME']))
sys.path.append('{}/src/allreduce'.format(os.environ['SIMHOME']))

from model import Model
from hmc import HMC
from hmc_fermat import HMC as HMC_fermat
from booksim import BookSim
from allreduce import construct_allreduce
from eventq import EventQueue
from message_buffer import MessageBuffer

logger = logging.getLogger(__name__)


def cleanup(args):
    cmd = 'mkdir ' + args.outdir + '/layer_wise'
    os.system(cmd)

    cmd = 'mv ' + args.outdir + '/*sram* ' + args.outdir + '/layer_wise'
    os.system(cmd)

    cmd = 'mv ' + args.outdir + '/*dram* ' + args.outdir + '/layer_wise'
    os.system(cmd)

    if args.dump == False:
        cmd = 'rm -rf ' + args.outdir + '/layer_wise'
        os.system(cmd)


def init():
    parser = argparse.ArgumentParser()

    parser.add_argument('--arch-config', default='{}/src/SCALE-Sim/configs/google.cfg'.format(os.environ['SIMHOME']),
                        help='accelerator architecture file, '
                             'default=SCALE-Sim/configs/express_64.cfg')
    parser.add_argument('--num-hmcs', default=64, type=int,
                        help='number of hybrid memory cubes, default=16')
    parser.add_argument('--num-vaults', default=16, type=int,
                        help='number of vaults per hybrid memory cube')
    parser.add_argument('--mini-batch-size', default=1024, type=int,
                        help='number of mini batch size for all hmc accelerator, distributed to all vault npu of each accelerator')
    parser.add_argument('--network', default='SCALE-Sim/topologies/mlperf/Resnet152.csv',
                        help='neural network architecture topology file, '
                             'default=SCALE-Sim/topologies/conv_nets/Googlenet.csv')
    parser.add_argument('--run-name', default='bb',
                        help='naming for this experiment run, default is empty')
    parser.add_argument('-d', '--outdir', default='{}/results/mesh_logs'.format(os.environ['SIMHOME']),
                        help='naming for the output directory, default is empty')
    parser.add_argument('--dump', default=False, action='store_true',
                        help='dump memory traces, default=False')
    parser.add_argument('--allreduce', default='mesh_overlap_2d_1',
                        help='allreduce shedule (multitree|mxnettree|ring|dtree|hdrm|ring2d|mesh_fermat|mesh_overlap_2d_1|mesh_overlap_2d_2|ring_odd|ring_bi|ring_odd_bi|ring2dn), default=multitree')
    parser.add_argument('--booksim-config',
                        default='{}/src/booksim2/runfiles/mesh/anynet_mesh_64_200.cfg'.format(os.environ['SIMHOME']),
                        required=False,
                        help='required config file for booksim')
    parser.add_argument('--booksim-network', default='mesh',
                        help='network topology (torus|mesh|bigraph|mesh_fermat), default is torus')
    parser.add_argument('-v', '--verbose', default=True, action='store_true',
                        help='Set the log level to debug, printing out detailed messages during execution.')
    parser.add_argument('--only-compute', default=False, action='store_true',
                        help='Set the flag to only run training computation without allreduce')
    parser.add_argument('--only-allreduce', default=True, action='store_true',
                        help='Set the flag to only run allreduce communication')
    parser.add_argument('--only-reduce-scatter', default=False, action='store_true',
                        help='Set the flag to only run reduce-scatter communication')
    parser.add_argument('--only-all-gather', default=False, action='store_true',
                        help='Set the flag to only run all-gather communication')
    parser.add_argument('--message-buffer-size', default=32, type=int,
                        help='message buffer size, default is 0 (infinite)')
    parser.add_argument('--message-size', default=8192, type=int,
                        help='size of a message, default is 256 bytes, 0 means treat the whole chunk of gradients as a message')
    parser.add_argument('--sub-message-size', default=8192, type=int,
                        help='size of a sub message, default is 256 bytes')
    parser.add_argument('--synthetic-data-size', default=0, type=int,
                        help='synthetic data size in number of parameters, default is 0 (run model)')
    parser.add_argument('--flits-per-packet', default=16, type=int,
                        help='Number of payload flits per packet, packet header is not considered here, that will be added in booksim')
    parser.add_argument('--bandwidth', default=200, type=int,
                        help='On chip BW between chiplets')
    parser.add_argument('--load-tree', default=False, action='store_true',
                        help='Whether just build tree or run full simulation')
    # parser.add_argument('--total-full-trees', default=2, type=int,
    #                     help='Total number of full trees in mesh')
    # parser.add_argument('--total-partial-trees', default=1, type=int,
    #                     help='Total number of partial trees in mesh')
    parser.add_argument('-k', '--kary', default=5, type=int,
                        help='generay kary allreduce trees, default is 2 (binary)')
    parser.add_argument('--radix', default=4, type=int,
                        help='node radix connected to router (end node NIs), default is 4')
    parser.add_argument('--chunk-size', default=0, type=int, help='Chunk size for overlapped algorithm')
    parser.add_argument('--strict-schedule', default=False, action='store_true',
                        help='strict schedule, default=False')
    parser.add_argument('--prioritize-schedule', default=False, action='store_true',
                        help='prioritize for arbitration to enforce schedule sequencing, default=False')
    parser.add_argument('--oracle-lockstep', default=False, action='store_true',
                        help='magic lockstep with zero overhead')
    parser.add_argument('--estimate-lockstep', default=False, action='store_true',
                        help='estimate message finished time based on data size to achieve lockstep')
    parser.add_argument('-l', '--enable-logger', default=[], action='append',
                        help='Enable logging for a specific module, append module name')
    parser.add_argument('--save-link-utilization', default=False, action='store_true',
                        help='Save link utilization info')
    parser.add_argument('--layer-by-layer', default=False, action='store_true',
                        help='Whether to simulate allreduce layer by layer or not')
    parser.add_argument('--layer-number', default="0_1_2_3_4", help='Layer number to simulate for the case of layer by layer allreduce')

    args = parser.parse_args()
    if args.flits_per_packet != 16:
        raise RuntimeError('Warnings: Flits per packet is not 16, be cautious with floating point calculation')
    args.latency = math.ceil((args.message_size * 8 / args.flits_per_packet) / args.bandwidth)
    args.per_message_time = args.latency * (args.flits_per_packet + 1)
    args.layer_number_list = [int(i) for i in args.layer_number.split('_')]

    for scope in args.enable_logger:
        debug_logger = logging.getLogger(scope)
        debug_logger.setLevel(logging.DEBUG)

    config = cp.ConfigParser()
    config.read(args.arch_config)

    if not args.run_name:
        args.run_name = config.get('general', 'run_name')

    net_name = args.network.split('/')[-1].split('.')[0]
    if not args.run_name:
        args.run_name = net_name + args.data_flow

    config_name = args.arch_config.split('/')[-1].split('.')[0]

    path = './outputs/' + args.run_name
    args.outdir = '{}/outputs/{}'.format(args.outdir, args.run_name)

    arch_sec = 'architecture_presets'

    args.pe_array_height = int(config.get(arch_sec, 'ArrayHeight'))
    args.pe_array_width = int(config.get(arch_sec, 'ArrayWidth'))

    args.ifmap_sram_size = int(config.get(arch_sec, 'IfmapSramSz')) << 10  # * 1024
    args.filter_sram_size = int(config.get(arch_sec, 'FilterSramSz')) << 10  # * 1024
    args.ofmap_sram_size = int(config.get(arch_sec, 'OfmapSramSz')) << 10  # * 1024

    args.ifmap_offset = int(config.get(arch_sec, 'IfmapOffset'))
    args.filter_offset = int(config.get(arch_sec, 'FilterOffset'))
    args.ofmap_offset = int(config.get(arch_sec, 'OfmapOffset'))
    args.ifmap_grad_offset = int(config.get(arch_sec, 'IfmapGradOffset'))
    args.filter_grad_offset = int(config.get(arch_sec, 'FilterGradOffset'))
    args.ofmap_grad_offset = int(config.get(arch_sec, 'OfmapGradOffset'))

    args.data_flow = config.get(arch_sec, 'Dataflow')

    # Create output directory
    if args.dump:
        if not os.path.exists(args.outdir):
            os.system('mkdir -p {}'.format(args.outdir))
        elif os.path.exists(args.outdir):
            t = time.time()
            old_path = args.outdir + '_' + str(t)
            os.system('mv ' + args.outdir + ' ' + old_path)

    logger.info("====================================================")
    logger.info("******************* SCALE SIM **********************")
    logger.info("====================================================")
    logger.info("Array Size:    {} x {}".format(args.pe_array_height, args.pe_array_width))
    logger.info("SRAM IFMAP:    {}".format(args.ifmap_sram_size))
    logger.info("SRAM Filter:   {}".format(args.filter_sram_size))
    logger.info("SRAM OFMAP:    {}".format(args.ofmap_sram_size))
    logger.info("CSV file path: {}".format(args.network))
    logger.info("Dataflow:      {}".format(args.data_flow))
    logger.info("====================================================\n")

    global_eventq = EventQueue()

    model = Model(args)

    if args.outdir:
        args.logdir = args.outdir
    else:
        logpath = '{}/results/logs'.format(os.environ['SIMHOME'])
        args.logdir = logpath
    os.system('mkdir -p {}'.format(args.outdir))
    link_utilization_file = None
    if args.layer_by_layer:
        if args.allreduce == 'mesh_fermat' or args.allreduce == 'mesh_overlap_2d_1':
            logfile = '{}/{}_{}_{}_{}_{}_{}_layer_{}.log'.format(args.logdir, args.run_name, args.allreduce, args.num_hmcs,
                                                        args.booksim_network, net_name, config_name, args.layer_number)
            jsonfile = '{}/{}_{}_{}_{}_{}_{}_layer_{}.json'.format(args.logdir, args.run_name, args.allreduce, args.num_hmcs,
                                                          args.booksim_network, net_name, config_name, args.layer_number)
            link_utilization_file = '{}/{}_{}_{}_{}.pkl'.format(args.logdir, args.run_name, args.allreduce,
                                                                args.num_hmcs,
                                                                args.booksim_network)
        else:
            logfile = '{}/{}_{}_{}_{}_{}_{}_{}_layer_{}.log'.format(args.logdir, args.run_name, args.allreduce, args.num_hmcs,
                                                           args.booksim_network, args.bandwidth, net_name, config_name, args.layer_number)
            jsonfile = '{}/{}_{}_{}_{}_{}_{}_{}_layer_{}.json'.format(args.logdir, args.run_name, args.allreduce, args.num_hmcs,
                                                             args.booksim_network, args.bandwidth, net_name,
                                                             config_name, args.layer_number)
            link_utilization_file = '{}/{}_{}_{}_{}_{}.pkl'.format(args.logdir, args.run_name, args.allreduce,
                                                                   args.num_hmcs,
                                                                   args.booksim_network, args.bandwidth)
    elif args.only_compute:
        if args.allreduce == 'mesh_fermat' or args.allreduce == 'mesh_overlap_2d_1':
            logfile = '{}/{}_{}_{}_{}_{}_{}.log'.format(args.logdir, args.run_name, args.allreduce, args.num_hmcs,
                                                        args.booksim_network, net_name, config_name)
            jsonfile = '{}/{}_{}_{}_{}_{}_{}.json'.format(args.logdir, args.run_name, args.allreduce, args.num_hmcs,
                                                          args.booksim_network, net_name, config_name)
            link_utilization_file = '{}/{}_{}_{}_{}.pkl'.format(args.logdir, args.run_name, args.allreduce, args.num_hmcs,
                                                    args.booksim_network)
        else:
            logfile = '{}/{}_{}_{}_{}_{}_{}_{}.log'.format(args.logdir, args.run_name, args.allreduce, args.num_hmcs,
                                                     args.booksim_network, args.bandwidth, net_name, config_name)
            jsonfile = '{}/{}_{}_{}_{}_{}_{}_{}.json'.format(args.logdir, args.run_name, args.allreduce, args.num_hmcs,
                                                       args.booksim_network, args.bandwidth, net_name, config_name)
            link_utilization_file = '{}/{}_{}_{}_{}_{}.pkl'.format(args.logdir, args.run_name, args.allreduce, args.num_hmcs,
                                                       args.booksim_network, args.bandwidth)
    else:
        if args.allreduce == 'mesh_fermat' or args.allreduce == 'mesh_overlap_2d_1':
            logfile = '{}/{}_{}_{}_{}.log'.format(args.logdir, args.run_name, args.allreduce, args.num_hmcs,
                                                        args.booksim_network)
            jsonfile = '{}/{}_{}_{}_{}.json'.format(args.logdir, args.run_name, args.allreduce, args.num_hmcs,
                                                          args.booksim_network)
            link_utilization_file = '{}/{}_{}_{}_{}.pkl'.format(args.logdir, args.run_name, args.allreduce, args.num_hmcs,
                                                    args.booksim_network)
        else:
            logfile = '{}/{}_{}_{}_{}_{}.log'.format(args.logdir, args.run_name, args.allreduce, args.num_hmcs,
                                                     args.booksim_network, args.bandwidth)
            jsonfile = '{}/{}_{}_{}_{}_{}.json'.format(args.logdir, args.run_name, args.allreduce, args.num_hmcs,
                                                       args.booksim_network, args.bandwidth)
            link_utilization_file = '{}/{}_{}_{}_{}_{}.pkl'.format(args.logdir, args.run_name, args.allreduce, args.num_hmcs,
                                                       args.booksim_network, args.bandwidth)

    # if os.path.exists(logfile) or os.path.exists(jsonfile):
    #     raise RuntimeError('Warn: {} or {} already existed, may overwritten'.format(logfile, jsonfile))

    if args.verbose:
        logging.basicConfig(filename=logfile, format='%(message)s', level=logging.DEBUG)
    else:
        logging.basicConfig(filename=logfile, format='%(message)s', level=logging.INFO)

    logger.info('NN model size: {} parameters'.format(model.size))
    if args.allreduce == 'mesh_fermat':
        args.per_dim_nodes = int(math.sqrt(args.num_hmcs))
        if args.chunk_size == 0:
            args.chunk_size = max(args.pe_array_height, args.pe_array_width) * args.num_vaults * args.num_hmcs
        # total_possible_trees_in_both_dims = math.ceil(model.size / args.chunk_size)
        # total_possible_trees_in_both_dims = 12
        args.total_chunks = math.ceil(model.size / args.chunk_size)
        args.per_dim_chunks = math.ceil(args.total_chunks / 2)
        # total_possible_trees = args.per_dim_chunks
        args.total_multiplied = math.floor(args.per_dim_nodes / 2)
        if args.per_dim_chunks >= args.total_multiplied:
            args.total_full_trees = args.total_multiplied
        else:
            args.total_full_trees = args.per_dim_chunks
        # if args.per_dim_chunks < 2 * args.total_full_trees:
        #     raise RuntimeError("Total number of chunks are less than 2 * total full trees")
        args.total_partial_trees = args.per_dim_chunks - args.total_full_trees
        args.full_tree_message_rs1 = math.ceil(
            (args.chunk_size * 4) / (args.message_size * args.per_dim_nodes))
        args.full_tree_message_rs2_for_full_tree = math.ceil(
            (args.chunk_size * 4) / (args.message_size * args.per_dim_nodes * args.per_dim_nodes))
        args.full_tree_message_rs2_for_partial_tree = math.ceil(
            (args.chunk_size * 4) / (args.message_size * args.per_dim_nodes * 2))
        args.partial_tree_message_rs1 = math.ceil(
            (args.chunk_size * 4) / (args.message_size * 2))
        args.partial_tree_message_rs2 = math.ceil(
            (args.chunk_size * 4) / (args.message_size * 4))
        logger.info('Per dim nodes {}'.format(args.per_dim_nodes))
        logger.info('Chunk size {}'.format(args.chunk_size))
        logger.info('Total chunks {}'.format(args.total_chunks))
        logger.info('Per dim chunks {}'.format(args.per_dim_chunks))
        logger.info('Total multiplied {}'.format(args.total_multiplied))
        logger.info('Total full trees {}'.format(args.total_full_trees))
        logger.info('Total partial trees {}'.format(args.total_partial_trees))
        logger.info('RS1 message in full tree {}'.format(args.full_tree_message_rs1))
        logger.info('RS2 message in full tree for full tree {}'.format(args.full_tree_message_rs2_for_full_tree))
        logger.info('RS2 message in full tree for partial tree {}'.format(args.full_tree_message_rs2_for_partial_tree))
        logger.info('RS1 message in partial tree {}'.format(args.partial_tree_message_rs1))
        logger.info('RS2 message in partial tree {}'.format(args.partial_tree_message_rs2))
    elif args.allreduce == 'mesh_overlap_2d_1':
        if args.chunk_size == 0:
            if config_name == 'express_8':
                args.chunk_size = max(args.pe_array_height, args.pe_array_width) * args.num_vaults * 3 * 2 * 16
            elif config_name == 'express_16':
                args.chunk_size = max(args.pe_array_height, args.pe_array_width) * args.num_vaults * 3 * 2 * 8
            elif config_name == 'express':
                args.chunk_size = max(args.pe_array_height, args.pe_array_width) * args.num_vaults * 3 * 2 * 4
            elif config_name == 'express_64':
                args.chunk_size = max(args.pe_array_height, args.pe_array_width) * args.num_vaults * 3 * 2 * 2
            elif config_name == 'express_128':
                args.chunk_size = max(args.pe_array_height, args.pe_array_width) * args.num_vaults * 3 * 2
            elif config_name == 'google':
                args.chunk_size = max(args.pe_array_height, args.pe_array_width) * args.num_vaults * 3 * 2
            else:
                args.chunk_size = math.ceil((args.message_size * 3) / 4)
        if args.layer_by_layer:
            if model.size < args.chunk_size:
                args.chunk_size = model.size
        args.total_partial_trees = math.ceil(model.size / args.chunk_size)
        args.partial_tree_message = math.ceil((args.chunk_size * 4) / (args.message_size * 3)) # Here 3 is due to number of trees in TTO
        logger.info('Chunk size {}'.format(args.chunk_size))
        logger.info('Total partial trees {}'.format(args.total_partial_trees))
        logger.info('Message in partial tree {}'.format(args.partial_tree_message))

    network = BookSim(args, global_eventq)
    if args.message_buffer_size == 0:
        inject_buf_size = network.booksim.GetInjectBufferSize()
        msg_buf_size = network.booksim.GetMessageBufferSize()
        if inject_buf_size != 0 or msg_buf_size != 0:
            raise RuntimeError('Message buffer is set to 0 (infinite) here,'
                               ' but message buffer size and inject buffer size are set'
                               'to {} and {} in booksim config (should set to 0 for'
                               ' infinite)'.format(msg_buf_size, inject_buf_size))

    allreduce = construct_allreduce(args)
    allreduce.compute_schedule(args.kary, verbose=args.verbose)
    assert not (args.only_compute and args.only_allreduce)

    if args.allreduce == 'dtree' and args.num_hmcs % 2 != 0:
        args.num_hmcs = args.num_hmcs - 1

    if args.allreduce == 'mesh_fermat' or args.allreduce == 'mesh_overlap_2d_1':
        link_dict = {}
        messages_sent = {}
        sending = {}
        for i in range(args.num_hmcs):
            messages_sent[i] = [0] * args.radix
            sending[i] = [None for j in range(args.radix)]
            link_dict[i] = {}
            for key in allreduce.reduce_scatter_schedule[i].keys():
                link_dict[i][key] = False
        allreduce.link_dict = link_dict
        allreduce.messages_sent = messages_sent
        allreduce.sending = sending

    optimal_messages_sent = []
    optimal_sending = []
    optimal_free_nis = []
    for i in range(args.num_hmcs):
        optimal_messages_sent.append([0] * args.radix)
        optimal_sending.append([None for i in range(args.radix)])
        optimal_free_nis.append(set([i for i in range(args.radix)]))

    hmcs = []
    from_network_message_buffers = []
    to_network_message_buffers = []
    for i in range(args.num_hmcs):
        if args.allreduce == 'mesh_fermat' or args.allreduce == 'mesh_overlap_2d_1':
            hmcs.append(HMC_fermat(i, args, global_eventq))
        else:
            hmcs.append(HMC(i, args, global_eventq))
        hmcs[i].load_model(model)
        hmcs[i].startup()
        # connect with network
        from_network_message_buffers.append([])
        to_network_message_buffers.append([])
        for j in range(args.radix):
            from_network_message_buffers[i].append(
                MessageBuffer('from_network_node{}_ni{}'.format(i, j), args.message_buffer_size))
            to_network_message_buffers[i].append(
                MessageBuffer('to_network_node{}_ni{}'.format(i, j), args.message_buffer_size))
            from_network_message_buffers[i][j].set_consumer(hmcs[i])
            to_network_message_buffers[i][j].set_consumer(network)
        hmcs[i].set_message_buffers(from_network_message_buffers[i],
                                    to_network_message_buffers[i])
        # hmcs[i].set_optimal_params(optimal_messages_sent, optimal_sending, optimal_free_nis)
        hmcs[i].set_allreduce(allreduce)

    network.set_message_buffers(to_network_message_buffers,
                                from_network_message_buffers)
    # network.set_parameters(allreduce.reduce_scatter_time_track_dict, allreduce.all_gather_time_track_dict, hmcs[0].per_message_max_latency, allreduce.network.links_usage, allreduce.network.link_start_times, allreduce.network.link_end_times)

    # return args, global_eventq, model, hmcs, network, allreduce.trees, allreduce.timesteps, logfile, jsonfile, link_utilization_file
    return args, global_eventq, model, hmcs, network, jsonfile, link_utilization_file


def do_sim_loop(eventq):
    while not eventq.empty():
        cur_cycle, events = eventq.next_events()

        for event in events:
            event.process(cur_cycle)


def compute_link_idle_cycles(link_start_times, link_end_times, start, end):
    final_unused_cycles = set(list(range(start, end)))
    for key in link_start_times.keys():
        used_cycles = get_used_cycles(link_start_times[key], link_end_times[key], start)
        final_unused_cycles = final_unused_cycles - used_cycles
    return len(final_unused_cycles)


def get_used_cycles(link_start_time, link_end_time, current_last):
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


def main():
    start_time = time.time()
    # args, global_eventq, model, hmcs, network, trees, timesteps, logfile_name, jsonfile_name, link_utilization_file = init()
    args, global_eventq, model, hmcs, network, jsonfile_name, link_utilization_file = init()
    # if args.allreduce == 'multitree' and not args.load_tree:
    #     print("Returned")
    #     return
    do_sim_loop(global_eventq)

    logger.debug('booksim network idle? {}'.format(network.booksim.Idle()))
    for i, hmc in enumerate(hmcs):
        logger.debug('HMC {}:'.format(i))
        logger.debug('   reduce-scatter-schedule:')
        for schedule in hmc.reduce_scatter_schedule:
            logger.debug('       {}'.format(schedule))
        logger.debug('   all-gather-schedule:')
        for schedule in hmc.all_gather_schedule:
            logger.debug('       {}'.format(schedule))
        logger.debug('   from network message buffers:')
        for i, message_buffer in enumerate(hmc.from_network_message_buffers):
            logger.debug('       {}-{}: has {} messages'.format(i, message_buffer.name, message_buffer.size))
        logger.debug('   to network message buffers:')
        for i, message_buffer in enumerate(hmc.to_network_message_buffers):
            logger.debug('       {}-{}: has {} messages'.format(i, message_buffer.name, message_buffer.size))

    compute_cycles = hmcs[0].compute_cycles
    allreduce_compute_cycles = 0
    for hmc in hmcs:
        if allreduce_compute_cycles < hmc.allreduce_compute_cycles:
            allreduce_compute_cycles = hmc.allreduce_compute_cycles
    cycles = global_eventq.cycles
    # pure_allreduce_compute_cycles = cycles
    # for hmc in hmcs:
    #     if hmc.just_allreduce_compute_cycles < pure_allreduce_compute_cycles:
    #         pure_allreduce_compute_cycles = hmc.just_allreduce_compute_cycles
    allreduce_cycles = cycles - compute_cycles
    pure_communication_cycles = allreduce_cycles - allreduce_compute_cycles
    # computation_communication_overlap = allreduce_compute_cycles - pure_allreduce_compute_cycles
    # pure_allreduce_compute_cycles_2 = compute_link_idle_cycles(network.link_start_times, network.link_end_times, compute_cycles + 1, cycles)
    # computation_communication_overlap_2 = allreduce_compute_cycles - pure_allreduce_compute_cycles_2

    # TODO: workaround to reduce simulation time for one-shot training,
    #       need to change for layer-wise training
    if args.allreduce == 'ring2d':
        cycles *= 2
        allreduce_cycles *= 2
        pure_communication_cycles *= 2

    compute_percentile = compute_cycles / cycles * 100
    allreduce_percentile = allreduce_cycles / cycles * 100
    allreduce_compute_percentile = allreduce_compute_cycles / cycles * 100
    # pure_communication_percentile = allreduce_percentile - allreduce_compute_percentile
    # pure_allreduce_compute_percentile = pure_allreduce_compute_cycles / cycles * 100
    # computation_communication_overlap_percentile = computation_communication_overlap / cycles * 100
    # pure_allreduce_compute_percentile_2 = pure_allreduce_compute_cycles_2 / cycles * 100
    # computation_communication_overlap_percentile_2 = computation_communication_overlap_2 / cycles * 100

    # logger.info('\n======== Link Utilization ========')
    # for key in network.link_start_times.keys():
    #     logger.info("Link " + str(key) + ":")
    #     logger.info("Start time " + str(network.link_start_times[key]))
    #     logger.info("End time " + str(network.link_end_times[key]))
    #
    if args.save_link_utilization:
        save_object = {}
        save_object['link_start_time'] = network.link_start_times
        save_object['link_end_time'] = network.link_end_times
        pickle.dump(save_object, open(link_utilization_file, "wb"))

    logger.info('\n======== Simulation Summary ========')
    logger.info('Training epoch runtime: {} cycles'.format(cycles))
    logger.info(' - computation: {} cycles ({:.2f}%)'.format(compute_cycles, compute_percentile))
    logger.info(' - allreduce: {} cycles ({:.2f}%)'.format(allreduce_cycles, allreduce_percentile))
    # logger.info('     - overlapped computation: {} cycles ({:.2f}%)'.format(allreduce_compute_cycles, allreduce_compute_percentile))
    # logger.info('     - overlapped computation communication: {} cycles ({:.2f}%)'.format(computation_communication_overlap, computation_communication_overlap_percentile))
    # logger.info('     - overlapped computation communication 2: {} cycles ({:.2f}%)'.format(computation_communication_overlap_2, computation_communication_overlap_percentile_2))
    # logger.info('     - pure communication: {} cycles ({:.2f}%)'.format(pure_communication_cycles, pure_communication_percentile))
    # logger.info('     - pure allreduce computation: {} cycles ({:.2f}%)'.format(pure_allreduce_compute_cycles, pure_allreduce_compute_percentile))
    # logger.info('     - pure allreduce computation 2: {} cycles ({:.2f}%)'.format(pure_allreduce_compute_cycles_2, pure_allreduce_compute_percentile_2))
    total_messages_sent = 0
    for i, hmc in enumerate(hmcs):
        logger.debug(' - HMC {} sends {} messages'.format(i, hmc.total_messages_sent))
        total_messages_sent += hmc.total_messages_sent
    logger.info('Total number of messages: {}\n'.format(total_messages_sent))

    assert network.booksim.Idle()
    for i, hmc in enumerate(hmcs):
        if args.only_reduce_scatter:
            assert len(hmc.pending_aggregations) == 0
            assert len(hmc.reduce_scatter_schedule) == 0
        elif args.only_all_gather:
            assert len(hmc.all_gather_schedule) == 0
        elif not args.only_compute:
            assert len(hmc.pending_aggregations) == 0
            if args.allreduce == 'mesh_fermat' or args.allreduce == 'mesh_overlap_2d_1':
                all_ag_done = True
                for key in hmc.link_dict[hmc.id].keys():
                    if len(hmc.all_gather_schedule[key]) > 0:
                        all_ag_done = False
                        break
                assert all_ag_done is True
                all_rs_done = True
                for key in hmc.link_dict[hmc.id].keys():
                    if len(hmc.reduce_scatter_schedule[key]) > 0:
                        all_rs_done = False
                        break
                assert all_rs_done is True
            else:
                assert len(hmc.reduce_scatter_schedule) == 0
                assert len(hmc.all_gather_schedule) == 0
        for i, message_buffer in enumerate(hmc.from_network_message_buffers):
            assert message_buffer.size == 0
        for i, message_buffer in enumerate(hmc.to_network_message_buffers):
            assert message_buffer.size == 0

    if args.dump:
        cleanup(args)

    # dump configuration and results
    sim = {}
    sim['configuration'] = vars(args)
    sim['results'] = {}

    sim['results']['performance'] = {}
    sim['results']['performance']['training'] = compute_cycles
    sim['results']['performance']['training_by_layer'] = hmcs[0].back_time
    sim['results']['performance']['allreduce'] = {}
    sim['results']['performance']['allreduce']['computation'] = allreduce_compute_cycles
    sim['results']['performance']['allreduce']['pure_communication'] = pure_communication_cycles
    # sim['results']['performance']['allreduce']['pure_computation'] = pure_allreduce_compute_cycles
    # sim['results']['performance']['allreduce']['overlap'] = computation_communication_overlap
    # sim['results']['performance']['allreduce']['pure_computation_2'] = pure_allreduce_compute_cycles_2
    # sim['results']['performance']['allreduce']['overlap_2'] = computation_communication_overlap_2
    sim['results']['performance']['allreduce']['total'] = allreduce_cycles
    sim['results']['performance']['total'] = cycles

    network.booksim.CalculatePower()
    net_dyn_power = network.booksim.GetNetDynPower()
    net_leak_power = network.booksim.GetNetLeakPower()
    router_dyn_power = network.booksim.GetRouterDynPower()
    router_leak_power = network.booksim.GetRouterLeakPower()
    link_dyn_power = network.booksim.GetLinkDynPower()
    link_leak_power = network.booksim.GetLinkLeakPower()
    net_link_activities = network.booksim.GetNetLinkActivities()
    # TODO: workaround to reduce simulation time for one-shot training
    #       need to change for layer-wise training
    if args.allreduce == 'ring2d':
        net_link_activities *= 2

    sim['results']['power'] = {}
    sim['results']['power']['network'] = {}
    sim['results']['power']['network']['dynamic'] = net_dyn_power
    sim['results']['power']['network']['static'] = net_leak_power
    sim['results']['power']['network']['total'] = net_dyn_power + net_leak_power
    sim['results']['power']['network']['router'] = {}
    sim['results']['power']['network']['router']['dynamic'] = router_dyn_power
    sim['results']['power']['network']['router']['static'] = router_leak_power
    sim['results']['power']['network']['router']['total'] = router_dyn_power + router_leak_power
    sim['results']['power']['network']['link'] = {}
    sim['results']['power']['network']['link']['dynamic'] = link_dyn_power
    sim['results']['power']['network']['link']['static'] = link_leak_power
    sim['results']['power']['network']['link']['total'] = link_dyn_power + link_leak_power
    sim['results']['power']['network']['link']['flits'] = net_link_activities

    with open(jsonfile_name, 'w') as simfile:
        json.dump(sim, simfile, indent=4)
        simfile.close()
    # enforce_ordering = "order_false"
    # immediate_aggregation = "imm_false"
    # if args.enforce_ordering:
    #     enforce_ordering = "order_true"
    # if args.immediate_aggregation:
    #     immediate_aggregation = "imm_true"
    # timed_treepath = '{}/src/Timed_Trees/{}'.format(os.environ['SIMHOME'], args.tree_type)
    # timed_tree_name = '{}/{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(timed_treepath, args.run_name, args.allreduce, args.bw_type,
    #                                                            args.num_hmcs, args.total_dimensions, args.nodes_in_dimension_list,
    #                                                            args.topology_in_dimension_list, args.bandwidth_list,
    #                                                            args.multiplier_calculator, '_'.join(str(e) for e in args.latency_multiplier),
    #                                                            args.booksim_network, enforce_ordering, immediate_aggregation)
    # stall_in_tree = False
    # if args.bw_type == 'heterogeneous' and (
    #         args.tree_type == 'optimal' or args.tree_type == 'base-3-shift-conflict'):
    #     stall_in_tree = True
    # draw_time_tree(timesteps, trees, args.num_hmcs, logfile_name, timed_tree_name, stall_in_tree)
    logger.info('Simulation Done \n')
    logger.info("Total time " + str(time.time() - start_time))


def reduce_scatter_start_components(line):
    splitted_string = line.split()
    start_time = splitted_string[0]
    child = splitted_string[2].split('-')[1]
    flow = splitted_string[8]
    parent = splitted_string[14].split('-')[1]
    return int(child), int(parent), int(flow), int(start_time)


def reduce_scatter_end_components(line):
    splitted_string = line.split()
    start_time = splitted_string[0]
    parent = splitted_string[2].split('-')[1]
    flow = splitted_string[9]
    child = splitted_string[-1].split('-')[1]
    return int(child), int(parent), int(flow), int(start_time)


def all_gather_start_components(line):
    splitted_string = line.split()
    start_time = splitted_string[0]
    parent = splitted_string[2].split('-')[1]
    flow = splitted_string[8]
    child = splitted_string[14].split('-')[1]
    return int(child), int(parent), int(flow), int(start_time)


def all_gather_end_components(line):
    splitted_string = line.split()
    start_time = splitted_string[0]
    child = splitted_string[2].split('-')[1]
    flow = splitted_string[9]
    parent = splitted_string[-1].split('-')[1]
    return int(child), int(parent), int(flow), int(start_time)


def draw_time_tree(timesteps, trees, nodes, logfile_name, timed_tree_name, stall_in_tree):
    file1 = open(logfile_name, 'r')
    lines = file1.readlines()

    reduce_scatter_start_hmap = {}
    reduce_scatter_end_hmap = {}
    all_gather_start_hmap = {}
    all_gather_end_hmap = {}
    for i in range(nodes):
        reduce_scatter_start_hmap[i] = {}
        reduce_scatter_end_hmap[i] = {}
        all_gather_start_hmap[i] = {}
        all_gather_end_hmap[i] = {}

    for line in lines:
        if 'start reducing for flow' in line:
            child, parent, flow, start_time = reduce_scatter_start_components(line)
            reduce_scatter_start_hmap[flow][child, parent] = start_time
        elif 'receives full reduce for flow' in line:
            child, parent, flow, start_time = reduce_scatter_end_components(line)
            reduce_scatter_end_hmap[flow][child, parent] = start_time
        elif 'start gathering for flow' in line:
            child, parent, flow, start_time = all_gather_start_components(line)
            all_gather_start_hmap[flow][child, parent] = start_time
        elif 'receives full gather for flow' in line:
            child, parent, flow, start_time = all_gather_end_components(line)
            all_gather_end_hmap[flow][child, parent] = start_time

    new_trees = {}
    for tree_id in trees:
        tree = trees[tree_id]
        new_tree = []
        for edge in tree:
            child = edge[0]
            parent = edge[1]
            timestep = edge[2]
            distance = None
            if stall_in_tree:
                distance = edge[3]
            reduce_scatter_start = reduce_scatter_start_hmap[tree_id][child, parent]
            reduce_scatter_end = reduce_scatter_end_hmap[tree_id][child, parent]
            all_gather_start = all_gather_start_hmap[tree_id][child, parent]
            all_gather_end = all_gather_end_hmap[tree_id][child, parent]
            if not stall_in_tree:
                new_tree.append((child, parent, timestep, reduce_scatter_start, reduce_scatter_end, all_gather_start,
                                 all_gather_end))
            else:
                new_tree.append(
                    (child, parent, timestep, distance, reduce_scatter_start, reduce_scatter_end, all_gather_start,
                     all_gather_end))
        new_trees[tree_id] = new_tree
    if not stall_in_tree:
        generate_timed_per_tree_dotfile(nodes, timesteps, new_trees, timed_tree_name)
    else:
        generate_timed_per_tree_dotfile_with_stall(nodes, timesteps, new_trees, timed_tree_name)


def generate_timed_per_tree_dotfile(nodes, timesteps, new_trees, filename):
    cmd = 'mkdir ' + filename
    os.system(cmd)

    # color palette for ploting nodes of different tree levels
    colors = ['#f7f4f9', '#e7e1ef', '#d4b9da', '#c994c7', '#df65b0',
              '#e7298a', '#ce1256', '#980043', '#67001f']

    header = 'digraph tree {\n'
    header += '  rankdir = BT;\n'
    header += '  subgraph {\n'

    # group nodes with same rank (same tree level/iteration)
    # and set up the map for node name and its rank in node_rank
    ranks = {}
    node_rank = {}
    trees = {}
    for root in range(nodes):
        ranks[root] = {}
        node_rank[root] = {}
        for rank in range(timesteps + 1):
            ranks[root][rank] = []

    for root in range(nodes):
        minrank = timesteps
        for edge in new_trees[root]:
            child = '"{}-{}"'.format(root, edge[0])
            rank = edge[2] + 1
            ranks[root][rank].append(child)
            node_rank[root][child] = rank
            if edge[1] == root and rank - 1 < minrank:
                minrank = rank - 1
        ranks[root][minrank].append('"{}-{}"'.format(root, root))
        node_rank[root]['"{}-{}"'.format(root, root)] = minrank

    for root in range(nodes):
        trees[root] = header + '    /* tree {} */\n'.format(root)
        for edge in new_trees[root]:
            child = '"{}-{}"'.format(root, edge[0])
            parent = '"{}-{}"'.format(root, edge[1])
            cycle = timesteps - edge[2]
            label = str(cycle) + "\n (" + str(edge[3]) + "-" + str(edge[4]) + ")\n(" + str(edge[5]) + "-" + str(
                edge[6]) + ")"
            minlen = node_rank[root][child] - node_rank[root][parent]  # for strict separation of ranks
            trees[root] += ''.join('    {} -> {} [ label="{}" minlen={} ];\n'.format(child, parent, label, minlen))

    for root in range(nodes):
        trees[root] += '    // note that rank is used in the subgraph\n'
        for rank in range(timesteps + 1):
            if ranks[root][rank]:
                level = '    {rank = same;'
                for node in ranks[root][rank]:
                    level += ' {};'.format(node)
                level += '}\n'
                trees[root] += level

        trees[root] += '    // node colors\n'
        style = '    {} [style="filled", fillcolor="{}"];\n'
        for rank in range(timesteps + 1):
            if ranks[root][rank]:
                trees[root] += ''.join(style.format(node, colors[rank % len(colors)]) for node in ranks[root][rank])

        trees[root] += '  } /* closing subgraph */\n'
        trees[root] += '}\n'

        f = open('{}/tree-{}.dot'.format(filename, root), 'w')
        f.write(trees[root])
        f.close()


def generate_timed_per_tree_dotfile_with_stall(nodes, timesteps, new_trees, filename):
    cmd = 'mkdir ' + filename
    os.system(cmd)

    # color palette for ploting nodes of different tree levels
    colors = ['#f7f4f9', '#e7e1ef', '#d4b9da', '#c994c7', '#df65b0',
              '#e7298a', '#ce1256', '#980043', '#67001f']

    header = 'digraph tree {\n'
    header += '  rankdir = BT;\n'
    header += '  subgraph {\n'

    # self.timesteps += 4

    # group nodes with same rank (same tree level/iteration)
    # and set up the map for node name and its rank in node_rank
    ranks = {}
    node_rank = {}
    trees = {}
    for root in range(nodes):
        ranks[root] = {}
        node_rank[root] = {}
        for rank in range(timesteps + 1):
            ranks[root][rank] = []

    for root in range(nodes):
        minrank = timesteps
        for edge in new_trees[root]:
            child = '"{}-{}"'.format(root, edge[0])
            rank = edge[2] + edge[3]
            ranks[root][rank].append(child)
            node_rank[root][child] = rank
            if edge[1] == root and rank - edge[3] < minrank:
                minrank = rank - edge[3]
        ranks[root][minrank].append('"{}-{}"'.format(root, root))
        node_rank[root]['"{}-{}"'.format(root, root)] = minrank

    for root in range(nodes):
        trees[root] = header + '    /* tree {} */\n'.format(root)
        for edge in new_trees[root]:
            child = '"{}-{}"'.format(root, edge[0])
            parent = '"{}-{}"'.format(root, edge[1])
            stall = '"{}-{}-{}-stall"'.format(root, edge[0], edge[1])
            cycle = timesteps - edge[2] - (edge[3] - 1)
            label = str(cycle) + "\n (" + str(edge[4]) + "-" + str(edge[5]) + "-" + str(
                edge[5] - edge[4]) + ")\n(" + str(edge[6]) + "-" + str(
                edge[7]) + "-" + str(edge[7] - edge[6]) + ")"
            minlen = node_rank[root][child] - node_rank[root][parent]  # for strict separation of ranks
            if minlen == edge[3]:
                trees[root] += ''.join(
                    '    {} -> {} [ label="{}({})" minlen={} ];\n'.format(child, parent, label, edge[3], minlen))
            elif minlen > edge[3]:
                trees[root] += ''.join(
                    '    {} -> {} [ label="{}({})" minlen={} ];\n'.format(child, stall, label, edge[3], edge[3]))
                trees[root] += ''.join(
                    '    {} -> {} [ label="{}({})" minlen={} style="dashed" ];\n'.format(stall, parent,
                                                                                         cycle + edge[3],
                                                                                         minlen - edge[3],
                                                                                         minlen - edge[3]))
            else:
                raise Exception("minlen should bot be less than edge weight")

    for root in range(nodes):
        trees[root] += '    // note that rank is used in the subgraph\n'
        for rank in range(timesteps + 1):
            if ranks[root][rank]:
                level = '    {rank = same;'
                for node in ranks[root][rank]:
                    level += ' {};'.format(node)
                level += '}\n'
                trees[root] += level

        trees[root] += '    // node colors\n'
        style = '    {} [style="filled", fillcolor="{}"];\n'
        for rank in range(timesteps + 1):
            if ranks[root][rank]:
                trees[root] += ''.join(style.format(node, colors[rank % len(colors)]) for node in ranks[root][rank])

        trees[root] += '  } /* closing subgraph */\n'
        trees[root] += '}\n'

        f = open('{}/tree-{}.dot'.format(filename, root), 'w')
        f.write(trees[root])
        f.close()


if __name__ == '__main__':
    main()
