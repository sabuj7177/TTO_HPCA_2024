# needs dimensions, number of chunks, number of npus in each dimesnion, bw of each dimension, topology of each dimension, scf/fifo

import math
import numpy as np
from tabulate import tabulate

# dimensions = 3
# chunks = 5
# npus = [4,8,16]
# bw = [3,4,1]
# topology = ["ring", "fc", "switch"]
# scf = 1

message_size = 8480
flits_per_packet = 16

def get_bw_ratio(bw, topology):
    bw_new = []
    for i in range(len(topology)):
        if topology[i] == "switch":
            bw_new.append(int(bw[i]/2))
        else:
            bw_new.append(bw[i])
    gcd = bw_new[0]
    for i in range(len(bw_new)-1):
        gcd = np.gcd(gcd, bw_new[i+1])
    ans = []
    for i in range(len(bw_new)):
        ans.append(int(bw_new[i]/gcd))
    return ans

def get_configuration_values(size):
    # d = int(input("\nEnter Number of Dimensions: "))
    # c = int(input("\nEnter Number of Chunks: "))
    d = 4
    c = 64
    # n = []
    # b = []
    # t = []
    # for i in range(d):
    #     n.append(int(input("\nEnter Number of NPUs in Dimension " + str(i) + " : ")))
    # for i in range(d):
    #     b.append(int(input("\nEnter Bandwidth in Dimension " + str(i) + " : ")))
    # for i in range(d):
    #     t.append(input("\nEnter Topology in Dimension " + str(i) + " (ring, switch, fc) : "))
    n = [4, 4, 8, 8]
    b = [2000, 1600, 800, 400]
    t = ['ring', 'switch', 'switch', 'switch']
    # s = int(input("\nIntra Dimension Schedule ( 1 if SCF , 0 if FIFO ): "))
    # size = int(input("\nSynthetic Data Size (Number of Parameters): "))
    s = 1
    # size = 268435456
    return [c, d, n, b, t, s, size]

# Predicts latency of current chunk based on chunksize, npus, bw and topology
def latency_predictor(datasize, current_npus, current_bw, current_topology):
    if current_topology == "ring":
        return (current_npus-1)*1/(current_npus)*datasize*(1/current_bw)
    elif current_topology == "fc":
        return (1)*1/(current_npus)*datasize*(1/current_bw)
    elif current_topology == "switch":
        return (current_npus-1)*1/(current_npus)*datasize*(1/current_bw)
    else:
        raise RuntimeError("TOPOLOGY UNDEFINED\nChoose one of these:ring,fc,switch\n")

# Inter Dimension Scheduler - THEMIS Algorithm
def inter_scheduler(C, chunk_id, dim_wise_load, inter_schedule, D, N, B, T):
    # index sorted : lowest load to highest load, start with highest bandwidth for 1st chunk
    if chunk_id == 0:
        sorted_schedule = np.argsort(B)[::-1]
    else:
        sorted_schedule = np.argsort(dim_wise_load)

    # If difference between max and min is less than threshold, then assign load in order of decreasing bandwidth. Threshold as mentioned in THEMIS paper.
    # We set the Threshold to be the estimated runtime (predicted by the Latency Model) when running an RS/AG of size chunkSize/16 on the dimension with the lowest current load.
    threshold = latency_predictor(1/(16*C), N[sorted_schedule[0]], B[sorted_schedule[0]], T[sorted_schedule[0]])
    if max(dim_wise_load)-min(dim_wise_load) < threshold:
        sorted_schedule = np.argsort(B)[::-1]
    
    # Update schedule for each dimension by going through the sorted list of dimensions
    chunk_size = 1/C
    ag = D*2
    for i in range(D):
        # dimension of current chunk:
        d = sorted_schedule[i]
        # returns load for dimension d according to topology, bw and no. of npus
        load = latency_predictor(chunk_size, N[d], B[d], T[d])
        inter_schedule[d].append([d, chunk_id, i+1, chunk_size, load, 0])   #RS
        inter_schedule[d].append([d, chunk_id, ag, chunk_size, load, 1])    #AG
        ag = ag-1
        load = 2*load
        chunk_size = chunk_size/N[d]
        # updates load list
        dim_wise_load[d] = dim_wise_load[d] + load
    return

# Intra Scheduler
def intra_scheduler(inter_schedule, S, C, D):
    output = []
    done = []
    current_t = []
    for d in range(D):
        output.append([])       # final schedule with both intra and inter algos
        current_t.append(0)     # current time in dimension
    for c in range(C):
        done.append([0,0])      # chunk c has done 0 timesteps and is busy till 0.
    
    chunks_left = True          # bool to see if need to stop the algorithm or not
    while chunks_left:
        all_empty = True        # bool to check if all dimension are empty
        for d in range(D):
            if len(inter_schedule[d])!=0:
                all_empty = False   # if any dim non empty change all_empty bool
                available = []
                # find all chunks whose previous ts has already been scheduled
                for chunk in inter_schedule[d]:
                    if done[chunk[1]][0]+1 == chunk[2]:
                        # if this chunk is next in the order of timesteps
                        # available has the available chunks and when it can schedule it at the earliest
                        if done[chunk[1]][1] < current_t[d]:
                            done[chunk[1]][1] = current_t[d]
                        available.append([chunk, done[chunk[1]][1]])
                # if none of the chunks available now, move on to another dimensions
                if len(available) == 0:
                    continue
                # get chunks with minimum wait time
                # print(available)
                min_wait = min(x[1] for x in available)
                min_wait_chunks = [x[0] for x in available if x[1] == min_wait]
                if S:
                    # choose chunk with lowest timeload
                    min_timeload = min(x[4] for x in min_wait_chunks)
                    min_timeload_chunks = [x for x in min_wait_chunks if x[4] == min_timeload]
                    next = min_timeload_chunks[0]
                    output[d].append(next)
                    done[next[1]][0] = done[next[1]][0] + 1
                    done[next[1]][1] = min_wait + next[4]
                    current_t[d] = done[next[1]][1]
                    inter_schedule[d].remove(next)
                else:
                    # choose first chunk available
                    next = min_wait_chunks[0]
                    output[d].append(next)
                    done[next[1]][0] = done[next[1]][0] + 1
                    done[next[1]][1] = min_wait + next[4]
                    current_t[d] = done[next[1]][1]
                    inter_schedule[d].remove(next)
        if all_empty:
            chunks_left = False     # all chunks have been intra scheduled in all dimensions
    # print(done)
    last_end = 0
    for c in range(C):
        last_end = done[c][1] if done[c][1] > last_end else last_end
    # print("ALL CHUNKS FINISH AT " + str(last_end))
    return output

# Scheduler
def get_themis_schedule(C, D, N, B, T, S):
    # dim_wise_load : load
    dim_wise_load = np.zeros(D, dtype = float)
    # inter dimension schedule found using THEMIS algorithm
    inter_schedule = []
    for d in range(D):
        inter_schedule.append([])
    for chunk_id in range(C):
        inter_scheduler(C, chunk_id, dim_wise_load, inter_schedule, D, N, B, T)
    # print(dim_wise_load)
    # inter_schedule has : dimension, chunkid, timestep, datasize, timeload
    intra_schedule = intra_scheduler(inter_schedule, S, C, D)
    return intra_schedule

def get_latencies(bw):
    latency_list = []
    for i in range(len(bw)):
        computed_latency = math.ceil((message_size * 8 / flits_per_packet) / bw[i])
        latency_list.append(computed_latency)
    return latency_list

def get_theoretical_analysis(schedule, topology, npus, latencies):
    per_dimension_time = []
    for d in range(len(schedule)):
        per_dimension_time.append(0)
        for chunk in schedule[d]:
            if topology[d] == "ring":
                # FOR RING : Time = (number of messages) X (Datachunk to be handled/n) X (n-1)
                per_dimension_time[d] = per_dimension_time[d] + (flits_per_packet+1)*latencies[d]*math.ceil(num_messages*chunk[3])*(npus[d]-1)/npus[d]
            if topology[d] == "switch":
                # FOR SWITCH : Time = (number of messages) X (Datasize to be handled/n) X (n-1)
                per_dimension_time[d] = per_dimension_time[d] + (flits_per_packet+1)*latencies[d]*math.ceil(num_messages*chunk[3])*(npus[d]-1)/npus[d]
            if topology[d] == "fc":
                # FOR FC : Time = (number of messages) X (Datasize to be handled/n) X 1
                per_dimension_time[d] = per_dimension_time[d] + (flits_per_packet+1)*latencies[d]*math.ceil(num_messages*chunk[3])*1/npus[d]
    return per_dimension_time


# get_themis_schedule(chunks, dimensions, npus, bw, topology, scf)
configuration = get_configuration_values()

chunks = configuration[0]
dimensions = configuration[1]
npus = configuration[2]
bw = get_bw_ratio(configuration[3], configuration[4])
topology = configuration[4]
scf = configuration[5]
latencies = get_latencies(configuration[3])
model_size = configuration[6]
num_messages = math.ceil((model_size * 4) /message_size)

f = open("result.txt", "w")
themis_schedule = get_themis_schedule(chunks, dimensions, npus, bw, topology, scf)

f.write("\n-------------------------------------------------------------------------------------------------------\n")
f.write("CONFIGURATIONS")
f.write("\n-------------------------------------------------------------------------------------------------------\n")
# f.write("\tChunks : " + str(chunks))
# f.write("\n\tDimensions : " + str(dimensions))
# f.write("\n\tNPUS : " + str(npus))
# f.write("\n\tBandwidths : " + str(configuration[3]))
# f.write("\n\tBandwidth Ratio : " + str(bw))
# f.write("\n\tTopology : " + str(topology))
# f.write("\n\tIntra-Dimension Scheduling : " + ("SCF" if scf==1 else "FIFO"))
# f.write("\n\tMessage Size : " + str(message_size))
# f.write("\n\tFlits per Packet : " + str(flits_per_packet))
# f.write("\n\tLatencies Calculated : " + str(get_latencies(configuration[3])) + "\n")

mydata = [["Chunks", str(chunks)],
          ["Dimensions", str(dimensions)],
          ["NPUS", str(npus)],
          ["Bandwidths", str(configuration[3])],
          ["Bandwidth Ratio", str(bw)],
          ["Topology", str(topology)],
          ["Intra-Dimension Scheduling", ("SCF" if scf==1 else "FIFO")],
          ["Message Size", str(message_size)],
          ["Flits per Packet", str(flits_per_packet)],
          ["Latencies Calculated", str(latencies)],
          ["Parameters in Model", str(model_size)],
          ["Number of messages in Model", str(num_messages)]]
f.write(tabulate(mydata, tablefmt="plain"))

f.write("\n\n-------------------------------------------------------------------------------------------------------\n")
f.write("ANALYTICAL TIME CALCULATION")
f.write("\n-------------------------------------------------------------------------------------------------------\n")

dim_wise_time = get_theoretical_analysis(themis_schedule, topology, npus, latencies)

f.write("Total Time : " + str(np.max(dim_wise_time)) + "\n\n")

for d in range(len(themis_schedule)):
    f.write("\tTime in Dimension " + str(d) + " : " + str(dim_wise_time[d]) + "\n")

f.write("\nRING Calculation :\n\tNumber of messages to be sent = (Number of messages to be handled in current dimension / n)\n\tTotal Time for either RS or AG = (Number of messages to be sent)*(n-1)\n\n")
f.write("SWITCH Calculation :\n\tTotal Time for either RS or AG = (Number of messages to be handled in current dimension)*(1/2 + 1/4 + 1/8 + ... + 1/2^logn)\n\tTotal Time for either RS or AG = (Number of messages to be handled in current dimension)*(n-1/n)\n\n")
f.write("FC Calculation :\n\tNumber of messages to be sent = (Number of messages to be handled in current dimension / n)\n\tTotal Time for either RS or AG = (Number of messages to be sent)*(1)")

f.write("\n\n-------------------------------------------------------------------------------------------------------\n")
# f.write("CHUNKS SCHEDULED BY THEMIS")
# f.write("\n-------------------------------------------------------------------------------------------------------\n")

# f.write("(dimension, chunkid, timestep for chunk, datasize, timeload, rs/ag)\n\n")
# for d in range(len(themis_schedule)):
#     f.write("Dimension : " + str(d) + "\n")
#     for c in themis_schedule[d]:
#         f.write("\t" + str(c) + "\n")

# f.write("\n-------------------------------------------------------------------------------------------------------\n")