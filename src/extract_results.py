

def main():
    total_nodes = 16
    filename = "../results/mesh_logs/_multitree_16_heterogeneous_beta_alexnet.log"
    file1 = open(filename, 'r')
    lines = file1.readlines()
    for flow in range(total_nodes):
        print("Flow " + str(flow))
        print("Reduce scatter")
        for i in range(total_nodes):
            for j in range(total_nodes):
                if i != j:
                    parent = " HMC-" + str(i) + " "
                    child = " HMC-" + str(j) + " "
                    parent_name = " parent" + parent
                    child_name = " child" + child
                    start_reduce_part = " start reducing for flow " + str(flow) + " "
                    receive_reduce_part = " receives full reduce for flow " + str(flow) + " "
                    start_reduce = None
                    receive_reduce = None
                    for line in lines:
                        line = line.strip()
                        line = line + " "
                        if start_reduce_part in line and child in line and parent_name in line:
                            start_reduce = line
                        if receive_reduce_part in line and parent in line and child_name in line:
                            receive_reduce = line
                    if start_reduce is not None and receive_reduce is not None:
                        print("Child: " + str(j) + ", Parent: " + str(i) + ", Start: " + start_reduce.split()[0] + ", End: " + receive_reduce.split()[0])

        for i in range(total_nodes):
            for j in range(total_nodes):
                if i != j:
                    parent = " HMC-" + str(i) + " "
                    child = " HMC-" + str(j) + " "
                    parent_name = " parent" + parent
                    child_name = " child" + child
                    start_gather_part = " start gathering for flow " + str(flow) + " "
                    receive_gather_part = " receives full gather for flow " + str(flow) + " "
                    start_gather = None
                    receive_gather = None
                    for line in lines:
                        line = line.strip()
                        line = line + " "
                        if start_gather_part in line and parent in line and child_name in line:
                            start_gather = line
                        if receive_gather_part in line and child in line and parent_name in line:
                            receive_gather = line
                    if start_gather is not None and receive_gather is not None:
                        print("Child: " + str(j) + ", Parent: " + str(i) + ", Start: " + start_gather.split()[0] + ", End: " + receive_gather.split()[0])


if __name__ == '__main__':
    main()
