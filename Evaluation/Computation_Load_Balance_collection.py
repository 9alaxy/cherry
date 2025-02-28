import csv

def computation_collection(log_file):
    computation_node = []

    with open(log_file) as file:
        for line in file:
            if 'Number of nodes for computation during this epoch:' in line.strip():
                computation_node.append(float(line.split()[-1]))
    
    return computation_node

def memory_collection(log_file):
    memory = []

    with open(log_file) as file:
        for line in file:
            if ' max memory allocated' in line.strip():
                memory.append(float(line.split()[5]))
    
    return memory

if __name__ == "__main__":
    methods = ['Cherry', 'REG']
    path = f'./log/ogbn-arxiv/'

    all_computation_nodes = []
    all_memory = []

    for mtd in methods:
        log_file = path + f'{mtd}-4-batch-3-layer-256-hid-SAGE-ogbn-arxiv.log'
        all_computation_nodes.append(computation_collection(log_file))
        all_memory.append(memory_collection(log_file))

    file_name = './data_collection/computation_nodes_arxiv.csv'
    all_computation_nodes = zip(*all_computation_nodes)

    with open(file_name, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(all_computation_nodes)

    print("saved in ", file_name)
    
    file_name = './data_collection/memory_arxiv.csv'
    all_memory = zip(*all_memory)

    with open(file_name, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(all_memory)
    
    print("saved in ", file_name)
