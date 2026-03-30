import csv
import numpy as np

def computation_collection(log_file):
    """Collect computation nodes per epoch."""
    computation_node = []

    with open(log_file) as file:
        for line in file:
            if 'Number of nodes for computation during this epoch:' in line.strip():
                computation_node.append(float(line.split()[-1]))

    return computation_node


def memory_collection(log_file):
    """Collect peak memory usage per epoch."""
    memory = []

    with open(log_file) as file:
        for line in file:
            if ' max memory allocated' in line.strip():
                memory.append(float(line.split()[4]))

    return memory


def edge_cut_collection(log_file):
    """Collect edge cut metrics from log file."""
    edge_cut = []
    edge_cut_ratio = []

    with open(log_file) as file:
        for line in file:
            if 'Edge cut:' in line.strip():
                edge_cut.append(float(line.split()[-1]))
            elif 'Edge cut ratio:' in line.strip():
                edge_cut_ratio.append(float(line.split()[-1]))

    return edge_cut, edge_cut_ratio


def replication_factor_collection(log_file):
    """Collect replication factor from log file."""
    replication_factor = []

    with open(log_file) as file:
        for line in file:
            if 'Replication Factor:' in line.strip():
                replication_factor.append(float(line.split()[-1]))

    return replication_factor


def calculate_load_balance_std(partition_src_len_list):
    """Calculate load balance standard deviation.

    Args:
        partition_src_len_list: List of source node counts per micro-batch

    Returns:
        Tuple of (standard deviation, coefficient of variation)
    """
    if not partition_src_len_list:
        return 0.0, 0.0

    arr = np.array(partition_src_len_list)
    std = np.std(arr)
    mean = np.mean(arr)
    cv = std / mean if mean > 0 else 0  # coefficient of variation

    return std, cv


def partition_len_collection(log_file):
    """Collect partition source length per micro-batch.

    This shows the computation load distribution across micro-batches.
    Supports both old format ("train node:") and new format ("compute nodes:").
    """
    partition_lens = []

    with open(log_file) as file:
        for line in file:
            # New format: "Micro-batch-X max memory allocated: Y GB, compute nodes: Z"
            if 'Micro-batch-' in line.strip() and 'compute nodes:' in line.strip():
                parts = line.strip().split()
                try:
                    # Find the index of 'compute' and get the next value
                    idx = parts.index('compute')
                    node_count = int(parts[idx + 2])
                    partition_lens.append(node_count)
                except (IndexError, ValueError):
                    continue
            # Old format: "Micro-batch-X train node: Y"
            elif 'Micro-batch-' in line.strip() and 'train node:' in line.strip():
                parts = line.strip().split()
                try:
                    node_count = int(parts[-1])
                    partition_lens.append(node_count)
                except (IndexError, ValueError):
                    continue

    return partition_lens


if __name__ == "__main__":
    # Example usage
    methods = ['Cherry', 'REG']
    path = f'./log/ogbn-arxiv/'

    all_computation_nodes = []
    all_memory = []
    all_partition_lens = []

    for mtd in methods:
        log_file = path + f'{mtd}-4-batch-3-layer-256-hid-SAGE-ogbn-arxiv.log'
        all_computation_nodes.append(computation_collection(log_file))
        all_memory.append(memory_collection(log_file))
        all_partition_lens.append(partition_len_collection(log_file))

        # Calculate load balance std
        if all_partition_lens[-1]:
            std, cv = calculate_load_balance_std(all_partition_lens[-1])
            print(f"{mtd} - Load Balance Std: {std:.2f}, CV: {cv:.4f}")

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
