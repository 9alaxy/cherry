import csv

def memory_collection(log_file):
    memory = []

    with open(log_file) as file:
        for line in file:
            if ' max memory allocated' in line.strip():
                memory.append(float(line.split()[5]))
    
    # return max(memory), min(memory), sum(memory)/len(memory)
    return max(memory)

if __name__ == "__main__":
    # path = './log/ogbn-products/'
    # log_file = path + 'REG-4-batch-2-layer-192-hid-SAGE-ogbn-products-mean.log'
    # print(memory_collection(log_file))

    # log_file = path + 'REG-4-batch-2-layer-192-hid-SAGE-ogbn-products-pool.log'
    # print(memory_collection(log_file))
    # memory = []

    # path = './log/ogbn-products/'
    # fanout = [64, 96, 132, 164]

    # for fan in fanout:
    #     log_file = path + f'Cherry-4-batch-1-layer-256-hid-SAGE-ogbn-products-lstm-{fan}.log'
    #     memory.append(memory_collection(log_file))
    # aggregators = ['mean', 'pool', 'lstm']

    # for aggr in aggregators:
    #     log_file = path + f'Cherry-4-batch-2-layer-192-hid-SAGE-ogbn-products-{aggr}.log'
    #     memory.append(memory_collection(log_file))
    
    # print(memory)

    # path = './log/reddit/'

    # log_file = path + 'Cherry-16-batch-3-layer-256-hid-SAGE-reddit.log'
    # print(memory_collection(log_file))

    # log_file = path + 'Cherry-32-batch-3-layer-256-hid-SAGE-reddit.log'
    # print(memory_collection(log_file))

    model = ['SAGE', 'GCN', 'GAT']
    dataset = ['reddit', 'ogbn-arxiv', 'ogbn-products']

    all_memory = []

    for data in dataset:
        path = f'./log/{data}/'
        memory = []
        for md in model:
            if md == 'GAT':
                log_file = path + f'vanilla-8-batch-3-layer-128-hid-{md}-{data}.log'
            else:
                log_file = path + f'vanilla-8-batch-3-layer-256-hid-{md}-{data}.log'
            memory.append(memory_collection(log_file))
        all_memory.append(memory)
    
    file_name = './data_collection/vanilla_memory.csv'

    with open(file_name, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(all_memory)
    
    print("saved in ", file_name)