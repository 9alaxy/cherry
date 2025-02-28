import csv

def betty_time_collection(log_file):
    connect_t = []
    block_gen_t = []
    batch_gen_t = []
    partition_t = []
    gen_dist_t = []
    betty_pre_t = []
    total_t = []

    with open(log_file) as file:
        for line in file:
            if 'total_time:' in line.strip():
                total_t.append(float(line.split()[1]))
            elif 'conncect_check_time:' in line.strip():
                connect_t.append(float(line.split()[1]))
            elif 'block_gen_time_total:' in line.strip():
                block_gen_t.append(float(line.split()[1]))
            elif 'batch_blocks_gen_time:' in line.strip():
                batch_gen_t.append(float(line.split()[1]))
            elif 'partition_time:' in line.strip():
                partition_t.append(float(line.split()[1]))
            elif 'gen dist nodes time:' in line.strip():
                gen_dist_t.append(float(line.split()[4]))
            elif 'Betty prepare time:' in line.strip():
                betty_pre_t.append(float(line.split()[3]))
    
    connect_t.pop(0)
    block_gen_t.pop(0)
    batch_gen_t.pop(0)
    partition_t.pop(0)
    gen_dist_t.pop(0)
    betty_pre_t.pop(0)
    total_t.pop(0)

    time = []
    time.append(sum(total_t)/len(total_t))
    time.append(sum(connect_t)/len(connect_t))
    time.append(sum(block_gen_t)/len(block_gen_t))
    time.append(sum(batch_gen_t)/len(batch_gen_t))
    time.append(sum(partition_t)/len(partition_t))
    time.append(sum(gen_dist_t)/len(gen_dist_t))
    time.append(sum(total_t)/len(total_t) - sum(betty_pre_t)/len(betty_pre_t))

    return time

def betty_tm_collection(log_file):
    betty_pre_t = []
    total_t = []
    memory = []

    with open(log_file) as file:
        for line in file:
            if 'total_time:' in line.strip():
                total_t.append(float(line.split()[1]))
            elif 'Betty prepare time:' in line.strip():
                betty_pre_t.append(float(line.split()[3]))
            elif ' max memory allocated:' in line.strip():
                memory.append(float(line.split()[5]))
            
    
    train_t = (sum(total_t)/len(total_t) - sum(betty_pre_t)/len(betty_pre_t))

    return train_t, memory

if __name__ == "__main__":
    dataset = ['reddit', 'ogbn-arxiv', 'ogbn-products']
    all_time = []

    for data in dataset:
        path = f'./log/{data}/'
        log_file = path + f'REG-4-batch-3-layer-256-hid-SAGE-{data}.log'
        all_time.append(betty_time_collection(log_file))

    file_name = './data_collection/observation_1.csv'

    with open(file_name, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(all_time)
    
    # path = "./log/ogbn-products/"
    # method = 'REG'
    # num_batch = [2, 3, 4, 5, 6, 7, 8]
    # all_time = []
    # all_mem = []
    # blank = []
    # for nb in num_batch:
    #     log_file = path + f'{method}-{nb}-batch-3-layer-256-hid-SAGE-ogbn-products.log'
    #     train_t, memory = betty_tm_collection(log_file)
    #     all_time.append(train_t)
    #     all_mem.append(memory)
    
    # file_name = './data_collection/observation_2.csv'

    # transposed_memory = zip(*all_mem)

    # with open(file_name, 'w', newline='') as csv_file:
    #     writer = csv.writer(csv_file)
    #     writer.writerow(all_time)
    #     writer.writerow(blank)
    
    # with open(file_name, 'a', newline='') as csv_file:
    #     writer = csv.writer(csv_file)
    #     writer.writerows(all_mem)

    print("saved in ", file_name)


