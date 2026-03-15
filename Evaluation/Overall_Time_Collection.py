import csv

def total_time_collection(log_file):
    total_t = []

    with open(log_file) as file:
        for line in file:
            if 'total_time:' in line.strip():
                total_t.append(float(line.split()[1]))
    
    print(log_file)
    if len(total_t) == 0:
        return 0
    if len(total_t) == 1:
        return total_t[0]
    total_t.pop(0)
    avg = sum(total_t)/len(total_t)
    return avg

if __name__ == "__main__":
    dataset = ['reddit', 'ogbn-arxiv', 'ogbn-products']
    method = ['berry']
    model = ['SAGE', 'GCN', 'GAT']
    total_time = []

    for da in dataset:
        path = f'./log/{da}/'
        for md in model:
            temp_time = []
            for mtd in method:
                log_file = path + f'{mtd}-4-batch-2-layer-'
                if md == 'GAT':
                    log_file = log_file + f'128-hid-{md}-{da}.log'
                else:
                    log_file = log_file + f'64-hid-{md}-{da}.log'
                temp_time.append(total_time_collection(log_file))
            total_time.append(temp_time)
    
    file_name = './data_collection/berry_time.csv'

    with open(file_name, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(total_time)