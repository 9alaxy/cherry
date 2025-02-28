import csv

def train_time_collection(log_file):
    load_block = []
    block_move = []
    forward = []
    backward = []
    total = []

    with open(log_file) as file:
        for line in file:
            if 'load_block_time:' in line.strip():
                load_block.append(float(line.split()[-1]))
            elif 'block_move_time:' in line.strip():
                block_move.append(float(line.split()[-1]))
            elif 'model_time:' in line.strip():
                forward.append(float(line.split()[-1]))
            elif 'loss_time:' in line.strip():
                backward.append(float(line.split()[-1]))
            elif 'total_time:' in line.strip():
                total.append(float(line.split()[-1]))
    
    time = []
    time.append(sum(load_block)/len(load_block) + sum(block_move)/len(block_move))
    time.append(sum(forward)/len(forward))
    time.append(sum(backward)/len(backward))
    time.append(sum(total)/len(total))
    
    return time

if __name__ == "__main__":
    all_time = []

    method = ['REG', 'Cherry']

    for mtd in method:
        log_file = f'./log/ogbn-products/{mtd}-4-batch-3-layer-256-hid-SAGE-ogbn-products.log'
        all_time.append(train_time_collection(log_file))
    
    file_name = './data_collection/train_time_breakdown.csv'

    with open(file_name, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(all_time)

    print("saved in ", file_name)