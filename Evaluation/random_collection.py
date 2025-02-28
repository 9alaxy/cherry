def pa_collection(log_file):
    pa_time = []

    with open(log_file) as file:
        for line in file:
            if 'one partition time:' in line.strip():
                pa_time.append(float(line.split()[-1]))

    return sum(pa_time) / len(pa_time)

if __name__ == "__main__":
    dataset = ['reddit', 'ogbn-arxiv', 'ogbn-products']

    for data in dataset:
        path = f'./log/{data}/'
        log_file = path + f'Random-8-batch-3-layer-256-hid-SAGE-{data}.log'
        print(pa_collection(log_file))