import csv

def train_time_collection(log_file):
    """Collect time breakdown from log file.

    Returns:
        List of [data_loading_time, compute_time, total_time]
        where data_loading_time = sampling_time + load_block_time + block_move_time
    """
    sampling = []
    load_block = []
    block_move = []
    forward = []
    backward = []
    total = []

    with open(log_file) as file:
        for line in file:
            if 'sampling_time:' in line.strip():
                sampling.append(float(line.split()[-1]))
            elif 'load_block_time:' in line.strip():
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
    # Data loading time = sampling + load_block + block_move
    data_loading = 0
    if sampling:
        data_loading += sum(sampling) / len(sampling)
    if load_block:
        data_loading += sum(load_block) / len(load_block)
    if block_move:
        data_loading += sum(block_move) / len(block_move)

    # Compute time = forward + backward
    compute_time = 0
    if forward:
        compute_time += sum(forward) / len(forward)
    if backward:
        compute_time += sum(backward) / len(backward)

    time.append(data_loading)
    time.append(compute_time)
    if total:
        time.append(sum(total) / len(total))

    return time


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Collect time breakdown from log files")
    parser.add_argument("--log-files", nargs="+", required=True, help="Log file paths")
    parser.add_argument("--output", required=True, help="Output CSV file path")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    all_time = []
    for log_file in args.log_files:
        all_time.append(train_time_collection(log_file))

    with open(args.output, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(all_time)

    print("saved in", args.output)
