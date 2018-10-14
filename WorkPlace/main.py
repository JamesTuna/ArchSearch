import argparse
import os
import subprocess


def read_block_file(file_name):
    train_X = []
    train_y = []
    with open(file_name, "r") as f:
        for line in f.readlines():
            train_X.append(line.split(" accuracy: ")[0])
            train_y.append(float(line.split(" accuracy: ")[1]))

    return train_X, train_y


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('file_name', type=str, help='structure indicators')
    parser.add_argument('start', type=int, help='structure indicators')
    parser.add_argument('end', type=int, help='structure indicators')
    parser.add_argument('out_file', type=str, help='structure indicators')
    args = parser.parse_args()

    train_X, train_y = read_block_file(args.file_name)
    # train_X = train_X[256:]
    train_X = train_X[args.start:args.end]

    for i in range(len(train_X)):
        subprocess.Popen(
                ["python3.6", os.path.join(os.getcwd(), "PNASsearch/search.py"), train_X[i], args.file_name, args.out_file])


if __name__ == "__main__":
    run()
