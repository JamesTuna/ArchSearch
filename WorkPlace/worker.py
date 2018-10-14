import argparse
import json

from WorkPlace import train_eval


def main():
    size = 4
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', type=str, help='structure indicators')
    parser.add_argument('index', type=int, help='structure indicators')
    parser.add_argument('total', type=int, help='structure indicators')
    args = parser.parse_args()

    archs = []
    with open(args.filename, "r") as f:
        for line in f.readlines():
            archs.append(line.split("\n")[0])

    length = len(archs)
    for i in range(args.index*length//args.total, (args.index+1)*length//args.total):
        accr = train_eval.main(size, archs[i]).item()
        # accr = 1234
        with open('block1.txt', 'a') as f:
            f.write(archs[i]+" accuracy: "+str(accr)+"\n")
    # genotypes = json.loads(args.genotype)
    # for genotype in genotypes:
    #     accr = train_eval.dist_main(size, genotype).item()
    #
    #     with open('block1.txt', 'a') as f:
    #         f.write(genotype+" accuracy: "+str(accr)+"\n")


if __name__ == "__main__":
    main()
