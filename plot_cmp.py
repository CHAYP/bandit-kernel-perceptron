import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys


def read_group(file_in, n):
    tmp = {}
    for i in range(n):
        filename = file_in.readline().split()[0]
        with open(filename, "rb") as f:
            num = np.load(f)
            for t, x in num:
                tmp[t] = tmp.get(t, 0) + x

    ret = []
    for i in tmp:
        ret.append((i, tmp[i] / n))

    return np.array(ret)


def main(args):

    for line in args.file:
        name, n = line.split()
        n = int(n)
        print(name, n)
        tmp = read_group(args.file, n)
        plt.plot(tmp[:, 0], tmp[:, 1], "-", label=name)

    plt.xlabel("Times")
    plt.ylabel("Mistakes")

    plt.legend()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "file", nargs="?", type=argparse.FileType("r"), default=sys.stdin
    )

    args = parser.parse_args()
    main(args)