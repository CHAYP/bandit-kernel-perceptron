import os
import argparse
import numpy as np
import random

import numba as nb

import matplotlib.pyplot as plt
import matplotlib as mpl


def unit_ball(N, d=2, r=1):
    def in_ball(x, origin, r):
        return sum([(i - j) ** 2 for i, j in zip(x, origin)]) <= r ** 2

    M = 0
    origin = [0] * d
    X = []
    while M < N:
        tmp = 2 * np.random.sample((N, d)) - 1
        tmp = [i for i in tmp if in_ball(i, origin, r)]
        tmp = tmp[: min(len(tmp), N - M)]
        X += tmp
        M += len(tmp)

    return np.array(X)


def filename_increment(fname):
    ftype = "npy"
    i = 1
    while os.path.exists(f"{fname}_{i}.{ftype}"):
        i += 1
    return f"{fname}_{i}.{ftype}"


def parse_ouput(fname):
    return filename_increment(fname)


def strongly(W, x, G):
    tmp = [(np.dot(vec, x), i) for i, vec in enumerate(W)]
    pos = []
    neg = []
    for v, i in tmp:
        if v >= G / 2:
            pos.append(i)
        elif v <= -G / 2:
            neg.append(i)
        else:
            return -1, -1, False
    if len(pos) == 1 and len(neg) == len(W) - 1:
        return pos[0], 0, True
    return -1, -1, False


def weakly(W, x, G):
    tmp = [np.dot(vec, x) for i, vec in enumerate(W)]
    for i in range(len(W)):
        p = True
        for j in range(len(W)):
            if i != j and p and tmp[i] < tmp[j] + G:
                p = False
        if p:
            return i, 0, True
    return -1, -1, False


def in_ball(x):
    return np.sum(x ** 2) <= 1


def gen_vec(n, d):
    vec = 2 * np.random.sample((n, d)) - 1  # [-1,1]^D
    norm = np.sqrt(np.sum(vec ** 2))
    vec_norm = vec / norm
    return vec_norm


def find_inner_class(v, x, G):
    for i, j in enumerate(v):
        jj = np.dot(j, j)
        jx = np.dot(j, x)
        if jx < jj - 2 * G:
            return i, True
        if jj - 2 * G <= jx and jx <= jj + 2 * G:
            return -1, False
    return len(v), True


def group_weakly(W, V, C_G, x, G):
    _y = weakly(W, x, G)
    if _y[2]:
        _g = find_inner_class(V[_y[0]], x, G)
        if _g[1]:
            tmp = sum(C_G[: _y[0]]) + _g[0] + _y[0], _y[0], True
            return tmp
    return -1, -1, False


def to_norm(vec):
    norm = np.sqrt(np.sum(vec ** 2))
    vec_norm = vec / norm
    return vec_norm


EXAM_W = [
    to_norm(np.array([np.array([1, 0]), np.array([-1, 0])])),
    to_norm(np.array([np.array([1, 1]), np.array([-1, 1])])),
    to_norm(np.array([np.array([1, 1]), np.array([-1, 1]), np.array([0, -1])])),
    to_norm(
        np.array([np.array([1, 1, 1]), np.array([-1, 1, 1]), np.array([0, -1, 1])])
    ),
]
EXAM_C = [2, 2, 3, 3]
EXAM_D = [2, 2, 2, 3]
EXAM_C_G = [[1, 0], [1, 1], [4, 2, 3], [4, 2, 3]]


def main(args):

    N = args.num
    C = args.cls
    D = args.dim
    G = args.gam

    if args.exam > 0:
        np.random.seed(0)
        W = EXAM_W[args.exam - 1]
        C = EXAM_C[args.exam - 1]
        D = EXAM_D[args.exam - 1]
        C_G = EXAM_C_G[args.exam - 1]
        U = W
        V = [
            np.linspace(0, 1, C_G[i] + 2)[1:-1].reshape(C_G[i], 1) * U[i]
            for i in range(C)
        ]
        V = [to_norm(i) for i in V]

    else:
        W = gen_vec(C, D)
        U = gen_vec(C, D)
        C_G = [i - 1 for i in args.group] if args.group else [1 for i in range(C)]
        V = [np.sort(np.random.sample((C_G[i], 1))) * U[i] for i in range(C)]
        V = [to_norm(i) for i in V]

    n = 0
    X = np.empty((0, D))
    Y = np.empty((0, 2))  # class, group
    while n < N:
        _x = 2 * np.random.sample((N, D)) - 1  # [-1,1]^D
        _x = np.array([x for x in _x if in_ball(x)])

        if args.sep == "strongly":
            _y = [strongly(W, x, G) for x in _x]
        elif args.sep == "weakly":
            _y = [weakly(W, x, G) for x in _x]
        elif args.sep == "group":
            _y = [group_weakly(W, V, C_G, x, G) for x in _x]

        _p = [i[2] for i in _y]
        _y = np.array(_y)[:, :-1]
        _y = _y[_p]
        _x = _x[_p]

        X = np.concatenate((X, _x[: min(_x.shape[0], N - n)]))
        Y = np.concatenate((Y, _y[: min(_y.shape[0], N - n)]))
        n = X.shape[0]

    with open(args.output, "wb") as f:
        tC = sum(C_G) + C if args.sep == "group" else C
        tmp = {}
        tmp["number"] = N
        tmp["class"] = tC
        tmp["dimension"] = D
        tmp["gamma"] = G
        tmp["X"] = X
        tmp["Y"] = Y
        tmp["W"] = W
        tmp["U"] = U
        np.save(f, tmp)

        print(f"Number of examples: {N}")
        print(f"Number of classes: {tC}")
        print(f"Example dimension: {D}")
        print(f"Gamma: {G}")
        print(f"Example seperable type: {args.sep}")

    print(f"saved to file {args.output}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-n", "--num", type=int, default=10000, help="number of examples"
    )

    parser.add_argument(
        "-c", "--cls", type=int, default=3, help="number of main classes"
    )

    parser.add_argument(
        "-d", "--dim", type=int, default=2, help="number of dimenssions"
    )

    parser.add_argument("-g", "--gam", type=float, default=0.01, help="gamma")

    parser.add_argument(
        "-e",
        "--exam",
        type=int,
        default=0,
        help="example of class vectors, fix dimension and number of classes. [1,4]",
    )
    parser.add_argument(
        "-G",
        "--group",
        type=eval,
        help="list specify classes in each groups eg. [1,2,3,4]",
    )

    parser.add_argument(
        "-s",
        "--sep",
        default="group",
        help="data seperable type (strongly, weakly, group)",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=parse_ouput,
        default="example",
        help="ouput file name (just name no file type)",
    )

    args = parser.parse_args()
    main(args)
