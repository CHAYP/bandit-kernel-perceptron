import numpy as np
import matplotlib.pyplot as plt
import argparse


def main(args):
    print(args)
    with open(args.file, "rb") as f:
        data = np.load(f, allow_pickle=True)[()]  # so weird
        N = data["number"]
        C = data["class"]
        D = data["dimension"]
        G = data["gamma"]
        X = data["X"]
        Y = data["Y"]
        name = data["name"]

    print(f"Number of examples: {N}")
    print(f"Number of classes: {C}")
    print(f"Example dimension: {D}")
    print(f"Gamma: {G}")

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])
    # ax.add_artist(plt.Circle((0, 0), 1, color="b", fill=False))
    ax.set_aspect("equal", "box")
    plt.grid(linestyle="--")
    plt.scatter(X[:, 0], X[:, 1], c=Y[:, 0], s=10)
    ax.set_title(name)
    plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-f",
        "--file",
        help="input file name",
    )

    args = parser.parse_args()
    main(args)