import numpy as np
from classifier import Perceptron, KernelPerceptron
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import argparse


def main(args):
    with open(args.data, "rb") as f:
        data = np.load(f, allow_pickle=True)[()]  # so weird
        N = data["number"]
        C = data["class"]
        D = data["dimension"]
        G = data["gamma"]
        X = data["X"]
        Y = data["Y"]

    classifier = KernelPerceptron("rational", C)
    classifier.load_model(args.model)

    h = 0.03
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # focus_class = 2
    # Z = np.array(
    #     [classifier.sum_kernel(focus_class, i) for i in np.c_[xx.ravel(), yy.ravel()]]
    # )
    Z = np.array([classifier.predict(i)[0] for i in np.c_[xx.ravel(), yy.ravel()]])

    fig, ax = plt.subplots()
    # bounds = np.linspace(Z.min(), Z.max(), 10)
    bounds = np.linspace(0, C - 1, C)
    norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)

    Z = Z.reshape(xx.shape)
    # pcm = plt.pcolormesh(xx, yy, Z, norm=norm, cmap="RdBu_r", shading="auto")
    pcm = plt.pcolormesh(xx, yy, Z, norm=norm, shading="auto")
    fig.colorbar(pcm, ax=ax, extend="both", orientation="vertical")

    # f = Y[:, 0] == focus_class
    # fX = X[f]
    # plt.scatter(fX[:, 0], fX[:, 1], cmap=norm)
    plt.scatter(X[:, 0], X[:, 1], c=Y[:, 0], edgecolors="k")

    plt.axis("tight")
    plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-d",
        "--data",
        help="data file name",
    )

    parser.add_argument(
        "-m",
        "--model",
        help="model file name",
    )

    args = parser.parse_args()
    main(args)