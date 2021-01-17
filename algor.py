from classifier import Perceptron, KernelPerceptron
import numpy as np
import argparse
import os


def filename_increment(fname):
    ftype = "npy"
    if os.path.exists(f"{fname}.{ftype}"):
        i = 1
        while os.path.exists(f"{fname}_{i}.{ftype}"):
            i += 1
        return f"{fname}_{i}.{ftype}"
    return f"{fname}.{ftype}"


def main(args):
    with open(args.data, "rb") as f:
        data = np.load(f, allow_pickle=True)[()]  # so weird
        N = data["number"]
        C = data["class"]
        D = data["dimension"]
        G = data["gamma"]
        X = data["X"]
        Y = data["Y"]

    N = int(N)
    T = N if args.num == 0 else args.num
    t = 0
    result = []
    correct = 0

    classifier = KernelPerceptron("rational", C)
    if args.kernel == "linear":
        classifier = Perceptron("linear", C)
    if args.model:
        classifier.load_model(args.model)

    while t < T:
        tx = t % N
        x = X[tx]
        y = int(Y[tx][0])

        yt, not_none = classifier.predict(x)

        if not_none:
            if yt != y:
                classifier.learn(x, yt, -1)
        else:
            if yt == y:
                classifier.learn(x, yt, 1)

        if yt == y:
            correct += 1

        if t % args.step == 0:
            print(f"{t+1}/{T}, correct {correct}/{t+1} ({correct/(t+1)*100}%)")

        if args.dump:
            result.append((t, t - correct))

        t += 1

    print(f"summary correct {correct}/{T} ({correct/T*100}%)")
    classifier.print_model_len()

    if args.save:
        if args.model:
            classifier.save_model(args.model)
            print(f"updated to file {args.model}")
        else:
            file_name = filename_increment("model")
            classifier.save_model(file_name)
            print(f"created file {file_name}")

    if args.dump:
        with open(args.dump, "wb") as f:
            np.save(f, result)
            print(f"dump to file {args.dump}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-n", "--num", type=int, default=0, help="number of loop")
    parser.add_argument("-s", "--step", type=int, default=10000, help="step interval")

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

    parser.add_argument("--save", action="store_true")
    parser.add_argument(
        "-x",
        "--dump",
        help="dump file name",
    )

    parser.add_argument(
        "-k",
        "--kernel",
        default="rational",
        help="(liner, rational)",
    )

    args = parser.parse_args()
    main(args)