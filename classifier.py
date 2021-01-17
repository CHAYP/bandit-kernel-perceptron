import numpy as np
import random


class Perceptron:
    def __init__(self, name, class_number):
        self.name = name
        self.class_number = int(class_number)
        self.init()
        self.count = {}
        self.count["update"] = 0
        self.count["negative"] = 0
        self.count["predict"] = 0

    def init(self):
        self.model = [[] for i in range(self.class_number)]

    def load_model(self, file_name):
        with open(file_name, "rb") as f:
            tmp = np.load(f, allow_pickle=True)[()]  # so wierd
            self.name = tmp["name"]
            self.class_number = tmp["class"]
            self.count = tmp["count"]
            self.model = tmp["model"]
            self.print_model_len()

    def save_model(self, file_name):
        with open(file_name, "wb") as f:
            tmp = {}
            tmp["name"] = self.name
            tmp["class"] = self.class_number
            tmp["count"] = self.count
            tmp["model"] = self.model
            np.save(f, tmp)
            # np.save(
            #     f, np.array([self.name, self.class_number, self.negative, self.update])
            # )
            # np.save(f, self.model)

    def sum_kernel(self, y, xt):
        return np.sum([self.kernel(x, xt) * sign for x, sign in self.model[y]])

    def predict(self, x):
        Y = [i for i in range(self.class_number) if self.sum_kernel(i, x) >= 0]
        self.count["predict"] += 1
        if len(Y) > 0:
            return random.choice(Y), True
        return np.random.randint(0, self.class_number), False

    def learn(self, x, y, sign=1):
        if sign == -1:
            self.count["negative"] += 1
        self.count["update"] += 1
        self.model[y].append((x, sign))

    def kernel(self, x, xt):
        return np.dot(x, xt)

    def print_model_len(self):
        print("=================== Model info ====================")
        print(f"Classes:\t{self.class_number}")
        print(f"Updates:\t{self.count['update']} with -1: {self.count['negative']}")
        [
            print(f"\tclass {i}:\t{len(self.model[i])} instances")
            for i in range(self.class_number)
        ]


class KernelPerceptron(Perceptron):
    def __init__(self, name, class_number):
        super().__init__(name, class_number)

    def kernel(self, x, xt):
        return 1 / (1 - 1 / 2 * np.dot(x, xt))
