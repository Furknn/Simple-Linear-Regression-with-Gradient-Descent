import numpy as np
import random
import matplotlib.pyplot as plt


class LRGD:
    a_current = 0
    b_current = 0
    error = []

    def train_with_gd(self, X, y, epochs, e):
        self.a_current = 0
        self.b_current = 0
        self.error = []
        N = float(len(y))
        for i in range(epochs):
            y_current = (self.a_current * X) + self.b_current
            cost = sum([data ** 2 for data in (y - y_current)]) / N
            a_gradient = -(2 / N) * sum(X * (y - y_current))
            b_gradient = -(2 / N) * sum(y - y_current)
            self.a_current = self.a_current - (e * a_gradient)
            self.b_current = self.b_current - (e * b_gradient)
            self.error.append(cost)

    def train_with_sgd(self, X, y, epochs, e):
        self.a_current = 0
        self.b_current = 0
        self.error = []
        rn = random.randint(0, len(y))
        for i in range(epochs):
            y_current = (self.a_current * X[rn]) + self.b_current
            cost = (y[rn] - y_current) ** 2
            a_gradient = -2 * X[rn] * (y[rn] - y_current)
            b_gradient = -2 * y[rn] - y_current
            self.a_current = self.a_current - (e * a_gradient)
            self.b_current = self.b_current - (e * b_gradient)
            self.error.append(cost)

    def train_with_minibatch(self, X, y, epochs, e):
        self.a_current = 0
        self.b_current = 0
        self.error = []
        N = 30
        rn = random.randint(0, len(y) - N)
        X = X[rn:rn + N]
        y = y[rn:rn + N]
        for i in range(epochs):
            y_current = (self.a_current * X) + self.b_current
            cost = sum([data ** 2 for data in (y - y_current)]) / N
            a_gradient = -(2 / N) * sum(X * (y - y_current))
            b_gradient = -(2 / N) * sum(y - y_current)
            self.a_current = self.a_current - (e * a_gradient)
            self.b_current = self.b_current - (e * b_gradient)
            self.error.append(cost)

    def predict(self, X):
        return self.a_current * X + self.b_current


if __name__ == '__main__':
    epsilon = 0.00001
    iterations = 70
    lgrd = LRGD()
    size = 1000
    # y = 3x + 7 + gaussian
    a = 3
    b = 7
    X = np.random.random(size) * 100
    Y = np.empty(size)
    for i in range(X.size):
        Y[i] = int(a * X[i] + b + random.gauss(100, 50))

    lgrd.train_with_gd(X, Y, iterations, epsilon)
    train_gd_b = lgrd.b_current
    train_gd_a = lgrd.a_current
    train_gd_error = lgrd.error

    lgrd.train_with_sgd(X, Y, iterations, epsilon)
    train_sgd_b = lgrd.b_current
    train_sgd_a = lgrd.a_current
    train_sgd_error = lgrd.error

    lgrd.train_with_minibatch(X, Y, iterations, epsilon)
    train_mb_b = lgrd.b_current
    train_mb_a = lgrd.a_current
    train_mb_error = lgrd.error

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.plot(X, Y)

    y_gd = train_gd_a * X + train_gd_b
    plt.plot(X, y_gd, "-r")

    y_sgd = train_sgd_a * X + train_sgd_b
    plt.plot(X, y_sgd, "-g")

    y_mb = train_mb_a * X + train_mb_b
    plt.plot(X, y_mb, "-m")

    plt.show()
    plt.plot(list(range(iterations)), train_sgd_error, "-r")
    plt.plot(list(range(iterations)), train_gd_error, "-g")
    plt.plot(list(range(iterations)), train_mb_error, "-m")
    plt.show()
