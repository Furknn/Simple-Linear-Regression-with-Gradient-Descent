import numpy as np
import random
import matplotlib.pyplot as plt


class LRGD:
    a_current = 0
    b_current = 0
    error = []

    def train_with_gd(self, X, y, iterations, e):
        self.a_current = 0
        self.b_current = 0
        self.error = []
        N = float(len(y))
        for i in range(iterations):
            y_current = (self.a_current * X) + self.b_current

            # add iter error
            self.error.append(sum([data ** 2 for data in (y - y_current)]) / N)

            # calculate gradients
            a_gradient = -(2 / N) * sum(X * (y - y_current))
            b_gradient = -(2 / N) * sum(y - y_current)
            self.a_current = self.a_current - (e * a_gradient)
            self.b_current = self.b_current - (e * b_gradient)

    def train_with_sgd(self, X, y, iterations, e):
        self.a_current = 0
        self.b_current = 0
        self.error = []
        N = len(y)
        for iter in range(iterations):
            for epoch in range(N):
                y_current = (self.a_current * X[epoch]) + self.b_current

                # calculate epoch error
                self.error.append((y[epoch] - y_current) ** 2)

                # calculate gradients
                a_gradient = -2 * X[epoch] * (y[epoch] - y_current)
                b_gradient = -2 * y[epoch] - y_current
                self.a_current = self.a_current - (e * a_gradient)
                self.b_current = self.b_current - (e * b_gradient)

    def train_with_minibatch(self, X, y, iterations, e, batch_size):
        self.a_current = 0
        self.b_current = 0
        self.error = []
        N = len(y)
        for iter in range(iterations):
            epochs = int(N / batch_size)
            for epoch in range(epochs):
                batch_X = X[epoch * batch_size:epoch * batch_size + batch_size]
                batch_Y = y[epoch * batch_size:epoch * batch_size + batch_size]
                y_current = (self.a_current * batch_X) + self.b_current

                # calculate epoch error
                self.error.append(sum(data ** 2 for data in (batch_Y - (self.a_current * batch_X) + self.b_current)) / batch_size)

                # calculate gradients
                a_gradient = -(2 / batch_size) * sum(batch_X * (batch_Y - y_current))
                b_gradient = -(2 / batch_size) * sum(batch_Y - y_current)
                self.a_current = self.a_current - (e * a_gradient)
                self.b_current = self.b_current - (e * b_gradient)

    def predict(self, X):
        return self.a_current * X + self.b_current


if __name__ == '__main__':
    lgrd = LRGD()
    size = 1000
    # y = 3x + 7 + gaussian
    a = 3
    b = 7
    X = np.random.random(size) * 100
    Y = np.empty(size)
    for i in range(X.size):
        Y[i] = int(a * X[i] + b + random.gauss(20, 10))

    lgrd.train_with_gd(X, Y, iterations=20, e=0.0001)
    train_gd_b = lgrd.b_current
    train_gd_a = lgrd.a_current
    train_gd_error = lgrd.error

    lgrd.train_with_sgd(X, Y, iterations=1, e=0.0001)
    train_sgd_b = lgrd.b_current
    train_sgd_a = lgrd.a_current
    train_sgd_error = lgrd.error

    lgrd.train_with_minibatch(X, Y, iterations=1, e=0.0001, batch_size=10)
    train_mb_b = lgrd.b_current
    train_mb_a = lgrd.a_current
    train_mb_error = lgrd.error

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.scatter(X, Y)
    y_gd = train_gd_a * X + train_gd_b
    plt.plot(X, y_gd, "-r", label="with GD")
    y_sgd = train_sgd_a * X + train_sgd_b
    plt.plot(X, y_sgd, "-g", label="with SGD")
    y_mb = train_mb_a * X + train_mb_b
    plt.plot(X, y_mb, "-b", label="with Minibatch")
    plt.legend(loc="upper left")
    plt.show()

    plt.plot(train_gd_error, "-r", label="with GD")
    plt.xlabel("iteration")
    plt.ylabel("error")
    plt.legend(loc="upper left")
    plt.show()
    plt.plot(train_sgd_error, "-g", label="with SGD")
    plt.xlabel("epoch")
    plt.ylabel("error")
    plt.legend(loc="upper left")
    plt.show()
    plt.plot(train_mb_error, "-b", label="with Minibatch")
    plt.xlabel("epoch")
    plt.ylabel("error")
    plt.legend(loc="upper left")
    plt.show()
