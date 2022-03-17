import numpy as np
import random
import matplotlib.pyplot as plt


class LRGD:
    m = 0
    c = 0

    def plot_data_label(self, x, y):
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.plot(x, y, 'bo')
        plt.show()

    def gradient_descent(self, X, Y, L):
        print("WIP")
        # Y_pred = self.m * X + self.c  # The current predicted value of Y
        # D_m = (-2 / len(X)) * sum(X * (Y - Y_pred))  # Derivative wrt m
        # D_c = (-2 / len(X)) * sum(Y - Y_pred)  # Derivative wrt c
        # self.m = self.m - L * D_m  # Update m
        # self.c = self.c - L * D_c  # Update c

    def generate_random_data(self, a, b):
        X = np.random.randint(100, size=1000)
        Y = np.empty(1000)
        for i in range(X.size):
            Y[i] = int(a * X[i] + b + random.gauss(100, 20))
        return X, Y

    def train_with_gd(self, X, Y, epoch, e):
        for i in range(epoch):
            self.gradient_descent(X,Y,e)


if __name__ == '__main__':
    epsilon = 0.001
    iterations = 1000
    lgrd = LRGD()
    X, Y = lgrd.generate_random_data(3, 7)
    lgrd.train(X,Y,iterations, epsilon)
    lgrd.plot_data_label(X, Y)
