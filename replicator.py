import numpy as np
import matplotlib.pyplot as plt
import argparse
from tqdm import trange

np.set_printoptions(precision=3)

def softmax(x):
    z = x - np.max(x)
    return np.exp(z) / np.sum(np.exp(z))


def replicator(A, B, x, y, delta=1.0):
    x, y = (softmax(x), softmax(y))

    delta_x = x * (A.dot(y) - x.T.dot(A).dot(y))
    delta_y = y * (x.T.dot(B).T - x.T.dot(B).dot(y))

    return delta_x, delta_y


def replicator2(A, B, X, Y, delta=1.0):
    x = np.array([[X, 1.0 - X]]).T
    y = np.array([[Y, 1.0 - Y]]).T

    delta_x = x * (A.dot(y) - x.T.dot(A).dot(y))
    delta_y = y * (x.T.dot(B).T - x.T.dot(B).dot(y))

    return delta_x, delta_y


def PrisonersDilemma():
    A = [
        [-1, -3],
        [0, -2]
    ]

    B = [
        [-1, 0],
        [-3, -2]
    ]

    return np.array(A), np.array(B)


def MatchingPennies():
    A = [
        [1, -1],
        [-1, 1]
    ]

    B = [
        [-1, 1],
        [1, -1]
    ]

    return np.array(A), np.array(B)


def Stag():
    A = [
        [2, 0],
        [1, 1]
    ]

    B = [
        [2, 1],
        [0, 1]
    ]

    return np.array(A), np.array(B)


def RandomZeroSum(size):
    A = np.random.rand(size, size)
    return A, -1 * A


def RandomSymmetric(size):
    A = np.random.rand(size, size)
    return A, A.T


def Random(size):
    return np.random.rand(size, size), np.random.rand(size, size)


if __name__ == '__main__':

    delta = 0.99
    size = 2

    parser = argparse.ArgumentParser()
    parser.add_argument("game", help="select the game to use ['prisoners', 'matching', 'stag', 'symmetric_random']")
    parser.add_argument("--phase", action="store_true", help="enable plotting phase diagram (only works for"
                                                             "2 player, 2 action games)")
    parser.add_argument("--size", type=int, help="if game is specified as random, this controls the size of the game")

    args = parser.parse_args()

    if args.size:
        size = args.size

    G = None
    if args.game == "prisoners":
        G = PrisonersDilemma()
    elif args.game == "matching":
        G = MatchingPennies()
    elif args.game == "stag":
        G = Stag()
    elif args.game == "symmetric_random":
        G = RandomSymmetric(size)
    else:
        raise ValueError("game argument must be 'prisoners', 'matching', or 'stag'.")

    A, B = G

    print("\nGame: {}\n".format(args.game))

    meta_a = softmax(np.random.rand(size)).T
    meta_b = softmax(np.random.rand(size)).T

    nb_episodes = 1000
    for i in trange(nb_episodes):
        dmeta_a, dmeta_b = replicator(A, B, meta_a, meta_b, delta)
        meta_a += dmeta_a
        meta_b += dmeta_b


    print("Done!\n")

    meta_a, meta_b = softmax(meta_a), softmax(meta_b)

    print("Policy A: {}".format(meta_a))
    print("Policy A Expected Utility: {:.3f}\n".format(meta_a.T.dot(A).dot(meta_b)))
    print("Policy B: {}".format(meta_b))
    print("Policy B Expected Utility: {:.3f}".format(meta_a.T.dot(B).dot(meta_b)))


    if args.phase:
        grid = np.mgrid[0:1:10j, 0:1:10j]
        stack = np.dstack((grid))

        grad = np.zeros((10, 10, 2))
        for i in range(10):
            for j in range(10):
                policy_1 = float(i / 9.0)
                policy_2 = float(j / 9.0)

                x, y = replicator2(A, B, policy_1, policy_2, delta)
                grad[i, j, 0] = x[0]
                grad[i, j, 1] = y[0]


        Y, X = grid
        V, U = grad[:, :, 0], grad[:, :, 1]

        plt.quiver(X, Y, U, V)
        plt.title(args.game)
        plt.xlabel("$x_1$")
        plt.ylabel("$y_1$")
        plt.show()

