import argparse
import numpy as np
from numpy.random import binomial


def parse_args():
    parser = argparse.ArgumentParser(
        "Run Gibbs sampling on Ising model")
    parser.add_argument("--iter",
                        type=int, default=1000,
                        help="Number of iterations")
    parser.add_argument("--size",
                        type=int, default=30,
                        help="Size of Ising model")
    parser.add_argument("--theta",
                        type=float, default=0.45,
                        help="Coupling parameter")
    parser.add_argument("--show-freq",
                        type=int, default=100,
                        help="Show frequency")
    return parser.parse_args()


def sample(num_iter, size, theta, show_freq):
    rng = np.random.RandomState(123)
    vars = (rng.binomial(1, 0.5, size=(size, size)) * 2) - 1
    ans = []
    for it in xrange(num_iter):
        for i in xrange(size):
            for j in xrange(size):
                sum = 0
                if i != 0:
                    sum += theta * vars[i - 1, j]
                if i != size - 1:
                    sum += theta * vars[i + 1, j]
                if j != 0:
                    sum += theta * vars[i, j - 1]
                if j != size - 1:
                    sum += theta * vars[i, j + 1]
                prob_neg = np.exp(-sum)
                prob_pos = np.exp(sum)
                sample = rng.binomial(1, prob_pos / (prob_neg + prob_pos))
                vars[i, j] = (sample * 2) - 1
        if it % show_freq == 0:
            ans += [vars.copy()]

    return ans


if __name__ == "__main__":
    args = parse_args()
    sample(args.iter, args.size, args.theta, args.show_freq)

