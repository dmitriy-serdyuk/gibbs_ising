import argparse
import numpy as np
import scipy.io


def read_data(filename):
    data = scipy.io.loadmat(filename)
    data = np.array(data['price_move'])[:, 0]
    n = len(data)
    return n, data


def parse_args():
    parser = argparse.ArgumentParser(
        "Run Gibbs sampling on Ising model")
    parser.add_argument("--input",
                        type=str, help="Input .mat file", default='file.mat')
    parser.add_argument("--q-value",
                        type=float, default=0.7,
                        help="Value of q")
    parser.add_argument("--prob-x1",
                        type=float, default=0.2,
                        help="Probability of x1")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args.input, args.q_value, args.prob_x1)

