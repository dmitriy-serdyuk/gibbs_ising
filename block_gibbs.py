__author__ = 'dima'

from itertools import tee, izip, izip_longest
import numpy as np
from numpy import random
from matplotlib import pyplot as plt
import matplotlib.cm as cm


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return izip(a, b)


def tree_iter(part, size):
    def func():
        if part == 0:
            for i in xrange(size):
                if i % 2 == 0:
                    gen = xrange(1)
                else:
                    gen = xrange(size - 1)
                for j in gen:
                    yield i, j
        else:
            for i in reversed(xrange(size)):
                if i % 2 == 1:
                    gen = [size - 1]
                else:
                    gen = reversed(xrange(1, size))
                for j in gen:
                    yield i, j
    return func


def rev_tree_iter(part, size):
    def func():
        if part == 0:
            for i in reversed(xrange(size)):
                if i % 2 == 0:
                    gen = xrange(1)
                else:
                    gen = reversed(xrange(size - 1))
                for j in gen:
                    yield i, j
        else:
            for i in xrange(size):
                if i % 2 == 1:
                    gen = [size - 1]
                else:
                    gen = xrange(1, size)
                for j in gen:
                    yield i, j
    return func


def neighbours(i, j, size):
    neigh = set([])
    if i != 0:
        neigh.add((i - 1, j))
    if j != 0:
        neigh.add((i, j - 1))
    if i != size - 1:
        neigh.add((i + 1, j))
    if j != size - 1:
        neigh.add((i, j + 1))
    return neigh


def prec(i, j, size):
    if j == size - 1:
        part = 1
    elif j == 0:
        part = 0
    elif i % 2 == 0:
        part = 1
    else:
        part = 0

    neib = neighbours(i, j, size)
    out = []
    if part == 1:
        for ni, nj in neib:
            if ni < i and j == size - 1:
                out += [(ni, nj)]
            if i % 2 == 0 and ni == i and nj < j and nj != 0:
                out += [(ni, nj)]
    else:
        for ni, nj in neib:
            if ni > i and j == 0:
                out += [(ni, nj)]
            if i % 2 == 1 and ni == i and nj > j and nj != size - 1:
                out += [(ni, nj)]
    return out


def sample(size, theta, num_iter, vis_step):
    rng = random.RandomState(123)
    vars = rng.binomial(1, 0.5, (size, size)) * 2. - 1.
    probs = np.zeros((size, size))

    vals = np.array([-1, 1])

    def do_ancestral_sampling(vars, part, theta):

        msg_back = np.ones((size, size, 2))  # (source_i, source_j, probs)
        msg_fort = np.ones((size, size, 2))  # (target_i, target_j, probs)

        cur_pos = rev_tree_iter(part, size)()
        cur_pos.next()
        prev_pos = rev_tree_iter(part, size)()
        # do backpropagation from leafs to the root
        for (i, j), (pi, pj) in izip(cur_pos, prev_pos):
            neib = neighbours(pi, pj, size).difference((i, j))
            energy = 0.
            for ni, nj in neib:
                energy += theta * vars[ni, nj]
            prob = np.array([np.exp(-energy), np.exp(energy)])
            for ppi, ppj in prec(pi, pj, size):
                prob[:] *= msg_back[ppi, ppj, :]
            msg_back[pi, pj, :] = (prob[0] * np.exp(-theta * vals[:]) +
                                   prob[1] * np.exp( theta * vals[:]))
            msg_back[pi, pj, :] /= msg_back[pi, pj, 0] + msg_back[pi, pj, 1]

        cur_pos = tree_iter(part=part, size=size)()
        prev_pos = tree_iter(part=part, size=size)()
        cur_pos.next()

        for (i, j), (pi, pj) in izip(cur_pos, prev_pos):
            neigh = neighbours(pi, pj, size).difference((i, j))
            energy = 0.0
            for ii, jj in neigh:
                energy += theta * vars[ii, jj]
            prob = np.array([np.exp(-energy), np.exp(energy)]) * msg_back[i, j, :]
            for ppi, ppj in prec(pi, pj, size):
                if (ppi, ppj) != (i, j):
                    prob[:] *= msg_back[ppi, ppj, :]
            #prob[:] *= msg_fort[pi, pj, :]
            msg_fort[i, j, :] = prob[:].copy()
            prob[:] *= msg_back[i, j, :]
            prob /= np.sum(prob)
            sample = rng.binomial(1, prob[1])
            vars[pi, pj] = (sample * 2) - 1
            msg_fort[pi, pj, :] *= np.exp(theta * vars[pi, pj] * vals[:])
            msg_fort[pi, pj, :] /= msg_fort[pi, pj, 0] + msg_fort[pi, pj, 1]
        neigh = neighbours(i, j, size).difference((pi, pj))
        energy = 0.0
        for ii, jj in neigh:
            energy += theta * vars[ii, jj]
        prob = np.array([np.exp(-energy), np.exp(energy)]) * msg_back[i, j, :]
        prob *= msg_fort[i, j, :]
        prob /= np.sum(prob)
        sample = rng.binomial(1, prob[1])
        vars[i, j] = (sample * 2) - 1

    ret = []
    for i in xrange(num_iter):
        do_ancestral_sampling(vars, part=0, theta=theta)
        do_ancestral_sampling(vars, part=1, theta=theta)

        if i % vis_step == vis_step - 1:
            ret += [vars.copy()]
    return ret


def iterate_grid(size):
    for j in xrange(size):
        for i in xrange(size):
            if i < size - 1:
                yield (i, j), (i + 1, j)
            if j < size - 1:
                yield (i, j), (i, j + 1)


def sum_grid(data):
    sum = 0
    size = data[0].shape[0]
    for (i, j), (ii, jj) in iterate_grid(size):
        for matr in data:
            sum += matr[i, j] * matr[ii, jj]
    return sum


def compute_grad(data, theta, sample_iter=100, sample_freq=10):
    size = data[0].shape[0]
    # compute sum
    sum = sum_grid(data) / len(data)

    samples = sample(num_iter=sample_iter, size=size, theta=theta,
                     vis_step=sample_freq)

    partition = sum_grid(samples) / len(samples)
    return sum - partition


def infer(data, start_theta, grad_step, sample_iter=100, sample_freq=10):
    theta = start_theta
    while True:
        grad_step /= 1.03
        grad = compute_grad(data, theta, sample_iter=sample_iter,
                            sample_freq=sample_freq)
        theta += grad_step * grad
        yield theta


def batched(data, batch_size):
    batch = []
    for i, data_point in enumerate(data):
        batch += [data_point]
        if i % batch_size == batch_size - 1:
            yield np.array(batch)
            batch = []


def batch_infer(data, start_theta, grad_step, sample_iter=100, sample_freq=10,
                batch_size=10):
    theta = start_theta
    while True:
        grad_step /= 1.03
        for data_batch in batched(data, batch_size):
            grad = compute_grad(data_batch, theta, sample_iter=sample_iter,
                                sample_freq=sample_freq)
            theta += grad_step * grad
        yield theta


if __name__ == "__main__":
    sample(size=30,
         theta=0.25,
         num_iter=1000,
         vis_step=100)
    plt.savefig("sample_block45.eps")
    plt.show()
