__author__ = 'dima'

from itertools import tee, izip


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return izip(a, b)


def tree_iter(part, size):
    def func():
        for i in xrange(size):
            if part == 0:
                if i % 2 == 0:
                    gen = xrange(1)
                else:
                    gen = xrange(size - 1)
            else:
                if i % 2 == 1:
                    gen = [size - 1]
                else:
                    gen = reversed(xrange(1, size))
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


def main():
    pass

if __name__ == "__main__":
    main()