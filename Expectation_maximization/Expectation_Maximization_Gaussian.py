#!/usr/bin/env python3
import numpy as np

DATA_PATH = "./"  # data path to points.dat


def parse_data(args):
    num = float
    dtype = np.float32
    data = []
    with open(args.data_file, 'r') as f:
        for line in f:
            data.append([num(t) for t in line.split()])
    dev_cutoff = int(.9 * len(data))
    train_xs = np.asarray(data[:dev_cutoff], dtype=dtype)
    dev_xs = np.asarray(data[dev_cutoff:], dtype=dtype) if not args.nodev else None
    return train_xs, dev_xs


def init_model(args):
    clusters = []
    if args.cluster_num:
        lambdas = np.zeros(args.cluster_num)
        mus = np.zeros((args.cluster_num, 2))
        if not args.tied:
            sigmas = np.zeros((args.cluster_num, 2, 2))
        else:
            sigmas = np.zeros((2, 2))
        lambdas = np.ones(args.cluster_num) / args.cluster_num
        mus = np.asmatrix(np.random.random(mus.shape))
        if not args.tied:
            sigmas = np.array([np.asmatrix(np.identity(2)) for i in range(args.cluster_num)])
        else:
            sigmas = np.array([np.asmatrix(np.identity(2))])
    else:
        lambdas = []
        mus = []
        sigmas = []
        with open(args.clusters_file, 'r') as f:
            for line in f:
                lambda_k, mu_k_1, mu_k_2, sigma_k_0_0, sigma_k_0_1, sigma_k_1_0, sigma_k_1_1 = map(float, line.split())
                lambdas.append(lambda_k)
                mus.append([mu_k_1, mu_k_2])
                sigmas.append([[sigma_k_0_0, sigma_k_0_1], [sigma_k_1_0, sigma_k_1_1]])
        lambdas = np.asarray(lambdas)
        mus = np.asmatrix(np.asarray(mus))
        sigmas = np.asarray(sigmas)
        args.cluster_num = len(lambdas)
    model = (lambdas, mus, sigmas)
    return model


def train_model(model, train_xs, dev_xs, args):
    from scipy.stats import multivariate_normal
    itra = 0
    lld = 1.0
    lld_prev = 0.0
    w = np.empty((len(train_xs), args.cluster_num), dtype=float)
    while (lld - lld_prev > 1e-4 and itra < args.iterations):
        lld_prev = average_log_likelihood(model, train_xs, args)
        Estep(model, train_xs, w, args)
        Mstep(model, train_xs, w, args)
        itra += 1
        lld = average_log_likelihood(model, train_xs, args)
    return model


def average_log_likelihood(model, data, args):
    from math import log
    from scipy.stats import multivariate_normal
    ll = 0.0
    for i in range(len(data)):
        tmp = 0
        for j in range(args.cluster_num):
            tmp += model[0][j] * multivariate_normal(mean=model[1][j].A1, cov=model[2][j], allow_singular=True).pdf(
                data[i])
        ll += np.log(tmp)
    return ll / len(data)


def Estep(model, train_xs, w, args):
    from scipy.stats import multivariate_normal
    for i in range(len(train_xs)):
        for j in range(args.cluster_num):
            w[i][j] = model[0][j] * multivariate_normal(mean=model[1][j].A1, cov=model[2][j], allow_singular=True).pdf(
                train_xs[i])
    wsum = w.sum(axis=1)
    for j in range(args.cluster_num):
        w[:, j] = w[:, j] / wsum


def Mstep(model, train_xs, w, args):
    for j in range(args.cluster_num):
        SUM = w[:, j].sum()
        model[0][j] = SUM / len(train_xs)
        mus_j = np.zeros(2)
        sigma_j = np.zeros((2, 2))
        for i in range(len(train_xs)):
            mus_j += (train_xs[i, :] * w[i][j])
        model[1][j] = mus_j / SUM
        for i in range(len(train_xs)):
            sigma_j += w[i][j] * ((train_xs[i, :] - model[1][j, :]).T * (train_xs[i, :] - model[1][j, :]))
        model[2][j] = sigma_j / SUM


def extract_parameters(model):
    lambdas = model[0]
    mus = np.asarray(model[1])
    sigmas = model[2]
    return lambdas, mus, sigmas


def main():
    import argparse
    import os
    print('Gaussian')
    parser = argparse.ArgumentParser(
        description='Expectation-maximization (EM) algorithm with Gaussian mixture models.')
    init_group = parser.add_mutually_exclusive_group(required=True)
    init_group.add_argument('--cluster_num', type=int, help='Randomly initialize this many clusters.')
    init_group.add_argument('--clusters_file', type=str, help='Initialize clusters from this file.')
    parser.add_argument('--nodev', action='store_true', help='If provided, no dev data will be used.')
    parser.add_argument('--data_file', type=str, default=os.path.join(DATA_PATH, 'points.dat'), help='Data file.')
    parser.add_argument('--print_params', action='store_true',
                        help='If provided, learned parameters will also be printed.')
    parser.add_argument('--iterations', type=int, default=1, help='Number of EM iterations to perform')
    parser.add_argument('--tied', action='store_true',
                        help='If provided, use a single covariance matrix for all clusters.')
    args = parser.parse_args()
    if args.tied and args.clusters_file:
        print(
            'You should not implement tied covariances when initializing from a file. Don\'t provide --tied and --clusters_file together.')
        exit(1)
    train_xs, dev_xs = parse_data(args)
    model = init_model(args)
    model = train_model(model, train_xs, dev_xs, args)
    ll_train = average_log_likelihood(model, train_xs, args)
    print('Train LL: {}'.format(ll_train))
    if not args.nodev:
        ll_dev = average_log_likelihood(model, dev_xs, args)
        print('Dev LL: {}'.format(ll_dev))
    lambdas, mus, sigmas = extract_parameters(model)
    if args.print_params:
        def intersperse(s):
            return lambda a: s.join(map(str, a))

        print('Lambdas: {}'.format(intersperse(' | ')(np.nditer(lambdas))))
        print('Mus: {}'.format(intersperse(' | ')(map(intersperse(' '), mus))))
        if args.tied:
            print('Sigma: {}'.format(intersperse(' ')(np.nditer(sigmas))))
        else:
            print('Sigmas: {}'.format(intersperse(' | ')(map(intersperse(' '), map(lambda s: np.nditer(s), sigmas)))))


if __name__ == '__main__':
    main()
