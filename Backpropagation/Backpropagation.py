#!/usr/bin/env python3
import numpy as np
from io import StringIO

NUM_FEATURES = 124  # features are 1 through 123 (123 only in test set), +1 for the bias
DATA_PATH = "./adult"


def parse_line(line):
    tokens = line.split()
    x = np.zeros(NUM_FEATURES)
    y = int(tokens[0])
    y = max(y, 0)  # treat -1 as 0 instead, because sigmoid's range is 0-1
    for t in tokens[1:]:
        parts = t.split(':')
        feature = int(parts[0])
        value = int(parts[1])
        x[feature - 1] = value
    x[-1] = 1  # bias
    return y, x


def parse_data(filename):
    with open(filename, 'r') as f:
        vals = [parse_line(line) for line in f]
        (ys, xs) = ([v[0] for v in vals], [v[1] for v in vals])
        return np.asarray([ys], dtype=np.float32).T, np.asarray(xs, dtype=np.float32).reshape(len(xs), NUM_FEATURES, 1)


def init_model(args):
    w1 = None
    w2 = None
    if args.weights_files:
        with open(args.weights_files[0], 'r') as f1:
            w1 = np.loadtxt(f1)
            o1 = np.zeros(len(w1))
        with open(args.weights_files[1], 'r') as f2:
            w2 = np.loadtxt(f2)
            w2 = w2.reshape(1, len(w2))
            o2 = np.zeros(len(w2))
            e2 = np.zeros(len(w2))
            d2 = np.zeros(len(w2))
    else:
        w1 = np.random.rand(args.hidden_dim, NUM_FEATURES)  # bias included in NUM_FEATURES
        o1 = np.zeros(len(w1))
        w2 = np.random.rand(1, args.hidden_dim + 1)  # add bias column
        o2 = np.zeros(len(w2))
        e2 = np.zeros(len(w2))
        d2 = np.zeros(len(w2))
    model = []
    model.append((w1, o1))
    model.append((w2, o2, e2, d2))
    return model


def train_model(model, train_ys, train_xs, dev_ys, dev_xs, args):
    devAcc = list()
    for itra in range(args.iterations):
        for dp in range(len(train_ys)):
            input = np.squeeze(train_xs[dp, :, :])
            for layer in range(len(model)):
                for neuron in range(len(model[layer][0])):
                    activation = np.sum(input * model[layer][0][neuron])
                    model[layer][1][neuron] = 1. / (1. + np.exp(-activation))
                input = np.append(model[layer][1], 1.0)
            for outNeuron in range(len(model[-1][2])):
                model[-1][2][outNeuron] = train_ys[dp] - model[-1][1][outNeuron]
                model[-1][3][outNeuron] = (model[-1][1][outNeuron] - train_ys[dp]) * model[-1][1][outNeuron] * (
                        1.0 - model[-1][1][outNeuron])
            for layerBP in range(len(model)):
                for neuronBP in range(len(model[layerBP][0])):
                    if layerBP == len(model) - 1:
                        inputBP = np.append(model[layerBP - 1][1], 1.0)
                        model[layerBP][0][neuronBP] = model[layerBP][0][neuronBP] - args.lr * inputBP * \
                                                      model[layerBP][3][neuronBP]
                    else:
                        weightBPls = []
                        for weightBP in model[-1][0]:
                            weightBPls.append(weightBP[neuronBP])
                        update = np.sum(model[-1][3].dot(weightBPls)) * model[layerBP][1][neuronBP] * (
                                1.0 - model[layerBP][1][neuronBP]) * train_xs[dp, :, :]
                        update = update.reshape(1, len(update))
                        model[layerBP][0][neuronBP] = model[layerBP][0][neuronBP] - args.lr * update
        if args.nodev is not True:
            devAcc.append(test_accuracy(model, dev_ys, dev_xs))
            print('Iteration: {}'.format(itra))
            print('Iteration accuracy: {}'.format(devAcc[itra]))
            if itra >= 10:
                if aver(devAcc, 0) < 0.95 * aver(devAcc, 5):
                    return model
    return model


def aver(devAcc, n):
    res = 0.0
    for i in range(n, n + 5):
        res += devAcc[len(devAcc) - i - 1]
    return res / 5.0


def test_accuracy(model, test_ys, test_xs):
    numPos = 0.0
    for dp in range(len(test_ys)):
        input = np.squeeze(test_xs[dp, :, :])
        for layer in range(len(model)):
            for neuron in range(len(model[layer][0])):
                activation = np.sum(input * model[layer][0][neuron])
                model[layer][1][neuron] = 1. / (1. + np.exp(-activation))
            input = np.append(model[layer][1], 1.0)
        if test_ys[dp] == 0 and model[-1][1][0] <= 0.5:
            numPos += 1
        elif test_ys[dp] == 1 and model[-1][1][0] >= 0.5:
            numPos += 1
    accuracy = numPos / float(len(test_ys))
    return accuracy


def extract_weights(model):
    w1 = model[0][0]
    w2 = model[1][0]
    return w1, w2


def main():
    import argparse
    import os

    parser = argparse.ArgumentParser(description='Backpropagation algorithm.')
    parser.add_argument('--nodev', action='store_true', default=False, help='If provided, no dev data will be used.')
    parser.add_argument('--iterations', type=int, default=1,
                        help='Number of iterations through the full training data to perform.')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate to use for update in training loop.')
    weights_group = parser.add_mutually_exclusive_group()
    weights_group.add_argument('--weights_files', nargs=2, metavar=('W1', 'W2'), type=str,
                               help='Files to read weights from. First is weights from input to hidden layer, second is from hidden to output.')
    weights_group.add_argument('--hidden_dim', type=int, default=5, help='Dimension of hidden layer.')
    parser.add_argument('--print_weights', action='store_true', default=False,
                        help='If provided, print final learned weights to stdout (used in autograding)')
    parser.add_argument('--train_file', type=str, default=os.path.join(DATA_PATH, 'a7a.train'),
                        help='Training data file.')
    parser.add_argument('--dev_file', type=str, default=os.path.join(DATA_PATH, 'a7a.dev'), help='Dev data file.')
    parser.add_argument('--test_file', type=str, default=os.path.join(DATA_PATH, 'a7a.test'), help='Test data file.')
    args = parser.parse_args()
    train_ys, train_xs = parse_data(args.train_file)
    dev_ys = None
    dev_xs = None
    if not args.nodev:
        dev_ys, dev_xs = parse_data(args.dev_file)
    test_ys, test_xs = parse_data(args.test_file)
    model = init_model(args)
    model = train_model(model, train_ys, train_xs, dev_ys, dev_xs, args)
    accuracy = test_accuracy(model, test_ys, test_xs)
    print('Test accuracy: {}'.format(accuracy))
    if args.print_weights:
        w1, w2 = extract_weights(model)
        with StringIO() as weights_string_1:
            np.savetxt(weights_string_1, w1)
            print('Hidden layer weights: {}'.format(weights_string_1.getvalue()))
        with StringIO() as weights_string_2:
            np.savetxt(weights_string_2, w2)
            print('Output layer weights: {}'.format(weights_string_2.getvalue()))


if __name__ == '__main__':
    main()
