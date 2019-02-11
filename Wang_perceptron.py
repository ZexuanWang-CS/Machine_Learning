import numpy as np

NUM_FEATURES = 124
DATA_PATH = "adult"

def parse_line(line):
    tokens = line.split()
    x = np.zeros(NUM_FEATURES)
    y = int(tokens[0])
    for t in tokens[1:]:
        parts = t.split(':')
        feature = int(parts[0])
        value = int(parts[1])
        x[feature-1] = value
    x[-1] = 1
    return y, x

def parse_data(filename):
    with open(filename, 'r') as f:
        vals = [parse_line(line) for line in f]
        (ys, xs) = ([v[0] for v in vals],[v[1] for v in vals])
        return np.asarray(ys), np.asarray(xs)

def perceptron(train_ys, train_xs, dev_ys, dev_xs, args):
    weights = np.zeros(NUM_FEATURES)
    devAcc = list()
    for iter in range(args.iterations):
        for dpInd in range(len(train_ys)):
            predict = Predict(weights, train_xs, dpInd)
            if not predict == train_ys[dpInd]:
                weights += args.lr * train_xs[dpInd] * train_ys[dpInd]
        if args.nodev is not True:
            devAcc.append(test_accuracy(weights, dev_ys, dev_xs))
            print('Iteration: {}'.format(iter))
            print('Iteration accuracy: {}'.format(devAcc[iter]))
            if iter >= 10:
                if aver(devAcc, iter, 0) < 0.95 * aver(devAcc, iter, 5):
                    return weights
    return weights

def aver(devAcc, iter, n):
    res = 0.0
    for i in range(n, n+5):
        res += devAcc[len(devAcc)-i-1]
    return res/5.0

def Predict(weights, xs, dpInd):
    predict = weights.dot(xs[dpInd])
    if predict > 0:
        return 1
    elif predict < 0:
        return -1
    else:
        return 0

def test_accuracy(weights, test_ys, test_xs):
    accuracy = 0.0
    numPos = 0.0
    for dpInd in range(len(test_ys)):
        predict = Predict(weights, test_xs, dpInd)
        if predict == test_ys[dpInd]:
            numPos += 1
    accuracy = numPos/float(len(test_ys))
    return accuracy

def main():
    import argparse
    import os

    parser = argparse.ArgumentParser(description='Basic perceptron algorithm.')
    parser.add_argument('--nodev', action='store_true', default=False, help='If provided, no dev data will be used.')
    parser.add_argument('--iterations', type=int, default=50, help='Number of iterations through the full training data to perform.')
    parser.add_argument('--lr', type=float, default=1.0, help='Learning rate to use for update in training loop.')
    parser.add_argument('--train_file', type=str, default=os.path.join(DATA_PATH,'a7a.train'), help='Training data file.')
    parser.add_argument('--dev_file', type=str, default=os.path.join(DATA_PATH,'a7a.dev'), help='Dev data file.')
    parser.add_argument('--test_file', type=str, default=os.path.join(DATA_PATH,'a7a.test'), help='Test data file.')
    args = parser.parse_args()

    """
    At this point, args has the following fields:

    args.nodev: boolean; if True, you should not use dev data; if False, you can (and should) use dev data.
    args.iterations: int; number of iterations through the training data.
    args.lr: float; learning rate to use for training update.
    args.train_file: str; file name for training data.
    args.dev_file: str; file name for development data.
    args.test_file: str; file name for test data.
    """
    train_ys, train_xs = parse_data(args.train_file)
    dev_ys = None
    dev_xs = None
    if not args.nodev:
        dev_ys, dev_xs= parse_data(args.dev_file)
    test_ys, test_xs = parse_data(args.test_file)
    weights = perceptron(train_ys, train_xs, dev_ys, dev_xs, args)
    accuracy = test_accuracy(weights, test_ys, test_xs)
    print('Test accuracy: {}'.format(accuracy))
    print('Feature weights (bias last): {}'.format(' '.join(map(str,weights))))

if __name__ == '__main__':
    main()
