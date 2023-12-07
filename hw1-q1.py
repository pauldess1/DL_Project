#!/usr/bin/env python

# Deep Learning Homework 1

import argparse

import numpy as np
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import utils


class LinearModel(object):
    def __init__(self, n_classes, n_features, **kwargs):
        self.W = np.zeros((n_classes, n_features))

    def update_weight(self, x_i, y_i, **kwargs):
        raise NotImplementedError

    def train_epoch(self, X, y, **kwargs):
        for x_i, y_i in zip(X, y):
            self.update_weight(x_i, y_i, **kwargs)

    def predict(self, X):
        """X (n_examples x n_features)"""
        scores = np.dot(self.W, X.T)  # (n_classes x n_examples)
        predicted_labels = scores.argmax(axis=0)  # (n_examples)
        return predicted_labels

    def evaluate(self, X, y):
        """
        X (n_examples x n_features):
        y (n_examples): gold labels
        """
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible


class Perceptron(LinearModel):
    def update_weight(self, x_i, y_i, **kwargs):
        """
        x_i (n_features): a single training example
        y_i (scalar): the gold label for that example
        other arguments are ignored
        """
        y_pred = self.predict(x_i)
        if y_pred != y_i:
            self.W[y_i] += x_i * y_i


class LogisticRegression(LinearModel):
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def update_weight(self, x_i, y_i, learning_rate=0.001):
        """
        x_i (n_features): a single training example
        y_i: the gold label for that example
        learning_rate (float): keep it at the default value for your plots
        """
        z = np.dot(self.W, x_i)
        h = self.sigmoid(z)
        gradient = np.outer((h - (np.arange(len(h)) == y_i)), x_i)
        self.W = self.W - learning_rate * gradient

class MLP(object):
    def __init__(self, n_classes, n_features, hidden_size):
        self.hidden_units = hidden_size
        self.weights_hidden = np.random.normal(0.1, 0.1, (n_features, hidden_size))
        self.bias_hidden = np.zeros(hidden_size)
        self.weights_output = np.random.normal(0.1, 0.1, (hidden_size, n_classes))
        self.bias_output = np.zeros(n_classes)

    def predict(self, X):
        hidden_layer_input = np.dot(X, self.weights_hidden) + self.bias_hidden
        hidden_layer_output = self.relu(hidden_layer_input)
        output_layer_input = np.dot(hidden_layer_output, self.weights_output) + self.bias_output
        output_probs = self.softmax(output_layer_input)
        return output_probs

    def evaluate(self, X, y):
        y_hat = np.argmax(self.predict(X), axis=1)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible

    def train_epoch(self, X, y, learning_rate=0.001):
        y_pred = self.predict(X)
        loss = self.loss(y, y_pred)
        grad_loss_output = self.loss_derivative(y, y_pred)
        grad_weights_output = np.dot(self.relu_derivative(np.dot(X, self.weights_hidden) + self.bias_hidden).T, grad_loss_output)
        grad_bias_output = np.sum(grad_loss_output, axis=0)

        grad_hidden = np.dot(grad_loss_output, self.weights_output.T) * self.relu_derivative(np.dot(X, self.weights_hidden) + self.bias_hidden)
        grad_weights_hidden = np.dot(X.T, grad_hidden)
        grad_bias_hidden = np.sum(grad_hidden, axis=0)

        self.weights_output -= learning_rate * grad_weights_output
        self.bias_output -= learning_rate * grad_bias_output
        self.weights_hidden -= learning_rate * grad_weights_hidden
        self.bias_hidden -= learning_rate * grad_bias_hidden

        return loss

    def relu(self, x):
        return np.maximum(0, x)

    def softmax(self, x):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

    def relu_derivative(self, x):
        return np.where(x <= 0, 0, 1)

    def loss(self, y_true, y_pred):
        num_classes = y_pred.shape[1]
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        y_true_one_hot = self.to_one_hot(y_true, num_classes)
        loss = -y_true_one_hot * np.log(y_pred)

        return np.sum(loss) / len(y_true)

    def loss_derivative(self, y_true, y_pred):
        num_classes = y_pred.shape[1]
        y_true_one_hot = self.to_one_hot(y_true, num_classes)
    
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        derivative = -y_true_one_hot / y_pred
    
        return derivative
    
    def to_one_hot(self, labels, num_classes):
        num_samples = len(labels)
        one_hot_labels = np.zeros((num_samples, num_classes))
        one_hot_labels[np.arange(num_samples), labels] = 1
        return one_hot_labels


def plot(epochs, train_accs, val_accs, pdf):
    plt.figure(figsize=(10, 5))
    
    max_val = 1
    min_val = 0
    
    plt.plot(epochs, train_accs, 'b', label='Train')
    plt.plot(epochs, val_accs, 'r', label='Validation') 

    plt.ylim(min_val, max_val)
    
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    pdf.savefig()

def plot_loss(epochs, loss, pdf):
    fig, ax = plt.subplots()
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')

    ax.plot(epochs,loss, label='train')
    print(loss)

    ax.legend()
    pdf.savefig(fig)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model',
                        choices=['perceptron', 'logistic_regression', 'mlp'],
                        help="Which model should the script run?")
    parser.add_argument('-epochs', default=20, type=int,
                        help="""Number of epochs to train for. You should not
                        need to change this value for your plots.""")
    parser.add_argument('-hidden_size', type=int, default=200,
                        help="""Number of units in hidden layers (needed only
                        for MLP, not perceptron or logistic regression)""")
    parser.add_argument('-learning_rate', type=float, default=0.001,
                        help="""Learning rate for parameter updates (needed for
                        logistic regression and MLP, but not perceptron)""")
    opt = parser.parse_args()

    utils.configure_seed(seed=42)

    add_bias = opt.model != "mlp"
    data = utils.load_oct_data(bias=add_bias)
    train_X, train_y = data["train"]
    dev_X, dev_y = data["dev"]
    test_X, test_y = data["test"]
    n_classes = np.unique(train_y).size
    n_feats = train_X.shape[1]

    # initialize the model
    if opt.model == 'perceptron':
        model = Perceptron(n_classes, n_feats)
    elif opt.model == 'logistic_regression':
        model = LogisticRegression(n_classes, n_feats)
    else:
        model = MLP(n_classes, n_feats, opt.hidden_size)
    epochs = np.arange(1, opt.epochs + 1)
    train_loss = []
    valid_accs = []
    train_accs = []
    
    for i in epochs:
        print('Training epoch {}'.format(i))
        train_order = np.random.permutation(train_X.shape[0])
        train_X = train_X[train_order]
        train_y = train_y[train_order]
        if opt.model == 'mlp':
            loss = model.train_epoch(
                train_X,
                train_y,
                learning_rate=opt.learning_rate
            )
        else:
            model.train_epoch(
                train_X,
                train_y,
                learning_rate=opt.learning_rate
            )
        
        train_accs.append(model.evaluate(train_X, train_y))
        valid_accs.append(model.evaluate(dev_X, dev_y))
        if opt.model == 'mlp':
            print('loss: {:.4f} | train acc: {:.4f} | val acc: {:.4f}'.format(
                loss, train_accs[-1], valid_accs[-1],
            ))
            train_loss.append(loss)
        else:
            print('train acc: {:.4f} | val acc: {:.4f}'.format(
                 train_accs[-1], valid_accs[-1],
            ))
    print('Final test acc: {:.4f}'.format(
        model.evaluate(test_X, test_y)
        ))

    with PdfPages('plots.pdf') as pdf:
        plot(epochs, train_accs, valid_accs, pdf)
        if opt.model == 'mlp':
            plot_loss(epochs, train_loss, pdf)


if __name__ == '__main__':
    main()
