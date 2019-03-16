from Perceptron import Perceptron
import random
from math import log, e
import time
import numpy
import matplotlib.pyplot as plt
import sys


def get_weighted_sample_indices(size_of_sample, weights):
    indices = random.choices([i for i in range(size_of_sample)], weights, k=size_of_sample)
    return indices

def get_samples(x, indices):
    samples = [x[i] for i in indices]
    return samples

def get_labels(y, indices):
    labels = [y[i] for i in indices]
    return labels

def read_file(filename):
    checkpoints_from_file = []
    y = []
    x = []
    with open(filename, "r") as f:
        lines = f.readlines()
        for line in lines:
            line_arr = [float(l.strip("\n")) for l in line.split(",")]
            y.append(line_arr[0])
            x.append(line_arr[1:])
            checkpoints_from_file.append(line_arr)
    return x, y

def usage():
    print("Usage: AdaBoost.py [data_file_name] [training_size]\n\n\
        \tdata_file_name  -> Name of file with dataset\n\
        \ttraining_size   -> how many training data (the rest is used for testing)\n")

class Adaboost:
    def __init__(self, number_of_classifiers, max_iter):
        self.no_of_classifies = number_of_classifiers
        self.classifiers = []
        self.max_iterations = max_iter
        self.classifier_alphas = [0 for _ in range(number_of_classifiers)]

        for i in range(0, number_of_classifiers):
            self.classifiers.append(Perceptron())

    def get_training_error(self, predicted, y, weights):
        len_predicted = len(predicted)
        error = 0
        for i in range(len_predicted):
            if predicted[i] == y[i]:
                y_i = 0
            else:
                y_i = 1

            error += (weights[i] * y_i)
        return error

    def get_test_accuracy(self, y, predictions):
        test_error = 0

        for i in range(len(y)):
            if y[i] == predictions[i]:
                test_error += 1

        return test_error / len(y)

    def compute_new_weights(self, curr_wt, alpha, y, predictions):
        len_y = len(y)
        normalization_factor = 0

        for i in range(len_y):
            curr_wt[i] = curr_wt[i] * (e **(- 1 * alpha * y[i] * predictions[i]))
            normalization_factor += curr_wt[i]

        new_weights = [(curr_wt[i] / normalization_factor) for i in range(len_y)]
        return new_weights

    def ensemble_classifier(self, alphas, hypotheses):
        len_sample = len(hypotheses[0])
        ensemble_preds = [0 for _ in range(len_sample)]

        for i in range(len_sample):
            pred_sum = 0
            for j in range(self.no_of_classifies):
                pred_sum += (alphas[j] * hypotheses[j][i])

            if pred_sum < 0:
                ensemble_preds[i] = -1
            else:
                ensemble_preds[i] = 1

        return ensemble_preds

    def adabtrain(self, x, y):
        weights = [1.0/len(x) for _ in x]
        classifier_errors = [0 for _ in self.classifiers]
        # classifier_weights = []
        hypotheses = []
        no_of_samples = len(x)

        for i in range(0, self.no_of_classifies):
            # classifier_weights.append(weights)
            indices = get_weighted_sample_indices(no_of_samples, weights)
            samples = get_samples(x, indices)
            labels = get_labels(y, indices)

            predictions = self.classifiers[i].train(samples, labels, self.max_iterations)
            hypotheses.append(predictions)
            classifier_errors[i] = self.get_training_error(predictions, y, weights)

            self.classifier_alphas[i] = 0.5 * log(((1 - classifier_errors[i]) / classifier_errors[i]))

            weights = self.compute_new_weights(weights.copy(), self.classifier_alphas[i], y, predictions)
        # classifier_weights.append(weights)

        ensem_predictions = self.ensemble_classifier(self.classifier_alphas, hypotheses)
        train_acc = self.get_test_accuracy(y, ensem_predictions)
        return train_acc

    def adabpredict(self, x, y):
        hypotheses = []
        for i in range(0, self.no_of_classifies):
            hypotheses.append(self.classifiers[i].predict(x))

        final_predictions = self.ensemble_classifier(self.classifier_alphas, hypotheses)
        accuracy = self.get_test_accuracy(y, final_predictions)
        return accuracy


if __name__ == '__main__':

    if len(sys.argv) != 3:
        usage()

    else:
        try:
            filename = sys.argv[1]
            x_data, y_data = read_file(filename)
            no_of_training_set = int(sys.argv[2])
            is_valid = 1
        except:
            print("Wrong input parameters!")
            is_valid = 0

        if is_valid == 1:
            k_class = [(i + 1) * 10 for i in range(10)]
            train_accuracies = [0 for i in range(10)]
            accuracies = [0 for i in range(10)]
            training_time = [0 for i in range(10)]

            start_of_test_set = no_of_training_set
            max_iter = 10000

            for i in range(len(k_class)):
                adab = Adaboost(k_class[i], max_iter)
                t = time.time()
                train_accuracies[i] = adab.adabtrain(x_data[:no_of_training_set - 1], y_data[:no_of_training_set - 1])
                training_time[i] = time.time() - t
                accuracies[i] = adab.adabpredict(x_data[start_of_test_set:], y_data[start_of_test_set:])
                print("Accuracy: Training {}% Testing {}%, Training Time = {}s k = {} size of training set = {}".format
                      (train_accuracies[i] * 100, accuracies[i] * 100, training_time[i], k_class[i], no_of_training_set))

            best_adab_index = numpy.argmax(accuracies)
            best_adab_config = k_class[best_adab_index]
            print("Best config: k = {}, Accuracy = {}%".format(best_adab_config, accuracies[best_adab_index] * 100))

            plt.figure(0)
            plt.plot(k_class, train_accuracies)
            plt.title("Training Accuracy for {}".format(filename))

            plt.figure(1)
            plt.plot(k_class, accuracies)
            plt.title("Testing Accuracy {}".format(filename))

            plt.figure(2)
            plt.plot(k_class, training_time)
            plt.title("Training Time {}".format(filename))

            plt.show()









