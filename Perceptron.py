import random


def vector_sum(x, weights):
    activation = 0

    for i in range(0, len(weights)):
        if i == 0:
            activation += weights[0]
        else:
            activation += (x[i - 1] * weights[i])

    return activation


class Perceptron:
    learning_rate = 0.01

    def __init__(self):
        self.weights = []

    def train(self, x, y, max_iterations):
        no_of_features = len(x[0])
        no_of_samples = len(x)
        n_v = 0
        n_w = 0

        self.weights = [0 for _ in range(len(x[0]) + 1)]
        self.weights[0] = 1
        predictions = [-1 for _ in range(no_of_samples)]
        current_weights = [0 for _ in range(len(x[0]) + 1)]
        current_weights[0] = 1

        for count in range(0, max_iterations):
            random_index = random.randint(0, no_of_samples - 1)
            activation = vector_sum(x[random_index], current_weights)

            if activation >= 0:
                prediction = 1
            else:
                prediction = -1
            predictions[random_index] = prediction

            if prediction == y[random_index]:
                n_v += 1
            else:
                if n_v > n_w:
                    self.weights = current_weights
                    n_w = n_v

                current_weights[0] = current_weights[0] + y[random_index]
                for i in range(0, no_of_features):
                    current_weights[i + 1] = current_weights[i + 1] + (y[random_index] * x[random_index][i])
                n_v = 0
        return predictions

    def predict(self, x):
        no_of_samples = len(x)
        predictions = [-1 for _ in range(no_of_samples)]

        for count in range (0, no_of_samples):
            sample = x[count]
            activation = vector_sum(sample, self.weights)
            if activation >= 0:
                prediction = 1
            else:
                prediction = -1
            predictions[count] = prediction

        return predictions


perc = Perceptron()
x_t = [[-3.08984, -0.83169], [-2.98083, -0.15958], [-2.836, -0.177], [-2.2843, -1.01894], [-2.252, -1.963],
       [-2.1905, -1.3004]]
y_t = [-1.0, 1.0, -1.0, -1.0, 1.0, 1.0]
perc.train(x_t, y_t, 10000)






