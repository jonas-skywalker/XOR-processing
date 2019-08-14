"""
Neural Network class file to create a neural network with a list as the structure
Example:
    list = [100, 20, 30, 10]
    neural_net = NeuralNetwork(list)
This example creates a Neural Network with 100 input nodes, two hidden layers with 20 and 30 nodes
and finally 10 output nodes.

To do:
    - input handling
    - Ruckwartsabfrage
    - optimisation methods of adjusting the learning rate
    - non-stochastic gradient descent
    - epoch training in library
    - unsupervised learning
    - store and restore network data
"""
import my_matrix_lib as matrix
import math
import json


# Activation Function
def sigmoid(x):
    return 1/(1 + math.e**(-x))


class NeuralNetwork:
    def __init__(self, structure, lr=0.1, af=sigmoid):
        self.structure = structure
        self.outputs = []
        self.errors = []
        self.weights = []
        self.biases = []
        self.lr = lr
        self.af = af
        # check if structure is ok
        if len(self.structure) < 2:
            print(">>> Cannot create Neural Network from given Structure!")
        elif len(self.structure) == 2:
            self.input_nodes = self.structure[0]
            self.output_nodes = self.structure[1]
            self.weights.append(matrix.Matrix(self.output_nodes, self.input_nodes, mat_type="random"))
            self.biases.append(matrix.Matrix(self.output_nodes, 1, mat_type="random"))

        else:
            # Declaration of input layer
            self.input_nodes = self.structure[0]
            self.first_hidden_nodes = self.structure[1]
            self.weights.append(matrix.Matrix(self.first_hidden_nodes, self.input_nodes, mat_type="random"))
            self.biases.append(matrix.Matrix(self.first_hidden_nodes, 1, mat_type="random"))

            # Declaration of hidden layers
            for i in range(1, len(structure) - 2):
                self.weights.append(matrix.Matrix(self.structure[i + 1], self.structure[i], mat_type="random"))
                self.biases.append(matrix.Matrix(self.structure[i + 1], 1, mat_type="random"))

            # Declaration of output layer
            self.output_nodes = self.structure[-1]
            self.last_hidden_nodes = self.structure[-2]
            self.weights.append(matrix.Matrix(self.output_nodes, self.last_hidden_nodes, mat_type="random"))
            self.biases.append(matrix.Matrix(self.output_nodes, 1, mat_type="random"))

    def feed_forward(self, list_input, soft_max=False):

        output = None

        # Convert inputs into input vector
        mat_input = matrix.Matrix.create_vector(list_input)

        old_input = mat_input
        self.outputs.append(old_input)

        # loop through the matrices and biases and feed the input through the neural network
        for i in range(len(self.weights)):
            product = matrix.Matrix.mat_mul(self.weights[i], old_input)
            new_input = matrix.Matrix.vector_add(product, self.biases[i])
            output = matrix.Matrix.apply_func(new_input, self.af)
            old_input = output
            self.outputs.append(output)
        if soft_max:
            return output.soft_max()
        else:
            return output

    def linear_regression_gradient_descent(self, list_input, target_input, soft_max=False):
        self.outputs = []
        self.errors = []

        # Convert target to vector object
        target = matrix.Matrix.create_vector(target_input)

        # feed forward the input
        output = self.feed_forward(list_input, soft_max)

        # calculate the output error
        error = matrix.Matrix.vector_sub(output, target)
        self.errors.append(error)

        # calculate all errors
        for i in range(len(self.weights)-1, 0, -1):
            error = matrix.Matrix.mat_mul(matrix.Matrix.transpose(self.weights[i]), self.errors[0])
            self.errors.insert(0, error)

        # backpropagate and start with the adjustment of the last weight matrix
        for i in range(len(self.weights) - 1, -1, -1):

            # updating weights and biases(refactor this!!!!!!)

            # calculating the derivative of the activation function
            first = matrix.Matrix.mult(self.outputs[i + 1], -1)

            second = matrix.Matrix.add(first, 1)

            derivative = matrix.Matrix.vector_mult(self.outputs[i + 1], second)

            gradient = matrix.Matrix.vector_mult(self.errors[i], derivative)

            fourth = matrix.Matrix.mat_mul(gradient, matrix.Matrix.transpose(self.outputs[i]))

            # update weights with deltas
            delta_weights = matrix.Matrix.mult(fourth, self.lr)
            self.weights[i] = matrix.Matrix.mat_add(self.weights[i], delta_weights)

            # updating biases with deltas
            delta_biases = matrix.Matrix.mult(gradient, self.lr)
            self.biases[i] = matrix.Matrix.vector_add(self.biases[i], delta_biases)

    def save_json(self, name):
        weights_data = []
        biases_data = []
        for mat in self.weights:
            weights_data.append(mat.matrix_data)

        for vector in self.biases:
            biases_data.append(vector.matrix_data)

        data = {"weights": weights_data, "biases": biases_data, "structure": self.structure, "lr": self.lr}
        with open(name, "w") as outfile:
            json.dump(data, outfile, indent=4)


def load_json(name):
    with open(name) as infile:
        data = json.load(infile)

    nn = NeuralNetwork(data["structure"], lr=data["lr"])

    nn.weights = []
    nn.biases = []

    for weight in data["weights"]:
        nn.weights.append(matrix.Matrix(data=weight))

    for bias in data["biases"]:
        nn.biases.append(matrix.Matrix(data=bias))
    return nn
