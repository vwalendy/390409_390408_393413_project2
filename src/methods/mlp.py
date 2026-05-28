import numpy as np

class MLP:
    def __init__(self, dimensions, activations):
        """
        :param dimensions: list of dimensions of the neural net. (input, hidden layer, ... ,hidden layer, output)
        :param activations: list of activation functions. Must contain N-1 activation function, where N = len(dimensions).

        Example of one hidden layer with
        - 2 inputs
        - 10 hidden nodes
        - 5 outputs
        layers -->    [0,        1,          2]
        ----------------------------------------
        dimensions =  (2,     10,          5)
        activations = (      Sigmoid,      Sigmoid)
        """

        ### WRITE YOUR CODE HERE
        self.dimensions = dimensions
        self.activations = activations
        self.n_layers = len(dimensions) - 1

        # Weights and biases, indexed by layer number 1..n_layers.
        self.W = {}
        self.b = {}
        for i in range(1, self.n_layers + 1):
            d_in = dimensions[i - 1]
            d_out = dimensions[i]
            # He-style scaling keeps activations from vanishing/exploding.
            self.W[i] = np.random.randn(d_in, d_out) * np.sqrt(2.0 / d_in)
            self.b[i] = np.zeros((1, d_out))

        self.learning_rate = 1e-3

    def feed_forward(self, x):
        """
        Execute a forward feed through the network.
        :param x: (array) Batch of input data vectors.
        :return: (tpl) Node outputs and activations per layer. The numbering of the output is equivalent to the layer numbers.
        """

        ### WRITE YOUR CODE HERE
        a = {}
        z = {0: x}
        for i in range(1, self.n_layers + 1):
            a[i] = z[i - 1] @ self.W[i] + self.b[i]
            z[i] = self.activations[i - 1].forward(a[i])
        return a, z


    def predict(self, x):
        """
        :param x: (array) Containing parameters
        :return: (array) A 2D array of shape (n_cases, n_classes).
        """

        ### WRITE YOUR CODE HERE
        _, z = self.feed_forward(x)
        return z[self.n_layers]


    def back_prop(self, z, a, y_true, loss):
        """
        The input dicts keys represent the layers of the net.
        a = { 0: x,
              1: f(w1(x) + b1)
              2: f(w2(a2) + b2)
              }
        :param a: (dict) w^T@x + b
        :param z: (dict) f(a)
        :param y_true: (array) One hot encoded truth vector.
        :param loss: Loss class with a static .gradient(y_true, y_pred) method.
        :return:
        """

        ### WRITE YOUR CODE HERE
        L = self.n_layers
        y_pred = z[L]

        delta = loss.gradient(y_true, y_pred) * self.activations[L - 1].gradient(a[L])

        for i in range(L, 0, -1):
            dw = z[i - 1].T @ delta
            db = np.sum(delta, axis=0, keepdims=True)
            if i > 1:
                delta = (delta @ self.W[i].T) * self.activations[i - 2].gradient(a[i - 1])
            self.update_w_b(i, dw, db)


    def update_w_b(self, index, dw, delta):
        """
        Update weights and biases.
        :param index: (int) Number of the layer
        :param dw: (array) Partial derivatives
        :param delta: (array) Delta error.
        """

        ### WRITE YOUR CODE HERE
        self.W[index] -= self.learning_rate * dw
        self.b[index] -= self.learning_rate * delta

    def fit(self, x, y_true, loss, epochs, batch_size, learning_rate=1e-3):
        """
        :param x: (array) Containing parameters
        :param y_true: (array) Containing one hot encoded labels.
        :param loss: Loss class (MSE, CrossEntropy etc.)
        :param epochs: (int) Number of epochs.
        :param batch_size: (int)
        :param learning_rate: (flt)
        """

        ### WRITE YOUR CODE HERE
        self.learning_rate = learning_rate
        N = x.shape[0]

        for _ in range(epochs):
            perm = np.random.permutation(N)
            x_shuf = x[perm]
            y_shuf = y_true[perm]
            for start in range(0, N, batch_size):
                xb = x_shuf[start:start + batch_size]
                yb = y_shuf[start:start + batch_size]
                a, z = self.feed_forward(xb)
                self.back_prop(z, a, yb, loss)
        return self
