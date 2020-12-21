# Patel, Meetkumar
# 1001-750-000
# 2020-10-12
# Assignment-02-01

import numpy as np


class LinearAssociator(object):
    def __init__(self, input_dimensions=2, number_of_nodes=4, transfer_function="Hard_limit"):
        """
        Initialize linear associator model
        :param input_dimensions: The number of dimensions of the input data
        :param number_of_nodes: number of neurons in the model
        :param transfer_function: Transfer function for each neuron. Possible values are:
        "Hard_limit", "Linear".
        """
        self.input_dimensions = input_dimensions
        # print(self.input_dimensions)
        self.number_of_nodes = number_of_nodes
        self.transfer_function = transfer_function.lower()
        self.initialize_weights()

    def initialize_weights(self, seed=None):
        """
        Initialize the weights, initalize using random numbers.
        If seed is given, then this function should
        use the seed to initialize the weights to random numbers.
        :param seed: Random number generator seed.
        :return: None
        """
        if seed != None:
            np.random.seed(seed)
        self.weights = np.array(np.random.randn(self.number_of_nodes,self.input_dimensions))

    def set_weights(self, W):
        """
         This function sets the weight matrix.
         :param W: weight matrix
         :return: None if the input matrix, w, has the correct shape.
         If the weight matrix does not have the correct shape, this function
         should not change the weight matrix and it should return -1.
         """
        if W.shape == (self.number_of_nodes,self.input_dimensions):
            self.weights = W
            return None
        else:
            return -1

    def get_weights(self):
        """
         This function should return the weight matrix(Bias is included in the weight matrix).
         :return: Weight matrix
         """
        return self.weights

    def predict(self, X):
        """
        Make a prediction on an array of inputs
        :param X: Array of input [input_dimensions,n_samples].
        :return: Array of model outputs [number_of_nodes ,n_samples]. This array is a numerical array.
        """
        self.input_matrix = X
        #calculation    
        self.output = np.dot(self.weights,self.input_matrix)
        if self.transfer_function == "hard_limit":
        # using hardlimit as activation function 
            for i in range(len(self.output)):
                for j in range(len(self.output[0])):
                    if self.output[i][j] >= 0:
                        self.output[i][j] = 1
                    else:
                        self.output[i][j] = 0
            prediction = self.output.astype(int)
        else:
            # using linear as activation function
            prediction = self.output
        return prediction

    def fit_pseudo_inverse(self, X, y):
        """
        Given a batch of data, and the targets,
        this function adjusts the weights using pseudo-inverse rule.
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [number_of_nodes,n_samples]
        """

        # making pseudo inverse of Input X
        self.X = X
        self.y = y
        inverse_X = np.linalg.inv(np.dot(self.X,self.X.transpose()))
        pseudoinv_X = np.dot(inverse_X,self.X)
        self.weights = np.dot(self.y,pseudoinv_X.transpose())


    def train(self, X, y, batch_size=5, num_epochs=10, alpha=0.1, gamma=0.9, learning="Delta"):
        """
        Given a batch of data, and the necessary hyperparameters,
        this function adjusts the weights using the learning rule.
        Training should be repeated num_epochs time.
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [number_of_nodes,n_samples].
        :param num_epochs: Number of times training should be repeated over all input data
        :param batch_size: number of samples in a batch
        :param num_epochs: Number of times training should be repeated over all input data
        :param alpha: Learning rate
        :param gamma: Controls the decay
        :param learning: Learning rule. Possible methods are: "Filtered", "Delta", "Unsupervised_hebb"
        :return: None
        """
        self.X = X
        self.batches = self.X.shape[1] // batch_size
        #handling case sensitive strings
        self.learning = learning.lower()
        
        for i in range(num_epochs):
            for j in range(self.batches):
                X_test = self.X[:,j*self.batches:(j+1)*self.batches]
                y_actual = y[:,j*self.batches:(j+1)*self.batches]
                Y_predict = self.predict(X_test)

                if self.learning =="filtered":
                    self.weights = (1-gamma)*self.weights + alpha*np.dot(y_actual,X_test.transpose())
                elif self.learning == "unsupervised_hebb":
                    self.weights = self.weights + alpha*(np.dot(Y_predict,X_test.transpose()))
                elif self.learning == "delta":
                    self.weights = np.add(self.weights,alpha*np.dot(y_actual-Y_predict,X_test.transpose()))
                else:
                    # for all other learning choices
                    exit("Invalid choice")
            
    def calculate_mean_squared_error(self, X, y):
        """
        Given a batch of data, and the targets,
        this function calculates the mean squared error (MSE).
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [number_of_nodes,n_samples]
        :return mean_squared_error
        """
        mean_sqre_err = np.mean((y - self.predict(X) )**2)
        return mean_sqre_err


input_dimensions = 5
number_of_nodes = 5
for i in range(10):
    model = LinearAssociator(input_dimensions=input_dimensions, number_of_nodes=number_of_nodes,
                    transfer_function="Linear")
    model.initialize_weights(seed=i+1)
    X_train = np.random.randn(input_dimensions, 100)
    out = model.predict(X_train)

    model.set_weights(np.random.randn(*model.get_weights().shape))

    model.train(X_train, out, batch_size=10, num_epochs=50, alpha=0.1, gamma=0.1, learning="delta")
    new_out = model.predict(X_train)
