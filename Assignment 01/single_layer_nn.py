# Patel, Meetkumar
# 1001-750-000
# 2020-09-28
# Assignment-01-01

import numpy as np

class SingleLayerNN(object):
    def __init__(self, input_dimensions=2,number_of_nodes=4):
        """
        Initialize SingleLayerNN model and set all the weights and biases to random numbers.
        :param input_dimensions: The number of dimensions of the input data
        :param number_of_nodes: Note that number of neurons in the model is equal to the number of classes.
        """
        self.input_dimensions = input_dimensions
        self.number_of_nodes = number_of_nodes
        self.initialize_weights()

    def initialize_weights(self,seed=None):

        """
        Initialize the weights, initalize using random numbers.
        If seed is given, then this function should
        use the seed to initialize the weights to random numbers.
        :param seed: Random number generator seed.
        :return: None
        """
        #weights = weight matrix

        if seed != None:
            np.random.seed(seed)
        self.weights = np.array(np.random.randn(self.number_of_nodes,self.input_dimensions+1))

        # self.bias = np.random.randn(self.input_dimensions,1)
        # self.weights = np.c_[self.bias,self.weights]

    def set_weights(self, W):
        """
        This function sets the weight matrix (Bias is included in the weight matrix).
        :param W: weight matrix
        :return: None if the input matrix, w, has the correct shape.
        If the weight matrix does not have the correct shape, this function
        should not change the weight matrix and it should return -1.
        """
        #append weight matrix to biases

        if W.shape == (self.number_of_nodes,(self.input_dimensions+1)):
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

    def ip_matrix(self,X):
        self.batch_size = X.shape[1]
        array_of_ones = np.ones((1,self.batch_size),dtype=int)
        input_new = np.r_[array_of_ones,X]
        return input_new

    def predict(self, X):
        """
        Make a prediction on a batach of inputs.
        :param X: Array of input [input_dimensions,n_samples]
        :return: Array of model [number_of_nodes ,n_samples]
        Note that the activation function of all the nodes is hard limit.
        """
        # as bias is included in the weight matrix, input matrix is being set according to it 
        # by adding a row of '1' as the first row 

        input2 = self.ip_matrix(X)

        #calculation
        self.output = np.dot(self.weights,input2)

        # using hardlimit as activation function 
        for i in range(len(self.output)):
            for j in range(len(self.output[0])):
                if self.output[i][j] >= 0:
                    self.output[i][j] = 1
                else:
                    self.output[i][j] = 0

        self.prediction = self.output.astype(int)
        # print("\n Prediction:\n",self.prediction)
        return self.prediction

    def train(self, X, Y, num_epochs=10,  alpha=0.1):
        """
        Given a batch of input and desired outputs, and the necessary hyperparameters (num_epochs and alpha),
        this function adjusts the weights using Perceptron learning rule.
        Training should be repeated num_epochs times.
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [number_of_nodes ,n_samples]
        :param num_epochs: Number of times training should be repeated over all input data
        :param alpha: Learning rate
        :return: None
        """
        # self.batch_size = X.shape[1]
        # array_of_ones = np.ones((1,self.batch_size),dtype=int)
        # input_new = np.r_[array_of_ones,X]

        input1 = self.ip_matrix(X)

        for i in range(num_epochs):
            predict_new = self.predict(X)
            error = Y - predict_new
            # print("Error Matrix",error)
            self.weights = self.weights + alpha*np.dot(error,input1.T)

    def calculate_percent_error(self,X, Y):
        """
        Given a batch of input and desired outputs, this function calculates percent error.
        For each input sample, if the output is not the same as the desired output, Y,
        then it is considered one error. Percent error is 100*(number_of_errors/ number_of_samples).
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [number_of_nodes ,n_samples]
        :return percent_error
        """

        new_prediction = self.predict(X)
        error_new = Y - new_prediction
        counter = 0
        for i in range(len(error_new[0])):
            for j in range(len(error_new)):
                if error_new[j][i] != 0:
                    counter = counter + 1
                    break
                else:
                    counter = counter

        percentage_error = counter/len(error_new[0]) * 100
        return percentage_error


if __name__ == "__main__":
    input_dimensions = 2
    number_of_nodes = 2

    model = SingleLayerNN(input_dimensions=input_dimensions, number_of_nodes=number_of_nodes)
    model.initialize_weights(seed=2)
    X_train = np.array([[-1.43815556, 0.10089809, -1.25432937, 1.48410426],
                        [-1.81784194, 0.42935033, -1.2806198, 0.06527391]])
    print("\nInitial Prediction: \n",model.predict(X_train))
    Y_train = np.array([[1, 0, 0, 1], [0, 1, 1, 0]])
    print("\n****** Model weights ******\n",model.get_weights())
    print("\n****** Input samples ******\n",X_train)
    print("\n****** Desired Output ******\n",Y_train)
    percent_error=[]
    for k in range (20):
        model.train(X_train, Y_train, num_epochs=1, alpha=0.1)
        percent_error.append(model.calculate_percent_error(X_train,Y_train))
    print("******  Percent Error ******\n",percent_error)
    print("****** Model weights ******\n",model.get_weights())

    predict_final = model.predict(X_train) 
    print("\nFinal prediction: \n",predict_final)