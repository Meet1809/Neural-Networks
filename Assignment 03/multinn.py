# Patel, Meetkumar
# 1001-750-000
# 2020_10_12
# Assignment-03-01

# %tensorflow_version 2.x
import tensorflow as tf
import numpy as np
import sys
import os
import imp
from sklearn.metrics import confusion_matrix


class MultiNN(object):
    def __init__(self, input_dimension):
        """
        Initialize multi-layer neural network
        :param input_dimension: The number of dimensions for each input data sample
        """
        self.input_dimension = input_dimension
        self.weights = []
        self.biases = []
        self.layers_nodes=[]
        # self.weights = []
        self.transfer_functions_list = []
        self.counter = 0

    def add_layer(self, num_nodes, transfer_function="Linear"):
        """
         This function adds a dense layer to the neural network
         :param num_nodes: number of nodes in the layer
         :param transfer_function: Activation function for the layer. Possible values are:
        "Linear", "Relu","Sigmoid".
         :return: None
         """
        #layers_nodes is list, stores all nodes values
        self.layers_nodes.append(num_nodes)
        self.transfer_functions_list.append(transfer_function.lower())

        self.layer_no = self.counter
        self.counter = self.counter + 1

        if self.layer_no ==0:
            temp_inp = self.input_dimension
        else:
            temp_inp = self.layers_nodes[self.counter - 2]

        weights = tf.Variable(tf.random.normal(shape=(temp_inp,num_nodes)))
        biases = tf.Variable(tf.random.normal(shape=(1,num_nodes)))   
        self.weights.append(weights)
        self.biases.append(biases) 

    def get_weights_without_biases(self, layer_number):
        """
        This function should return the weight matrix (without biases) for layer layer_number.
        layer numbers start from zero.
         :param layer_number: Layer number starting from layer 0. This means that the first layer with
          activation function is layer zero
         :return: Weight matrix for the given  layer (not including the biases). Note that the shape of the weight matrix should be
          [input_dimensions][number of nodes]
         """

        return (self.weights[layer_number])

    def get_biases(self, layer_number):
        """
        This function should return the biases for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
         :param layer_number: Layer number starting from layer 0
         :return: Weight matrix for the given layer (not including the biases).
         Note that the biases shape should be [1][number_of_nodes]
         """

        return (self.biases[layer_number])

    def set_weights_without_biases(self, weights, layer_number):
        """
        This function sets the weight matrix for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
         :param weights: weight matrix (without biases). Note that the shape of the weight matrix should be
          [input_dimensions][number of nodes]
         :param layer_number: Layer number starting from layer 0
         :return: none
         """

        self.weights[layer_number] = weights

    def set_biases(self, biases, layer_number):
        """
        This function sets the biases for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
        :param biases: biases. Note that the biases shape should be [1][number_of_nodes]
        :param layer_number: Layer number starting from layer 0
        :return: none
        """

        self.biases[layer_number] = biases

    def calculate_loss(self, y, y_hat):
        """
        This function calculates the sparse softmax cross entropy loss.
        :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
         the desired (true) class.
        :param y_hat: Array of actual output values [n_samples][number_of_classes].
        :return: loss
        """

        return tf.reduce_mean(tf.softmax_cross_entropy_with_logits(labels=y, logits=y_hat))

    def transferfunc_output(self,net,trans_func):
        fun = trans_func
        net = net
        if fun == "relu":
            # using ReLu as activation function 
            prediction = tf.nn.relu(net)
        elif fun == "sigmoid":
            # using sigmoid as activation function
            prediction = tf.math.sigmoid(net)
        elif fun == "linear":
            #using linear as activation function
            prediction = net
        else:
            print("Invalid choice of transfer function !!!")
            return -1
        return prediction

    def predict(self, X):
        """
        Given array of inputs, this function calculates the output of the multi-layer network.
        :param X: Array of input [n_samples,input_dimensions].
        :return: Array of outputs [n_samples,number_of_classes ]. This array is a numerical array.
        """
        self.input_matrix = X
        self.transition_function = self.transfer_functions_list[self.layer_no]
        number_of_samples = X.shape[0]
        #making biases' size

        for i in range(len(self.biases)):
            self.biases[i] = tf.repeat(self.biases[i],repeats=number_of_samples,axis=0)

        for i in range(len(self.weights)):
            if i ==0:
                add = tf.Variable(tf.matmul(self.input_matrix,self.weights[i]))
                net = add.assign_add(self.biases[i])
                self.output = self.transferfunc_output(net,self.transition_function)

            else:
                add = tf.Variable(tf.matmul(self.output,self.weights[i]))
                net = add.assign_add(self.biases[i])
                self.output = self.transferfunc_output(net,self.transition_function)

        return self.output

    def train(self, X_train, y_train, batch_size, num_epochs, alpha=0.8):
        """
         Given a batch of data, and the necessary hyperparameters,
         this function trains the neural network by adjusting the weights and biases of all the layers.
         :param X: Array of input [n_samples,input_dimensions]
         :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
         the desired (true) class.
         :param batch_size: number of samples in a batch
         :param num_epochs: Number of times training should be repeated over all input data
         :param alpha: Learning rate
         :return: None
         """

        self.batches = X_train.shape[1] // batch_size
        for i in range(num_epochs):
            # for j in range(self.batches):
            with tf.GradientTape as gt:
                
                X_test = X_train[:,i*self.batches:(i+1)*self.batches]
                y_test = y_train[:,i*self.batches:(i+1)*self.batches]
                # gt.watch(self.weights)
                # gt.watch(self.biases)
                Y_predict = self.predict(X_test)
                loss = self.calculate_loss(y_test,Y_predict)
                dloss_dw, dloss_db = gt.gradient(loss, self.weights,self.biases)  

            # for each layer
            for k in range(len(self.weights)):   
                self.weights[k].assign_sub(alpha*dloss_dw[k])
                self.biases[k].assign_sub(alpha*dloss_db[k])

    def calculate_percent_error(self, X, y):
        """
        Given input samples and corresponding desired (true) output as indexes,
        this method calculates the percent error.
        For each input sample, if the predicted class output is not the same as the desired class,
        then it is considered one error. Percent error is number_of_errors/ number_of_samples.
        Note that the predicted class is the index of the node with maximum output.
        :param X: Array of input [n_samples,input_dimensions]
        :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
        the desired (true) class.
        :return percent_error
        """

        y_predict = self.predict(X)
        err = y-y_predict
        counter = 0
        for i in range(len(err[0])):
            for j in range(len(err)):
                if err!=0:
                    counter = counter + 1
                    break
                else:
                    counter = counter 
        percent_err = counter / len(err[0]) * 100
        return percent_err

    def calculate_confusion_matrix(self, X, y):
        """
        Given input samples and corresponding desired (true) outputs as indexes,
        this method calculates the confusion matrix.
        :param X: Array of input [n_samples,input_dimensions]
        :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
        the desired (true) class.
        :return confusion_matrix[number_of_classes,number_of_classes].
        Confusion matrix should be shown as the number of times that
        an image of class n is classified as class m.
        """

        number_of_classes = self.weights[-1].shape[1]
        confusion_matrix = np.zeros(shape = (number_of_classes, number_of_classes))



        #some encoding, that i learned from some websites
        # new_y = []
        # for i in y:
        #     output = [0 for _ in range(number_of_classes)]
        #     output[i] = 1
        #     new_y.append(output)
        # new_y = np.array(new_y)

        prediction = self.predict(X)
        prediction = tf.argmax(prediction,1)

        for i in range(y.shape[0]):
            confusion_matrix[y[i],prediction[i]] +=1
        return confusion_matrix
