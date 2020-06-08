# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 11:51:57 2020

@author: Michael
"""

import numpy as np
import copy
from math import exp
from random import random
from random import seed

###############################################################################
############################## FUNCTIONS ######################################

# use this function to create a weight vector and storage vector for a layer
def Layer(Input,output,Layer):
    neuron_outputs = []
    for i in np.arange(0,output):
        temp = []
        temp2 = []
        # Bias term adds an extra weight
        for j in np.arange(0,(Input+1)):
            # Generate the random weight
            temp.append(random())
            # generate the blank slate vector used for propagation later
            temp2.append(0)
        neuron_outputs.append(temp2)
        Layer.append(temp)
    return(Layer,neuron_outputs)

def neuron_output(input_signal, weights):
    output = 0
    # now we loop through the weights and compute the dot product against the 
    # input
    for i in range(len(weights)-1):
        output = output + input_signal[i] * weights[i]
    # the final weight is bias term, we so make sure include it
    output = output + weights[-1]
    return(output)
        
def Forward_Propagate(input_row, MLP, neuron_outputs):
    signal = input_row
    for i in np.arange(0,len(MLP)):
        # holds the input signal temporarily
        prop_inputs = []
        for j in np.arange(0,len(MLP[i])):
            # compute initial neuron output
            output = neuron_output(signal,MLP[i][j])
            # now feed output through the activation function
            neuron_outputs[i][j] = sigmoid(output)
            prop_inputs.append(neuron_outputs[i][j])
        signal = prop_inputs
    return(signal,neuron_outputs)

# Sigmoid activation function
def sigmoid(x):
    return(1.0/(1.0 + exp(-x)))
# derivative of the activation function that we need to create the error function
def derivative(x):
    return(x*(1.0-x))
  
# Softmax activation function for the output
def softmax(x):
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exps/np.sum(exps, axis=1, keepdims=True)


# error prediction function
def error(pred, real):
    n_samples = real.shape[0]
    logp = - np.log(pred[np.arange(n_samples), real.argmax(axis=1)])
    loss = np.sum(logp)/n_samples
    return loss


# back propagation algorithm
def Back_Propagate(Expected_output, neuron_outputs, output_changes, MLP):
    for i in reversed(np.arange(0,len(MLP))):
        #print(MLP[i])
        errors = list()
        # for the end of network
        if i == (len(MLP)-1):
            for j in np.arange(0,len(MLP[i])):
                # compare the neuron output against the actual output
                errors.append(Expected_output[j] - neuron_outputs[i][j])
        else:
            for j in np.arange(0,len(MLP[i])):
                error = 0
                for k in np.arange(0,len(MLP[i+1])):
                    # accumulate the error
                    error = error + MLP[i+1][k][j]*output_changes[i+1][k]
                errors.append(error)
        for j in np.arange(0,len(MLP[i])):
            # use the derivative of the activation function
            output_changes[i][j] = errors[j] * derivative(neuron_outputs[i][j])
    
    
# Weight updating function -> input for the output layer is outputs from hidden
def update(MLP, neuron_outputs, output_changes, input_row, Learning_Rate):
    # input_row is = row of data
    for i in np.arange(0,len(MLP)):
        # gives us inputs in reverse order (last to first)
        inputs = input_row[:-1]
        # if this isn't the first layer (ie. now comparing hidden to output layer)
        if i != 0:
            # take the outputs from the last layer
            inputs = []
            for m in np.arange(0,len(MLP[i-1])):
                inputs.append(neuron_outputs[i-1][m])
            #inputs = [neuron['output'] for neuron in network[i - 1]]
            #inputs = [neuron_outputs[i-1][0] for N in MLP[i-1]]
        # loop through the layer and update each neuron with respect to the inputs from the last year
        for k in np.arange(0,len(MLP[i])):
            # loop through the inputs
            for j in np.arange(0,len(inputs)):
                MLP[i][k][j] = MLP[i][k][j] + Learning_Rate*output_changes[i][k]*inputs[j]
            MLP[i][k][-1] = MLP[i][k][-1] + Learning_Rate*output_changes[i][k]


def train(MLP, neuron_outputs, output_changes, train_data, Learning_Rate, n_epochs, n_outputs):
    for epoch in range(n_epochs):
        error = 0
        for row in train_data:
            signal_output, neuron_outputs = Forward_Propagate(row,MLP,neuron_outputs)
            Expected_output = [0 for i in np.arange(0,n_outputs)]
            Expected_output[row[-1]] = 1
            error = error + sum([(Expected_output[i]-signal_output[i])**2 for i in range(len(Expected_output))])
            Back_Propagate(Expected_output, neuron_outputs, output_changes, MLP)
            update(MLP, neuron_outputs, output_changes, row, Learning_Rate)

        print('>epoch=%d, Learning_rate=%.3f, error=%.3f' % (epoch, Learning_Rate, error))

def predict(input_row, MLP, neuron_outputs):
    signal_outputs, neuron_outputs = Forward_Propagate(input_row, MLP, neuron_outputs)
    return(signal_outputs.index(max(signal_outputs)))

###############################################################################
#################                 SCRIPT                  #####################

# First Step: import training data/labels using numpy
#train_data = np.loadtxt(open("train_data.csv", "rb"), delimiter=",")
#train_labels = np.loadtxt(open("train_labels.csv", "rb"), delimiter=",")

seed(1)
dataset = [[2.7810836,2.550537003,0],
    [1.465489372,2.362125076,0],
    [3.396561688,4.400293529,0],
    [1.38807019,1.850220317,0],
    [3.06407232,3.005305973,0],
    [7.627531214,2.759262235,1],
    [5.332441248,2.088626775,1],
    [6.922596716,1.77106367,1],
    [8.675418651,-0.242068655,1],
    [7.673756466,3.508563011,1]]


n_inputs = len(dataset[0]) - 1
n_outputs = len(set([row[-1] for row in dataset]))

# now that we have the data in matrix we can initialize our MLP network
nn = []

# I have decided to set the HL to be 15 neurons
H = 2
HL = []
OL = []
neuron_outputs = []


# We need to randomly initialize the sets of weights in between the input and hidden 
# layer and hidden layer and output layer using our layer function
#HL = Layer(784,H,HL)
HL,o1 = Layer(n_inputs,H,HL)
neuron_outputs.append(o1)
nn.append(HL)

# Repeat for the output layer
#OL = Layer(15,4,OL)
OL,o2 = Layer(H,n_outputs,OL)
neuron_outputs.append(o2)
nn.append(OL)
output_changes = copy.deepcopy(neuron_outputs)


train(nn, neuron_outputs, output_changes, dataset, 0.5, 20, n_outputs)

for row in dataset:
    prediction = predict(row, nn, neuron_outputs)
    print('Expected=%d, Got=%d' % (row[-1], prediction))

# Now we implement Forward propagation through the MLP. Remember the activation 
# for each neuron is simply just the sum of the weights(for that neuron)
# multiplied by the input. This gives us the weighted input signal
