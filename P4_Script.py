# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 11:51:57 2020

Inspiration Reference: https://www.tensorflow.org/api_docs/python/tf/keras/Model

@author: Michael
"""
import numpy as np
from random import seed
import copy
########################### CLASS FOR MLP MODEL ###############################

# we are going to create a class to store our MLP variables (weights, bias)
class MLP:
    def __init__(self, input_data, hidden_dim, output_labels):
        
        # store the input data, and output_labels
        self.input_data = input_data
        self.output_labels = output_labels
        
        input_dim = input_data.shape[1]
        output_dim = output_labels.shape[1]
        
        # Hidden layer weight initialization + bias term initialization
        self.w1 = np.random.randn(input_dim, hidden_dim)
        self.b1 = np.zeros((1, hidden_dim))
        
        # Output layer weight initialization + bias term initialization
        self.w2 = np.random.randn(hidden_dim, output_dim)
        self.b2 = np.zeros((1, output_dim))



######################### FUNCTIONS FOR MLP MODEL #############################

# Sigmoid activation function
def sigmoid(x):
    return(1.0/(1.0 + np.exp(-x)))
# derivative of the activation function that we need to create the error function
def derivative(x):
    return(x*(1.0-x))

# softmax activation function for the output layer
def softmax(x):
    exponent = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exponent/np.sum(exponent, axis=1, keepdims=True)

# cross entropy loss functions instead of MSE
    
def cross_entropy(pred, real):
    n_samples = real.shape[0]
    res = pred - real
    return res/n_samples
# calculate error/loss 
def error(pred, real):
    n_samples = real.shape[0]
    logp = - np.log(pred[np.arange(n_samples), real.argmax(axis=1)])
    loss = np.sum(logp)/n_samples
    return loss

def Forward_Propagate(MLP):
    
    # Forward propagation is just the inputs dot producted with the weights
    # Each "neuron" has a number of inputs = to the previous layers neurons/outputs
    # To do this we use the matrix product (can also use a loop but its slower)
    Layer_Output_1 = np.dot(MLP.input_data, MLP.w1) + MLP.b1
    MLP.O1 = sigmoid(Layer_Output_1)
    
    Layer_Output_2 = np.dot(MLP.O1, MLP.w2) + MLP.b2
    MLP.O2 = softmax(Layer_Output_2)
    
def Back_Propagate(MLP,LR):
    
    # for back propagation we go backwards from the output towards the input 
    # First use cross_entropy to compute the error between actual vs pred output
    
    # Error for W2
    error_2 = cross_entropy(MLP.O2, MLP.output_labels)
    
    # Signal change sent back
    O2_delta = np.dot(error_2, MLP.w2.T)
    
    # Error for W1
    error_1 = O2_delta * derivative(MLP.O1)

    MLP.w2 = MLP.w2 - LR * np.dot(MLP.O1.T, error_2)
    MLP.b2 = MLP.b2 - LR * np.sum(error_2, axis=0, keepdims=True)
    MLP.w1 = MLP.w1 - LR * np.dot(MLP.input_data.T, error_1)
    MLP.b1 = MLP.b1 - LR * np.sum(error_1, axis=0)

def predict(MLP, data):
    
    MLP.input_data = data
    Forward_Propagate(MLP)
    
    return(MLP.O2.argmax())

def train(MLP_Model,n_epochs,LR):
    
    val_accuracy = 0
    # Run for each of the epochs
    for i in np.arange(0,n_epochs):
        # For each epoch we forward propagate data to achieve output and then 
        # backpropagate with the error to adjust weights
        Forward_Propagate(MLP_Model)
        
        # after forward propagation calculate the error or loss at the output
        loss = error(MLP_Model.O2, MLP_Model.output_labels)
        print('Error :', loss)
        
        # Now use back prop to update the weights
        Back_Propagate(MLP_Model,LR)
        
        MLP_Copy = copy.deepcopy(MLP_Model)
        temp = get_acc(x_val, np.array(y_val),MLP_Copy)
        
        # If validation accuracy drops below 0.5% threshold the training ends early 
        if((temp - val_accuracy < -0.5) and (i > 10)):
            val_accuracy = temp
            print("Training accuracy : ", get_acc(x_train, np.array(y_train)), MLP_Copy)
            print("Validation accuracy : ", val_accuracy)
            break
        else:
            val_accuracy = temp
            print("Training accuracy : ", get_acc(x_train, np.array(y_train)), MLP_Copy)
            print("Validation accuracy : ", val_accuracy)
		
def get_acc(data, labels, MLP):
    Accuracy = 0
    for x,y in zip(data, labels):
        s = predict(MLP, x)
        if s == np.argmax(y):
            Accuracy = Accuracy + 1
    return(Accuracy/len(data)*100)

def test_train_split(train_data,train_labels,Split):
    
    Train_size = round(len(train_data)*Split)
    
    # random permutation of the data in order to create a validation/test sets
    idx = np.random.permutation(train_data.shape[0])
    training_idx, test_idx = idx[:Train_size], idx[Train_size:]
    x_train, x_test = train_data[training_idx,:], train_data[test_idx,:]
    y_train, y_test = train_labels[training_idx,:], train_labels[test_idx,:]
    
    return(x_train,y_train,x_test, y_test)
    

############################### SCRIPT ########################################

# Load training data and labels
train_data = np.loadtxt(open("train_data.csv", "rb"), delimiter=",")
train_labels = np.loadtxt(open("train_labels.csv", "rb"), delimiter=",")

# random seed
seed(42)

# create test/train split
[x_train, y_train, x_test, y_test] = test_train_split(train_data,train_labels,0.9)

# create val/train split
[x_train, y_train, x_val, y_val] = test_train_split(x_train,y_train,0.8)

LR = 0.5
Hidden_Layer_Dim = 640
n_epochs = 250

# Initialize the MLP model using the input data, hidden layer, and # of outputs
# to shape the layers correctly
MLP_Model = MLP(x_train, Hidden_Layer_Dim, np.array(y_train))

# Now train the model
train(MLP_Model, n_epochs, LR)

	
print("\n\nFinal Training accuracy : ", get_acc(x_train, np.array(y_train)),MLP_Model)
print("Final Validation accuracy : ", get_acc(x_val, np.array(y_val)),MLP_Model)
print("Test accuracy : ", get_acc(x_test, np.array(y_test)),MLP_Model)