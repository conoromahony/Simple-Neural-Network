# Implements a simple two-layer neural network.
# Input layer 𝑎[0] will have 784 units corresponding to the 784 pixels in each 28x28 input image. 
# A hidden layer 𝑎[1] will have 10 units with ReLU activation, 
# and finally our output layer 𝑎[2] will have 10 units corresponding to the ten digit classes with softmax activation.

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


# Create the initial weights and biases for the neural network.
def init_params():
    # Defines the weights for the conections to the nodes in layer 1. W1 is a 10 x 784 matrix with random values.
    W1 = np.random.rand(10, 784) - 0.5
    # Defines the biases for the nodes in layer 1. b1 is a 10 x 1 matrix with random values.
    b1 = np.random.rand(10, 1) - 0.5
    # Defines the weights for the conections to the nodes in layer 2. W2 is a 10 x 10 matrix with random values.
    W2 = np.random.rand(10, 10) - 0.5
    # Defines the biases for the nodes in layer 2. W1 is a 10 x 1 matrix with random values.
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2


# Implement the Rectified Linear Unit (ReLU) function. That is, the output is:
#  x if x > 0
#  0 if x <= 0
def ReLU(Z):
    return np.maximum(Z, 0)


# Implement the softmax function. That is, it translates the values to probabilities, between 0 and 1, that all add up to 1.
def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A

# Perform forward propagation for the hidden and ouput layers.
#   𝑍[1] = 𝑊[1]𝑋+𝑏[1]
#   𝐴[1] = 𝑔ReLU(𝑍[1]))
#   𝑍[2] = 𝑊[2]𝐴[1]+𝑏[2]
#   𝐴[2] = 𝑔softmax(𝑍[2])
def forward_prop(W1, b1, W2, b2, X):
    # Calculate the node values for layer 1 (the hiden layer).
    Z1 = W1.dot(X) + b1
    # Apply the activation function. We are using the Rectified Linear Unit (ReLU) function.
    A1 = ReLU(Z1)
    # Calculate the node values for layer 2 (the output layer).
    Z2 = W2.dot(A1) + b2
    # Apply the softmax function. The softmax function turns the output values into probabilities.
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2


# Implement the derivative of the activation function (i. the ReLU function).
def ReLU_deriv(Z):
    return Z > 0


# Implement "one hot" encoding for the labels in the training data. 
# That is, create a matrix with 1 in the position of the label, and 0's in all other positions.
def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

# Perform back propagation through the neural network. 
# Here are the calcuations for the weights and biases for layer 2 (i.e. the output layer)
#   𝑑𝑍[2]=𝐴[2]−𝑌      To determine the error for the output layer during training (i.e. dZ2), subtract a "one hot encoding" of the label from the probabilities.
#   𝑑𝑊[2]=1/𝑚 . 𝑑𝑍[2]𝐴[1]𝑇      That is, the average of the error values.
#   𝑑𝐵[2]=1/𝑚 . Σ𝑑𝑍[2]        
# Here are the calcuations for the weights and biases for layer 1 (i.e. the hidden layer)
#   𝑑𝑍[1]=𝑊[2]𝑇 . 𝑑𝑍[2].∗𝑔[1]′(𝑧[1])     Taking error from layer 2 (i.e. dZ2), and applying weights to it in reverse (i.e. transpose of W2). g' is the drivative of the activation function.
#   𝑑𝑊[1]=1/𝑚 . 𝑑𝑍[1]𝐴[0]𝑇
#   𝑑𝐵[1]=1/𝑚 . Σ𝑑𝑍[1]
def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2


# Update our parameters as follows:
#   𝑊[2]:=𝑊[2]−𝛼𝑑𝑊[2]
#   𝑏[2]:=𝑏[2]−𝛼𝑑𝑏[2]
#   𝑊[1]:=𝑊[1]−𝛼𝑑𝑊[1]
#   𝑏[1]:=𝑏[1]−𝛼𝑑𝑏[1]
# Alpha is the learning rate. Alpha is a hyper parameter (i.e. it is not trained by the model).
def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1    
    W2 = W2 - alpha * dW2  
    b2 = b2 - alpha * db2    
    return W1, b1, W2, b2


def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            print(get_accuracy(predictions, Y))
    return W1, b1, W2, b2




W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.10, 500)



def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

def test_prediction(index, W1, b1, W2, b2):
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()


# We are using the MNIST digit recognizer dataset.
# Use pandas  read the CSV file with the data.
data = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
data = np.array(data)
m, n = data.shape

# Shuffle before splitting into dev and training sets.
np.random.shuffle(data)

# Create the dev data from the first 1,000 items.
data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255.

# Create the training data from the remaining data.
data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.
_,m_train = X_train.shape
