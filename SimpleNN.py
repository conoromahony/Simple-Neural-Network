# Implements a simple two-layer neural network. Input layer 𝑎[0] will have 784 units corresponding to the 784 pixels in each 28x28 input image. 
# A hidden layer 𝑎[1] will have 64 units with ReLU activation, and finally our output layer 𝑎[2] will have 10 units corresponding to the ten digit
# classes with softmax activation.
# Video: https://www.youtube.com/watch?v=w8yWXqWQYmU
# Blog post: https://www.samsonzhang.com/2020/11/24/understanding-the-math-behind-neural-networks-by-building-one-from-scratch-no-tf-keras-just-numpy

# This code uses Gradient Descent. The basic idea of gradient descent is to figure out what direction each parameter can go in to decrease error
# by the greatest amount, then nudge each parameter in its corresponding direction over and over again until the parameters for minimum error and
# highest accuracy are found. In a neural network, gradient descent is carried out via a process called backward propagation. We take a prediction,
# calculate an error of how off it was from the actual value, then run this error backwards through the network to find out how much each weight
# and bias parameter contributed to this error. Once we have these error derivative terms, we can nudge our weights and biases accordingly to improve
# our model. Do it enough times, and we'll have a neural network that can recognize handwritten digits accurately.

# To run:
#  - Go to Programming > SimpleNN
#  - Type "python SimpleNN.py"
#  - Edit with IDLE
#  - git commit -m "message"
#  - git push

# TODO:
#  - Refactor the code so the number of layers is not hardcoded.
#  - Add other activation functions.
#  - Consider fancier architectures like CNNs.
#  - Consider variations of gradient descent that improve training efficiency: gradient descent with momentum, RMSProp, and Adam optimization. 
#  - Create a package for the refactored neural network code.
#  - Create a way to visualize what's happening in the neural network.

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


# Create the initial weights and biases for the neural network.
# Note: it's best practice to initialize your weights/biases close to 0, otherwise your gradients get really small really quickly:
# https://stackoverflow.com/questions/47240308/differences-between-numpy-random-rand-vs-numpy-random-randn-in-python
def init_params():
    # Defines the weights for the conections to the nodes in layer 1. W1 is a 64 x 784 matrix with random values.
    # We subtract 0.5 from the random values so we end up with numbers between -0.5 and 0.5 (rather than 0 and 1),
    W1 = np.random.rand(64, 784) - 0.5
    # Defines the biases for the nodes in layer 1. b1 is a 64 x 1 matrix with random values.
    b1 = np.random.rand(64, 1) - 0.5
    # Defines the weights for the conections to the nodes in layer 2. W2 is a 10 x 64 matrix with random values.
    W2 = np.random.rand(10, 64) - 0.5
    # Defines the biases for the nodes in layer 2. W1 is a 10 x 1 matrix with random values.
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2


# Implement the Rectified Linear Unit (ReLU) function. That is, a simple linear function that returns:
#   x if x > 0
#   0 if x <= 0
def ReLU(Z):
    return np.maximum(Z, 0)


# Implement the softmax function. That is, it translates the values to probabilities, between 0 and 1, that all add up to 1.
# Softmax takes a column of data at a time, taking each element in the column and outputting the exponential of that element divided by the
# sum of the exponentials of each of the elements in the input column.
def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A


# Perform forward propagation for the hidden and ouput layers.
#   𝑍[1] = 𝑊[1]𝑋+𝑏[1]
#   𝐴[1] = 𝑔ReLU(𝑍[1]))
#   𝑍[2] = 𝑊[2]𝐴[1]+𝑏[2]
#   𝐴[2] = 𝑔softmax(𝑍[2])
def forward_prop(W1, b1, W2, b2, X):
    # Calculate the node values for layer 1 (the hiden layer). Remember W1 is a numpy array, so we can use .dot for matrix operations.
    Z1 = W1.dot(X) + b1
    # Apply the activation function. We are using the Rectified Linear Unit (ReLU) function.
    A1 = ReLU(Z1)
    # Calculate the node values for layer 2 (the output layer).
    Z2 = W2.dot(A1) + b2
    # Apply the softmax function. The softmax function turns the output values into probabilities.
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2


# Implement the derivative of the activation function (i. the ReLU function).
# Note that the slope of the ReLU function when X is less than zero is 0, and the slope of the ReLU function when X is greater than zero is 1.
def ReLU_deriv(Z):
    # When booleans convert to numbers, true converts to 1 and false converts to 0.
    # Z > 0 is true when any one element of Z is greater than 0 (ie. the function returns 1)
    # Z > 0 is false when no element of Z is greater than 0 (i.e. the function returns 0)
    return Z > 0


# Implement "one hot" encoding for the labels in the training data. That is, create a matrix for all images, where each column represents an image label.
# Put 1 in the position of the label, and 0's in all other positions.
def one_hot(Y):
    # Create an m x 10 matrix.  Y.size is m.  Y.max() is 9 (i.e. the biggest value when working with the digits 0-9 is 9).
    # Initialize the matrix to have zeros in all positions.
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    # For each row identified by np.arange(Y.size), change the value in column Y to 1.
    one_hot_Y[np.arange(Y.size), Y] = 1
    # Transpose the matrix, so each column represents an image label. That is, return a 10 x m matrix.
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
# Note that one commenter wrote that... I believe dZ[2] should be 2(A[2]−Y) because the error/cost at the final output layer should be (A[2]−Y)^2. 
def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    m = Y.size
    one_hot_Y = one_hot(Y)
    # The closer the prediction probability is to 1, the closer the loss is to 0. By minimizing the cost function, we improve the accuracy of our model.
    # We do so by substracting the derivative of the loss function with respect to each parameter from that parameter over many rounds of graident descent.
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


# Return the first column of A2.
def get_predictions(A2):
    return np.argmax(A2, 0)


# Get the accuracy between the predictions (i.e. A2) and Y (i.e. the labels).
def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size


# This pulls everything together. It initializes the parameters, performs the forward propagation, the backward propagation, and dates the parameters.
# It does this iteration times, and it prints out an update every 10 iterations.
def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            print("Iteration: ", i)
            # A2 are the predictions that come out the other end of forward propagation.
            predictions = get_predictions(A2)
            # Y are the image labels.
            print(get_accuracy(predictions, Y))
    return W1, b1, W2, b2


def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions


# Test the neural network's prediction for the image at the "index" parameter.
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


# We are using the MNIST digit recognizer dataset. MNIST ("Modified National Institute of Standards and Technology") is the de facto “hello world”
# dataset of computer vision. Since its release in 1999, this classic dataset of handwritten images has served as the basis for benchmarking
# classification algorithms. Use pandas to read the CSV file with the data.
data = pd.read_csv('train.csv')
# Use numpy to load the CSV data into an array.
data = np.array(data)
# Get the dimensions of the array. There are m rows (i.e. images). Each image has n (i.e. 785 values; one for the label and 784 for the pixels)
m, n = data.shape

# Shuffle the data before splitting into dev and training sets.
np.random.shuffle(data)

# Create the dev data (i.e. validation data) from the first 1,000 images.
# Remember to transpose the matrix, so each column (rather than row) is an image.
data_dev = data[0:1000].T
# Now, Y_dev (i.e. the image label) will just be the first row.
Y_dev = data_dev[0]
# And X_dev will be the image pixels.
X_dev = data_dev[1:n]
X_dev = X_dev / 255.

# Create the training data from the remaining images. There are something like 41,000 of them.
# Again, remember to transpose the matrix so each column is an image.
data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.

# Run the neural network for 500 iterations on the training set, with an alpha of 0.1.
W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.10, 500)


# Test the neural network's prediction for the images at indexes 0, 1, 2, and 3.
test_prediction(0, W1, b1, W2, b2)
test_prediction(1, W1, b1, W2, b2)
test_prediction(2, W1, b1, W2, b2)
test_prediction(3, W1, b1, W2, b2)

# Let's have a look at the accuracy for the validation set.
dev_predictions = make_predictions(X_dev, W1, b1, W2, b2)
validation_accuracy = get_accuracy(dev_predictions, Y_dev)
print(validation_accuracy)

