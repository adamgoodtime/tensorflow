import numpy as np

# define XOR training data
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
])

y = np.atleast_2d([0, 1, 1, 0]).T

print('X.shape:', X.shape)
print('y.shape:', y.shape)

# defining network parameters
# [2, 2, 1] will also work for the XOR problem presented
LAYERS = [2, 2, 2, 1]
ETA = .1
THETA = []

# sigmoid activation function
def sigmoid(x):
    return 1/(1+np.exp(-x))

# derivative of sigmoid activation function
def sigmoid_prime(x):
    return sigmoid(x) * (1-sigmoid(x))

# calculating cost given the prediction and actual vectors
def cost(y_hat, y):
    return np.mean([_ * _ for _ in (y_hat - y)])

# initialing THETA params for all the layers
def initialize_parameters():
    for idx in range(1, len(LAYERS)):
        THETA.append(np.random.rand(LAYERS[idx], LAYERS[idx-1]+1))

# vectorized forward propagation
def forward_propagation(X,initialize=True):
    if initialize:
        initialize_parameters()
    # adding bias column to the input X
    A = [np.hstack((np.ones((X.shape[0],1)), X))]
    Z = []
    activate = False
    for idx, theta in enumerate(THETA):
        Z.append(np.matmul(A[-1], theta.T))
        # adding bias column to the output of previous layer
        A.append(np.hstack((np.ones((Z[-1].shape[0],1)), sigmoid(Z[-1]))))
    # bias is not needed in the final output
    A[-1] = A[-1][:, 1:]
    y_hat = A[-1]
    return A, Z, y_hat

# vectorized backpropagation
def back_propagation(X, y, initialize=True, debug=False, verbose=False):
    # run a forward pass
    A, Z, y_hat = forward_propagation(X, initialize)
    # calculate delta at final output
    del_ = [(y_hat - y) * sigmoid_prime(Z[len(Z)-1])]
    if verbose:
        print(cost(y_hat, y))
    # flag to signify whether a layer has bias column of not
    bias_free = True
    # running in reverse because delta is propagated backwards
    for idx in reversed(range(1, len(THETA))):
        if bias_free:
            # true only for the final layer where there is no bias
            temp = np.matmul(del_[0], THETA[idx]) * np.hstack((np.ones((Z[idx-1].shape[0], 1)), sigmoid_prime(Z[idx-1])))
            bias_free=False
        else:
            # true for all the layers except the input and output layer
            temp = np.matmul(del_[0][:,1:], THETA[idx]) * np.hstack((np.ones((Z[idx-1].shape[0], 1)), sigmoid_prime(Z[idx-1])))
        del_ = [temp] + del_
    del_theta = []
    bias_free = True
    # calculation for the delta in the parameters
    for idx in reversed(range(len(del_))):
        if bias_free:
            # true only for the final layer where there is no bias
            del_theta = [-ETA * np.matmul(del_[idx].T, A[idx])] + del_theta
            bias_free = False
        else:
            # true for all the layers except the input and output layer
            del_theta = [-ETA * np.matmul(del_[idx][:, 1:].T, A[idx])] + del_theta
    # update parameters
    for idx in range(len(THETA)):
        # asserting that the matrix sizes are same
        assert THETA[idx].shape == del_theta[idx].shape
        THETA[idx] = THETA[idx] + del_theta[idx]
    if debug:
        return (A, Z, y_hat, del_, del_theta)

# training epochs
initialize=True
verbose=True
THETA=[]
for i in range(10000):
    if i % 1000 == 0:
        verbose=True
    back_propagation(X, y, initialize, debug=False, verbose=verbose)
    verbose=False
    initialize=False

# inference after training
A, Z, y_hat = forward_propagation(X, initialize=False)

# final output of the network
print(y_hat)
