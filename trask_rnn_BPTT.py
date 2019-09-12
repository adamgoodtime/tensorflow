import copy, numpy as np

np.random.seed(0)


# compute sigmoid nonlinearity
def sigmoid(x):
    output = 1 / (1 + np.exp(-x))
    return output


# convert output of sigmoid function to its derivative
def sigmoid_output_to_derivative(output):
    return output * (1 - output)


# training dataset generation
int2binary = {}
binary_dim = 8

largest_number = pow(2, binary_dim)
binary = np.unpackbits(
    np.array([range(largest_number)], dtype=np.uint8).T, axis=1)
for i in range(largest_number):
    int2binary[i] = binary[i]

# input variables
alpha = 0.1
input_dim = 2
hidden_dim = 16
output_dim = 1

# initialize neural network weights
synapse_0 = 2 * np.random.random((input_dim, hidden_dim)) - 1
synapse_1 = 2 * np.random.random((hidden_dim, output_dim)) - 1
synapse_h = 2 * np.random.random((hidden_dim, hidden_dim)) - 1

synapse_0_update = np.zeros_like(synapse_0)
synapse_1_update = np.zeros_like(synapse_1)
synapse_h_update = np.zeros_like(synapse_h)

def trask_BPTT(connection_list, in_dim, h_dim, o_dim, error_return=False):
    global synapse_0, synapse_1, synapse_h, input_dim, hidden_dim, output_dim

    input_dim = in_dim
    hidden_dim = h_dim
    output_dim = o_dim

    # synapse_0 = connection_matrix[0:in_dim]#.T  # or flipped
    # synapse_h = connection_matrix[in_dim:in_dim+h_dim]#.T
    # synapse_1 = connection_matrix[in_dim+h_dim:]#.T
    synapse_0 = connection_list[0:(input_dim * hidden_dim)].reshape(input_dim, hidden_dim)
    synapse_h = connection_list[(input_dim * hidden_dim):(input_dim * hidden_dim) + (hidden_dim * hidden_dim)].reshape(hidden_dim, hidden_dim)
    synapse_1 = connection_list[(input_dim * hidden_dim) + (hidden_dim * hidden_dim):].reshape(hidden_dim, output_dim)

    synapse_0_update = np.zeros_like(synapse_0)
    synapse_1_update = np.zeros_like(synapse_1)
    synapse_h_update = np.zeros_like(synapse_h)

    np.random.seed(27272)
    # generate a simple addition problem (a + b = c)
    a_int = np.random.randint(largest_number / 2)  # int version
    a = int2binary[a_int]  # binary encoding

    b_int = np.random.randint(largest_number / 2)  # int version
    b = int2binary[b_int]  # binary encoding

    # true answer
    c_int = a_int + b_int
    c = int2binary[c_int]

    # where we'll store our best guess (binary encoded)
    d = np.zeros_like(c)

    overallError = 0

    layer_2_deltas = list()
    layer_1_values = list()
    layer_1_values.append(np.zeros(hidden_dim))

    # moving along the positions in the binary encoding
    for position in range(binary_dim):
        # generate input and output
        X = np.array([[a[binary_dim - position - 1], b[binary_dim - position - 1]]])
        y = np.array([[c[binary_dim - position - 1]]]).T

        # hidden layer (input ~+ prev_hidden)
        layer_1 = sigmoid(np.dot(X, synapse_0) + np.dot(layer_1_values[-1], synapse_h))

        # output layer (new binary representation)
        layer_2 = sigmoid(np.dot(layer_1, synapse_1))

        # did we miss?... if so, by how much?
        layer_2_error = y - layer_2
        layer_2_deltas.append((layer_2_error) * sigmoid_output_to_derivative(layer_2))
        overallError += np.abs(layer_2_error[0])

        # decode estimate so we can print it out
        d[binary_dim - position - 1] = np.round(layer_2[0][0])

        # store hidden layer so we can use it in the next timestep
        layer_1_values.append(copy.deepcopy(layer_1))

    if error_return:
        return overallError

    future_layer_1_delta = np.zeros(hidden_dim)

    for position in range(binary_dim):
        X = np.array([[a[position], b[position]]])
        layer_1 = layer_1_values[-position - 1]
        prev_layer_1 = layer_1_values[-position - 2]

        # error at output layer
        layer_2_delta = layer_2_deltas[-position - 1]
        # error at hidden layer
        layer_1_delta = (future_layer_1_delta.dot(synapse_h.T) + layer_2_delta.dot(
            synapse_1.T)) * sigmoid_output_to_derivative(layer_1)

        # let's update all our weights so we can try again
        synapse_1_update += np.atleast_2d(layer_1).T.dot(layer_2_delta)
        synapse_h_update += np.atleast_2d(prev_layer_1).T.dot(layer_1_delta)
        synapse_0_update += X.T.dot(layer_1_delta)

        future_layer_1_delta = layer_1_delta

    synapse_0 += synapse_0_update * alpha
    synapse_1 += synapse_1_update * alpha
    synapse_h += synapse_h_update * alpha

    return np.concatenate([synapse_0_update.reshape(input_dim * hidden_dim),
                           synapse_h_update.reshape(hidden_dim * hidden_dim),
                           synapse_1_update.reshape(hidden_dim * output_dim)])

if __name__ == '__main__':
    # training logic
    for j in range(10000):

        # generate a simple addition problem (a + b = c)
        a_int = np.random.randint(largest_number / 2)  # int version
        a = int2binary[a_int]  # binary encoding

        b_int = np.random.randint(largest_number / 2)  # int version
        b = int2binary[b_int]  # binary encoding

        # true answer
        c_int = a_int + b_int
        c = int2binary[c_int]

        # where we'll store our best guess (binary encoded)
        d = np.zeros_like(c)

        overallError = 0

        layer_2_deltas = list()
        layer_1_values = list()
        layer_1_values.append(np.zeros(hidden_dim))

        # moving along the positions in the binary encoding
        for position in range(binary_dim):
            # generate input and output
            X = np.array([[a[binary_dim - position - 1], b[binary_dim - position - 1]]])
            y = np.array([[c[binary_dim - position - 1]]]).T

            # hidden layer (input ~+ prev_hidden)
            layer_1 = sigmoid(np.dot(X, synapse_0) + np.dot(layer_1_values[-1], synapse_h))

            # output layer (new binary representation)
            layer_2 = sigmoid(np.dot(layer_1, synapse_1))

            # did we miss?... if so, by how much?
            layer_2_error = y - layer_2
            layer_2_deltas.append((layer_2_error) * sigmoid_output_to_derivative(layer_2))
            overallError += np.abs(layer_2_error[0])

            # decode estimate so we can print it out
            d[binary_dim - position - 1] = np.round(layer_2[0][0])

            # store hidden layer so we can use it in the next timestep
            layer_1_values.append(copy.deepcopy(layer_1))

        future_layer_1_delta = np.zeros(hidden_dim)

        for position in range(binary_dim):
            X = np.array([[a[position], b[position]]])
            layer_1 = layer_1_values[-position - 1]
            prev_layer_1 = layer_1_values[-position - 2]

            # error at output layer
            layer_2_delta = layer_2_deltas[-position - 1]
            # error at hidden layer
            layer_1_delta = (future_layer_1_delta.dot(synapse_h.T) + layer_2_delta.dot(
                synapse_1.T)) * sigmoid_output_to_derivative(layer_1)

            # let's update all our weights so we can try again
            synapse_1_update += np.atleast_2d(layer_1).T.dot(layer_2_delta)
            synapse_h_update += np.atleast_2d(prev_layer_1).T.dot(layer_1_delta)
            synapse_0_update += X.T.dot(layer_1_delta)

            future_layer_1_delta = layer_1_delta

        synapse_0 += synapse_0_update * alpha
        synapse_1 += synapse_1_update * alpha
        synapse_h += synapse_h_update * alpha

        synapse_0_update *= 0
        synapse_1_update *= 0
        synapse_h_update *= 0

        # print out progress
        if (j % 1000 == 0):
            print "Error:" + str(overallError)
            print "Pred:" + str(d)
            print "True:" + str(c)
            out = 0
            for index, x in enumerate(reversed(d)):
                out += x * pow(2, index)
            print str(a_int) + " + " + str(b_int) + " = " + str(out)
            print "------------"


