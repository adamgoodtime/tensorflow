import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import warnings

warnings.filterwarnings("error")


class sigmoid_neuron(object):

    def __init__(self, forward_weights, backward_weights, output=False):
        self.forward_weights = forward_weights
        self.num_inputs = len(forward_weights)

        self.backward_errors = [0.0 for i in range(self.num_inputs)]
        self.backward_weights = backward_weights
        self.error = 0.0
        self.gradient = 0.0

        self.output = output

        self.internal_value = 0.0
        self.activation = self.H(self.internal_value)
        self.input = [0.0 for i in range(self.num_inputs)]

    # activation function
    def H(self, x, dev=False, gammma=0.1, sigmoid=True, internal=True):
        if self.output:
            if dev:
                return 1
            else:
                return x
        if not sigmoid:
            if dev:
                return (gammma / self.dt) * max(0, 1 - abs(x))
            else:
                return gammma * max(0, 1 - abs(x))
        if dev:
            # x *= 3
            # return np.exp(-x) / ((1 + np.exp(-x)) ** 2)
            if internal:
                value = self.H(x, dev=False, sigmoid=sigmoid)
            else:
                value = x
            # print "H dev - og:", np.exp(-x) / ((1 + np.exp(-x))**2), "gb:", value / (1 - value)
            return value * (1 - value)
        else:
            if x > 700:
                return 1
            elif x < -700:
                return 0
            else:
                return 1 / (1 + np.exp(-x))

    def forward_step(self):
        self.internal_value = self.integrate_inputs()
        self.activation = self.H(self.internal_value)

    def integrate_inputs(self):
        inputs = [self.forward_weights[i] * self.input[i] for i in range(len(self.forward_weights))]
        return sum(inputs)

    def backward_step(self, internal_value):
        # dodw2 = input_value * \
        #         forward_weight * drv_sig(backward_weight * input_value) * \
        #         drv_sig(forward_weight * sig(input_value * backward_weight))

        self.error = self.integrate_errors(internal_value)

    def integrate_errors(self, internal_value):
        # outputs = [self.backward_weights[i] * self.backward_errors[i] * (activation*(1-activation)) for i in range(len(self.backward_weights))]
        outputs = [self.backward_weights[i] * self.backward_errors[i] * self.H(internal_value, dev=True) for i in
                   range(len(self.backward_weights))]
        return sum(outputs)

    def delta_w(self, activations):
        delta_w = [0.0 for i in range(self.num_inputs)]
        for i in range(self.num_inputs):
            if self.forward_weights[i]:
                delta_w[i] += l_rate * self.error * activations[i]  # * self.forward_weights[i] #self.error
        return delta_w


class Network(object):

    def __init__(self, weight_matrix, bias=False):
        # network variables
        self.weight_matrix = weight_matrix
        self.number_of_neurons = len(weight_matrix)
        self.neuron_list = []
        self.inputs = [0.0 for i in range(self.number_of_neurons)]
        self.activations = [0.0 for i in range(self.number_of_neurons)]
        self.internal_values = [0.0 for i in range(self.number_of_neurons)]
        # matrix multiplication of w[] and z[]
        self.errors = [0.0 for i in range(self.number_of_neurons)]
        self.bias = bias

        # initialise the network of neurons connected
        for neuron in range(self.number_of_neurons):
            if neuron >= self.number_of_neurons - output_neurons:
                self.neuron_list.append(
                    sigmoid_neuron(np.take(self.weight_matrix, neuron, axis=1), self.weight_matrix[neuron],
                                   output=False))
            else:
                self.neuron_list.append(
                    sigmoid_neuron(np.take(self.weight_matrix, neuron, axis=1), self.weight_matrix[neuron]))
            self.activations[neuron] = self.neuron_list[neuron].activation
        if self.bias:
            self.activations.append(bias_value)

    # step all neurons and save state
    def forward_step(self):
        activations = []
        internal_values = []
        for neuron in self.neuron_list:
            neuron.input = np.add(np.array(self.activations), np.array(self.inputs)).tolist()
            neuron.forward_step()
            activations.append(neuron.activation)
            internal_values.append(neuron.internal_value)
        if self.bias:
            activations.append(bias_value)
        self.activations = activations
        self.internal_values = internal_values

    def backward_step(self, internal_values, activations, delta):
        gradients = [0.0 for i in range(number_of_neurons)]
        for idx, neuron in reversed(list(enumerate(self.neuron_list))):
            if idx >= number_of_neurons - output_neurons:
                # this way would need to be done per synapse, move first half to the neuron? or pos whole thing?
                neuron.gradient = (internal_values[idx]) * neuron.H(internal_values[idx])
            else:
                neuron.backward_errors = self.errors
                # neuron.backward_step(activations[idx])
                neuron.backward_step(internal_values[idx])
            errors[idx] = neuron.error
        self.errors = errors

    def weight_update(self, internal_values, activations, deltas):
        self.backward_step(internal_values, activations, deltas)
        update_weight_matrix = []
        for neuron in self.neuron_list:
            update_weight_matrix.append(neuron.delta_w(activations))
        return update_weight_matrix


def bp_and_error(weight_matrix, error_return=False, print_update=False):
    global number_of_neurons, epoch_errors, neuron_output
    number_of_neurons = len(weight_matrix)
    # weight_matrix.tolist()
    all_errors = []
    all_activations = []
    all_internal_values = []
    output_activation = []
    output_delta = []
    network = Network(weight_matrix)
    np.random.seed(272727)
    for step in range(steps):
        inputs = [0.0 for i in range(number_of_neurons)]
        for i in range(input_neurons):
            inputs[i] = float(step) / float(steps)  # np.random.random()
        network.inputs = inputs
        network.forward_step()
        all_activations.append(network.activations)
        all_internal_values.append(network.internal_values)
        output_activation.append(all_activations[step][-1])
        if learn == 'hz':
            # error = 0.5 * np.power(output_activation[-1] - target_hz, 2)
            error = output_activation[-1] - target_hz
        else:
            error = 0.5 * np.power(output_activation[-1] - target_sine_wave[step], 2) * np.sign(
                output_activation[-1] - target_sine_wave[step])
            # error = output_activation[-1] - target_sine_wave[step]
        all_errors.append(error)
        output_delta.append(error * output_activation[-1] * (1 - output_activation[-1]))
    print "error:", sum(all_errors)
    epoch_errors.append(sum(all_errors))
    neuron_output = output_activation
    if error_return:
        return sum(all_errors)
    weight_update = np.zeros([number_of_neurons, number_of_neurons])
    for step in reversed(range(steps)):
        if learn == 'sine' and not total:
            new_weight_update = np.array(
                network.weight_update(all_internal_values[step], all_activations[step], output_delta[step])).transpose()
        else:
            new_weight_update = np.array(
                network.weight_update(all_internal_values[step], all_activations[step], output_delta[step])).transpose()
            # new_weight_update = np.array(network.weight_update(all_internal_values[step], all_activations[step], error)).transpose()
        weight_update += new_weight_update
        if print_update:
            print "weight update:", new_weight_update
    if print_update:
        print "total update:", weight_update
    return weight_update


# Feedforward network
bias = False
neurons_per_layer = 1
hidden_layers = 1
input_neurons = 1
output_neurons = 1
weight_scale = np.sqrt(neurons_per_layer)
number_of_neurons = input_neurons + (hidden_layers * neurons_per_layer) + output_neurons
weight_matrix = np.zeros([number_of_neurons, number_of_neurons]).tolist()
for i in range(input_neurons):
    for j in range(neurons_per_layer):
        weight_matrix[i][j + input_neurons] = np.random.randn() / weight_scale
        # print "ii=", i, "\tj=", j+input_neurons
for i in range(hidden_layers - 1):
    for j in range(neurons_per_layer):
        for k in range(neurons_per_layer):
            weight_matrix[(i * neurons_per_layer) + input_neurons + k][
                j + ((i + 1) * neurons_per_layer) + input_neurons] = np.random.randn() / weight_scale
            # print "hi=", (i*neurons_per_layer)+input_neurons+k, "\tj=", j+((i+1)*neurons_per_layer)+input_neurons
for i in range(neurons_per_layer):
    for j in range(output_neurons):
        weight_matrix[((hidden_layers - 1) * neurons_per_layer) + input_neurons + i][
            number_of_neurons - j - 1] = np.random.randn() / weight_scale
        # print "oi=", ((hidden_layers-1)*neurons_per_layer)+input_neurons+i, "\tj=", number_of_neurons-j-1
# weight_matrix = np.transpose(weight_matrix).tolist()

# Recurrent network
number_of_neurons = 10
input_neurons = 2
weight_scale = np.sqrt(number_of_neurons)
weight_matrix = [[np.random.randn() / weight_scale for i in range(number_of_neurons)] for j in
                 range(number_of_neurons)]
# connection_prob = 1
# for i in range(number_of_neurons):
#     for j in range(number_of_neurons):
#         if np.random.random() > connection_prob:
#             weight_matrix[i][j] = 0.0
#

if bias:
    weight_matrix.append(np.ones(number_of_neurons).tolist())

biases = [0.0 for i in range(number_of_neurons)]
for i in range(input_neurons):
    biases[i] = 1
bias_value = 1.0

epochs = 100
l_rate = 0.01
max_l_rate = 0.05
min_l_rate = 0.00001
# Duration of the simulation in ms
T = 200
# Duration of each time step in ms
dt = 1
# Number of iterations = T/dt
steps = int(T / dt)

learn = 'hz'
target_hz = 0.58
sine_rate = 100
sine_scale = 0.25
total = False
target_sine = lambda x: sine_scale * (np.sin(sine_rate * x)) + 0.5
target_sine_wave = [target_sine(t / 1000.0) for t in range(steps)]

epoch_errors = []
neuron_output = []

if __name__ == "__main__":
    for epoch in range(epochs):
        weight_update = bp_and_error(weight_matrix, error_return=False, print_update=False)
        weight_matrix = (np.array(weight_matrix) - weight_update).tolist()

        if epoch % 100 == 0 or abs(epoch_errors[-1]) < 0.001:
            plt.figure()
            plt.title('target sine')
            plt.xlabel('Time (msec)')
            plt.plot(target_sine_wave)
            plt.axhline(y=target_hz, color='r', linestyle='-')
            plt.plot(neuron_output)
            plt.show()

    plt.figure()
    plt.title('target sine')
    plt.xlabel('Time (msec)')
    plt.plot(target_sine_wave)
    plt.axhline(y=target_hz, color='r', linestyle='-')
    plt.plot(neuron_output)
    plt.show()
    print "done"
