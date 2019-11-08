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
        self.neuron_gradient = 0.0
        self.weight_gradients = [0.0 for i in range(self.num_inputs)]

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
        self.neuron_gradients = [0.0 for i in range(self.number_of_neurons)]
        self.bias = bias
        self.gradient_matrix = []

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
        self.total_inputs = np.add(np.array(self.activations), np.array(self.inputs)).tolist()
        for neuron in self.neuron_list:
            neuron.input = self.total_inputs
            neuron.forward_step()
            activations.append(neuron.activation)
            internal_values.append(neuron.internal_value)
        if self.bias:
            activations.append(bias_value)
        self.activations = activations
        self.internal_values = internal_values

    def calculate_all_gradients(self, activations, internal_values, error=False):
        neuron_gradients = [0.0 for i in range(number_of_neurons)]
        # update internal neuron gradients
        for idx, neuron in enumerate(self.neuron_list):
            # maybe make the first gradient the gradient of the output function ie 0.5x^2
            if idx >= number_of_neurons - output_neurons:
                if not isinstance(error, bool):
                    neuron_gradients[idx] = neuron.H(internal_values[idx], dev=True) * error
                else:
                    neuron_gradients[idx] = neuron.H(internal_values[idx], dev=True)
            else:
                gradient = 0.0
                for post_neuron in range(self.number_of_neurons):
                    gradient += self.weight_matrix[idx][post_neuron] * self.neuron_gradients[post_neuron] * neuron.H(internal_values[idx], dev=True)
                neuron_gradients[idx] = gradient
        # pass new gradients into the neurons
        self.neuron_gradients = neuron_gradients
        # update synapses
        synapse_gradients = self.calc_synapse_gradients(activations)
        return synapse_gradients

    def calc_synapse_gradients(self, activations):
        synapse_gradients = np.zeros([self.number_of_neurons, self.number_of_neurons])
        for i in range(self.number_of_neurons):
            for j in range(self.number_of_neurons):
                if self.weight_matrix[i][j]:
                    synapse_gradients[i][j] = self.neuron_gradients[j] * activations[i]
        return synapse_gradients

    def backward_step(self, activations, internal_values, error):
        self.calculate_all_gradients(activations, internal_values)
        errors = [0.0 for i in range(number_of_neurons)]
        for idx, neuron in reversed(list(enumerate(self.neuron_list))):
            if idx >= number_of_neurons - output_neurons:
                neuron.error = error
            else:
                neuron.backward_errors = self.errors
                backward_gradients = np.take(self.gradient_matrix, idx, axis=1)
                neuron.backward_step(backward_gradients)
            errors[idx] = neuron.error
        self.errors = errors
        return errors


def gradient_and_error(weight_matrix, error_return=False, print_update=True, test=False):
    global number_of_neurons, epoch_errors, neuron_output
    number_of_neurons = len(weight_matrix)
    # weight_matrix.tolist()
    all_errors = []
    all_drv_errors = []
    all_activations = []
    all_internal_values = []
    all_inputs = []
    output_activation = []
    output_delta = []
    network = Network(weight_matrix)
    np.random.seed(2727)
    all_errors.append(0.0)
    all_drv_errors.append(0.0)
    all_activations.append(network.activations)
    all_internal_values.append(network.internal_values)
    all_inputs.append(network.inputs)
    for step in range(steps):
        inputs = [0.0 for i in range(number_of_neurons)]
        if not test:
            for i in range(input_neurons):
                inputs[i] = np.random.random() # float(step) / float(steps) #
        network.inputs = inputs
        network.forward_step()
        all_activations.append(network.activations)
        all_internal_values.append(network.internal_values)
        all_inputs.append(network.inputs)
        output_activation.append(all_activations[step+1][-1])
        if learn == 'hz':
            error = 0.5 * np.power(output_activation[-1] - target_hz, 2) #* np.sign(output_activation[-1] - target_hz)
            drv_error = output_activation[-1] - target_hz
        else:
            error = 0.5 * np.power(output_activation[-1] - target_sine_wave[step], 2)
            drv_error = output_activation[-1] - target_sine_wave[step]
            # error = output_activation[-1] - target_sine_wave[step]
        all_errors.append(error)
        all_drv_errors.append(drv_error)
    if not test:
        print epoch, "/", epochs, "error:", np.average(all_errors)
    epoch_errors.append(np.average(all_errors))
    neuron_output = output_activation
    if error_return:
        return (all_errors[-1])
        return np.average(all_errors)
    weight_update = np.zeros([number_of_neurons, number_of_neurons])
    for step in reversed(range(steps+1)):
        new_weight_update = np.array(network.calculate_all_gradients(all_activations[step], all_internal_values[step], error=all_drv_errors[step]))
        # new_weight_update = np.array(network.calculate_all_gradients(all_activations[step], all_internal_values[step]))
        weight_update += new_weight_update #* l_rate
        if print_update:
            print "weight update:\n", new_weight_update
        # if step < 2: #steps / 2:
        #     return new_weight_update
    weight_update /= steps + 1
    if print_update:
        print "total update:\n", weight_update
    return weight_update


# Feedforward network
bias = False
neurons_per_layer = 3
hidden_layers = 3
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
number_of_neurons = 50
input_neurons = 1
weight_scale = np.sqrt(number_of_neurons)
weight_matrix = [[np.random.randn() / weight_scale for i in range(number_of_neurons)] for j in
                 range(number_of_neurons)]
# connection_prob = 1
# for i in range(number_of_neurons):
#     for j in range(number_of_neurons):
#         if np.random.random() > connection_prob:
#             weight_matrix[i][j] = 0.0
#

np.random.seed(272727)

if bias:
    weight_matrix.append(np.ones(number_of_neurons).tolist())

biases = [0.0 for i in range(number_of_neurons)]
for i in range(input_neurons):
    biases[i] = 1
bias_value = 1.0

epochs = 10000
l_rate = 0.1
max_l_rate = 0.05
min_l_rate = 0.00001
# Duration of the simulation in ms
T = 200
# Duration of each time step in ms
dt = 1
# Number of iterations = T/dt
steps = int(T / dt)

learn = 'sine'
target_hz = 0.28
sine_rate = 5.0
sine_scale = 0.5
min_error = 0.00000001
total = False
target_sine = lambda x: sine_scale * (np.sin(sine_rate * x)) + 0.5
target_sine_wave = [target_sine((float(t) / 1000.0) * 2.0 * np.pi) for t in range(steps+1)]

epoch_errors = []
neuron_output = []

if __name__ == "__main__":
    for epoch in range(epochs):
        weight_update = gradient_and_error(weight_matrix, error_return=False, print_update=False)
        weight_matrix = (np.array(weight_matrix) - (weight_update * l_rate)).tolist()

        if epoch % 100 == 0 or abs(epoch_errors[-1]) < min_error:
            plt.figure()
            plt.title('target sine')
            plt.xlabel('Time (msec)')
            plt.plot(target_sine_wave)
            plt.axhline(y=target_hz, color='r', linestyle='-')
            plt.plot(neuron_output)
            plt.show()
            if abs(epoch_errors[-1]) < min_error:
                break

    plt.figure()
    plt.title('Final target sine etc')
    plt.xlabel('Time (msec)')
    plt.plot(target_sine_wave)
    plt.axhline(y=target_hz, color='r', linestyle='-')
    plt.plot(neuron_output)
    plt.show()
    print "done"
