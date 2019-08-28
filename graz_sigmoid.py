import numpy as np
from copy import deepcopy
import warnings
warnings.filterwarnings("error")

class sigmoid_neuron(object):

    def __init__(self, forward_weights, backward_weights):
        self.forward_weights = forward_weights
        self.num_inputs = len(forward_weights)

        self.backward_errors = [0.0 for i in range(self.num_inputs)]
        self.backward_weights = backward_weights
        self.error = 0.0

        self.voltage = 0.0
        self.activation = self.H(self.voltage)
        self.input = [0.0 for i in range(self.num_inputs)]


    # activation function
    def H(self, x, dev=False, gammma=0.1, sigmoid=True):
        if not sigmoid:
            if dev:
                return (gammma / self.dt) * max(0, 1 - abs(x))
            else:
                return gammma * max(0, 1 - abs(x))
        if dev:
            # x *= 3
            return np.exp(-x) / ((1 + np.exp(-x)) ** 2)
            value = self.H(x, dev=False, sigmoid=True)
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

    def backward_step(self, activation):
        self.error = self.integrate_errors(activation)

    def integrate_errors(self, activation):
        outputs = [self.backward_weights[i] * self.backward_errors[i] * self.H(activation, dev=True) for i in range(len(self.backward_weights))]
        return sum(outputs)

    def delta_w(self, activations):
        delta_w = [0.0 for i in range(self.num_inputs)]
        for i in range(self.num_inputs):
            if self.forward_weights[i]:
                delta_w[i] += l_rate * activations[i] * self.error
        return delta_w

class Network(object):

    def __init__(self, weight_matrix):
        # network variables
        self.weight_matrix = weight_matrix
        self.number_of_neurons = number_of_neurons
        self.neuron_list = []
        self.activations = [0.0 for i in range(self.number_of_neurons)]
        self.voltages = [0.0 for i in range(self.number_of_neurons)]
        # matrix multiplication of w[] and z[]
        self.errors = [0.0 for i in range(self.number_of_neurons)]

        # initialise the network of neurons connected
        for neuron in range(self.number_of_neurons):
            self.neuron_list.append(sigmoid_neuron(np.take(self.weight_matrix, neuron, axis=1), self.weight_matrix[neuron]))
            self.activations[neuron] = self.neuron_list[neuron].activation
        self.activations.append(bias_value)

    # step all neurons and save state
    def forward_step(self):
        activations = []
        voltages = []
        for neuron in self.neuron_list:
            neuron.input = self.activations
            neuron.forward_step()
            activations.append(neuron.activation)
            voltages.append(neuron.voltage)
        activations.append(bias_value)
        self.activations = activations
        self.voltages = voltages

    def backward_step(self, activations, error):
        errors = []
        for idx, neuron in reversed(list(enumerate(self.neuron_list))):
            if idx >= number_of_neurons - output_neurons:
                neuron.error = error * neuron.H(activations[idx], dev=True)
            else:
                neuron.backward_errors = self.errors
                neuron.backward_step(activations[idx])
            errors.append(neuron.error)
        self.errors = errors

    def weight_update(self, activations, error):
        self.backward_step(activations, error)
        update_weight_matrix = []
        for neuron in self.neuron_list:
            update_weight_matrix.append(neuron.delta_w(activations))
        return update_weight_matrix

def sine_error(activations, total=False, current_step=0):
    if total:
        errors = [(activations[i] - target_sine_wave[i])**2 for i in range(steps)]
        # sign = np.sign(sum([(activations[i] - target_sine_wave[i]) for i in range(steps)]))
        total_error = sum(errors) / steps
        # total_error *= sign
        return total_error
    else:
        # errors = [(activations[i] - target_sine_wave[i])**2 for i in range(steps)]
        errors = [(activations[i] - target_sine_wave[i])**2 * np.sign(activations[i] - target_sine_wave[i]) for i in range(steps)]
        # errors = [(activations[i] - target_sine_wave[i])**2 for i in range(steps)]
        return errors

def hz_error(activations, dev=False):
    if dev:
        return (sum(activations) / steps) - target_hz
    else:
        return sum([activations[i] - target_hz for i in range(len(activations))]) / steps
        return 0.5 * (((sum(activations) / steps) - target_hz)**2)

# Feedforward network
neurons_per_layer = 4
hidden_layers = 3
input_neurons = 3
output_neurons = 1
weight_scale = np.sqrt(neurons_per_layer)
number_of_neurons = input_neurons + (hidden_layers * neurons_per_layer) + output_neurons
weight_matrix = np.zeros([number_of_neurons, number_of_neurons]).tolist()
for i in range(input_neurons):
    for j in range(neurons_per_layer):
        weight_matrix[i][j+input_neurons] = np.random.randn() / weight_scale
        # print "ii=", i, "\tj=", j+input_neurons
for i in range(hidden_layers-1):
    for j in range(neurons_per_layer):
        for k in range(neurons_per_layer):
            weight_matrix[(i*neurons_per_layer)+input_neurons+k][j+((i+1)*neurons_per_layer)+input_neurons] = np.random.randn() / weight_scale
            # print "hi=", (i*neurons_per_layer)+input_neurons+k, "\tj=", j+((i+1)*neurons_per_layer)+input_neurons
for i in range(neurons_per_layer):
    for j in range(output_neurons):
        weight_matrix[((hidden_layers-1)*neurons_per_layer)+input_neurons+i][number_of_neurons-j-1] = np.random.randn() / weight_scale
        # print "oi=", ((hidden_layers-1)*neurons_per_layer)+input_neurons+i, "\tj=", number_of_neurons-j-1
# weight_matrix = np.transpose(weight_matrix).tolist()

# Recurrent network
# number_of_neurons = 15
# input_neurons = 0
# weight_scale = np.sqrt(number_of_neurons)
# weight_matrix = [[np.random.randn() / weight_scale for i in range(number_of_neurons)] for j in
#                  range(number_of_neurons)]
# connection_prob = 1
# for i in range(number_of_neurons):
#     for j in range(number_of_neurons):
#         if np.random.random() > connection_prob:
#             weight_matrix[i][j] = 0.0
#
weight_matrix.append(np.ones(number_of_neurons).tolist())

biases = [0.0 for i in range(number_of_neurons)]
for i in range(input_neurons):
    biases[i] = 1
bias_value = 1.0

epochs = 100
l_rate = 0.03
max_l_rate = 0.05
min_l_rate = 0.00001
# Duration of the simulation in ms
T = 1000
# Duration of each time step in ms
dt = 1
# Number of iterations = T/dt
steps = int(T / dt)

learn = 'sine'
target_hz = 0.78
sine_rate = 100
sine_scale = 0.25
total = False
target_sine = lambda x: sine_scale * (np.sin(sine_rate * x)) + 0.5
target_sine_wave = [target_sine(t / 1000.0) for t in range(steps)]

all_errors = []
for epoch in range(epochs):
    all_activations = []
    all_voltages = []
    output_activation = []
    network = Network(weight_matrix)
    np.random.seed(272727)
    for step in range(steps):
        all_activations.append(network.activations)
        all_voltages.append(network.voltages)
        output_activation.append(all_activations[step][-2])
        network.forward_step()
        bias_value = abs(np.random.random())#(steps - step) / steps + 0.5
    if learn == 'hz':
        error = hz_error(output_activation)
        all_errors.append(error)
        print error
    else:
        error = sine_error(output_activation, total=total)
        if not total:
            all_errors.append(sum(error))
            print sum(error)
        else:
            all_errors.append(error)
            print error
    if abs(all_errors[-1]) < 1e-3:
        print all_errors
        if learn == 'hz':
            error = hz_error(output_activation)
        else:
            error = sine_error(output_activation, total=True)
        break
    weight_update = np.zeros([number_of_neurons+1, number_of_neurons])
    for step in reversed(range(steps)):
        if learn == 'sine' and not total:
            weight_update += np.array(network.weight_update(all_activations[step], error[step])).transpose()
        else:
            weight_update += np.array(network.weight_update(all_activations[step], error)).transpose()
    weight_matrix = (np.array(weight_matrix) - weight_update).tolist()

if learn == 'hz':
    error = hz_error(output_activation)
else:
    error = sine_error(output_activation, total=True)
print "done"
