import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import warnings

warnings.filterwarnings("error")


class spiking_neuron(object):

    def __init__(self, forward_weights, backward_weights, delays, backward_delays, id, gradient_decay=0.9, readout=False, poisson=False,
                 v_rest=0.0, cm=1.0, tau_m=20.0, tau_refract=1.0, v_thresh=1.0, v_reset=0.0, i_offset=0.0):
        self.forward_weights = forward_weights
        self.old_weights = deepcopy(forward_weights)
        self.num_inputs = len(forward_weights)
        self.delays = delays
        self.backward_delays = backward_delays
        self.spike_buffer = [[] for i in range(self.num_inputs)]

        self.gradient_decay = gradient_decay
        self.old_error = 0.0
        self.new_error = 0.0
        self.dE = 0.0
        self.old_dEdw = [0.0 for i in range(self.num_inputs)]
        self.new_dEdw = [0.0 for i in range(self.num_inputs)]
        self.old_dE2dw2 = [0.0 for i in range(self.num_inputs)]
        self.new_dE2dw2 = [0.0 for i in range(self.num_inputs)]
        self.first_update = True

        #deprecated
        self.backward_errors = [0.0 for i in range(self.num_inputs)]
        self.backward_weights = backward_weights
        self.error = 0.0
        self.neuron_gradient = 0.0
        self.post_gradients = [0.0 for i in range(self.num_inputs)]
        self.gradient_buffer = [[] for i in range(self.num_inputs)]

        self.id = id
        self.poisson = poisson
        self.readout = readout

        self.alpha = np.exp(- dt / tau_m)
        # neuron variable
        self.v_rest = v_rest  # * 10**-3
        self.cm = cm  # * 10**-3
        self.tau_m = tau_m  # * 10**-3
        self.r_mem = 1.0  # self.tau_m / cm
        self.tau_refract = tau_refract
        self.v_thresh = v_thresh  # * 10**-3
        self.v_reset = v_reset  # * 10**-3
        self.i_offset = i_offset
        # state variables
        self.v = self.v_rest
        if self.readout and learn == 'sine':
            self.v = sine_offset
        self.scaled_v = (self.v - self.v_thresh) / self.v_thresh
        self.t_rest = 0.0
        self.i = self.i_offset

        self.activation = False
        self.input = [0.0 for i in range(self.num_inputs)]

    # activation function
    def H(self, x, dev=False, gammma=0.1, sigmoid=False, internal=True, scaled=False):
        if not sigmoid:
            if not scaled:
                x = (x - self.v_thresh) / self.v_thresh
            if dev:
                return (gammma / dt) * max(0, 1 - abs(x))
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
        if self.poisson:
            if np.random.random() < self.poisson:
                self.activation = True
            else:
                self.activation = False
        else:
            if not self.t_rest:
                current = self.integrate_inputs()
                self.v = (self.alpha * (self.v - self.v_rest)) + ((1 - self.alpha) * self.r_mem * current)
                self.scaled_v = (self.v - self.v_thresh) / self.v_thresh
                self.activation = self.did_it_spike()
            else:
                self.activation = False
                self.t_rest -= 1.0

    def did_it_spike(self):
        if self.readout:
            if self.v >= self.v_thresh:
                self.v = self.v_thresh
            elif self.v <= 0:
                self.v = 0
            self.scaled_v = (self.v - self.v_thresh) / self.v_thresh
            return False
        if self.scaled_v >= 0:
            self.v = self.v_reset
            self.scaled_v = (self.v - self.v_thresh) / self.v_thresh
            self.t_rest = self.tau_refract
            return True
        else:
            return False

    def integrate_inputs(self):
        for input in range(self.num_inputs):
            if self.input[input] and self.forward_weights[input]:
                self.spike_buffer[input].append(self.delays[input])
        did_spike_arrive = [False for i in range(self.num_inputs)]
        for input in range(self.num_inputs):
            for spike in range(len(self.spike_buffer[input])):
                self.spike_buffer[input][spike] -= dt
                if self.spike_buffer[input][spike] <= 0:
                    did_spike_arrive[input] = True
        inputs = 0.0
        for neuron in range(self.num_inputs):
            if did_spike_arrive[neuron]:
                inputs += self.forward_weights[neuron]
                del self.spike_buffer[neuron][0]
        return inputs + self.i_offset

    def gradient_selection(self, synapse):
        random_variable = np.random.rand() - 0.45
        gradient = (self.new_dEdw[synapse] + random_variable) #* self.new_dE2dw2
        return gradient

    def update_weights(self, new_error):
        if self.first_update:
            self.first_update = False
            self.new_error = new_error
            self.old_error = self.new_error
        else:
            self.calculate_gradient(new_error)
        for weight in range(self.num_inputs):
            self.old_weights[weight] = self.forward_weights[weight]
            # print self.old_weights[weight], self.forward_weights[weight]
            gradient = self.gradient_selection(weight)
            if self.forward_weights[weight]:
                self.forward_weights[weight] = self.forward_weights[weight] + gradient * l_rate
            # print self.old_weights[weight], self.forward_weights[weight], gradient
        return self.forward_weights

    def calculate_gradient(self, new_error):
        self.old_error = self.new_error
        self.new_error = new_error
        self.dE = self.new_error - self.old_error
        # if self.dE:
        #     print "error changed"
        for weight in range(self.num_inputs):
            if self.forward_weights[weight]:
                self.old_dEdw[weight] = self.new_dEdw[weight]
                dEdw = self.dE / (self.forward_weights[weight] - self.old_weights[weight])
                self.new_dEdw[weight] = (self.gradient_decay * self.new_dEdw[weight]) + dEdw
                self.old_dE2dw2[weight] = self.new_dE2dw2[weight]
                self.new_dE2dw2[weight] = self.new_dEdw[weight] - self.old_dEdw[weight]

class Network(object):

    def __init__(self, weight_matrix, delay_matrix, bias=False):
        # network variables
        self.weight_matrix = weight_matrix
        self.delay_matrix = delay_matrix
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
            if neuron >= self.number_of_neurons - output_neurons and learn == 'sine':
                self.neuron_list.append(
                    spiking_neuron(np.take(self.weight_matrix, neuron, axis=1), self.weight_matrix[neuron],
                                   np.take(self.delay_matrix, neuron, axis=1), self.delay_matrix[neuron], id=neuron,
                                   readout=True))
            elif neuron < input_neurons:
                self.neuron_list.append(
                    spiking_neuron(np.take(self.weight_matrix, neuron, axis=1), self.weight_matrix[neuron],
                                   np.take(self.delay_matrix, neuron, axis=1), self.delay_matrix[neuron], id=neuron,
                                   poisson=0.1))
            else:
                self.neuron_list.append(
                    spiking_neuron(np.take(self.weight_matrix, neuron, axis=1), self.weight_matrix[neuron],
                                   np.take(self.delay_matrix, neuron, axis=1), self.delay_matrix[neuron], id=neuron))
            self.activations[neuron] = self.neuron_list[neuron].activation
        if self.bias:
            self.activations.append(bias_value)

    # step all neurons and save state
    def forward_step(self):
        activations = []
        internal_values = []
        # self.activations = np.add(np.array(self.activations), np.array(self.inputs)).tolist()
        self.total_inputs = self.activations
        for id, neuron in enumerate(self.neuron_list):
            neuron.input = self.total_inputs
            neuron.i_offset = self.inputs[id]
            neuron.forward_step()
            activations.append(neuron.activation)
            internal_values.append(neuron.v)
        if self.bias:
            activations.append(bias_value)
        self.activations = activations
        self.internal_values = internal_values

    def calc_weight_updates(self, error=False):
        for neuron in self.neuron_list:
            neuron.update_weights(error)

    def return_weight_matrix(self):
        weight_matrix = []
        for neuron in self.neuron_list:
            weight_matrix.append(neuron.forward_weights)
        return np.array(weight_matrix).transpose()


def gradient_and_error(network, error_return=False, print_update=False, test=False):
    global number_of_neurons, epoch_errors, neuron_output
    # if delay_matrix == []:
    #     min_delay = 1
    #     max_delay = 16
    #     delay_matrix = deepcopy(weight_matrix)
    #     for i in range(len(delay_matrix)):
    #         for j in range(len(delay_matrix[0])):
    #             if delay_matrix[i][j]:
    #                 delay_matrix[i][j] = 1  # np.random.randint(min_delay, max_delay)
    number_of_neurons = network.number_of_neurons
    # weight_matrix.tolist()
    all_errors = []
    all_drv_errors = []
    all_activations = []
    all_internal_values = []
    all_inputs = []
    output_activation = []
    output_v = []
    output_delta = []
    np.random.seed(2727)
    all_errors.append(0.0)
    all_drv_errors.append(0.0)
    all_activations.append(network.activations)
    all_internal_values.append(network.internal_values)
    all_inputs.append(network.inputs)
    # forward propagation of the network
    for step in range(steps):
        inputs = [False for i in range(number_of_neurons)]
        for i in range(input_neurons):
            # if np.random.random() < dt / input_rate:
            #     inputs[i] = True
            inputs[i] = 0#np.random.random() * 2 # 2 * float(step) / float(steps) #
        network.inputs = inputs
        network.forward_step()
        all_activations.append(network.activations)
        all_internal_values.append(network.internal_values)
        all_inputs.append(network.inputs)
        output_activation.append(all_activations[step+1][-1])
        output_v.append(all_internal_values[step+1][-1])
        if learn == 'hz':
            current_hz = (float(sum(output_activation)) / float(len(output_activation))) * 1000
            error = 0.5 * np.power(current_hz - target_hz, 2)
            drv_error = current_hz - target_hz
        else:
            error = 0.5 * np.power(output_v[-1] - target_sine_wave[step], 2)
            drv_error = output_v[-1] - target_sine_wave[step]
            network.calc_weight_updates(error=drv_error)
        all_errors.append(error)
        all_drv_errors.append(drv_error)
    if not test:
        print epoch, "/", epochs, "ave error:", np.average(all_errors), "final error:", all_errors[-1]
    if learn == 'sine':
        epoch_errors.append(np.average(all_errors))
    else:
        epoch_errors.append(all_errors[-1])
        network.calc_weight_updates(error=np.average(all_errors))
    neuron_output = output_activation
    if error_return:
        return (all_errors[-1])
        return np.average(all_errors)

    weight_update = network.return_weight_matrix()
    if print_update:
        print "total update:\n", weight_update
        print "errors:", epoch_errors
    if test:
        return weight_update
    else:
        return all_activations, output_v


np.random.seed(272727)
# Feedforward network
bias = False
neurons_per_layer = 4
hidden_layers = 3
input_neurons = 5
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
# number_of_neurons = 10
# input_neurons = 5
# weight_scale = np.sqrt(number_of_neurons)
# weight_matrix = [[np.random.randn() / weight_scale for i in range(number_of_neurons)] for j in
#                  range(number_of_neurons)]
# connection_prob = 1
# for i in range(number_of_neurons):
#     for j in range(number_of_neurons):
#         if np.random.random() > connection_prob:
#             weight_matrix[i][j] = 0.0
#

# Create delay matrix
min_delay = 1
max_delay = 16
delay_matrix = deepcopy(weight_matrix)
for i in range(len(delay_matrix)):
    for j in range(len(delay_matrix[0])):
        if delay_matrix[i][j]:
            delay_matrix[i][j] = np.random.randint(min_delay, max_delay)

if bias:
    weight_matrix.append(np.ones(number_of_neurons).tolist())

biases = [0.0 for i in range(number_of_neurons)]
for i in range(input_neurons):
    biases[i] = 1
bias_value = 1.0

epochs = 10000
l_rate = 0.001
max_l_rate = 0.05
min_l_rate = 0.00001
# Duration of the simulation in ms
T = 1000
# Duration of each time step in ms
dt = 1.0
# Number of iterations = T/dt
steps = int(T / dt)

learn = 'sine'
target_hz = 10
sine_rate = 5.0
sine_scale = 0.5
sine_offset = 0.5
min_error = 0.00000001
total = False
target_sine = lambda x: sine_scale * (np.sin(sine_rate * x)) + sine_offset
target_sine_wave = [target_sine((float(t) / 1000.0) * 2.0 * np.pi) for t in range(steps+1)]

epoch_errors = []
neuron_output = []

if __name__ == "__main__":
    print "no. neurons = ", number_of_neurons, "\tno. inputs = ", input_neurons, "\tLR = ", l_rate
    network = Network(weight_matrix, delay_matrix)
    for epoch in range(epochs):
        activations, output_v = gradient_and_error(network, error_return=False, print_update=True)

        if epoch % 20 == 0 or abs(epoch_errors[-1]) < min_error:
            fig, axs = plt.subplots(3)
            plt.title(("no. neurons = ", number_of_neurons, "\tno. inputs = ", input_neurons, "\tLR = ", l_rate))
            plt.xlabel('Time (msec)')
            axs[0].plot(target_sine_wave)
            axs[0].plot(output_v)
            axs[1].axhline(y=target_hz, color='r', linestyle='-')
            axs[1].plot(neuron_output)
            axs[2].plot(activations)
            plt.show()
            if abs(epoch_errors[-1]) < min_error:
                break

    fig, axs = plt.subplots(3)
    plt.title(("no. neurons = ", number_of_neurons, "\tno. inputs = ", input_neurons, "\tLR = ", l_rate))
    plt.xlabel('Time (msec)')
    axs[0].plot(target_sine_wave)
    axs[0].plot(output_v)
    axs[1].axhline(y=target_hz, color='r', linestyle='-')
    axs[1].plot(neuron_output)
    axs[2].plot(activations)
    plt.show()
    print "done"
