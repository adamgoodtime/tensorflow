import numpy as np
import warnings
warnings.filterwarnings("error")

'''
pass values between neurons
progress until a set time
output error
randomly perturb weight
output new error
calculate dE/dw
use dE/dw to determine new weight update
new error
calculate new dE/dw and add to 0.9*old
calculate dE2/dw2 to moderate learning rule
keep going
maybe each neuron sees different amount of error
'''

class neuron(object):

    def __init__(self, momentum, weights, gradient_decay, gradient2_decay, learning_rate, random_feedback):
        self.momentum = momentum
        self.weights = weights
        self.num_inputs = len(weights)
        self.old_weights = [0.0 for i in range(self.num_inputs)]
        self.gradient_decay = gradient_decay
        self.gradient2_decay = gradient2_decay
        self.learning_rate = learning_rate
        self.random_feedback = random_feedback

        self.activation = 0.0
        self.input = [0.0 for i in range(self.num_inputs)]

        self.old_error = 0.0
        self.new_error = 0.0
        self.dE = 0.0
        self.old_dEdw = [0.0 for i in range(self.num_inputs)]
        self.new_dEdw = [0.0 for i in range(self.num_inputs)]
        self.old_dE2dw2 = ['empty' for i in range(self.num_inputs)]
        self.new_dE2dw2 = ['empty' for i in range(self.num_inputs)]


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
            value = self.H(x)
            # print "H dev - og:", np.exp(-x) / ((1 + np.exp(-x))**2), "gb:", value / (1 - value)
            return value * (1 - value)
        else:
            return 1 / (1 + np.exp(-x))

    def step(self):
        self.activation += self.integrate_inputs()

    def integrate_inputs(self):
        inputs = [self.weights[i] * self.input[i] for i in range(len(self.weights))]
        return sum(inputs)

    def perturb_weight(self):
        for i in range(self.num_inputs):
            if self.weights[i] != 0:
                self.old_weights[i] = self.weights[i]
                self.weights[i] = np.random.randn() / weight_scale

    def update_gradients(self, error):
        self.old_error = self.new_error
        self.new_error = error * self.random_feedback
        self.dE = self.new_error - self.old_error
        for synapse in range(self.num_inputs):
            if self.weights[synapse] != 0:
                self.update_synapse(synapse)

    def update_synapse(self, synapse):
        dw = self.weights[synapse] - self.old_weights[synapse]
        self.old_dEdw[synapse] = self.new_dEdw[synapse]
        if dw == 0:
            # print "um what?"
            self.new_dEdw[synapse] = (self.gradient_decay * self.old_dEdw[synapse])
        else:
            self.new_dEdw[synapse] = (self.gradient_decay * self.old_dEdw[synapse]) + (self.dE / dw)
        self.old_weights[synapse] = self.weights[synapse]
        # add a learning rate or deminish amount added
        self.weights[synapse] -= self.new_error / self.new_dEdw[synapse] * self.learning_rate
        if self.old_weights[synapse] == self.weights[synapse]:
            # print "",#heeeeellll no"
            a = None
        # find someway to use the second derivative to moderate learning rate
        if self.new_dE2dw2[synapse] == 'empty':
            self.new_dE2dw2[synapse] = 0.0
        else:
            self.old_dE2dw2[synapse] = self.new_dE2dw2[synapse]
            gradient_change = self.new_dEdw[synapse] - self.old_dEdw[synapse]
            self.new_dE2dw2[synapse] = (self.gradient2_decay * self.old_dE2dw2[synapse]) + gradient_change
            # self.new_dEdw[synapse] = (self.gradient2_decay * self.old_dE2dw2[synapse]) + gradient_change

def hz_error(activations, target_value):
    return ((sum(activations) / steps) - target_value)

def sine_error(activations, rate, scale, starting_value=0):
    target_sine = lambda x: scale * (np.sin(rate * x)) + starting_value
    target_sine_wave = [target_sine(t / 1000.0) for t in range(steps)]
    errors = [(activations[i] - target_sine_wave[i])**2 for i in range(steps)]
    total_error = sum(errors) / steps
    return total_error

neurons_per_layer = 21
hidden_layers = 4
input_neurons = 1
output_neurons = 1
weight_scale = np.sqrt(neurons_per_layer)
number_of_neurons = input_neurons + (hidden_layers * neurons_per_layer) + output_neurons
weight_matrix = np.zeros([number_of_neurons, number_of_neurons])
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
number_of_neurons = 25
weight_scale = np.sqrt(number_of_neurons)
weight_matrix = [[np.random.randn() / weight_scale for i in range(number_of_neurons)] for j in
                 range(number_of_neurons)]

epochs = 1000
steps = 100
target_value = 100
sine_rate = 60
sine_scale = 0.25
learn = 'hz'
learning_rate = 0.9
gamma = 0.001
error_over_time = []

neuron_list = []
for i in range(number_of_neurons):
    random_feedback = 1 #+ (np.random.randn() * 0.1)
    neuron_list.append(neuron(0, np.take(weight_matrix, i, axis=1), gamma, 0.999, learning_rate, random_feedback))
    # neuron_list.append(neuron(0, weight_matrix[i], 0.9, 0.999, 0.1, 1))

for epoch in range(epochs):
    error = []
    for i in range(input_neurons):
        neuron_list[i].activation = 1
    for i in range(input_neurons, number_of_neurons):
        neuron_list[i].activation = 0
    for step in range(steps):
        activations = []
        for i in range(number_of_neurons):
            activations.append(neuron_list[i].activation)
        for i in range(number_of_neurons):
            neuron_list[i].input = activations
            neuron_list[i].step()
        error.append(neuron_list[number_of_neurons-1].activation)
    if learn == 'hz':
        error_over_time.append(hz_error(error, target_value))
    else:
        error_over_time.append(sine_error(error, sine_rate, sine_scale))
    print error_over_time[epoch]
    if abs(error_over_time[epoch]) < 0.01:
        break
    for i in range(number_of_neurons):
        if epoch == 0:
            neuron_list[i].new_error = error_over_time[epoch]
            neuron_list[i].perturb_weight()
        else:
            neuron_list[i].update_gradients(error_over_time[epoch])

print "done"