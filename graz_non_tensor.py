import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

class Graz_LIF(object):

    def __init__(self, weights, dt=1.0, v_rest=0.0, cm=1.0, tau_m=20.0, tau_refract=5.0, v_thresh=1.0, v_reset=0.0, i_offset=0.0, alpha=0.05):
        self.alpha = alpha
        # passed in varaibles of the neuron
        self.dt = dt
        self.v_rest = v_rest #* 10**-3
        self.cm = cm #* 10**-3
        self.tau_m = tau_m #* 10**-3
        self.r_mem = 1.0 #self.tau_m / cm
        self.tau_refract = tau_refract
        self.v_thresh = v_thresh #* 10**-3
        self.v_reset = v_reset #* 10**-3
        self.i_offset = i_offset
        # state variables
        self.v = self.v_rest
        self.scaled_v = (self.v - self.v_thresh) / self.v_thresh
        self.t_rest = 0.0
        self.i = self.i_offset
        # network variables
        self.weights = weights
        self.has_spiked = False
        self.received_spikes = [False for i in range(len(weights))]

    # activation function
    def H(self, x, dev=False, heavy=True, gammma=0.1):
        if heavy:
            if dev:
                return (gammma/self.dt) * max(0, 1-x)
            else:
                return gammma * max(0, 1 - x)
        if dev:
            x *= 3
            return np.exp(-x) / ((1 + np.exp(-x))**2)
        else:
            return 1 / (1 + np.exp(-x))

    # operation to be performed when not spiking
    def integrating(self):
        self.has_spiked = False
        current = self.return_current()
        self.scaled_v = (self.v - self.v_thresh) / self.v_thresh
        z = self.H(self.scaled_v) * (1 / self.dt)
        update = (self.alpha * (self.v - self.v_rest)) + ((1 - self.alpha) * self.r_mem * current)# - (self.dt * self.v_thresh * z)
        self.v = update

    # to be perfromed once threshold crossed
    def spiked(self):
        self.has_spiked = True
        self.v = self.v_rest
        self.t_rest = self.tau_refract

    # refractory behaviour
    def refracting(self):
        self.has_spiked = False
        self.t_rest -= 1.0

    # collect current input from spikes
    def sum_the_spikes(self):
        total_current = 0.0
        for neuron in range(len(self.received_spikes)):
            if self.received_spikes[neuron]:
                total_current += self.weights[neuron]
        return total_current

    # calculates the current for all inputs
    def return_current(self):
        total = self.sum_the_spikes()
        return self.i_offset + total

    # step the neuron
    def step(self):
        if self.v > self.v_thresh:
            self.spiked()
        elif self.t_rest > 0:
            self.refracting()
        else:
            self.integrating()

    def return_differentials(self, error):
        dEdz = error
        dzdu = 0
        dEdVt1 = 0
        dEdVt = (dEdz * dzdu / network.v_thresh) + (dEdVt1 * self.alpha)
        dEdzt = dEdz - (dEdVt1 * dt * self.v_thresh) + (dEdVt1 * self.weights["blah"] * (1 - self.alpha) * self.r_mem)

class Network(object):

    def __init__(self, weight_matrix, dt=1.0, v_rest=0.0, cm=1.0, tau_m=20.0, tau_refract=5.0, v_thresh=1.0, v_reset=0.0, i_offset=0.0):
        self.alpha = dt / tau_m
        # neuron variable
        self.dt = dt
        self.v_rest = v_rest #* 10**-3
        self.cm = cm #* 10**-3
        self.tau_m = tau_m #* 10**-3
        self.r_mem = 1.0 #self.tau_m / cm
        self.tau_refract = tau_refract
        self.v_thresh = v_thresh #* 10**-3
        self.v_reset = v_reset #* 10**-3
        self.i_offset = i_offset
        # network variables
        self.weight_matrix = weight_matrix
        self.number_of_neurons = len(weight_matrix)
        self.neuron_list = []
        self.did_it_spike = [False for i in range(self.number_of_neurons)]

        # initialise the network of neurons connected
        for neuron in range(self.number_of_neurons):
            self.neuron_list.append(Graz_LIF(self.weight_matrix[neuron], dt, v_rest, cm, tau_m, tau_refract, v_thresh, v_reset, i_offset, self.alpha))

    # step all neurons and save state
    def step(self):
        spike_tracking = []
        for neuron in self.neuron_list:
            neuron.received_spikes = self.did_it_spike
            neuron.step()
            spike_tracking.append(neuron.has_spiked)
        self.did_it_spike = spike_tracking

def calc_error(spike_history, single_error_neuron=True, quadratic=False):
    target_hz = 20
    if single_error_neuron:
        actual_hz = float(sum(spike_history[number_of_neurons-1])) / (float(T) / 1000.0)
    else:
        actual_hz = float(sum([sum(spike_history[n]) for n in range(number_of_neurons)])) / (float(T) / 1000.0)
    if quadratic:
        error = 0.5 * (target_hz - actual_hz)**2
    else:
        error = actual_hz - target_hz
        # error = target_hz - actual_hz
    print "Error for the last iteration:", error, " with target:", target_hz, "and actual:", actual_hz
    return error

# error = 0.5 (y - y_target)^2
def back_prop(spike_history, voltage_history, network):
    error = calc_error(spike_history)
    new_weight_matrix = deepcopy(network.weight_matrix)
    update_weight_matrix = np.zeros([number_of_neurons, number_of_neurons])
    dEdz = [[0.0 for i in range(T/dt + 1)] for neuron in range(number_of_neurons)]
    dEdV = [[0.0 for i in range(T/dt + 1)] for neuron in range(number_of_neurons)]
    dEdWi = [[0.0 for pre in range(number_of_neurons)] for post in range(number_of_neurons)]
    dEdWr = [[0.0 for pre in range(number_of_neurons)] for post in range(number_of_neurons)]
    for t in range(T/dt-1, -1, -1):
        for neuron in range(number_of_neurons):
            this_neuron = network.neuron_list[neuron]
            pseudo_derivative = this_neuron.H(voltage_history[neuron][t], dev=True)
            print "psd =", pseudo_derivative
            if pseudo_derivative:
                leak = np.exp(-network.dt / network.tau_m)
                p_dEdz = error * network.weight_matrix[neuron][number_of_neurons-1] * leak
                sum_dEdV = sum([dEdV[n][t+1] *
                                weight_matrix[neuron][n] *
                                # weight_matrix[n][neuron] *
                                (1-network.neuron_list[n].alpha) *
                                this_neuron.r_mem
                                for n in range(number_of_neurons)])
                dEdz[neuron][t] = p_dEdz - (dEdV[neuron][t+1] * dt * this_neuron.v_thresh) + sum_dEdV
                dEdV[neuron][t] = dEdz[neuron][t] * pseudo_derivative * (1/this_neuron.v_thresh) + \
                                  (dEdV[neuron][t+1] * this_neuron.alpha)

    for pre in range(number_of_neurons):
        for post in range(number_of_neurons):
            # dEdWi[pre][post] = sum([dEdV[post][t] * spike_history[pre][t] for t in range(T/dt)])
            # dEdWr[pre][post] = sum([dEdV[post][t] * spike_history[pre][t] for t in range(T/dt)])
            # new_weight_matrix[pre][post] += l_rate * dEdWr[pre][post]
            # update_weight_matrix[pre][post] += l_rate * dEdWr[pre][post]
            dEdWi[post][pre] = sum([dEdV[post][t] * spike_history[pre][t] for t in range(T/dt)])
            dEdWr[post][pre] = sum([dEdV[post][t] * spike_history[pre][t] for t in range(T/dt)])
            new_weight_matrix[post][pre] += l_rate * dEdWr[post][pre]
            update_weight_matrix[post][pre] += l_rate * dEdWr[post][pre]

    print "\n", network.weight_matrix
    print update_weight_matrix
    print new_weight_matrix
    return new_weight_matrix

# Network
# Feedforward network
neurons_per_layer = 3
hidden_layers = 1
input_neurons = 0
output_neurons = 1
starting_weight = np.sqrt(neurons_per_layer)
number_of_neurons = input_neurons + (hidden_layers * neurons_per_layer) + output_neurons
weight_matrix = np.zeros([number_of_neurons, number_of_neurons])
for i in range(input_neurons):
    for j in range(neurons_per_layer):
        weight_matrix[i][j+input_neurons] = starting_weight * np.random.random()
        # print "ii=", i, "\tj=", j+input_neurons
for i in range(hidden_layers-1):
    for j in range(neurons_per_layer):
        for k in range(neurons_per_layer):
            weight_matrix[(i*neurons_per_layer)+input_neurons+k][j+((i+1)*neurons_per_layer)+input_neurons] = starting_weight * np.random.random()
            # print "hi=", (i*neurons_per_layer)+input_neurons+k, "\tj=", j+((i+1)*neurons_per_layer)+input_neurons
for i in range(neurons_per_layer):
    for j in range(output_neurons):
        weight_matrix[((hidden_layers-1)*neurons_per_layer)+input_neurons+i][number_of_neurons-j-1] = starting_weight * np.random.random()
        # print "oi=", ((hidden_layers-1)*neurons_per_layer)+input_neurons+i, "\tj=", number_of_neurons-j-1
# weight_matrix = np.transpose(weight_matrix).tolist()

# Recurrent network
# number_of_neurons = 20
# max_weight = np.sqrt(number_of_neurons)
# weight_matrix = [[np.random.random() * max_weight for i in range(number_of_neurons)] for j in
#                  range(number_of_neurons)]
# weight_matrix = []

epochs = 100
l_rate = 1
# Duration of the simulation in ms
T = 200
# Duration of each time step in ms
dt = 1
# Number of iterations = T/dt
steps = int(T / dt)
plot = True
plot = not plot

if weight_matrix != []:

    for epoch in range(epochs):

        network = Network(weight_matrix=weight_matrix, tau_refract=0)
        spikes = [False for neuron in range(number_of_neurons)]
        all_spikes = []
        # Output variables
        I = []
        V = []
        scaled_V = []
        spike_history_index = []
        spike_history_time = []

        for step in range(steps):

            all_spikes.append(spikes)

            t = step * dt

            network.did_it_spike = spikes
            network.step()

            # Set input current in mA and save var
            i = []
            v = []
            sc = []
            spikes = []
            for neuron in network.neuron_list:
                neuron.i_offset = 1.1 * (1.0 - (float(step)/float(steps)))  # np.random.random() * 1.1#0.06
                i.append(neuron.i_offset)
                v.append(neuron.v)
                sc.append(neuron.scaled_v)
                spikes.append(neuron.has_spiked)
            I.append(i)
            V.append(v)
            scaled_V.append(sc)

            for neuron in range(number_of_neurons):
                if spikes[neuron]:
                    spike_history_index.append(neuron)
                    spike_history_time.append(t)

        I = np.transpose(I).tolist()
        V = np.transpose(V).tolist()
        scaled_V = np.transpose(scaled_V).tolist()
        all_spikes = np.transpose(all_spikes).tolist()

        weight_matrix = back_prop(all_spikes, scaled_V, network)

        if plot or epoch == 0 or epoch == epochs-1:
            plt.rcParams["figure.figsize"] = (12, 6)
            # Draw the input current and the membrane potential
            plt.figure()
            for neuron in range(number_of_neurons):
                plt.plot([i for i in I[neuron]])
            plt.title('Square input stimuli')
            plt.ylabel('Input current (I)')
            plt.xlabel('Time (msec)')
            plt.figure()
            for neuron in range(number_of_neurons):
                plt.plot([v for v in V[neuron]])
            plt.axhline(y=network.v_thresh, color='r', linestyle='-')
            plt.title('LIF response')
            plt.ylabel('Membrane Potential (mV)')
            plt.xlabel('Time (msec)')
            plt.figure()
            plt.axis([0, T, -0.5, number_of_neurons])
            plt.title('Synaptic spikes')
            plt.ylabel('spikes')
            plt.xlabel('Time (msec)')
            plt.scatter(spike_history_time, spike_history_index)
            plt.show()

else:
    # Output variables
    I = []
    U = []

    neuron = Graz_LIF(weights=[], dt=dt, tau_refract=0)

    for step in range(steps):

        t = step * dt
        # Set input current in mA
        if t > 10 and t < 30:
            i_app = 0.01
        elif t > 50 and t < 100:
            i_app = 0.06
        elif t > 120 and t < 180:
            i_app = 0.08
        else:
            i_app = 0.0
        i_app = np.random.random() * 0.1

        neuron.i_offset = i_app
        neuron.step()

        I.append(i_app)
        U.append(neuron.v)



    plt.rcParams["figure.figsize"] =(12,6)
    # Draw the input current and the membrane potential
    plt.figure()
    plt.plot([i for i in I])
    plt.title('Square input stimuli')
    plt.ylabel('Input current (I)')
    plt.xlabel('Time (msec)')
    plt.figure()
    plt.plot([u for u in U])
    plt.axhline(y=neuron.v_thresh, color='r', linestyle='-')
    plt.title('LIF response')
    plt.ylabel('Membrane Potential (mV)')
    plt.xlabel('Time (msec)')
    plt.show()

print "done"