import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

class SpiNN_LIF_curr(object):

    def __init__(self, weights, dt=1.0, v_rest=-65.0, cm=1.0, tau_m=20.0, tau_refract=5.0, v_thresh=-50.0, v_reset=0.0, i_offset=0.0, alpha=0.05):
        self.alpha = alpha
        # passed in varaibles of the neuron
        self.dt = dt
        self.v_rest = v_rest #* 10**-3
        self.cm = cm #* 10**-3
        self.tau_m = tau_m #* 10**-3
        self.r_mem = self.tau_m / cm
        self.tau_refract = tau_refract
        self.v_thresh = v_thresh #* 10**-3
        self.v_reset = v_reset #* 10**-3
        self.i_offset = i_offset
        # state variables
        self.v = self.v_rest
        self.scaled_v = (self.v - self.v_thresh) / self.v_thresh
        self.t_rest = 0.0
        self.i_in = self.i_offset
        # network variables
        self.weights = weights
        self.has_spiked = False
        self.received_spikes = [False for i in range(len(weights))]

    # activation function
    def H(self, x):
        return 1 / (1 + np.exp(-x))

    # operation to be performed when not spiking
    def integrating(self):
        self.has_spiked = False
        current = self.return_current()
        alpha = (current * self.r_mem) + self.v_rest
        tau = np.exp(-self.dt / (self.r_mem * self.cm))
        self.v = alpha - (tau * (alpha - self.v))

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
        self.i_in *= 0.8187307530779818
        for neuron in range(len(self.received_spikes)):
            if self.received_spikes[neuron]:
                self.i_in += self.weights[neuron] * 0.9063462346100909
        return self.i_in

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
        self.r_mem = self.tau_m / cm
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
            self.neuron_list.append(SpiNN_LIF_curr(self.weight_matrix[neuron], dt, v_rest, cm, tau_m, tau_refract, v_thresh, v_reset, i_offset, self.alpha))

    # step all neurons and save state
    def step(self):
        spike_tracking = []
        for neuron in self.neuron_list:
            neuron.received_spikes = self.did_it_spike
            neuron.step()
            spike_tracking.append(neuron.has_spiked)
        self.did_it_spike = spike_tracking

def calc_error(spike_history, single_error_neuron=False):
    target_hz = 10.0
    if single_error_neuron:
        actual_hz = float(sum(spike_history[number_of_neurons-1])) / float(T)
    else:
        actual_hz = float(sum([sum(spike_history[n]) for n in range(number_of_neurons)])) / float(T)
    error = 0.5 * (target_hz - actual_hz)**2
    return error

# error = 0.5 (y - y_target)^2
def back_prop(spike_history, voltage_history, network):
    error = calc_error(spike_history)
    print "Error for the last iteration is", error
    new_weight_matrix = deepcopy(network.weight_matrix)
    # theta = 1
    # dEdz = theta * error
    # dzdu = 0
    # dEdVt1 = 0
    # dEdVt = (dEdz * dzdu / network.v_thresh) + (dEdVt1 * network.alpha)
    # dEdzt = dEdz - (dEdVt1 * dt * network.v_thresh) + \
    #         sum([dEdVt1 * network.weight_matrix["blah"] * (1 - network.alpha) * network.r_mem for neuron in network.neuron_list])
    dEdz = [[0.0 for i in range(T/dt + 1)] for neuron in range(number_of_neurons)]
    dEdV = [[0.0 for i in range(T/dt + 1)] for neuron in range(number_of_neurons)]
    # dEdWi = [[[0.0 for i in range(T/dt + 1)] for pre in range(number_of_neurons)] for post in range(number_of_neurons)]
    # dEdWr = [[[0.0 for i in range(T/dt + 1)] for pre in range(number_of_neurons)] for post in range(number_of_neurons)]
    dEdWi = [[0.0 for pre in range(number_of_neurons)] for post in range(number_of_neurons)]
    dEdWr = [[0.0 for pre in range(number_of_neurons)] for post in range(number_of_neurons)]
    for t in range(T/dt-1, -1, -1):
        for neuron in range(number_of_neurons):
            this_neuron = network.neuron_list[neuron]
            if spike_history[neuron][t]:
                p_dEdz = error
                sum_dEdV = sum([dEdV[n][t+1] *
                                weight_matrix[neuron][n] *
                                (1-network.neuron_list[n].alpha) *
                                this_neuron.r_mem
                                for n in range(number_of_neurons)])
                dEdz[neuron][t] = p_dEdz - (dEdV[neuron][t+1] * dt * this_neuron.v_thresh) + sum_dEdV
                dEdV[neuron][t] = dEdz[neuron][t] * ((1/dt)/voltage_history[neuron][t]) * (1/this_neuron.v_thresh) + \
                                  (dEdV[neuron][t+1] * this_neuron.alpha)

    for pre in range(number_of_neurons):
        for post in range(number_of_neurons):
            dEdWi[pre][post] = sum([dEdV[post][t] * spike_history[pre][t] for t in range(T/dt)])
            dEdWr[pre][post] = sum([dEdV[post][t] * spike_history[pre][t] for t in range(T/dt)])

            new_weight_matrix[pre][post] += l_rate * dEdWi[pre][post]

    return new_weight_matrix

# Network
max_weight = 0.001
number_of_neurons = 6
weight_matrix = [[np.random.random() * max_weight for i in range(number_of_neurons)] for j in range(number_of_neurons)]
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
# plot = not plot

if weight_matrix:

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
                neuron.i_offset = np.random.random() * 1
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

    neuron = SpiNN_LIF_curr(weights=[], dt=dt, tau_refract=0)

    for step in range(steps):

        t = step * dt
        # Set input current in mA
        if t > 10 and t < 30:
            i_app = 0.5
        elif t > 50 and t < 100:
            i_app = 1.0
        elif t > 120 and t < 180:
            i_app = 1.5
        else:
            i_app = 0.0
        # i_app = np.random.random() * 1

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