import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

class SpiNN_LIF_curr(object):

    def __init__(self, weights, exc_or_inb='excitatory', dt=1.0, v_rest=-65.0, cm=1.0, tau_m=20.0, tau_syn_E=5.0, tau_syn_I=5.0, tau_refract=5.0, v_thresh=-50.0, v_reset=0.0, i_offset=0.0, alpha=0.05):
        self.alpha = alpha
        self.dt = dt
        # passed in varaibles of the neuron
        self.exc_or_inb = exc_or_inb
        self.v_rest = v_rest #* 10**-3
        self.cm = cm #* 10**-3
        self.tau_m = tau_m #* 10**-3
        self.r_mem = self.tau_m / cm
        self.tau_refract = tau_refract
        self.v_thresh = v_thresh #* 10**-3
        self.v_reset = v_reset #* 10**-3
        self.i_offset = i_offset
        self.curr_init_E = np.exp(-dt / tau_syn_E)
        self.curr_decay_E = (tau_syn_E / dt) * (1.0 - np.exp(-dt / tau_syn_E))
        self.curr_init_I = np.exp(-dt / tau_syn_I)
        self.curr_decay_I = (tau_syn_I / dt) * (1.0 - np.exp(-dt / tau_syn_I))
        # state variables
        self.v = self.v_rest
        self.scaled_v = (self.v - self.v_thresh) / self.v_thresh
        self.t_rest = 0.0
        self.i_in = self.i_offset
        # network variables
        self.weights = weights
        self.has_spiked = False
        self.received_spikes = [False for i in range(len(weights))]

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
        if self.exc_or_inb == 'excitatory':
            self.i_in *= self.curr_decay_E
        else:
            self.i_in *= self.curr_decay_I
        for neuron in range(len(self.received_spikes)):
            if self.received_spikes[neuron]:
                if self.weights[neuron] > 0:
                    self.i_in += self.weights[neuron] * self.curr_init_E
                else:
                    self.i_in += self.weights[neuron] * self.curr_init_I
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
        
class xor_neuron(object):
    
    def __init__(self, rate, stochastic):
        self.rate = rate
        self.dt = 0
        self.has_spiked = False
        self.stochastic = stochastic
        self.received_spikes = []
        
    def step(self):
        if self.stochastic:
            if np.random.random() < float(self.dt) / float(self.rate):
                self.has_spiked = True
            else:
                self.has_spiked = False
        else:
            if self.dt % self.rate == 0:
                self.has_spiked = True
            else:
                self.has_spiked = False
        self.dt += 1
        
        
class Network(object):

    def __init__(self, weight_matrix, e_size=0, i_size=0, dt=1.0, v_rest=-65.0, cm=1.0, tau_m=20.0, tau_syn_E=5.0, tau_syn_I=5.0, tau_refract=5.0, v_thresh=-55.0, v_reset=0.0, i_offset=0.0):
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
        self.e_size = e_size
        self.i_size = i_size
        self.weight_matrix = weight_matrix
        self.number_of_neurons = len(weight_matrix)
        self.neuron_list = []
        self.did_they_spike = [False for i in range(self.number_of_neurons)]

        # initialise the network of neurons connected
        for neuron in range(self.i_size):
            self.neuron_list.append(SpiNN_LIF_curr(self.weight_matrix[neuron], 'inhibitory', dt, v_rest, cm, tau_m, tau_syn_E, tau_syn_I, tau_refract, v_thresh, v_reset, i_offset, self.alpha))
        for neuron in range(self.i_size, self.number_of_neurons):
            self.neuron_list.append(SpiNN_LIF_curr(self.weight_matrix[neuron], 'excitatory', dt, v_rest, cm, tau_m, tau_syn_E, tau_syn_I, tau_refract, v_thresh, v_reset, i_offset, self.alpha))

    # step all neurons and save state
    def step(self):
        spike_tracking = []
        for neuron in self.neuron_list:
            neuron.received_spikes = self.did_they_spike
            neuron.step()
            spike_tracking.append(neuron.has_spiked)
        self.did_they_spike = spike_tracking


class Network_in_xor(object):

    def __init__(self, weight_matrix, delay_matrix, on_rate, off_rate, score_delay, xor_input, stochastic=False, excite_params={}, inhib_params={},
                 dt=1.0, v_rest=-65.0, cm=1.0, tau_m=20.0,  tau_syn_E=5.0, tau_syn_I=5.0, tau_refract=5.0, v_thresh=-50.0, v_reset=-65.0, i_offset=0.0):
        self.alpha = dt / tau_m
        self.excite_params = excite_params
        self.inhib_params = inhib_params
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
        self.did_they_spike = [False for i in range(self.number_of_neurons)]
        # environment variable
        self.inputs = 2
        self.outputs = 2
        self.on_rate = on_rate
        self.off_rate = off_rate
        self.stochastic = stochastic
        self.score_delay = score_delay
        self.score_tracking = []
        self.output_spike_count = [0 for i in range(self.outputs)]
        if xor_input[0] == xor_input[1]:
            self.xor_output = 0
        else:
            self.xor_output = 1

        # initialise the network of neurons connected
        for env_neuron in range(self.inputs):
            if xor_input[env_neuron]:
                self.neuron_list.append(xor_neuron(on_rate, stochastic))
            else:
                self.neuron_list.append(xor_neuron(off_rate, stochastic))
        for neuron in range(self.inputs, len(self.excite_params) + self.inputs):
            self.neuron_list.append(SpiNN_LIF_curr(self.weight_matrix[neuron],
                                                   'excitatory',
                                                   excite_params["dt"],
                                                   excite_params["v_rest"],
                                                   excite_params["cm"],
                                                   excite_params["tau_m"],
                                                   excite_params["tau_syn_E"],
                                                   excite_params["tau_syn_I"],
                                                   excite_params["tau_refract"],
                                                   excite_params["v_thresh"],
                                                   excite_params["v_reset"],
                                                   excite_params["i_offset"],
                                                   self.alpha))
        for neuron in range(len(self.excite_params) + self.inputs, len(self.inhib_params) + len(self.excite_params) + self.inputs):
            self.neuron_list.append(SpiNN_LIF_curr(self.weight_matrix[neuron],
                                                   'excitatory',
                                                   inhib_params["dt"],
                                                   inhib_params["v_rest"],
                                                   inhib_params["cm"],
                                                   inhib_params["tau_m"],
                                                   inhib_params["tau_syn_E"],
                                                   inhib_params["tau_syn_I"],
                                                   inhib_params["tau_refract"],
                                                   inhib_params["v_thresh"],
                                                   inhib_params["v_reset"],
                                                   inhib_params["i_offset"],
                                                   self.alpha))
        for neuron in range(len(self.inhib_params) + len(self.excite_params) + self.inputs, self.number_of_neurons):
            self.neuron_list.append(SpiNN_LIF_curr(self.weight_matrix[neuron], 'excitatory', dt, v_rest, cm, tau_m,
                                                   tau_syn_E, tau_syn_I, tau_refract, v_thresh, v_reset, i_offset,
                                                   self.alpha))

    def add_delays(self):
        # need to get rid of connection matrix as that fucks multapses
        return "it's fucked"

    # step all neurons and save state
    def step(self, current_dt):
        spike_tracking = []
        for neuron in self.neuron_list:
            neuron.received_spikes = self.did_they_spike
            neuron.step()
            spike_tracking.append(neuron.has_spiked)
        self.output_spike_count[0] += self.did_they_spike[number_of_neurons-2]
        self.output_spike_count[1] += self.did_they_spike[number_of_neurons-1]
        self.score(current_dt)
        self.did_they_spike = spike_tracking

    def score(self, current_dt):
        if current_dt % self.score_delay == 0 and current_dt:
            if self.output_spike_count[0] > self.output_spike_count[1] and self.xor_output:
                score = 0.0
            elif self.output_spike_count[0] < self.output_spike_count[1] and self.xor_output:
                score = 1.0
            elif self.output_spike_count[0] > self.output_spike_count[1] and not self.xor_output:
                score = 1.0
            elif self.output_spike_count[0] < self.output_spike_count[1] and not self.xor_output:
                score = 0.0
            else:
                score = 0.0
            self.score_tracking.append(score)
            self.output_spike_count = [0 for i in range(self.outputs)]


# Network
max_weight = 0.001
number_of_neurons = 6
weight_matrix = [[np.random.random() * max_weight for i in range(number_of_neurons)] for j in range(number_of_neurons)]
# weight_matrix = []

epochs = 0
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

        network = Network(weight_matrix=weight_matrix, tau_refract=0, dt=dt)
        spikes = [False for neuron in range(number_of_neurons)]
        # Output variables
        I = []
        V = []
        spike_history_index = []
        spike_history_time = []
        all_spikes = []

        for step in range(steps):

            all_spikes.append(spikes)

            t = step * dt

            network.did_they_spike = spikes
            network.step()

            # Set input current in mA and save var
            i = []
            v = []
            spikes = []
            for neuron in network.neuron_list:
                neuron.i_offset = np.random.random() * 1
                i.append(neuron.i_offset)
                v.append(neuron.v)
                spikes.append(neuron.has_spiked)
            I.append(i)
            V.append(v)

            for neuron in range(number_of_neurons):
                if spikes[neuron]:
                    spike_history_index.append(neuron)
                    spike_history_time.append(t)

        I = np.transpose(I).tolist()
        V = np.transpose(V).tolist()
        all_spikes = np.transpose(all_spikes).tolist()

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