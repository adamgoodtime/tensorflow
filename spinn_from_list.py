import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

class SpiNN_LIF_curr(object):

    def __init__(self, connections, exc_or_inb='excitatory', dt=1.0, v_rest=-65.0, cm=1.0, tau_m=20.0, tau_syn_E=5.0,
                 tau_syn_I=5.0, tau_refract=5.0, v_thresh=-50.0, v_reset=0.0, i_offset=0.0, max_delay=127):
        self.dt = dt
        # passed in varaibles of the neuron
        self.exc_or_inb = exc_or_inb
        self.v_rest = v_rest  # * 10**-3
        self.cm = cm  # * 10**-3
        self.tau_m = tau_m  # * 10**-3
        self.r_mem = self.tau_m / cm
        self.tau_refract = tau_refract
        self.v_thresh = v_thresh  # * 10**-3
        self.v_reset = v_reset  # * 10**-3
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
        self.connections = connections
        self.has_spiked = False
        self.received_spikes = [False for i in range(self.connections[0])]
        del self.connections[0]
        self.spike_train = []

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

    def extract_the_spikes(self):
        for conn in self.connections:
            if self.received_spikes[conn[0]]:
                if conn[1] > 0:
                    update = conn[1] * self.curr_init_E
                else:
                    update += conn[2] * self.curr_init_I
                curr_param = {'update': update, 'delay': conn[2]}
                self.spike_train.append(curr_param)

    # collect current input from spikes
    def sum_the_spikes(self):
        if self.exc_or_inb == 'excitatory':
            self.i_in *= self.curr_decay_E
        else:
            self.i_in *= self.curr_decay_I
        self.extract_the_spikes()
        to_be_deleted = []
        for spike in self.spike_train:
            if spike['delay'] <= 0:
                self.i_in += spike['update']
                to_be_deleted.append(spike)
            else:
                spike['delay'] -= self.dt
        for deletable in to_be_deleted:
            del self.spike_train[self.spike_train.index(deletable)]
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

class Network(object):
    def __init__(self, conn_list, e_size=0, i_size=0, max_delay=127, dt=1.0, excite_params={}, inhib_params={}, v_rest=-65.0, cm=1.0,
                 tau_m=20.0, tau_syn_E=5.0, tau_syn_I=5.0, tau_refract=5.0, v_thresh=-55.0, v_reset=0.0, i_offset=0.0):
        self.max_delay = max_delay
        # neuron variable
        self.excite_params = excite_params
        self.inhib_params = inhib_params
        self.dt = dt
        self.v_rest = v_rest  # * 10**-3
        self.cm = cm  # * 10**-3
        self.tau_m = tau_m  # * 10**-3
        self.r_mem = self.tau_m / cm
        self.tau_syn_E = tau_syn_E
        self.tau_syn_I = tau_syn_I
        self.tau_refract = tau_refract
        self.v_thresh = v_thresh  # * 10**-3
        self.v_reset = v_reset  # * 10**-3
        self.i_offset = i_offset
        # network variables
        self.e_size = e_size
        self.i_size = i_size
        self.conn_list = conn_list
        self.number_of_neurons = self.e_size + self.i_size
        self.neuron_list = []
        self.did_they_spike = [False for i in range(self.number_of_neurons)]
        self.split_list = [[self.number_of_neurons] for i in range(self.number_of_neurons)]

        # split and reverse conn list so only list of neurons received from is left
        for conn in conn_list:
            reverse_conn = deepcopy(conn)
            reverse_conn[0] = reverse_conn[1]
            reverse_conn[1] = conn[0]
            self.split_list[reverse_conn[0]].append(reverse_conn[1:])

        # initialise the network of neurons connected
        for neuron in range(len(self.inhib_params)):
            self.neuron_list.append(
                SpiNN_LIF_curr(self.split_list[neuron], 'inhibitory', max_delay, dt,
                               inhib_params["v_rest"][neuron],
                               inhib_params["cm"][neuron],
                               inhib_params["tau_m"][neuron],
                               inhib_params["tau_syn_E"][neuron],
                               inhib_params["tau_syn_I"][neuron],
                               inhib_params["tau_refract"][neuron],
                               inhib_params["v_thresh"][neuron],
                               inhib_params["v_reset"][neuron],
                               inhib_params["i_offset"][neuron]))
        for neuron in range(len(self.inhib_params), self.i_size):
            self.neuron_list.append(
                SpiNN_LIF_curr(self.split_list[neuron], 'inhibitory', max_delay, dt, v_rest, cm, tau_m, tau_syn_E,
                               tau_syn_I, tau_refract, v_thresh, v_reset, i_offset))
        for neuron in range(self.i_size, self.i_size + len(self.excite_params)):
            self.neuron_list.append(
                SpiNN_LIF_curr(self.split_list[neuron], 'excitatory', dt,
                               excite_params["v_rest"][neuron - self.i_size],
                               excite_params["cm"][neuron - self.i_size],
                               excite_params["tau_m"][neuron - self.i_size],
                               excite_params["tau_syn_E"][neuron - self.i_size],
                               excite_params["tau_syn_I"][neuron - self.i_size],
                               excite_params["tau_refract"][neuron - self.i_size],
                               excite_params["v_thresh"][neuron - self.i_size],
                               excite_params["v_reset"][neuron - self.i_size],
                               excite_params["i_offset"][neuron - self.i_size]))
        for neuron in range(self.i_size + len(self.excite_params), self.number_of_neurons):
            self.neuron_list.append(
                SpiNN_LIF_curr(self.split_list[neuron], 'excitatory', max_delay, dt, v_rest, cm, tau_m, tau_syn_E,
                               tau_syn_I, tau_refract, v_thresh, v_reset, i_offset))

