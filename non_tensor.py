import numpy as np
import matplotlib.pyplot as plt

class Graz_LIF(object):

    def __init__(self, weights, dt=1.0, v_rest=0.0, cm=1.0, tau_m=20.0, tau_refract=5.0, v_thresh=1.0, v_reset=0.0, i_offset=0.0, alpha=0.05):
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
        self.t_rest = 0.0
        self.i = self.i_offset
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
        scaled_v = (self.v - self.v_thresh) / self.v_thresh
        z = self.H(scaled_v) * (1 / self.dt)
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
            self.neuron_list.append(Graz_LIF(self.weight_matrix[neuron], dt, v_rest, cm, tau_m, tau_refract, v_thresh, v_reset, i_offset, self.alpha))

    # step all neurons and save state
    def step(self):
        spike_tracking = []
        for neuron in self.neuron_list:
            neuron.received_spikes = self.did_it_spike
            neuron.step()
            spike_tracking.append(neuron.has_spiked)
        self.did_it_spike = spike_tracking

def calc_error(spike_history, single_error_neuron=True):
    target_hz = 100.0
    spike_history_index = spike_history[0]
    spike_history_time = spike_history[1]
    if single_error_neuron:
        number_of_spikes = 0.0
        for spike in spike_history_index:
            if spike == number_of_neurons - 1:
                number_of_spikes += 1.0
        actual_hz = number_of_spikes / float(T)
    else:
        actual_hz = float(len(spike_history_time)) / float(T)
    error = 0.5 * (target_hz - actual_hz)**2
    return error

# error = 0.5 (y - y_target)^2
def back_prop(spike_history, network):
    error = calc_error(spike_history)
    theta = 1
    dEdz = theta * error
    dzdu = 0
    dEdVt1 = 0
    dEdVt = (dEdz * dzdu / network.v_thresh) + (dEdVt1 * network.alpha)
    dEdzt = dEdz - (dEdVt1 * dt * network.v_thresh) + \
            sum([dEdVt1 * network.weight_matrix["blah"] * (1 - network.alpha) * network.r_mem for neuron in network.neuron_list])


# Network
max_weight = 0.01
number_of_neurons = 6
weight_matrix = [[np.random.random() * max_weight for i in range(number_of_neurons)] for j in range(number_of_neurons)]

epochs = 10
# Duration of the simulation in ms
T = 200
# Duration of each time step in ms
dt = 1
# Number of iterations = T/dt
steps = int(T / dt)

if weight_matrix:

    for epoch in range(epochs):

        network = Network(weight_matrix=weight_matrix, tau_refract=0)
        spikes = [False for neuron in range(number_of_neurons)]
        # Output variables
        I = []
        V = []
        spike_history_index = []
        spike_history_time = []

        for step in range(steps):

            t = step * dt

            network.did_it_spike = spikes
            network.step()

            # Set input current in mA and save var
            i = []
            v = []
            spikes = []
            for neuron in network.neuron_list:
                neuron.i_offset = np.random.random() * 0.06
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