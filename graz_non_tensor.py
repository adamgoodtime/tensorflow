import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import warnings
warnings.filterwarnings("error")

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
        self.can_it_spike = True
        self.poisson_rate = 0.0

    # activation function
    def H(self, x, dev=False, gammma=0.1):
        if not sigmoid:
            if dev:
                return (gammma/self.dt) * max(0, 1-abs(x))
            else:
                return gammma * max(0, 1 - abs(x))
        if dev:
            # x *= 3
            return np.exp(-x) / ((1 + np.exp(-x))**2)
            value = self.H(x)
            # print "H dev - og:", np.exp(-x) / ((1 + np.exp(-x))**2), "gb:", value / (1 - value)
            return value * (1 - value)
        else:
            return 1 / (1 + np.exp(-x))

    # operation to be performed when not spiking
    def integrating(self):
        self.has_spiked = False
        current = self.return_current()
        # z = self.H(self.scaled_v) * (1 / self.dt)
        if self.can_it_spike:
            update = (self.alpha * (self.v - self.v_rest)) + ((1 - self.alpha) * self.r_mem * current)# - (self.dt * self.v_thresh * z)
        else:
            update = self.v + (self.r_mem * current)
        update = (self.alpha * (self.v - self.v_rest)) + ((1 - self.alpha) * self.r_mem * current)# - (self.dt * self.v_thresh * z)
        self.v = update
        self.scaled_v = (self.v - self.v_thresh) / self.v_thresh

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
        # for neuron in range(len(self.received_spikes)):
        #     if self.received_spikes[neuron]:
        #         total_current += self.weights[neuron] * (1 / self.dt)
        for neuron in range(len(self.received_spikes)):
            total_current += self.weights[neuron] * (1 / self.dt) * self.received_spikes[neuron]
        return total_current

    # calculates the current for all inputs
    def return_current(self):
        total = self.sum_the_spikes()
        return self.i_offset + total

    # step the neuron
    def step(self):
        if self.poisson_rate:
            if np.random.random() < self.poisson_rate * (self.dt / 1000.0):
                self.has_spiked = True
            else:
                self.has_spiked = False
        else:
            if self.v > self.v_thresh and self.can_it_spike and not sigmoid:
                self.spiked()
            elif self.t_rest > 0:
                self.refracting()
            else:
                self.integrating()

class Network(object):

    def __init__(self, weight_matrix, dt=1.0, v_rest=0.0, cm=1.0, tau_m=20.0, tau_refract=5.0, v_thresh=1.0, v_reset=0.0, i_offset=0.0):
        self.alpha = np.exp(- dt / tau_m)
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
        # matrix multiplication of w[] and z[]

        # initialise the network of neurons connected
        for neuron in range(self.number_of_neurons):
            self.neuron_list.append(Graz_LIF(np.take(self.weight_matrix, neuron, axis=1), dt, v_rest, cm, tau_m, tau_refract, v_thresh, v_reset, i_offset, self.alpha))
            # self.neuron_list.append(Graz_LIF(self.weight_matrix[neuron], dt, v_rest, cm, tau_m, tau_refract, v_thresh, v_reset, i_offset, self.alpha))

    # step all neurons and save state
    def step(self):
        spike_tracking = []
        for neuron in self.neuron_list:
            neuron.received_spikes = self.did_it_spike
            neuron.step()
            spike_tracking.append(neuron.has_spiked)
        self.did_it_spike = spike_tracking

def error_and_BP_gradients(weight_matrix, return_error=False, quadratic=False):
    network = Network(weight_matrix=weight_matrix, tau_refract=0)
    number_of_neurons = len(weight_matrix)
    spikes = [False for neuron in range(number_of_neurons)]
    activations = [network.neuron_list[0].H(-1.0) for neuron in range(number_of_neurons)]
    all_spikes = []
    all_activations = []
    # Output variables
    I = []
    V = []
    scaled_V = []
    spike_history_index = []
    spike_history_time = []

    for step in range(steps):

        all_spikes.append(spikes)
        all_activations.append(activations)

        t = step * dt

        if sigmoid:
            network.did_it_spike = activations
        else:
            network.did_it_spike = spikes
        network.step()

        # Set input current in mA and save var
        i = []
        v = []
        sc = []
        spikes = []
        activations = []
        np.random.seed(272727)
        for idx, neuron in enumerate(network.neuron_list):
            # if idx < number_of_neurons - 1:
                # neuron.i_offset = 20 #* np.random.random()  # * (2.0 - (float(step)/float(steps))) / 2  # np.random.random() * 1.1#0.06
                # neuron.v = 1.1
            # else:
            #     if optimize == 'sine':
            #         neuron.can_it_spike = False
            i.append(neuron.i_offset)
            v.append(neuron.v)
            sc.append(neuron.scaled_v)
            spikes.append(neuron.has_spiked)
            activations.append(neuron.H(neuron.scaled_v))
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
    all_activations = np.transpose(all_activations).tolist()

    if return_error:
        if sigmoid:
            return hz_error(all_activations, quadratic=quadratic)
        else:
            return hz_error(all_spikes, quadratic=quadratic)
    else:
        return back_prop(all_spikes, scaled_V, network, all_activations, check=True)

def hz_error(spike_history, single_error_neuron=True, quadratic=True):
    target_hz = 20
    number_of_neurons = len(spike_history)
    if single_error_neuron:
        actual_hz = float(sum(spike_history[number_of_neurons-1])) / (float(len(spike_history[number_of_neurons-1])) * (float(dt) / 1000.0))
    else:
        actual_hz = float(sum([sum(spike_history[n]) for n in range(number_of_neurons)])) / (float(T) / 1000.0)
    if quadratic:
        error = 0.5 * (target_hz - actual_hz)**2
    else:
        error = actual_hz - target_hz
        # error = target_hz - actual_hz
    print "Error for the last iteration:", error, " with target:", target_hz, "and actual:", actual_hz, "quad:", quadratic
    return error

def sine_error(voltage_history, quadratic=False):
    error = 0.0
    threshold = network.v_thresh
    if quadratic:
        error = [0.5 * (target_sine_wave[t] - ((voltage_history[number_of_neurons-1][t] * threshold) + threshold))**2 for t in range(T)]
    else:
        error = [((voltage_history[number_of_neurons-1][t] * threshold) + threshold) - target_sine_wave[t] for t in range(T)]
        # error = [target_sine_wave[t] - ((voltage_history[number_of_neurons-1][t] + threshold) * threshold) for t in range(T)]
        # error = target_hz - actual_hz
    print "Error for the last iteration:", sum(error)
    return error

error_tracker = []
# error = 0.5 (y - y_target)^2
def back_prop(spike_history, voltage_history, network, activations, check=False):
    number_of_neurons = len(spike_history)
    global error_tracker
    if optimize == 'sine':
        error = sine_error(voltage_history)
        error_tracker.append(sum([abs(error[i]) for i in range(len(error))]))
    else:
        if sigmoid:
            error = hz_error(activations, quadratic=False)
        else:
            error = hz_error(spike_history, quadratic=False)
        error_tracker.append(error)
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
            if pseudo_derivative:
                leak = np.exp(-network.dt / network.tau_m)
                if optimize == 'sine':
                    p_dEdz = error[t] * leak # * network.weight_matrix[neuron][number_of_neurons-1]
                    # p_dEdz = sum(error) * leak #* network.weight_matrix[neuron][number_of_neurons-1]
                else:
                    p_dEdz = error #* leak #* network.weight_matrix[neuron][number_of_neurons-1]
                sum_dEdV = sum([dEdV[n][t+1] *
                                # weight_matrix[neuron][n] *
                                network.weight_matrix[neuron][n] *
                                # network.weight_matrix[n][neuron] *
                                (1-network.neuron_list[n].alpha) *
                                this_neuron.r_mem
                                for n in range(number_of_neurons)])
                dEdz[neuron][t] = p_dEdz - (dEdV[neuron][t+1] * dt * this_neuron.v_thresh) + sum_dEdV
                # sum_dEdV = sum([(dEdV[n][t+1] *
                #                 # weight_matrix[neuron][n] *
                #                 # network.weight_matrix[neuron][n] *
                #                 network.weight_matrix[n][neuron] *
                #                 (1-network.neuron_list[n].alpha) *
                #                 this_neuron.r_mem) -
                #                 (dEdV[n][t + 1] * dt * this_neuron.v_thresh)
                #                 for n in range(number_of_neurons)])
                # dEdz[neuron][t] = p_dEdz + sum_dEdV
                dEdV[neuron][t] = dEdz[neuron][t] * pseudo_derivative * (1/this_neuron.v_thresh) + \
                                  (dEdV[neuron][t+1] * this_neuron.alpha)

    for pre in range(number_of_neurons):
        for post in range(number_of_neurons):
            if new_weight_matrix[pre][post]:
                if sigmoid:
                    dEdWi[pre][post] = sum([dEdV[post][t] * activations[pre][t] * (1 / dt) for t in range(T/dt)])
                    dEdWr[pre][post] = sum([dEdV[post][t] * activations[pre][t] * (1 / dt) for t in range(T/dt)])
                else:
                    dEdWi[pre][post] = sum([dEdV[post][t] * spike_history[pre][t] * (1 / dt) for t in range(T/dt)])
                    dEdWr[pre][post] = sum([dEdV[post][t] * spike_history[pre][t] * (1 / dt) for t in range(T/dt)])
                new_weight_matrix[pre][post] -= l_rate * dEdWr[pre][post]
                update_weight_matrix[pre][post] -= l_rate * dEdWr[pre][post]


    print "\noriginal\n", network.weight_matrix
    print "update\n", update_weight_matrix
    print "new\n", new_weight_matrix
    print "error", error_tracker[len(error_tracker)-1], "with LR", l_rate
    print error_tracker
    if check:
        return dEdWr
    else:
        return new_weight_matrix

# Network
sigmoid = True
# Feedforward network
neurons_per_layer = 1
hidden_layers = 1
input_neurons = 0
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
# number_of_neurons = 20
# weight_scale = np.sqrt(number_of_neurons)
# weight_matrix = [[np.random.randn() / weight_scale for i in range(number_of_neurons)] for j in
#                  range(number_of_neurons)]
# weight_matrix = 0

epochs = 100
l_rate = 0.1
max_l_rate = 0.0005
min_l_rate = 0.00001
# Duration of the simulation in ms
T = 2
# Duration of each time step in ms
dt = 1
# Number of iterations = T/dt
steps = int(T / dt)
plot = True
plot = not plot
end_at_best = True

optimize = 'hz'
starting_value = 0.0
target_sine = lambda x: 0.25 * (np.sin(10 * x)) + starting_value
target_sine_wave = [target_sine(t/1000.0) for t in range(T)]
poisson_rate = 100
number_of_poisson = 0 # number_of_neurons / 2

if __name__ == "__main__":

    if weight_matrix.tolist() != 0:

        for epoch in range(epochs):

            network = Network(weight_matrix=weight_matrix, tau_refract=0)
            spikes = [False for neuron in range(number_of_neurons)]
            all_spikes = []
            activations = [network.neuron_list[0].H(-1.0) for neuron in range(number_of_neurons)]
            all_activations = []
            # Output variables
            I = []
            V = []
            scaled_V = []
            spike_history_index = []
            spike_history_time = []
            l_rate = min_l_rate + ((1 - (float(epoch) / float(epochs))) * (max_l_rate - min_l_rate))

            np.random.seed(272727)
            for neuron in network.neuron_list:
                neuron.v = np.random.random()
            neuron.v = starting_value
            for step in range(steps):

                all_spikes.append(spikes)
                all_activations.append(activations)

                t = step * dt

                if sigmoid:
                    network.did_it_spike = activations
                else:
                    network.did_it_spike = spikes
                network.step()

                # Set input current in mA and save var
                i = []
                v = []
                sc = []
                spikes = []
                activations = []
                for idx, neuron in enumerate(network.neuron_list):
                    if idx < number_of_neurons - 1:
                        if optimize == 'sine':
                            # neuron.i_offset = 2 * np.random.random() #* (2.0 - (float(step)/float(steps))) / 2  # np.random.random() * 1.1#0.06
                            if idx < number_of_poisson:
                                neuron.poisson_rate = poisson_rate
                        else:
                            # neuron.i_offset = 2 * np.random.random() #* (2.0 - (float(step)/float(steps))) / 2  # np.random.random() * 1.1#0.06
                            if idx < number_of_poisson:
                                neuron.poisson_rate = poisson_rate
                    else:
                        if optimize == 'sine':
                            neuron.can_it_spike = False
                    i.append(neuron.i_offset)
                    v.append(neuron.v)
                    sc.append(neuron.scaled_v)
                    spikes.append(neuron.has_spiked)
                    activations.append(neuron.H(neuron.scaled_v))
                I.append(i)
                V.append(v)
                scaled_V.append(sc)

                for neuron in range(number_of_neurons):
                    if spikes[neuron]:
                        spike_history_index.append(neuron)
                        spike_history_time.append(t)

                # df = lambda x: gradients(x, np.transpose(scaled_V).tolist(), network)
                # check_gradient(f=hz_error, df=df, x0=np.transpose(all_spikes).tolist())

            I = np.transpose(I).tolist()
            V = np.transpose(V).tolist()
            scaled_V = np.transpose(scaled_V).tolist()
            all_spikes = np.transpose(all_spikes).tolist()
            all_activations = np.transpose(all_activations).tolist()

            weight_matrix = back_prop(all_spikes, scaled_V, network, all_activations)
            print "epoch", epoch, "/", epochs

            # df = lambda x: gradients(all_spikes, scaled_V, network, return_error=False)
            # check_gradient(f=hz_error, df=df, x0=weight_matrix)

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
                if optimize == 'sine':
                    plt.plot(target_sine_wave)
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

            if abs(error_tracker[len(error_tracker)-1]) < 0.1 and end_at_best:
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
                if optimize == 'sine':
                    plt.plot(target_sine_wave)
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
                break

        print "\n", error_tracker

    else:
        # Output variables
        I = []
        U = []

        neuron = Graz_LIF(weights=[], dt=dt, tau_refract=0)

        spike_history_index = []
        spike_history_time = []

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
            i_app = 1.05  # np.random.random() * 0.1

            neuron.i_offset = i_app
            neuron.step()

            if neuron.has_spiked:
                spike_history_index.append(1)
                spike_history_time.append(t)

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
        plt.figure()
        plt.axis([0, T, -0.5, number_of_neurons])
        plt.title('Synaptic spikes')
        plt.ylabel('spikes')
        plt.xlabel('Time (msec)')
        plt.scatter(spike_history_time, spike_history_index)
        plt.show()

    print "done"