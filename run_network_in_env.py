from spinn_non_tensor import Network_in_xor as XOR
import numpy as np
import matplotlib.pyplot as plt


def run_network():
    max_weight = 4.8
    max_delay = 25
    number_of_neurons = 6
    weight_matrix = [[np.random.random() * max_weight for i in range(number_of_neurons)] for j in
                     range(number_of_neurons)]
    delay_matrix = [[int(np.random.random() * max_weight) for i in range(number_of_neurons)] for j in
                     range(number_of_neurons)]
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

    rate_on = 20
    rate_off = 5
    score_delay = T
    xor_input = [0, 1]

    for epoch in range(epochs):

        network = XOR(weight_matrix, delay_matrix, rate_on, rate_off, score_delay, xor_input)
        spikes = [False for neuron in range(number_of_neurons)]
        # Output variables
        I = []
        V = []
        spike_history_index = []
        spike_history_time = []

        for step in range(steps+1):

            t = step * dt

            network.did_it_spike = spikes
            network.step(step)

            # Set input current in mA and save var
            i = []
            v = []
            spikes = []
            for neuron in network.neuron_list:
                neuron.i_offset = np.random.random() * 1
                # i.append(neuron.i_offset)
                # v.append(neuron.v)
                spikes.append(neuron.has_spiked)
            # I.append(i)
            # V.append(v)

            for neuron in range(number_of_neurons):
                if spikes[neuron]:
                    spike_history_index.append(neuron)
                    spike_history_time.append(t)

        # I = np.transpose(I).tolist()
        # V = np.transpose(V).tolist()
        #
        # if plot or epoch == 0 or epoch == epochs - 1:
        #     plt.rcParams["figure.figsize"] = (12, 6)
        #     # Draw the input current and the membrane potential
        #     plt.figure()
        #     for neuron in range(number_of_neurons):
        #         plt.plot([i for i in I[neuron]])
        #     plt.title('Square input stimuli')
        #     plt.ylabel('Input current (I)')
        #     plt.xlabel('Time (msec)')
        #     plt.figure()
        #     for neuron in range(number_of_neurons):
        #         plt.plot([v for v in V[neuron]])
        #     plt.axhline(y=network.v_thresh, color='r', linestyle='-')
        #     plt.title('LIF response')
        #     plt.ylabel('Membrane Potential (mV)')
        #     plt.xlabel('Time (msec)')
        plt.figure()
        plt.axis([0, T, -0.5, number_of_neurons])
        plt.title('Synaptic spikes')
        plt.ylabel('spikes')
        plt.xlabel('Time (msec)')
        plt.scatter(spike_history_time, spike_history_index)
        plt.show()

run_network()