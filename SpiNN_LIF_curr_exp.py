from __future__ import print_function

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

        #     v_rest = -65.0  # Resting membrane potential in mV.
        #     v_rest_stdev = 5
        #     cm = 1.0  # Capacity of the membrane in nF
        #     cm_stdev = 0.3
        #     tau_m = 20.0  # Membrane time constant in ms.
        #     tau_m_stdev = 5
        #     tau_refrac = 0.1  # Duration of refractory period in ms.
        #     tau_refrac_stdev = 0.03
        #     tau_syn_E = 5  # Rise time of the excitatory synaptic alpha function in ms.
        #     tau_syn_E_stdev = 1.6
        #     tau_syn_I = 5  # Rise time of the inhibitory synaptic alpha function in ms.
        #     tau_syn_I_stdev = 1.6
        #     e_rev_E = 0.0  # Reversal potential for excitatory input in mV
        #     e_rev_E_stdev = 0
        #     e_rev_I = -70.0  # Reversal potential for inhibitory input in mV
        #     e_rev_I_stdev = 3
        #     v_thresh = -50.0  # Spike threshold in mV.
        #     v_thresh_stdev = 5
        #     v_reset = -65.0  # Reset potential after a spike in mV.
        #     v_reset_stdev = 5
        #     i_offset = 0  # Offset current in nA
        #     i_offset_stdev = input_current_stdev
        #     v = -65.0  # 'v_starting'
        #     v_stdev = 5
        # elif neuron_type == 'IF_curr_exp':
        #     v_rest = -65.0  # Resting membrane potential in mV.
        #     v_rest_stdev = 5
        #     cm = 1.0  # Capacity of the membrane in nF
        #     cm_stdev = 0.3
        #     tau_m = 20.0  # Membrane time constant in ms.
        #     tau_m_stdev = 5
        #     tau_refrac = 0.1  # Duration of refractory period in ms.
        #     tau_refrac_stdev = 0.03
        #     tau_syn_E = 5  # Rise time of the excitatory synaptic alpha function in ms.
        #     tau_syn_E_stdev = 1.6
        #     tau_syn_I = 5  # Rise time of the inhibitory synaptic alpha function in ms.
        #     tau_syn_I_stdev = 1.6
        #     v_thresh = -50.0  # Spike threshold in mV.
        #     v_thresh_stdev = 5
        #     v_reset = -65.0  # Reset potential after a spike in mV.
        #     v_reset_stdev = 5
        #     i_offset = 0  # Offset current in nA
        #     i_offset_stdev = input_current_stdev
        #     v = -65.0  # 'v_starting'
        #     v_stdev = 5

class SpiNN_LIF_curr_exp(object):

    def __init__(self, v_rest=-65.0, cm=1.0, tau_m=20.0, tau_refract=5.0, v_thresh=-50.0, v_reset=-65.0, i_offest=0.0):
        self.v_rest = v_rest #* 10**-3
        self.cm = cm #* 10**-3
        self.tau_m = tau_m #* 10**-3
        self.r_mem = tau_m / cm
        self.tau_refract = tau_refract
        self.v_thresh = v_thresh #* 10**-3
        self.v_reset = v_reset #* 10**-3
        self.i_offest = i_offest

        self.graph = tf.Graph()

        # self.var1 = 0.0
        # self.var2 = 0.0
        # self.var3 = 0.0
        # self.var4 = 0.0

        # Build the graph
        with self.graph.as_default():
            # Variables and placeholders
            self.get_vars_and_ph()

            # Operations
            self.input = self.get_input_current()
            self.potential = self.get_potential_op()

    def get_vars_and_ph(self):
        # The current membrane potential
        self.v = tf.Variable(self.v_rest, dtype=tf.float32, name='v')
        # The duration left in the resting period (0 most of the time except after a neuron spike)
        self.t_rest = tf.Variable(0.0, dtype=tf.float32, name='t_rest')
        # Input current
        # self.i_curr = tf.Variable(self.i_offest, dtype=tf.float32, name='i_curr')
        self.i_curr = tf.placeholder(dtype=tf.float32, name='i_curr')
        # The chosen time interval for the stimulation in ms
        self.dt = tf.placeholder(dtype=tf.float32, name='dt')

    def get_input_current(self):
        return self.i_curr

    def get_integrating_op(self):
        alpha = tf.add(tf.multiply(self.get_input_current(), self.r_mem), self.v_rest)

        tau_mem = tf.math.exp(tf.negative(tf.divide(self.dt, tf.multiply(self.r_mem, self.cm))))

        v_op = self.v.assign(tf.multiply(self.dt, tf.subtract(alpha, tf.multiply(tau_mem, tf.subtract(alpha, self.v)))))

        t_rest_op = self.t_rest.assign(0.0)

        # with tf.control_dependencies([t_rest_op]):
        return v_op, t_rest_op

    # function to be called if the neuron fires
    def get_firing_op(self):
        v_op = self.v.assign(self.v_rest)

        t_rest_op = self.t_rest.assign(self.tau_refract)

        # with tf.control_dependencies([t_rest_op]):
        return v_op, t_rest_op

    # function to be called if in refractory period
    def get_resting_op(self):
        # Membrane potential stays at u_rest
        v_op = self.v.assign(self.v_reset)
        # Refractory period is decreased by dt
        t_rest_op = self.t_rest.assign_sub(self.dt)

        # with tf.control_dependencies([t_rest_op]):
        return v_op, t_rest_op

    def get_potential_op(self):
        return tf.case(
            [
                (self.t_rest > 0.0, self.get_resting_op),
                (self.v > self.v_thresh, self.get_firing_op),
            ],
            default=self.get_integrating_op
        )

class SpiNN_LIF_curr_exp_syn(SpiNN_LIF_curr_exp):

    def __init__(self, weights, neuron_id, v_rest=-65.0, cm=1.0, tau_m=20.0, tau_refract=5.0,
                 v_thresh=-50.0, v_reset=-65.0, i_offest=0.0):
        self.n_syn = len(weights)
        self.neuron_id = neuron_id
        self.weights = weights

        super(SpiNN_LIF_curr_exp_syn, self).__init__(v_rest, cm, tau_m, tau_refract, v_thresh, v_reset, i_offest)

    def get_vars_and_ph(self):
        # Get parent grah variables and placeholders
        super(SpiNN_LIF_curr_exp_syn, self).get_vars_and_ph()

        # A placeholder indicating which synapse spiked in the last time step
        self.syn_has_spiked = tf.placeholder(shape=[self.n_syn], dtype=tf.bool)
        # current information
        self.old_current = tf.Variable(0.0, dtype=tf.float32)
        self.new_current = tf.Variable(0.0, dtype=tf.float32)
        # the last time there was an update
        self.old_dt = tf.Variable(0.0, tf.float32)

    def add_spikes_to_current(self):
        tf.case([self.syn_has_spiked > 0, self.add_weight(self.syn_has_spiked)])
        # for neuron in range(self.n_syn):
        #     tf.case([(self.syn_has_spiked[neuron] is True, self.add_spike(neuron))])
            # if self.syn_has_spiked[neuron]:
            #     self.add_spike(neuron)

    def add_spike(self, index):
        init = 0.9063462346100909
        self.new_current.assign_add(tf.multiply(init, self.weights[self.neuron_id][index]))

    def add_weight(self, weight):
        init = 0.9063462346100909
        self.new_current.assign_add(tf.multiply(init, self.weights[self.neuron_id][index]))

    def get_total_current(self):
        decay = 0.8187307530779818
        time_elapsed = tf.subtract(self.dt, self.old_dt)
        self.old_dt.assign(self.dt)
        new_ic = self.new_current.assign_add(tf.multiply(self.old_current, tf.multiply(time_elapsed, decay)))
        return new_ic

    def get_input_current(self):
        # Update our memory of spike times with the new spikes
        t_spikes_op = self.add_spikes_to_current()

        i_op = self.get_total_current()

        self.old_current.assign(self.new_current)
        self.new_current.assign(0.0)

        return tf.add(self.i_offest, i_op)

#create a network of SpiNNaker neurons
class SpiNN_network(SpiNN_LIF_curr_exp_syn):
    def __init__(self, weights, v_rest=-65.0, cm=1.0, tau_m=20.0, tau_refract=5.0, v_thresh=-50.0,
                 v_reset=-65.0, i_offest=0.0):

        for neuron_id in range(len(weights)):
            super(SpiNN_network, self).__init__(weights, neuron_id, v_rest, cm, tau_m, tau_refract,
                                                v_thresh, v_reset, np.random.random()*2)



# Simulation with square input currents

# Duration of the simulation in ms
T = 200
# Duration of each time step in ms
dt = 1
# Number of iterations = T/dt
steps = int(T / dt)
# Output variables
I = []
U = []

network_size = 5
weight_matrix = [[np.random.randint(5) for i in range(network_size)] for j in range(network_size)]
network = SpiNN_network(weight_matrix)

with tf.Session(graph=SpiNN_network.graph) as sess:
    sess.run(tf.global_variables_initializer())
    spike_history = [False for i in range(network_size)]
    for step in range(steps):
        t = step * dt
        feed = {network.dt: dt, network.i_curr: 0.0, network.syn_has_spiked: spike_history}
        [membrane_v, resting_period] = sess.run(network.potential, feed_dict=feed)
        I.append(resting_period)
        U.append(membrane_v)


neuron = SpiNN_LIF_curr_exp()

with tf.Session(graph=neuron.graph) as sess:

    sess.run(tf.global_variables_initializer())

    for step in range(steps):

        t = step * dt
        # Set input current in mA
        if t > 10 and t < 30:
            i_app = 0.5
        elif t > 50 and t < 100:
            i_app = 1.2
        elif t > 120 and t < 180:
            i_app = 1.5
        else:
            i_app = 0.0

        feed = {neuron.i_curr: i_app, neuron.dt: dt}

        # output = sess.run(neuron.potential, feed_dict=feed)
        # membrane_v = output[0]
        # resting_period = output[1]
        [membrane_v, resting_period] = sess.run(neuron.potential, feed_dict=feed)

        # print("v:", neuron.v.eval())
        # # tf.print(neuron.get_integrating_op(), [neuron.v], message="tfv")
        # # print(sess.run(neuron.var1), "did it?")
        # # tf.print(neuron.get_integrating_op(), neuron.v)
        # # tf.print(neuron.v, neuron.get_integrating_op())
        # # tf.print(neuron.get_integrating_op(), [neuron.var1], message="tfv1")
        # print("v1:", neuron.var1.eval())
        # print("v2:", neuron.var2.eval())
        # print("v3:", neuron.var3.eval())
        # print("v4:", neuron.var4.eval())

        I.append(i_app)
        U.append(membrane_v)



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

print("did spin stuff")