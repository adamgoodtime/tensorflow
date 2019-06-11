from __future__ import print_function

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

            v_rest = -65.0  # Resting membrane potential in mV.
            v_rest_stdev = 5
            cm = 1.0  # Capacity of the membrane in nF
            cm_stdev = 0.3
            tau_m = 20.0  # Membrane time constant in ms.
            tau_m_stdev = 5
            tau_refrac = 0.1  # Duration of refractory period in ms.
            tau_refrac_stdev = 0.03
            tau_syn_E = 5  # Rise time of the excitatory synaptic alpha function in ms.
            tau_syn_E_stdev = 1.6
            tau_syn_I = 5  # Rise time of the inhibitory synaptic alpha function in ms.
            tau_syn_I_stdev = 1.6
            e_rev_E = 0.0  # Reversal potential for excitatory input in mV
            e_rev_E_stdev = 0
            e_rev_I = -70.0  # Reversal potential for inhibitory input in mV
            e_rev_I_stdev = 3
            v_thresh = -50.0  # Spike threshold in mV.
            v_thresh_stdev = 5
            v_reset = -65.0  # Reset potential after a spike in mV.
            v_reset_stdev = 5
            i_offset = 0  # Offset current in nA
            i_offset_stdev = input_current_stdev
            v = -65.0  # 'v_starting'
            v_stdev = 5
        elif neuron_type == 'IF_curr_exp':
            v_rest = -65.0  # Resting membrane potential in mV.
            v_rest_stdev = 5
            cm = 1.0  # Capacity of the membrane in nF
            cm_stdev = 0.3
            tau_m = 20.0  # Membrane time constant in ms.
            tau_m_stdev = 5
            tau_refrac = 0.1  # Duration of refractory period in ms.
            tau_refrac_stdev = 0.03
            tau_syn_E = 5  # Rise time of the excitatory synaptic alpha function in ms.
            tau_syn_E_stdev = 1.6
            tau_syn_I = 5  # Rise time of the inhibitory synaptic alpha function in ms.
            tau_syn_I_stdev = 1.6
            v_thresh = -50.0  # Spike threshold in mV.
            v_thresh_stdev = 5
            v_reset = -65.0  # Reset potential after a spike in mV.
            v_reset_stdev = 5
            i_offset = 0  # Offset current in nA
            i_offset_stdev = input_current_stdev
            v = -65.0  # 'v_starting'
            v_stdev = 5


u_rest=-65.0, u_thresh=-50.0, tau_rest=4.0, r=1.0, tau=10.0, q=1.5,
                 tau_syn=10.0)

class LIFNeuron(object):

    def __init__(self, u_rest=0.0, u_thresh=1.0, tau_rest=4.0, r=1.0, tau=10.0):
        # Membrane resting potential in mV
        self.u_rest = u_rest
        # Membrane threshold potential in mV
        self.u_thresh = u_thresh
        # Duration of the resting period in ms
        self.tau_rest = tau_rest
        # Membrane resistance in Ohm
        self.r = r
        # Membrane time constant in ms
        self.tau = tau

        # Instantiate a graph for this neuron
        self.graph = tf.Graph()

        # Build the graph
        with self.graph.as_default():
            # Variables and placeholders
            self.get_vars_and_ph()

            # Operations
            self.input = self.get_input_op()
            self.potential = self.get_potential_op()
            # Note that input is a prerequisite of potential, so it will
            # always be evaluated when potential is

    # Variables and placeholders
    def get_vars_and_ph(self):
        # The current membrane potential
        self.u = tf.Variable(self.u_rest, dtype=tf.float32, name='u')
        # The duration left in the resting period (0 most of the time except after a neuron spike)
        self.t_rest = tf.Variable(0.0, dtype=tf.float32, name='t_rest')
        # Input current
        self.i_app = tf.placeholder(dtype=tf.float32, name='i_app')
        # The chosen time interval for the stimulation in ms
        self.dt = tf.placeholder(dtype=tf.float32, name='dt')

    # Evaluate input current
    def get_input_op(self):
        return self.i_app

    # Neuron behaviour during integration phase (below threshold)
    def get_integrating_op(self):
        # Get input current
        i_op = self.get_input_op()

        # Update membrane potential
        du_op = tf.divide(tf.subtract(tf.multiply(self.r, i_op), self.u), self.tau)
        u_op = self.u.assign_add(du_op * self.dt)
        # Refractory period is 0
        t_rest_op = self.t_rest.assign(0.0)

        with tf.control_dependencies([t_rest_op]):
            return u_op

    # Neuron behaviour during firing phase (above threshold)
    def get_firing_op(self):
        # Reset membrane potential
        u_op = self.u.assign(self.u_rest)
        # Refractory period starts now
        t_rest_op = self.t_rest.assign(self.tau_rest)

        with tf.control_dependencies([t_rest_op]):
            return u_op

    # Neuron behaviour during resting phase (t_rest > 0)
    def get_resting_op(self):
        # Membrane potential stays at u_rest
        u_op = self.u.assign(self.u_rest)
        # Refractory period is decreased by dt
        t_rest_op = self.t_rest.assign_sub(self.dt)

        with tf.control_dependencies([t_rest_op]):
            return u_op

    def get_potential_op(self):
        return tf.case(
            [
                (self.t_rest > 0.0, self.get_resting_op),
                (self.u > self.u_thresh, self.get_firing_op),
            ],
            default=self.get_integrating_op
        )


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

neuron = LIFNeuron()

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

        feed = {neuron.i_app: i_app, neuron.dt: dt}

        u = sess.run(neuron.potential, feed_dict=feed)

        I.append(i_app)
        U.append(u)



plt.rcParams["figure.figsize"] =(12,6)
# Draw the input current and the membrane potential
plt.figure()
plt.plot([i for i in I])
plt.title('Square input stimuli')
plt.ylabel('Input current (I)')
plt.xlabel('Time (msec)')
plt.figure()
plt.plot([u for u in U])
plt.axhline(y=1.0, color='r', linestyle='-')
plt.title('LIF response')
plt.ylabel('Membrane Potential (mV)')
plt.xlabel('Time (msec)')
plt.show()

# Simulation with random input currents

# Duration of the simulation in ms
T = 200
# Duration of each time step in ms
dt = 1
# Number of iterations = T/dt
steps = int(T / dt)
# Output variables
I = []
U = []

neuron = LIFNeuron()

with tf.Session(graph=neuron.graph) as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(steps):

        t = step * dt
        if t > 10 and t < 180:
            i_app = np.random.normal(1.5, 1.0)
        else:
            i_app = 0.0

        feed = {neuron.i_app: i_app, neuron.dt: dt}

        u = sess.run(neuron.potential, feed_dict=feed)

        I.append(i_app)
        U.append(u)



plt.rcParams["figure.figsize"] =(12,6)
# Draw the input current and the membrane potential
plt.figure()
plt.plot([i for i in I])
plt.title('Random input stimuli')
plt.ylabel('Input current (I)')
plt.xlabel('Time (msec)')
plt.figure()
plt.plot([u for u in U])
plt.axhline(y=1.0, color='r', linestyle='-')
plt.title('LIF response')
plt.ylabel('Membrane Potential (mV)')
plt.xlabel('Time (msec)')
plt.show()


# A new neuron model derived from the LIF neuron
# It takes synaptic spikes as input and remember them over a specified time period
class LIFSynapticNeuron(LIFNeuron):

    def __init__(self, n_syn, w, max_spikes=50, u_rest=0.0, u_thresh=1.0, tau_rest=4.0, r=1.0, tau=10.0, q=1.5,
                 tau_syn=10.0):
        # Number of synapses
        self.n_syn = n_syn
        # Maximum number of spikes we remember
        self.max_spikes = max_spikes
        # The neuron synaptic 'charge'
        self.q = q
        # The synaptic time constant (ms)
        self.tau_syn = tau_syn
        # The synaptic efficacy
        self.w = w

        super(LIFSynapticNeuron, self).__init__(u_rest, u_thresh, tau_rest, r, tau)

    # Update the parent graph variables and placeholders
    def get_vars_and_ph(self):
        # Get parent grah variables and placeholders
        super(LIFSynapticNeuron, self).get_vars_and_ph()

        # Add ours

        # The history of synaptic spike times for the neuron
        self.t_spikes = tf.Variable(tf.constant(-1.0, shape=[self.max_spikes, self.n_syn], dtype=tf.float32))
        # The last index used to insert spike times
        self.t_spikes_idx = tf.Variable(self.max_spikes - 1, dtype=tf.int32)
        # A placeholder indicating which synapse spiked in the last time step
        self.syn_has_spiked = tf.placeholder(shape=[self.n_syn], dtype=tf.bool)

    # Operation to update spike times
    def update_spike_times(self):
        # Increase the age of older spikes
        old_spikes_op = self.t_spikes.assign_add(tf.where(self.t_spikes >= 0,
                                                          tf.constant(1.0,
                                                                      shape=[self.max_spikes, self.n_syn]) * self.dt,
                                                          tf.zeros([self.max_spikes, self.n_syn])))

        # Increment last spike index (modulo max_spikes)
        new_idx_op = self.t_spikes_idx.assign(tf.mod(self.t_spikes_idx + 1, self.max_spikes))

        # Create a list of coordinates to insert the new spikes
        idx_op = tf.constant(1, shape=[self.n_syn], dtype=tf.int32) * new_idx_op
        coord_op = tf.stack([idx_op, tf.range(self.n_syn)], axis=1)

        # Create a vector of new spike times (non-spikes are assigned a negative time)
        new_spikes_op = tf.where(self.syn_has_spiked,
                                 tf.constant(0.0, shape=[self.n_syn]),
                                 tf.constant(-1.0, shape=[self.n_syn]))

        # Replace older spikes by new ones
        return tf.scatter_nd_update(old_spikes_op, coord_op, new_spikes_op)

    # Override parent get_input_op method
    def get_input_op(self):
        # Update our memory of spike times with the new spikes
        t_spikes_op = self.update_spike_times()

        # Evaluate synaptic input current for each spike on each synapse
        i_syn_op = tf.where(t_spikes_op >= 0,
                            self.q / self.tau_syn * tf.exp(tf.negative(t_spikes_op / self.tau_syn)),
                            t_spikes_op * 0.0)

        # Add each synaptic current to the input current
        i_op = tf.reduce_sum(self.w * i_syn_op)

        return tf.add(self.i_app, i_op)


# Simulation with synaptic input currents

# Duration of the simulation in ms
T = 200
# Duration of each time step in ms
dt = 1
# Number of iterations = T/dt
steps = int(T / dt)
# Number of synapses
n_syn = 25
# Spiking frequency in Hz
f = 20
# We need to keep track of input spikes over time
syn_has_spiked = np.full((steps, n_syn), False)
# We define the synaptic efficacy as a random vector
W = np.random.normal(1.0, 0.5, size=n_syn)
# Output variables
I = []
U = []

# Instantiate our synaptic LIF neuron, with a memory of 200 events
# Note that in practice, a much shorter period is required as the
# contribution of each synapse decreases very rapidly
neuron = LIFSynapticNeuron(n_syn=n_syn, w=W, max_spikes=200)

with tf.Session(graph=neuron.graph) as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(steps):

        t = step * dt

        if t > 10 and t < 180:
            r = np.random.uniform(0, 1, size=(n_syn))
            syn_has_spiked[step, :] = r < f * dt * 1e-3

        feed = {neuron.i_app: 0.0, neuron.syn_has_spiked: syn_has_spiked[step], neuron.dt: dt}
        i, u = sess.run([neuron.input, neuron.potential], feed_dict=feed)

        I.append(i)
        U.append(u)

plt.rcParams["figure.figsize"] =(12,6)
# Draw spikes
spikes = np.argwhere(syn_has_spiked)
t, s = spikes.T
plt.figure()
plt.axis([0, T, 0, n_syn])
plt.title('Synaptic spikes')
plt.ylabel('spikes')
plt.xlabel('Time (msec)')
plt.scatter(t, s)
# Draw the input current and the membrane potential
plt.figure()
plt.plot([i for i in I])
plt.title('Synaptic input')
plt.ylabel('Input current (I)')
plt.xlabel('Time (msec)')
plt.figure()
plt.plot([u for u in U])
plt.axhline(y=1.0, color='r', linestyle='-')
plt.title('LIF response')
plt.ylabel('Membrane Potential (mV)')
plt.xlabel('Time (msec)')
plt.show()