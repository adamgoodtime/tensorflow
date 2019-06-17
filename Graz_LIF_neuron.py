from __future__ import print_function

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class Graz_LIF_curr_exp(object):

    def __init__(self, v_rest=-65.0, cm=1.0, tau_m=20.0, tau_refract=5.0, v_thresh=-50.0, v_reset=-65.0, i_offest=0.0):
        self.alpha = 0.9

        self.v_rest = v_rest #* 10**-3
        self.cm = cm #* 10**-3
        self.tau_m = tau_m * 10**-3
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
        self.var1 = tf.Variable(0.0, dtype=tf.float32, name='v1')
        self.var2 = tf.Variable(0.0, dtype=tf.float32, name='v1')
        self.var3 = tf.Variable(0.0, dtype=tf.float32, name='v1')
        self.var4 = tf.Variable(0.0, dtype=tf.float32, name='v1')
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
        # tau_mem = tf.math.exp(tf.divide(self.dt, tf.multiply(self.r_mem, self.cm)))
        term1 = tf.add(self.v_rest, tf.multiply(self.alpha, tf.subtract(self.v, self.v_rest)))
        term2 = tf.multiply(tf.subtract(1.0, self.alpha), tf.multiply(self.r_mem, self.get_input_current()))
        # term3 = tf.multiply(self.dt, tf.multiply()
        v_op = self.v.assign(tf.add(term1, term2))

        t_rest_op = self.t_rest.assign(0.0)

        # v1 = self.var1.assign(alpha)
        # self.var2.assign(tau_mem)
        v3 = self.var3.assign(v_op)
        v4 = self.var4.assign(t_rest_op)

        # print("tau_m:", tau_mem.eval(session=self.sess), "alpha:", alpha.eval(session=self.sess), "v:", self.v.eval(session=self.sess), "v_op:", v_op.eval(session=self.sess))

        # with tf.control_dependencies([t_rest_op]):
        return v_op, t_rest_op

    def get_firing_op(self):
        v_op = self.v.assign(self.v_rest)

        t_rest_op = self.t_rest.assign(self.tau_refract)

        # with tf.control_dependencies([t_rest_op]):
        return v_op, t_rest_op


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

neuron = Graz_LIF_curr_exp()

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

        u = sess.run(neuron.potential, feed_dict=feed)


        print("v:", neuron.v.eval())
        # tf.print(neuron.get_integrating_op(), [neuron.v], message="tfv")
        # print(sess.run(neuron.var1), "did it?")
        # tf.print(neuron.get_integrating_op(), neuron.v)
        # tf.print(neuron.v, neuron.get_integrating_op())
        # tf.print(neuron.get_integrating_op(), [neuron.var1], message="tfv1")
        print("v1:", neuron.var1.eval())
        print("v2:", neuron.var2.eval())
        print("v3:", neuron.var3.eval())
        print("v4:", neuron.var4.eval())

        I.append(i_app)
        U.append(u[0])



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