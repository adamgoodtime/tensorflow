import numpy as np
import matplotlib.pyplot as plt

class Graz_LIF(object):

    def __init__(self, dt=1.0, v_rest=-65.0, cm=1.0, tau_m=20.0, tau_refract=5.0, v_thresh=-50.0, v_reset=-65.0, i_offset=0.0):
        self.alpha = dt / tau_m
        # passed in varaibles of the neuron
        self.dt = dt
        self.v_rest = v_rest #* 10**-3
        self.cm = cm #* 10**-3
        self.tau_m = tau_m * 10**-3
        self.r_mem = tau_m / cm
        self.tau_refract = tau_refract
        self.v_thresh = v_thresh #* 10**-3
        self.v_reset = v_reset #* 10**-3
        self.i_offset = i_offset
        # state variables
        self.v = self.v_rest
        self.t_rest = 0.0
        self.i = self.i_offset

    # calculates the current for all inputs
    def return_current(self):
        # sum up the spikes and shit
        total = 0.0
        return self.i_offset + total

    # activation function
    def H(self, x):
        return 1 / (1 + np.exp(-x))

    # operation to be performed when not spiking
    def integrating(self):
        current = self.return_current()
        scaled_v = (self.v - self.v_thresh) / self.v_thresh
        z = self.H(scaled_v) * (1 / self.dt)
        update = (self.alpha * (self.v - self.v_rest)) + ((1 - self.alpha) * self.r_mem * current)# - (self.dt * self.v_thresh * z)
        self.v += update

    # to be perfromed once threshold crossed
    def spiked(self):
        self.v = self.v_rest
        self.t_rest = self.tau_refract

    #refractory behaviour
    def refracting(self):
        self.t_rest -= 1.0

    # step the neuron
    def time_step(self):
        if self.v > self.v_thresh:
            self.spiked()
        elif self.t_rest > 0:
            self.refracting()
        else:
            self.integrating()

# Duration of the simulation in ms
T = 200
# Duration of each time step in ms
dt = 1
# Number of iterations = T/dt
steps = int(T / dt)
# Output variables
I = []
U = []

neuron = Graz_LIF(dt=dt, tau_refract=0)

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

    neuron.i_offset = i_app
    neuron.time_step()

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
