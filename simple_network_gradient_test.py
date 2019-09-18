import numpy as np
from copy import deepcopy

def sig(x):
    return 1 / (1 + np.exp(-x))

def drv_sig(x):
    value = sig(x)
    return value * (1 - value)

def forward_step(weight_matrix):
    global activation_values
    new_values = []
    # step forward
    for id_post, neuron_post in enumerate(neuron_list):
        input_value = 0.0
        for id_pre, neuron_pre in enumerate(neuron_list):
            input_value += weight_matrix[id_pre][id_post] * neuron_pre["activation"]
        neuron_post["activation"] = sig(input_value)
        new_values.append(neuron_post["activation"])
    activation_values.append(new_values)
    output_value = neuron_list[-1]["activation"]
    print "out:\t", output_value
    return output_value

def backward_step(weight_matrix):
    hidden_value = activation_values[-1][-2]
    dodw = hidden_value * drv_sig(weight_matrix[-2][-1] * hidden_value)
    print "gradient:\t", dodw
    input_value = activation_values[-2][-3]
    dodw2 = input_value * \
            weight_matrix[-2][-1] * drv_sig(weight_matrix[-3][-2] * input_value) * \
            drv_sig(weight_matrix[-2][-1] * sig(input_value * weight_matrix[-3][-2]))
    print "gradient2:\t", dodw2
    return dodw + dodw2


weight_matrix = [[0, 1, 0],
                 [0, 0, 1],
                 [0, 0, 0]]

target_output = 0.78

activation_values = []
neuron_list = []
for neuron in range(len(weight_matrix)):
    neuron_list.append({"activation": sig(0.0)})
    activation_values.append(neuron_list[neuron]["activation"])
activation_values = [activation_values]

output_value = forward_step(weight_matrix)
delta = 1e-6
gradient = backward_step(weight_matrix)
new_weight_matrix = deepcopy(weight_matrix)
new_weight_matrix[0][1] += delta
new_weight_matrix[1][2] += delta
delta_output_value = forward_step(new_weight_matrix)
finite_difference = (delta_output_value - output_value) / delta

print "fd:\t", finite_difference

print "BP error:", np.log10(np.abs(finite_difference - gradient))

print "done"