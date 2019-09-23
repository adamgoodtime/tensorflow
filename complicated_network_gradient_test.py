import numpy as np
from copy import deepcopy

def sig(x):
    return 1 / (1 + np.exp(-x))

def drv_sig(x):
    value = sig(x)
    return value * (1 - value)

def qwk_drv(x):
    return x * (1 - x)

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
    forward_weight = weight_matrix[-2][-1]
    backward_weight = weight_matrix[-3][-2]
    dodw2 = input_value * \
            forward_weight * drv_sig(backward_weight * input_value) * \
            drv_sig(forward_weight * sig(input_value * backward_weight))
    print "gradient2:\t", dodw2
    print "total gradient:", dodw + dodw2
    return dodw + dodw2

def calc_gradients(weight_matrix):
    # calc neuron gradients first
    output = True
    for id_1, neuron_1 in reversed(list(enumerate(neuron_list))):
        if output:
            output = False
            neuron_1["gradient"] = qwk_drv(activation_values[-1][id_1])
        else:
            total_gradient = 0.0
            for id_2, neuron_2 in reversed(list(enumerate(neuron_list))):
                total_gradient += weight_matrix[id_1][id_2] * neuron_2["gradient"] * qwk_drv(activation_values[-1][id_1])
            neuron_1["gradient"] = total_gradient
    # use neuron gradients to calc synapse gradients
    weight_update = np.zeros([number_of_neurons, number_of_neurons])
    for i in range(number_of_neurons):
        for j in range(number_of_neurons):
            if weight_matrix[i][j]:
                weight_update[i][j] = neuron_list[j]["gradient"] * activation_values[-1][i]
    return weight_update

weight_matrix = [[0, 0, 0.25, 0, 0],
                 [0, 0, 0.5, 0, 0],
                 [0, 0, 0, 0.2, 0],
                 [0, 0, 0, 0, -0.8],
                 [0, 0, 0, 0, 0]]
# weight_matrix = [[0, 0, 0.25, 0],
#                  [0, 0, 0.5, 0],
#                  [0, 0, 0, 0.2],
#                  [0, 0, 0, 0]]
# weight_matrix = [[0, 0.25, 0],
#                  [0, 0, 0.5],
#                  [0, 0, 0]]

number_of_neurons = len(weight_matrix)

target_output = 0.78

activation_values = []
neuron_list = []
for neuron in range(number_of_neurons):
    neuron_list.append({"activation": sig(0.0), "gradient": 0.0})
    activation_values.append(neuron_list[neuron]["activation"])
activation_values = [activation_values]

output_value = forward_step(weight_matrix)
delta = 1e-6
gradients = calc_gradients(weight_matrix)
print "gradients:\n", gradients
new_weight_matrix = deepcopy(weight_matrix)
new_weight_matrix[0][2] += delta
new_weight_matrix[1][2] += delta
new_weight_matrix[2][3] += delta
new_weight_matrix[3][4] += delta
# new_weight_matrix[0][1] += delta
# new_weight_matrix[1][2] += delta
delta_output_value = forward_step(new_weight_matrix)
finite_difference = (delta_output_value - output_value) / delta

print "fd:\t", finite_difference

print "BP error:", np.log10(np.abs(finite_difference - sum(sum(gradients))))

print "done"