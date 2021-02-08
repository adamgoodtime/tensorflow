from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import tensorflow as tf
import numpy as np

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts

tf.compat.v1.enable_v2_behavior()


class xor_env(py_environment.PyEnvironment):

    def __init__(self, exposure_time=10):
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=4, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(1,), dtype=np.int32, minimum=0, maximum=1, name='observation')
        self._state = 0
        self._possible_states = [[0, 0], [0, 1], [1, 0], [1, 1]]
        self._correct_output = [0, 1, 1, 0]
        self._episode_ended = False
        self._dt = 1
        self._exposure_time = exposure_time
        self._current_time = 0

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._current_time = 0
        self._state = 0
        self._episode_ended = False
        return ts.restart(np.array([self._state], dtype=np.int32))

    def _step(self, action):
        if self._current_time >= self._exposure_time:
            self._current_time = 0
            self._state += 1

        if self._state >= len(self._correct_output):
            return self._reset()

        print("state {} - time {}".format(self._state, self._current_time))
        # Make sure episodes don't go on forever.
        if action == self._correct_output[self._state]:
            reward = 1
        else:
            reward = 0
            # raise ValueError('`action` should be 0 or 1.')
        self._current_time += self._dt

        if self._episode_ended:
            return ts.termination(np.array([self._state], dtype=np.int32), reward)
        else:
            return ts.transition(
                np.array([self._state], dtype=np.int32), reward=reward, discount=1.0)


# environment = xor_env()
# utils.validate_py_environment(environment, episodes=5)

out_zero = np.array(0, dtype=np.int32)
out_one = np.array(1, dtype=np.int32)

environment = xor_env()
time_step = environment.reset()
print(time_step)
cumulative_reward = time_step.reward

for _ in range(50):
    if np.random.random() > 0.5:
        time_step = environment.step(out_zero)
    else:
        time_step = environment.step(out_one)
    print(time_step)
    cumulative_reward += time_step.reward

# time_step = environment.step(end_round_action)
# print(time_step)
cumulative_reward += time_step.reward
print('Final Reward = ', cumulative_reward)
