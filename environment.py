import numpy as np
import torch
import math

from mec import MEC
from mtcd import MTCD
from experience import Experience


F_L = 1 * pow(10, 9)  # Local computation frequency (Hz-Hertz) - 1Ghz
MTCD_N = 10  # Number of MTCD devices
L_STATE = 6  # Length of the state vector
L_ACTION = 3  # Length of the action vector
L_Q = 1  # Length of the Q-Value
BATCH_S = 100  # Size of batch for random sampling training
RL_LEARN_RATE = 0.005  # Learning rate for RL Adam optimizer
HIDDEN_UNITS = 64  # Neural Network units in hidden layers


class Environment:
    def __init__(self, mtcd_count: int):
        self.mec = MEC(
            mtcd_count,
            input_dim=L_STATE + L_ACTION,
            hidden_units=HIDDEN_UNITS,
            output_dim=L_Q,
            learning_rate=RL_LEARN_RATE,
            batch_size=BATCH_S,
        )
        self.mtcds = [MTCD(i + 1, F_L, self.mec) for i in range(mtcd_count)]
        self.frame_number = 1
        self.average_rewards = []
        self.trans_queues = []  # Keeps track of queue lengths
        self.local_queues = []  # Keeps track of queue lengths
        self.average_rewards = []

    def perform_next_frame(self, random: bool, trans: bool, local: bool) -> float:
        attenuation = pow(
            np.clip(np.random.exponential(scale=1 / 0.8), 0, 1), 2
        )  # Random h with mean 0.8
        prob = torch.randint(low=0, high=100, size=(1,)).item() / 100
        reward_total = 0
        epsilon = 1 / math.sqrt(1+self.frame_number)
        transmit_queue = 0
        local_queue = 0

        for mtcd in self.mtcds:
            mtcd.generate_task()
            current_state = mtcd.get_current_state(attenuation)

            if random:
                action = mtcd.get_random_action(prob, current_state, epsilon)
            elif trans:
                action = mtcd.get_transmit_action(prob, current_state, epsilon)
            elif local:
                action = mtcd.get_local_action(prob, current_state, epsilon)
            else:
                action = mtcd.get_optimal_action(prob, current_state, epsilon)

            reward = mtcd.perform_action(action, attenuation, current_state)
            reward_total += reward.item()
            next_state = mtcd.get_current_state(attenuation)
            experience = Experience(current_state, action, reward, next_state)
            self.mec.add_experience(mtcd.get_device_id(), experience)
            transmit_queue += mtcd.get_queue_lengths()["trans"]
            local_queue += mtcd.get_queue_lengths()["local"]
            self.mec.train_network(mtcd.get_device_id())

        self.frame_number += 1
        self.average_rewards.append(reward_total / len(self.mtcds))
        self.trans_queues.append(transmit_queue / MTCD_N)
        self.local_queues.append(local_queue / MTCD_N)
        return reward_total / MTCD_N

    def get_frame_number(self) -> int:
        return self.frame_number

    def get_tasks_dropped(self) -> float:
        tasks_dropped = 0
        for mtcd in self.mtcds:
            tasks_dropped += mtcd.get_tasks_dropped()
        return tasks_dropped / MTCD_N

    def get_average_bit_rate(self) -> float:
        return np.mean(self.average_rewards)

    def get_queue_lengths(self):
        return [self.local_queues, self.trans_queues]

    def get_average_rewards(self):
        return self.average_rewards
