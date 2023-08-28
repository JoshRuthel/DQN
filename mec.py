import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from qnetwork import QNetwork
from experience import Experience

Q_LOCAL = 100 * pow(10, 6)  # Size of local queue (b-bits) - 10MB
Q_TRANS = 100 * pow(10, 6)  # Size of transmission queue (b-bits) - 10MB
Q_MEC = 1 * pow(10, 9)  # Size of MEC queue (b-bits) - 1GB
F_MEC = 10 * pow(10, 9)  # MEC computation frequency (Hz_Hertz) - 10Ghz
CYCLES_BITS = 10  # Local cycles per bit (c/b-cycles per bit)
M_TIME_BIT = CYCLES_BITS / F_MEC # MEC time to process one bit (s/b-seconds per bit)
MTCD_N = 10  # Number of MTCD devices
S_AVG = 1 * pow(10, 6)  # Mean task size of arrival task (b-bits) - 1MB
L_STATE = 6  # Length of the state vector
L_ACTION = 3  # Length of the action vector
L_Q = 1  # Length of the Q-Value
Z = 0.6  # RL discount rate
HIDDEN_UNITS = 64  # Neural Network units in hidden layers

def get_target_value(
    reward: torch.Tensor, next_state: torch.Tensor, q_network: QNetwork, learn_rate: int
) -> torch.Tensor:
    q_mec = next_state[3].item() * Q_MEC
    receive_time = get_receive_time(q_mec)
    transmit_time_max = 10 - receive_time
    action_space = get_action_space(transmit_time_max, next_state)
    input_vectors = torch.stack(
        [torch.cat((action, next_state), dim=0) for action in action_space]
    )
    q_values = q_network(input_vectors)
    max_q, _ = torch.max(q_values, dim=0)
    target_value = reward + (learn_rate * max_q)
    return target_value

def get_action_space(
    transmit_time_max: int, current_state: torch.Tensor
) -> torch.Tensor:
    action_list = []
    task_size = current_state[0].item() * S_AVG
    local_queue = current_state[1].item() * Q_LOCAL
    transmit_queue = current_state[2].item() * Q_TRANS

    if (task_size + local_queue > Q_LOCAL) and (
        task_size + transmit_queue > Q_TRANS
    ):  # Can only drop the task
        for transmit_time in range(transmit_time_max + 1):
            for local_time in range(11):
                action = torch.tensor(
                    [0, transmit_time_max - transmit_time, 10 - local_time],
                    dtype=torch.float32,
                )
                action_list.append(action)

    elif (task_size + local_queue > Q_LOCAL) and (
        task_size + transmit_queue <= Q_TRANS
    ):  # Can only add to transmit queue
        for transmit_time in range(transmit_time_max + 1):
            for local_time in range(11):
                action = torch.tensor(
                    [2, transmit_time_max - transmit_time, 10 - local_time],
                    dtype=torch.float32,
                )
                action_list.append(action)

    elif (task_size + transmit_queue > Q_TRANS) and (
        task_size + local_queue <= Q_LOCAL
    ):  # Can only add to local queue
        for transmit_time in range(transmit_time_max + 1):
            for local_time in range(11):
                action = torch.tensor(
                    [1, transmit_time_max - transmit_time, 10 - local_time],
                    dtype=torch.float32,
                )
                action_list.append(action)

    elif (task_size + transmit_queue <= Q_TRANS) and (
        task_size + local_queue <= Q_LOCAL
    ):  # Can choose either queue
        for n in range(1, 3):
            for transmit_time in range(transmit_time_max + 1):
                for local_time in range(11):
                    action = torch.tensor(
                        [n, transmit_time_max - transmit_time, 10 - local_time],
                        dtype=torch.float32,
                    )
                    action_list.append(action)

    return torch.stack(action_list)

def get_receive_time(q_mec: int) -> int:
    return 1 if q_mec / Q_MEC <= 0.33 else 2 if q_mec / Q_MEC <= 0.66 else 3


class MEC:
    def __init__(
        self,
        mtcd_count: int,
        input_dim: int,
        hidden_units: int,
        output_dim: int,
        learning_rate: float,
        batch_size: int,
    ):
        self.mtcd_count = mtcd_count  # Number of mtcds
        self.network_params = {
            i + 1: QNetwork(input_dim, hidden_units, output_dim)
            for i in range(mtcd_count)
        }  # Dictionary of QNetworks for each MTCD
        self.previous_network_params = {
            i + 1: self.network_params[i + 1].get_params() for i in range(mtcd_count)
        }
        self.optimizers = {
            i + 1: optim.Adam(self.network_params[i + 1].parameters(), lr=learning_rate)
            for i in range(mtcd_count)
        }  # Dictionary of optimizers for each MTCD
        self.experiences = {
            i + 1: [] for i in range(mtcd_count)
        }  # Dictionary storing the MTCD experiences
        self.batch_size = batch_size  # Batch size for training of networks
        # computation queue length [Initially starts half full]
        self.queue = Q_MEC / 2
        self.frame_number = 1

    def add_experience(
        self, mtcd_id: int, experience: Experience
    ):  # Store experience for mtcd
        self.experiences[mtcd_id].append(experience)

    def train_network(self, mtcd_id: int):  # Update mtcd QNetwork network parameters
        q_network = self.network_params[mtcd_id]
        q_target_network = QNetwork(L_STATE + L_ACTION, HIDDEN_UNITS, L_Q)
        q_target_network.set_params(self.previous_network_params[mtcd_id])

        if self.frame_number % 10 == 0:
            self.previous_network_params[mtcd_id] = q_network.get_params()
        experiences = self.experiences[mtcd_id]  # [{}]

        if len(self.experiences[mtcd_id]) <= self.batch_size:
            batch_indices = range(
                len(self.experiences[mtcd_id])
            )  # Train on all experiences
        else:
            batch_indices = np.random.choice(
                len(self.experiences[mtcd_id]), self.batch_size, replace=False
            )  # Selects a set of random indices of batch_size

        batch_experiences = [
            experiences[i] for i in batch_indices
        ]  # Training experiences [{}]
        batch_input_vectors = torch.stack(
            [
                torch.cat((experience.action, experience.current_state), dim=0)
                for experience in batch_experiences
            ]
        )  # Inputs for Q-values

        target_q_values = torch.stack(
            [
                get_target_value(
                    experience.reward, experience.next_state, q_target_network, Z
                )
                for experience in batch_experiences
            ]
        )
        target_q_values = target_q_values * pow(
            10, -3
        )  # Scale them down for the loss function
        q_values = q_network(batch_input_vectors)

        huber_loss = nn.SmoothL1Loss()
        self.optimizers[mtcd_id].zero_grad()  # Clear QNetwork gradients
        loss = huber_loss(q_values, target_q_values)
        loss.backward()  # Back propogate and calculate gradients
        self.optimizers[mtcd_id].step()  # Update network parameters
        self.frame_number += 1

    def get_q_network(self, mtcd_id: int) -> QNetwork:  # Return MTCD QNetwork
        return self.network_params[mtcd_id]

    def add_bits_to_queue(self, bits: int):  # Add transmitted bits to queue
        self.queue += bits

    # Simulates bits transmitted from other MTCDs
    def generate_random_queue_bits(self):
        if self.queue + random.randint(0, MTCD_N) * 100000 <= Q_MEC:
            self.queue += random.randint(0, MTCD_N) * 100000
        else:
            self.queue = Q_MEC

    def process_queue_bits(self, t_passed: int):  # Processes bits from queue
        if self.queue > t_passed / M_TIME_BIT:
            self.queue -= t_passed / M_TIME_BIT
        else:
            self.queue = 0

    def get_queue_length(self) -> int:
        return self.queue
