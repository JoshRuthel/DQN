import random
import numpy as np
import torch
import math
from mec import MEC

Q_LOCAL = 100 * pow(10, 6)  # Size of local queue (b-bits) - 10MB
Q_TRANS = 100 * pow(10, 6)  # Size of transmission queue (b-bits) - 10MB
Q_MEC = 1 * pow(10, 9)  # Size of MEC queue (b-bits) - 1GB 
Q_EN = 10 * pow(10, 6) # Max the energy queue can hold (pJ- picojoules) - 10000pJ
F_L = 1 * pow(10, 9)  # Local computation frequency (Hz-Hertz) - 1Ghz
F_MEC = 10 * pow(10, 9)  # MEC computation frequency (Hz_Hertz) - 10Ghz
CYCLES_BITS = 10  # Local cycles per bit (c/b-cycles per bit)
L_TIME_BIT = CYCLES_BITS / F_L # Local time to process one bit (s/b-seconds per bit)
M_TIME_BIT = CYCLES_BITS / F_MEC # MEC time to process one bit (s/b-seconds per bit)
B_MTCD = 10 * pow(10, 6)  # MTCD channel bandwith (Hz-Hertz) - 10Mhz
S_AVG = 1 * pow(10, 6)  # Mean task size of arrival task (b-bits) - 1MB
F_C = 2.4 * pow(10, 9) # The carrier frequency from the MEC (Hz-Hertz) - 2.4GHz
I_H = 0.9 # Conversion efficiency for the RF Harvesting unit (%-percentage) - 90%
D_MAX = 50  # Maximum distance of MTCD from MEC (m-meters) - 50m
P_BS = 500  # Base station power (W-watts) - 500W
U = 2.3  # Environment factor for energy harvesting
C = 3 * pow(10, 8)  # Speed of light (m/s-meters per second) - 3e10 m/s
K = pow(10, -30)  # Local computation energy consumption constant- 10e-30
P_T_MAX = 1  # Maximum MTCD transmit power (W-watts) - 1W
N_0 = pow(10, -12)  # The thermal noise (W/Hz-watts per hertz) - 10e-12

def get_receive_time(q_mec: int) -> int:
    return 1 if q_mec / Q_MEC <= 0.33 else 2 if q_mec / Q_MEC <= 0.66 else 3

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

def get_harvested_energy(t_h: int, d: int, h_mag_2: int) -> int:
    return round(
        (t_h * I_H * P_BS * pow(C / (4 * math.pi * d * F_C), U) * h_mag_2) * pow(10, 12)
    )

# The energy consumed in local computation - (nJ-picojoules)
def get_local_energy_consumed(t_L: int) -> int:
    return round(K * pow(F_L, 3) * t_L * pow(10, 12))

# The energy consumed in transmission - (nJ-picojoules)
def get_transmit_energy_consumed(p_t: int, t_O: int) -> int:
    return round(p_t * t_O * pow(10,12))

def get_transmit_channel_capacity(p_t: int, h_mag_2: int) -> int:
    return round(B_MTCD * math.log(1 + ((p_t * h_mag_2) / N_0), 2))

def get_consumed_energy(t_L: int, p_t: int, t_O: int) -> int:
    return get_local_energy_consumed(t_L) + get_transmit_energy_consumed(p_t, t_O)

def get_reward(
    current_state: torch.Tensor,
    action: torch.Tensor,
    bits_processed: int,
    transmit_power: float,
) -> torch.Tensor:
    energy_availible = current_state[4].item() * Q_EN
    energy_consumed = get_consumed_energy(
        round(action[2].item()) * pow(10, -3),
        transmit_power,
        round(action[1].item()) * pow(10, -3),
    )
    g_m = max(0, (energy_consumed - energy_availible)) + max(
        0, (transmit_power - P_T_MAX)
    )  # Constraint function

    reward = torch.tensor(
        bits_processed - g_m
    )
    return reward

class MTCD:
    def __init__(self, device_id: int, comp_freq: int, mec: MEC):
        self.device_id = device_id
        self.comp_freq = comp_freq
        # Starting state for the local queue (bits)
        self.local_queue = Q_LOCAL / 10
        self.transmit_queue = (
            Q_TRANS / 10
        )  # Starting state for the transmission queue (bits)
        self.energy_queue = Q_EN / 4  # Start with a quarter (pJ)
        self.awaiting_task_size = 0
        self.distance = random.randint(1, D_MAX)  # Distance from MEC
        self.tasks_dropped = 0
        self.mec = mec
        self.bits_processed = 0
        self.receive_time = 0  # Current receive time from MEC

    def get_optimal_action(
        self, p: float, current_state: torch.Tensor, epsilon: float
    ) -> torch.Tensor:
        q_network = self.mec.get_q_network(self.device_id)
        mec_queue = current_state[3].item() * Q_MEC
        receive_time = get_receive_time(mec_queue)
        self.receive_time = receive_time
        transmit_time_max = 10 - receive_time
        action_space = get_action_space(transmit_time_max, current_state)

        if p < epsilon:
            random_action = action_space[random.randint(0, len(action_space) - 1)]
            return random_action
        else:
            input_vectors = torch.stack(
                [torch.cat((action, current_state), dim=0) for action in action_space]
            )
            q_values = q_network(input_vectors)
            q_max, index = torch.max(q_values, dim=0)
            action = action_space[index.item()]
            return action

    def get_local_action(
        self, p: float, current_state: torch.Tensor, epsilon: float
    ) -> torch.Tensor:
        q_network = self.mec.get_q_network(self.device_id)
        mec_queue = current_state[3].item() * Q_MEC
        receive_time = get_receive_time(mec_queue)
        self.receive_time = receive_time

        action_list = []
        task_size = current_state[0].item() * S_AVG
        local_queue = current_state[1].item() * Q_LOCAL

        if task_size + local_queue > Q_LOCAL:  # Can only drop the task
            for local_time in range(11):
                action = torch.tensor([0, 0, 10 - local_time], dtype=torch.float32)
                action_list.append(action)

        else:  # Can only add to local queue
            for local_time in range(11):
                action = torch.tensor([1, 0, 10 - local_time], dtype=torch.float32)
                action_list.append(action)

        action_space = torch.stack(action_list)

        input_vectors = torch.stack(
            [torch.cat((action, current_state), dim=0) for action in action_space]
        )
        q_values = q_network(input_vectors)
        q_max, index = torch.max(q_values, dim=0)
        action = action_space[index.item()]
        return action

    def get_transmit_action(
        self, p: float, current_state: torch.Tensor, epsilon: float
    ) -> torch.Tensor:
        q_network = self.mec.get_q_network(self.device_id)
        mec_queue = current_state[3].item() * Q_MEC
        receive_time = get_receive_time(mec_queue)
        self.receive_time = receive_time
        transmit_time_max = 10 - receive_time

        action_list = []
        task_size = current_state[0].item() * S_AVG
        transmit_queue = current_state[2].item() * Q_TRANS

        if task_size + transmit_queue > Q_TRANS:  # Can only drop the task
            for transmit_time in range(transmit_time_max + 1):
                action = torch.tensor(
                    [0, transmit_time_max - transmit_time, 0], dtype=torch.float32
                )
                action_list.append(action)

        else:
            for transmit_time in range(transmit_time_max + 1):
                action = torch.tensor(
                    [2, transmit_time_max - transmit_time, 0], dtype=torch.float32
                )
                action_list.append(action)

        action_space = torch.stack(action_list)

        input_vectors = torch.stack(
            [torch.cat((action, current_state), dim=0) for action in action_space]
        )
        q_values = q_network(input_vectors)
        q_max, index = torch.max(q_values, dim=0)
        action = action_space[index.item()]
        return action

    def get_random_action(
        self, p: float, current_state: torch.Tensor, epsilon: float
    ) -> torch.Tensor:
        mec_queue = current_state[3].item() * Q_MEC
        receive_time = get_receive_time(mec_queue)
        self.receive_time = receive_time
        transmit_time_max = 10 - receive_time
        action_space = get_action_space(transmit_time_max, current_state)

        random_action = action_space[random.randint(0, len(action_space) - 1)]
        return random_action

    def process_local_bits(self, time: float):
        if self.local_queue > round(time / L_TIME_BIT):
            self.bits_processed += round(time / L_TIME_BIT)
            self.local_queue -= round(time / L_TIME_BIT)
            self.energy_queue -= get_local_energy_consumed(time)
        elif (self.local_queue <= round(time / L_TIME_BIT)) and self.local_queue > 0:
            self.bits_processed += self.local_queue
            self.local_queue = 0
            self.energy_queue -= get_local_energy_consumed(
                time
            )  # Assume the full energy is lost for simplicicity

    def process_transmit_bits(
        self, time: float, channel_capacity: float, transmit_power: float
    ):
        if self.transmit_queue > round(channel_capacity * time):
            self.bits_processed += round(channel_capacity * time)
            self.transmit_queue -= round(channel_capacity * time)
            self.energy_queue -= get_transmit_energy_consumed(transmit_power, time)
            self.mec.add_bits_to_queue(round(channel_capacity * time))
        elif (
            self.transmit_queue <= round(channel_capacity * time)
        ) and self.transmit_queue > 0:
            self.bits_processed += self.transmit_queue
            self.transmit_queue = 0
            self.energy_queue -= get_transmit_energy_consumed(
                transmit_power, time
            )  # Assume the full energy lost for simplicity
            self.mec.add_bits_to_queue(self.transmit_queue)

    def perform_action(
        self, action: torch.Tensor, h_mag_2: float, current_state: torch.Tensor
    ) -> torch.Tensor:
        self.bits_processed = 0
        local_time_passed = 0
        transmit_time_passed = 0

        # Determine frame times
        choice = round(action[0].item())  # Local/Offload/Drop
        transmit_time = round(action[1].item())
        local_time = round(action[2].item())
        receive_time = (
            self.receive_time if transmit_time != 0 else 0
        )  # If the MTCD doesn't transmit it won't receive
        harvest_time = 10 - max(transmit_time + receive_time, local_time)

        # Determine energy values
        availible_energy = round(current_state[4].item() * Q_EN)
        harvested_energy = get_harvested_energy(
            harvest_time * pow(10, -3), self.distance, h_mag_2
        )
        local_energy = get_local_energy_consumed(local_time * pow(10, -3))
        transmit_energy = max(0, availible_energy - local_energy)

        # Harvest energy
        if self.energy_queue + harvested_energy <= Q_EN:
            self.energy_queue += harvested_energy
        else:
            self.energy_queue = Q_EN

        self.mec.generate_random_queue_bits()  # Generate bits at MEC
        self.mec.process_queue_bits(harvest_time * pow(10, -3))  # Process bits at MEC

        # Determine channel capacity
        if transmit_time > 0:
            transmit_power = (transmit_energy * pow(10, -12)) / (
                transmit_time * pow(10, -3)
            )  # Watts
            channel_capacity = get_transmit_channel_capacity(
                transmit_power, h_mag_2
            )  # Bits/second
        else:
            transmit_power = 0
            channel_capacity = 0  # Doesn't transmit

        # Place the task
        if choice == 0:
            self.tasks_dropped += 1
        elif choice == 1:
            self.local_queue += self.awaiting_task_size
        elif choice == 2:
            self.transmit_queue += self.awaiting_task_size

        self.awaiting_task_size = 0  # Task has been handled

        for time in range(10 * (10 - harvest_time)):  # Executes every 0.1 milisecond
            if (get_local_energy_consumed(0.1 * pow(10, -3)) <= self.energy_queue) and (
                local_time_passed < local_time
            ):
                self.process_local_bits(0.1 * pow(10, -3))
                local_time_passed += 0.1
            if (
                get_transmit_energy_consumed(transmit_power, 0.1 * pow(10, -3))
                <= self.energy_queue
            ) and (transmit_time_passed < transmit_time):
                self.process_transmit_bits(
                    0.1 * pow(10, -3), channel_capacity, transmit_power
                )
                transmit_time_passed += 0.1
            if time % 10 == 0:  # Every millisecond
                # Adds bits to the queue from other MTCD devices
                self.mec.generate_random_queue_bits()
                self.mec.process_queue_bits(
                    1 * pow(10, -3)
                )  # Processes bits for the milisecond

        return get_reward(
            current_state,
            action,
            self.bits_processed,
            transmit_power,
        )

    def get_current_state(self, h_mag_2: float) -> torch.Tensor:
        current_state = torch.tensor(
            [
                self.awaiting_task_size / S_AVG,
                self.local_queue / Q_LOCAL,
                self.transmit_queue / Q_TRANS,
                self.mec.get_queue_length() / Q_MEC,
                self.energy_queue / Q_EN,
                h_mag_2,
            ],
            dtype=torch.float32,
        )
        return current_state

    def generate_task(self):
        self.awaiting_task_size = np.random.poisson(S_AVG)

    def get_device_id(self) -> int:
        return self.device_id

    def get_tasks_dropped(self) -> int:
        return self.tasks_dropped

    def get_queue_lengths(self):
        return {
            "local": self.local_queue / Q_LOCAL * 100,
            "trans": self.transmit_queue / Q_TRANS * 100,
        }
