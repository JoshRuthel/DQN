# Imports
import matplotlib.pyplot as plt
from environment import Environment


MTCD_N = 10  # Number of MTCD devices
N_FRAMES = 1000  # Number of frames to simulate


def Simulate():
    print("Starting simulation")
    optimal_environment = Environment(MTCD_N)
    random_environment = Environment(MTCD_N)
    transmit_environment = Environment(MTCD_N)
    local_environment = Environment(MTCD_N)
    optimal_rewards = []
    random_rewards = []
    transmit_rewards = []
    local_rewards = []

    for frame in range(N_FRAMES):
        optimal_frame_reward = optimal_environment.perform_next_frame(
            False, False, False
        )
        random_frame_reward = random_environment.perform_next_frame(True, False, False)
        transmit_frame_reward = transmit_environment.perform_next_frame(
            False, True, False
        )
        local_frame_reward = local_environment.perform_next_frame(False, False, True)
        optimal_rewards.append(optimal_frame_reward)
        random_rewards.append(random_frame_reward)
        local_rewards.append(local_frame_reward)
        transmit_rewards.append(transmit_frame_reward)
        print(f" Frame number {optimal_environment.get_frame_number()}")

    [optimal_local_queues,optimal_transmit_queues] = optimal_environment.get_queue_lengths()
    [random_local_queues,random_transmit_queues] = random_environment.get_queue_lengths()
    [local_local_queues, local_transmit_queues] = local_environment.get_queue_lengths()
    [transmit_local_queues,transmit_transmit_queues] = transmit_environment.get_queue_lengths()

    print( "Number of tasks dropped for optimal:", optimal_environment.get_tasks_dropped())
    print("Number of tasks dropped during random:", random_environment.get_tasks_dropped())
    print("Number of tasks dropped during transmit:",transmit_environment.get_tasks_dropped())
    print( "Number of tasks dropped during local:", local_environment.get_tasks_dropped())

    optimal_rewards_plot = []
    random_rewards_plot = []
    transmit_rewards_plot = []
    local_rewards_plot = []
    optimal_running_avg = 0
    random_running_avg = 0
    transmit_running_avg = 0
    local_running_avg = 0

    for i in range(N_FRAMES):
        optimal_running_avg += optimal_rewards[i]
        random_running_avg += random_rewards[i]
        transmit_running_avg += transmit_rewards[i]
        local_running_avg += local_rewards[i]
        if i % 50 == 0:
            new_avg = optimal_running_avg / 50
            optimal_rewards_plot.append(new_avg)
            new_avg = random_running_avg / 50
            random_rewards_plot.append(new_avg)
            new_avg = transmit_running_avg / 50
            transmit_rewards_plot.append(new_avg)
            new_avg = local_running_avg / 50
            local_rewards_plot.append(new_avg)
            optimal_running_avg = 0
            random_running_avg = 0
            transmit_running_avg = 0
            local_running_avg = 0

    time_plot = []
    for i in range(1, int((N_FRAMES / 50) + 1)):
        time_plot.append(i * 50)

    plt.figure()
    plt.plot(time_plot, optimal_rewards_plot, color="green", label="DQN")
    plt.plot(time_plot, random_rewards_plot, color="red", label="Random action")
    plt.plot(time_plot, transmit_rewards_plot, color="blue", label="Transmission only")
    plt.plot(time_plot, local_rewards_plot, color="orange", label="Local processing only")
    plt.xlabel("Frame Count")
    plt.ylabel("Average reward (bits/frame - constraints)")
    plt.title("Plot showing average frame rewards for different schemes for 5 MTCD devices")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure()
    plt.plot(range(1, N_FRAMES + 1),optimal_transmit_queues,color="blue",label="DQN Transmit queue",)
    plt.plot(range(1, N_FRAMES + 1),optimal_local_queues,color="red",label="DQN Local queue",)
    plt.plot(range(1, N_FRAMES + 1),random_transmit_queues,color="green",label="Random Transmit queue",)
    plt.plot(range(1, N_FRAMES + 1),random_local_queues,color="orange",label="Random Local queue",)
    plt.plot(range(1, N_FRAMES + 1),transmit_transmit_queues,color="black",label="Transmit only Transmit queue",)
    plt.plot(range(1, N_FRAMES + 1),transmit_local_queues,color="yellow",label="Transmit only Local queue",)
    plt.plot(range(1, N_FRAMES + 1),local_transmit_queues,color="brown",label="Local only Transmit queue",)
    plt.plot(range(1, N_FRAMES + 1),local_local_queues,color="purple",label="Local only Local queue",)
    plt.xlabel("Frame Count")
    plt.ylabel("Queue Capacity (%)")
    plt.title("Plot showing the average queue capacities throughput the simulation")
    plt.legend()
    plt.grid(True)
    plt.show()

    return


Simulate()
