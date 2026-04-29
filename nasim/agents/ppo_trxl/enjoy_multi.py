from dataclasses import dataclass
import numpy as np
import gymnasium as gym
import torch
import tyro
from ppo_trxl import Agent, make_env

import random

@dataclass
class Args:
    hub: bool = False
    """whether to load the model from the huggingface hub or from the local disk"""
    name: str = ""
    """path to the model file"""
    num_envs: int = 10  
    """number of environments to run in parallel"""
    num_episodes: int = 10
    """number of episodes to evaluate per environment"""
    min_num_hosts: int = 5
    """Minimum number of hosts in NASim scenarios"""
    max_num_hosts: int = 8
    """Maximum number of hosts in NASim scenarios"""
    render_first: bool = False
    """whether to render the first environment"""
    capture_video: bool = False
    """whether to capture videos of agent performance"""
    run_name: str = "enjoy_evaluation"
    """name for this evaluation run (used for video capture)"""
    seed: int = 4444
    """random seed"""
    env_seed: int = 2
    """seed for the environment"""
    visualize: bool = False
    """whether to visualize the action sequence"""
    save_trajectory: bool = False
    """whether to save the action success/reward trajectory"""
    traj_folder: str = "trajectories"
    """folder to save the action success/reward trajectory"""


def binary_array_to_int(arr):
    """We use this functino to obtain a unique integer representation of
    the starting state of the environment. We want to 
    """
    # Ensure array is 1D
    arr = arr.flatten()
    # Convert to integer
    return np.packbits(arr).dot(2**np.arange(len(np.packbits(arr)))[::-1])


def trajectory_visualization(network_size: int, trajectory: list[tuple]):
    """The network size is used to correctly map the chosen actions.
    The trajectories we obtain are a list of tuples. Each tuple is composed
    of action (int), action_success (bool), reward
    """
    import matplotlib.pyplot as plt

    env = gym.make('StochPO-v0')

    # Define action types and their colors
    action_types = {
        'service_scan': 'slateblue',
        'os_scan': 'skyblue',
        'subnet_scan': 'orange',
        'process_scan': 'seagreen',
        'e_srv_0_os_0': 'firebrick',
        'e_srv_1_os_0': 'firebrick',
        'e_srv_0_os_1': 'firebrick',
        'e_srv_1_os_1': 'firebrick',
        'pe_proc_0_os_0': 'violet',
        'pe_proc_1_os_0': 'violet',
        'pe_proc_0_os_1': 'violet',
        'pe_proc_1_os_1': 'violet',
        'no_op': 'gray'
    }

    # Translate trjactory to expected format
    # 1. Extract the data into separate lists
    actions = [item[0] for item in trajectory]
    action_success = [item[1] for item in trajectory]
    rewards = [item[2] for item in trajectory]

    """
    Possible actions in the environment:
        service_scan
        os_scan
        subnet_scan
        process_scan
        e_srv_0_os_0
        e_srv_1_os_0
        e_srv_0_os_1
        e_srv_1_os_1
        pe_proc_0_os_0
        pe_proc_1_os_0
        pe_proc_0_os_1
        pe_proc_1_os_1
    Total: 12
    """
    num_actions = 12
    max_action_idx = network_size * num_actions

    # 2. Map actions indices to names
    action_sequence = []
    for i in range(len(actions)):
        # We have 12 actions per host, if the action is > the total allowed in the
        # environment, we need to map to NoOP.
        if actions[i] > max_action_idx:
            a = 'no_op'
        else:
            a = env.action_space.get_action(actions[i]).name
        
        action_sequence.append((a, i, action_success[i])) # Insert the tuple into the list

    # Example data: for each algorithm, list of (action_type, start_step, success)
    # Where success is True/False
    algorithms = ["PPO-TrXL"]
    action_sequences = [
        # For each algorithm, list of (action_type, start_step, duration)
        # PPO-LSTM (brute force approach)
        #[
        #    ('service_scan', 0, True),
        #    ('exploit', 1, False),
        #    ('exploit', 2, False),
        #    ('exploit', 3, True),  # Gained user access
        #    ('process_scan', 4, True),
        #    ('priv_escalation', 5, False),
        #    ('priv_escalation', 6, False),
        #    ('priv_escalation', 7, True),  # Gained root access
        #],
        ## PPO-FrameStack (intermediate policy)
        #[
        #    ('service_scan', 0, True),
        #    ('exploit', 1, False),
        #    ('exploit', 2, True),  # Gained user access
        #    ('process_scan', 3, True),
        #    ('priv_escalation', 4, True),  # Gained root access
        #    ('os_scan', 5, True),
        #],
        # PPO-TrXL (intermediate policy)
        action_sequence,
    ]

    # Create the figure
    fig, ax = plt.subplots(figsize=(12, 4))

    # For each algorithm
    for i, (algo, actions) in enumerate(zip(algorithms, action_sequences)):
        # For each action in the sequence
        for j, (action_type, start_step, success) in enumerate(actions):
            # Base color from action type
            color = action_types[action_type]
            # Modify color based on success
            edge_color = 'black' if success else 'white'
            style = '-' if success else '--'
            
            # Plot the action as a horizontal bar
            ax.barh(i, width=1, height=0.3, left=start_step, color=color, 
                    edgecolor=edge_color, linestyle=style, alpha=0.7)
            
            # Add text label for the action
            ax.text(start_step + 0.5, i, action_type[:4], 
                    ha='center', va='center', color='black', fontsize=8)

    # Set labels and title
    ax.set_yticks(range(len(algorithms)))
    ax.set_yticklabels(algorithms)
    ax.set_xlabel('Step Number')
    ax.set_title('Action Sequence Comparison Across Algorithms')

    # Add a legend for action types
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, label=action) 
                    for action, color in action_types.items()]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.savefig('action_sequence_comparison.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    # Parse command line arguments and retrieve model path
    cli_args = tyro.cli(Args)
    if cli_args.hub:
        try:
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(repo_id="LilHairdy/cleanrl_memory_gym", filename=cli_args.name)
        except:
            raise RuntimeError(
                "Cannot load model from the huggingface hub. Please install the huggingface_hub pypi package and verify the model name. You can also download the model from the hub manually and load it from disk."
            )
    else:
        path = cli_args.name

    # Load the pre-trained model and the original args used to train it
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(path, map_location=device)
    args = checkpoint["args"]
    args = type("Args", (), args)

    # Initialize vector environment
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, seed=cli_args.env_seed, min_num_hosts=cli_args.min_num_hosts, max_num_hosts=cli_args.max_num_hosts) for _ in range(cli_args.num_envs)]
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(cli_args.seed)
    np.random.seed(cli_args.seed)
    torch.manual_seed(cli_args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # Determine maximum episode steps
    max_episode_steps = envs.envs[0].spec.max_episode_steps
    if not max_episode_steps:
        envs.envs[0].reset()  # Memory Gym envs need to be reset before accessing max_episode_steps
        max_episode_steps = envs.envs[0].max_episode_steps
    if max_episode_steps <= 0:
        max_episode_steps = 1024  # Memory Gym envs have max_episode_steps set to -1
        # Max episode impacts positional encoding, so make sure to set this accordingly

    # Setup agent and load its model parameters
    action_space_shape = (
        (envs.single_action_space.n,)
        if isinstance(envs.single_action_space, gym.spaces.Discrete) 
        else tuple(envs.single_action_space.nvec)
    )
    agent = Agent(args, envs.single_observation_space, action_space_shape, max_episode_steps)
    agent.load_state_dict(checkpoint["model_weights"])

    # Setup memory mask and indices (same for all envs)
    memory_mask = torch.tril(torch.ones((args.trxl_memory_length, args.trxl_memory_length)), diagonal=-1)
    repetitions = torch.repeat_interleave(
        torch.arange(0, args.trxl_memory_length).unsqueeze(0), 
        args.trxl_memory_length - 1, 
        dim=0
    ).long()
    memory_indices = torch.stack(
        [torch.arange(i, i + args.trxl_memory_length) 
         for i in range(max_episode_steps - args.trxl_memory_length + 1)]
    ).long()
    memory_indices = torch.cat((repetitions, memory_indices))
    
    # Statistics tracking
    episode_returns = []
    episode_lengths = []
    episode_actions = []
    action_tracking = [[] for _ in range(cli_args.num_envs)]
    reward_tracking = [[] for _ in range(cli_args.num_envs)]
    episode_stats_per_network_size = {}
    episodes_per_env = np.zeros(cli_args.num_envs, dtype=np.int32)
    total_episodes_completed = 0
    target_episodes = cli_args.num_envs * cli_args.num_episodes
    episode_trajectories = {}

    # Initialize memories and episode tracking
    memory = torch.zeros((cli_args.num_envs, max_episode_steps, args.trxl_num_layers, args.trxl_dim), dtype=torch.float32)
    env_steps = np.zeros(cli_args.num_envs, dtype=np.int32)
    current_returns = np.zeros(cli_args.num_envs)
    
    # Reset environments
    obs, _ = envs.reset()
    obs = torch.Tensor(obs)

    loop_counter = 1
    # Main evaluation loop
    while total_episodes_completed < target_episodes:
        # Identify active environments (those that haven't completed all episodes)
        active_envs = np.where(episodes_per_env < cli_args.num_episodes)[0]

        if len(active_envs) == 0:
            break
            
        # Prepare inputs for active environments
        active_obs = obs[active_envs]
        memory_windows = []
        masks = []
        indices_list = []
        
        for i in active_envs:
            t = env_steps[i]
            memory_window = memory[i:i+1, memory_indices[t]]
            mask = memory_mask[min(t, args.trxl_memory_length-1)].unsqueeze(0)
            index = memory_indices[t].unsqueeze(0)
            
            memory_windows.append(memory_window)
            masks.append(mask)
            indices_list.append(index)
        
        memory_batch = torch.cat(memory_windows, dim=0)
        mask_batch = torch.cat(masks, dim=0)
        indices_batch = torch.cat(indices_list, dim=0)
        
        # Get actions from the agent
        action, _, _, _, new_memory = agent.get_action_and_value(
            active_obs, memory_batch, mask_batch, indices_batch
        )
        
        # Update memories for active environments
        for idx, env_idx in enumerate(active_envs):
            memory[env_idx, env_steps[env_idx]] = new_memory[idx]
        
        # Create full action array
        full_actions = np.zeros((cli_args.num_envs,) + tuple(action.shape[1:]), dtype=np.int64)
        for idx, env_idx in enumerate(active_envs):
            full_actions[env_idx] = action[idx].cpu().numpy()
        
        # We have to access the network sizes before we step, because if we step
        # we reach the end of the environment, we will overwrite the actual nw size
        nw_size = envs.get_attr("current_num_hosts")

        # Step environments
        next_obs, rewards, terminations, truncations, infos = envs.step(full_actions)
        dones = np.logical_or(terminations, truncations)

        # Update returns and steps
        for i in active_envs:
            current_returns[i] += rewards[i]
            env_steps[i] += 1

            # Check if episode completed
            if dones[i]:
                # Record statistics
                episode_returns.append(current_returns[i])
                episode_lengths.append(env_steps[i])
                if nw_size[i] not in episode_stats_per_network_size:
                    episode_stats_per_network_size[nw_size[i]] = [(current_returns[i], env_steps[i])]
                else:
                    episode_stats_per_network_size[nw_size[i]].append((current_returns[i], env_steps[i]))
                
                action_success_reward_trajec = envs.get_attr('action_success_reward_trajec_pre_reset')

                if cli_args.visualize:
                    trajectory_visualization(nw_size[i], action_success_reward_trajec[i])

                episodes_per_env[i] += 1
                total_episodes_completed += 1

                # Store the trajectory
                if cli_args.save_trajectory:
                       # Just use the ep_len to have a unique key
                       episode_trajectories[f"traj_{total_episodes_completed}_{nw_size[i]}"] = action_success_reward_trajec[i]
                
                # Reset for next episode
                current_returns[i] = 0
                env_steps[i] = 0
                memory[i] = torch.zeros((max_episode_steps, args.trxl_num_layers, args.trxl_dim), dtype=torch.float32)
                action_tracking[i] = []
                reward_tracking[i] = []

        # Update observations
        obs = torch.Tensor(next_obs)
        loop_counter += 1


    # Close environments
    envs.close()
    
    # Print summary statistics
    if episode_returns:
        print("\nEvaluation Summary:")
        print(f"Total episodes: {len(episode_returns)}")
        print(f"Mean return: {np.mean(episode_returns):.2f} ± {np.std(episode_returns):.2f}")
        print(f"Min/Max return: {np.min(episode_returns):.2f} / {np.max(episode_returns):.2f}")
        print(f"Mean episode length: {np.mean(episode_lengths):.2f}")
    
        network_stats = {}
        
        for network_size, episodes in episode_stats_per_network_size.items():
            # Split the tuples into separate lists for returns and lengths
            returns = [episode[0] for episode in episodes]
            lengths = [episode[1] for episode in episodes]
            
            # Compute statistics
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            min_return = np.min(returns)
            max_return = np.max(returns)
            mean_length = np.mean(lengths)
            std_length = np.std(lengths)
            
            # Store statistics in the results dictionary
            network_stats[network_size] = {
                'count': len(episodes),
                'mean_return': mean_return,
                'std_return': std_return,
                'min_return': min_return,
                'max_return': max_return,
                'mean_length': mean_length,
                'std_length': std_length
            }
        
    # Print the results in a formatted way
    print("\nStatistics per Network Size:")
    for size, stats in sorted(network_stats.items()):
        print(f"\nNetwork Size: {size}")
        print(f"  Number of episodes: {stats['count']}")
        print(f"  Mean return: {stats['mean_return']:.2f} ± {stats['std_return']:.2f}")
        #print(f"  Min/Max return: {stats['min_return']:.2f} / {stats['max_return']:.2f}")
        #print(f"  Mean episode length: {stats['mean_length']:.2f} ± {stats['std_length']:.2f}")

    if cli_args.save_trajectory:
        import json
        import os
        file_path = os.path.join(f'{cli_args.traj_folder}', f'trajectoires_ppo-trxl.json')
        # Save to file
        with open(file_path, 'w') as f:
            json.dump(episode_trajectories, f)