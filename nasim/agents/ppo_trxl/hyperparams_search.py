import os
import sys
import logging
import torch
import random
import time
import tyro
from collections import deque
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ppo_trxl import Args
from ppo_trxl import make_env, batched_index_select, evaluate_policy
from ppo_trxl import Agent

import optuna
from optuna.storages import RetryFailedTrialCallback
from optuna.trial import TrialState
from optuna.study import MaxTrialsCallback
from optuna.pruners import MedianPruner
from sqlalchemy.pool import NullPool
from torch.utils.tensorboard import SummaryWriter

"""
This script performs a hyperparameter search for the PPO-TrXL algorithm.
What's expected as input is first and foremost a complete config, just
so that all the values can be set to perform some of the decisions.
Then, we sample the hyperparameters and create and instance of the
Learner class and train the agent. The reported score is the return
after the last performed evaluation during the training run. This 
value is then stored alongside other trial details in the database,
and also used to prune other trials.
"""

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 4444
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""

    # Algorithm specific arguments
    env_id: str = "StochPO-v0"
    """the id of the environment"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    num_evals: int = 1
    """Number of policy evaluations to perform during training."""
    num_eval_episodes: int = 10
    """Number of episodes to run during each policy evaluation."""
    init_lr: float = 2.75e-4
    """the initial learning rate of the optimizer"""
    final_lr: float = 1.0e-5
    """the final learning rate of the optimizer after linearly annealing"""
    num_envs: int = 1
    """the number of parallel game environments"""
    num_eval_envs: int = 1
    """the number of parallel game environments for evaluation"""
    num_steps: int = 256
    """the number of steps to run in each environment per policy rollout"""
    anneal_steps: int = num_envs * num_steps * 200 # We use approx 82% of the total timesteps for annealing
    """the number of steps to linearly anneal the learning rate and entropy coefficient from initial to final"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.99
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 8
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = False
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    init_ent_coef: float = 0.0001
    """initial coefficient of the entropy bonus"""
    final_ent_coef: float = 0.000001
    """final coefficient of the entropy bonus after linearly annealing"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.25
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # Transformer-XL specific arguments
    trxl_num_layers: int = 2
    """the number of transformer layers"""
    trxl_num_heads: int = 1
    """the number of heads used in multi-head attention"""
    trxl_dim: int = 384
    """the dimension of the transformer"""
    trxl_memory_length: int = 256
    """the length of TrXL's sliding memory window"""
    trxl_positional_encoding: str = "learned"
    """the positional encoding type of the transformer, choices: "", "absolute", "learned" """
    reconstruction_coef: float = 0.0
    """the coefficient of the observation reconstruction loss, if set to 0.0 the reconstruction loss is not used"""

    # To be filled on runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

    # Hyperparameter search specific arguments
    db_url: str = None
    """the database URL to store the hyperparameter search results"""
    trials: int = 1
    """the number of trials to run"""
    max_total_trials: int = None
    """the maximum total number of trials to run"""
    study_name: str = None
    """the name of the study to run"""
    pruner_warmup_steps: int = 400_000
    """the number of warmup steps for optuna.pruner.MedianPruner. Should be >= 40% of total_timesteps"""

def optimize_hyperparameters(optimize_trial, args):
    # Add stream handler of stdout to show the messages
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    print(f"Provided database: {args.db_url}")

    sqlite_timeout = 300
    engine_kwargs = None
    if "sqlite" in args.db_url:
        engine_kwargs={
            'connect_args': {'timeout': sqlite_timeout},
        }
    elif "postgresql" in args.db_url:
        engine_kwargs = {
            "poolclass": NullPool,
            "connect_args": {
                "connect_timeout": 60,
                "keepalives": 1,
                "keepalives_idle": 30,
                "keepalives_interval": 10
            }
        }
    print(f'Using {engine_kwargs} for engine_kwargs')

    storage = optuna.storages.RDBStorage(
        args.db_url,
        engine_kwargs=engine_kwargs,
        heartbeat_interval=60,
        grace_period=120,
        failed_trial_callback=RetryFailedTrialCallback(max_retry=2),
    )
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        load_if_exists=True,
        direction='maximize',
        pruner=MedianPruner(n_startup_trials=10, 
                            n_warmup_steps=args.pruner_warmup_steps),
    ) # No sampler is specified, so a default sampler (TPE) is used.
    
    if args.max_total_trials is not None:
        # Note: we count already running trials here otherwise we get
        #  (max_total_trials + number of workers) trials in total.
        counted_states = [
            TrialState.COMPLETE,
            TrialState.RUNNING,
            TrialState.PRUNED,
        ]
        completed_trials = len(study.get_trials(states=counted_states))
        if completed_trials < args.max_total_trials:
            study.optimize(
                optimize_trial,
                n_trials=args.trials,
                callbacks=[
                    MaxTrialsCallback(
                        args.max_total_trials,
                        states=counted_states,
                    )   
                ],
                gc_after_trial=True
            )
    else:
        study.optimize(optimize_trial, n_trials=args.trials, gc_after_trial=True)

    # Print results
    print(f"Best value (accuracy): {study.best_value:.4f}")
    print("\nBest hyperparameters:")
    for param, value in study.best_params.items():
        print(f"{param}: {value}")

    return study

def suggest_ppo_trxl_params(trial: optuna.Trial, args):
    """
    Suggest hyperparameters for the PPO-TrXL algorithm.
    """
    args.gamma = trial.suggest_categorical("gamma", [0.95, 0.99, 0.995, 0.999])                 # |V| = 4
    args.gae_lambda = trial.suggest_categorical("gae-lambda", [0.9, 0.95, 0.99])       # |V| = 4
    args.n_mini_batch = trial.suggest_categorical("n_mini_batch", [4, 8])                    # |V| = 2
    args.update_epochs = trial.suggest_categorical("epochs", [2, 3, 4])                      # |V| = 2
    args.norm_adv = trial.suggest_categorical("norm_adv", [True, False])                     # |V| = 2
    args.vf_coef = trial.suggest_categorical("vf_coef", [0.2, 0.3, 0.5])                     # |V| = 3
    args.max_grad_norm = trial.suggest_categorical("max_grad_norm", [0.25, 0.35, 0.5, 1.0])  # |V| = 4
    args.trxl_num_layers = trial.suggest_categorical("trxl_num_layers", [2, 3, 4])           # |V| = 3
    args.trxl_num_heads = trial.suggest_categorical("transformer_num_heads", [1, 4, 8])      # |V| = 3
    args.trxl_dim = trial.suggest_categorical("trxl_dim", [128, 256, 384])                   # |V| = 3
    args.trxl_memory_length = trial.suggest_categorical("transformer_memory_length", [128, 256, 512]) # |V| = 3
    args.trxl_positional_encoding = trial.suggest_categorical("transformer_positional_encoding", ["", "absolute", "learned"]) # |V| = 3
    args.init_lr = trial.suggest_categorical("learning_rate_initial", [2.0e-4, 2.75e-4, 3.0e-4, 3.5e-4]) # |V| = 4
    args.init_ent_coef = trial.suggest_categorical("entropy_coefficient_initial", [1.0e-4, 1.0e-3, 1.0e-2]) # |V| = 4
    args.clip_coef = trial.suggest_categorical("clip_coef", [0.1, 0.2, 0.3])                 # |V| = 3


if __name__ == '__main__':

    args = tyro.cli(Args)
    assert args.study_name is not None, "Please provide the study name"
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    eval_interval = args.num_iterations // args.num_evals

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # Determine the device to be used for training and set the default tensor type
    if args.cuda:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_default_device(device)
    else:
        device = torch.device("cpu")


    def optimize_trial(trial):
        run_name = f"{args.study_name}__{args.seed}__{trial.number}__{int(time.time())}"
        eval_counter = 0 # Number of already performed evaluations
        eval_counter_vec = [False] * args.num_evals
        eval_interval = args.total_timesteps // args.num_evals
        eval_steps = [step for step in range(eval_interval, args.total_timesteps+1, eval_interval)]
        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )

        # Sample hyperparameters
        suggest_ppo_trxl_params(trial, args)

        # Environment setup
        envs = gym.vector.SyncVectorEnv(
            [make_env(args.env_id) for _ in range(args.num_envs)],
        )
        # Evaluation environment setup
        # evaluation uses a separate vectorised env to keep training rollouts isolated.
        eval_envs = gym.vector.SyncVectorEnv(
            [make_env(args.env_id) for _ in range(args.num_envs)],
        )
        observation_space = envs.single_observation_space
        action_space_shape = (
            (envs.single_action_space.n,)
            if isinstance(envs.single_action_space, gym.spaces.Discrete)
            else tuple(envs.single_action_space.nvec)
        )
        env_ids = range(args.num_envs)
        env_current_episode_step = torch.zeros((args.num_envs,), dtype=torch.long)
        # Determine maximum episode steps
        max_episode_steps = envs.envs[0].max_episode_steps
        print(f"Max episode steps: {max_episode_steps}")
        args.trxl_memory_length = min(args.trxl_memory_length, max_episode_steps)

        agent = Agent(args, observation_space, action_space_shape, max_episode_steps).to(device)
        optimizer = optim.AdamW(agent.parameters(), lr=args.init_lr)
        bce_loss = nn.BCELoss()  # Binary cross entropy loss for observation reconstruction

        # ALGO Logic: Storage setup
        rewards = torch.zeros((args.num_steps, args.num_envs))
        actions = torch.zeros((args.num_steps, args.num_envs, len(action_space_shape)), dtype=torch.long)
        dones = torch.zeros((args.num_steps, args.num_envs))
        obs = torch.zeros((args.num_steps, args.num_envs) + observation_space.shape)
        log_probs = torch.zeros((args.num_steps, args.num_envs, len(action_space_shape)))
        values = torch.zeros((args.num_steps, args.num_envs))
        # The length of stored-memories is equal to the number of sampled episodes during training data sampling
        # (num_episodes, max_episode_length, num_layers, embed_dim)
        stored_memories = []
        # Memory mask used during attention
        stored_memory_masks = torch.zeros((args.num_steps, args.num_envs, args.trxl_memory_length), dtype=torch.bool)
        # Index to select the correct episode memory from stored_memories
        stored_memory_index = torch.zeros((args.num_steps, args.num_envs), dtype=torch.long)
        # Indices to slice the episode memories into windows
        stored_memory_indices = torch.zeros((args.num_steps, args.num_envs, args.trxl_memory_length), dtype=torch.long)

        # TRY NOT TO MODIFY: start the game
        global_step = 0
        start_time = time.time()
        episode_infos = deque(maxlen=100)  # Store episode results for monitoring statistics
        next_obs, _ = envs.reset(seed=args.seed)
        next_obs = torch.Tensor(next_obs).to(device)
        next_done = torch.zeros(args.num_envs)
        # Setup placeholders for each environments's current episodic memory
        next_memory = torch.zeros((args.num_envs, max_episode_steps, args.trxl_num_layers, args.trxl_dim), dtype=torch.float32)
        # Generate episodic memory mask used in attention
        memory_mask = torch.tril(torch.ones((args.trxl_memory_length, args.trxl_memory_length)), diagonal=-1)
        """ e.g. memory mask tensor looks like this if memory_length = 6
        0, 0, 0, 0, 0, 0
        1, 0, 0, 0, 0, 0
        1, 1, 0, 0, 0, 0
        1, 1, 1, 0, 0, 0
        1, 1, 1, 1, 0, 0
        1, 1, 1, 1, 1, 0
        """
        # Setup memory window indices to support a sliding window over the episodic memory
        repetitions = torch.repeat_interleave(
            torch.arange(0, args.trxl_memory_length).unsqueeze(0), args.trxl_memory_length - 1, dim=0
        ).long()
        memory_indices = torch.stack(
            [torch.arange(i, i + args.trxl_memory_length) for i in range(max_episode_steps - args.trxl_memory_length + 1)]
        ).long()
        memory_indices = torch.cat((repetitions, memory_indices))
        """ e.g. the memory window indices tensor looks like this if memory_length = 4 and max_episode_length = 7:
        0, 1, 2, 3
        0, 1, 2, 3
        0, 1, 2, 3
        0, 1, 2, 3
        1, 2, 3, 4
        2, 3, 4, 5
        3, 4, 5, 6
        """

        for iteration in range(1, args.num_iterations + 1):
            sampled_episode_infos = []

            # Annealing the learning rate and entropy coefficient if instructed to do so
            do_anneal = args.anneal_steps > 0 and global_step < args.anneal_steps
            frac = 1 - global_step / args.anneal_steps if do_anneal else 0
            lr = (args.init_lr - args.final_lr) * frac + args.final_lr
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
            ent_coef = (args.init_ent_coef - args.final_ent_coef) * frac + args.final_ent_coef

            # Init episodic memory buffer using each environments' current episodic memory
            stored_memories = [next_memory[e] for e in range(args.num_envs)]
            for e in range(args.num_envs):
                stored_memory_index[:, e] = e

            for step in range(args.num_steps):
                global_step += args.num_envs

                # ALGO LOGIC: action logic
                with torch.no_grad():
                    obs[step] = next_obs
                    dones[step] = next_done
                    stored_memory_masks[step] = memory_mask[torch.clip(env_current_episode_step, 0, args.trxl_memory_length - 1)]
                    stored_memory_indices[step] = memory_indices[env_current_episode_step]
                    # Retrieve the memory window from the entire episodic memory
                    memory_window = batched_index_select(next_memory, 1, stored_memory_indices[step])
                    action, logprob, _, value, new_memory = agent.get_action_and_value(
                        next_obs, memory_window, stored_memory_masks[step], stored_memory_indices[step]
                    )
                    next_memory[env_ids, env_current_episode_step] = new_memory
                    # Store the action, log_prob, and value in the buffer
                    actions[step], log_probs[step], values[step] = action, logprob, value

                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
                next_done = np.logical_or(terminations, truncations)
                rewards[step] = torch.tensor(reward).to(device).view(-1)
                next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

                # Reset and process episodic memory if done
                for id, done in enumerate(next_done):
                    if done:
                        # Reset the environment's current timestep
                        env_current_episode_step[id] = 0
                        # Break the reference to the environment's episodic memory
                        mem_index = stored_memory_index[step, id]
                        stored_memories[mem_index] = stored_memories[mem_index].clone()
                        # Reset episodic memory
                        next_memory[id] = torch.zeros(
                            (max_episode_steps, args.trxl_num_layers, args.trxl_dim), dtype=torch.float32
                        )
                        if step < args.num_steps - 1:
                            # Store memory inside the buffer
                            stored_memories.append(next_memory[id])
                            # Store the reference of to the current episodic memory inside the buffer
                            stored_memory_index[step + 1 :, id] = len(stored_memories) - 1
                    else:
                        # Increment environment timestep if not done
                        env_current_episode_step[id] += 1

                if "final_info" in infos:
                    for info in infos["final_info"]:
                        if info and "episode" in info:
                            sampled_episode_infos.append(info["episode"])

            # Bootstrap value if not done
            with torch.no_grad():
                start = torch.clip(env_current_episode_step - args.trxl_memory_length, 0)
                end = torch.clip(env_current_episode_step, args.trxl_memory_length)
                indices = torch.stack([torch.arange(start[b], end[b]) for b in range(args.num_envs)]).long()
                memory_window = batched_index_select(next_memory, 1, indices)  # Retrieve the memory window from the entire episode
                next_value = agent.get_value(
                    next_obs,
                    memory_window,
                    memory_mask[torch.clip(env_current_episode_step, 0, args.trxl_memory_length - 1)],
                    stored_memory_indices[-1],
                )
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values

            # Flatten the batch
            b_obs = obs.reshape(-1, *obs.shape[2:])
            b_logprobs = log_probs.reshape(-1, *log_probs.shape[2:])
            b_actions = actions.reshape(-1, *actions.shape[2:])
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)
            b_memory_index = stored_memory_index.reshape(-1)
            b_memory_indices = stored_memory_indices.reshape(-1, *stored_memory_indices.shape[2:])
            b_memory_mask = stored_memory_masks.reshape(-1, *stored_memory_masks.shape[2:])
            stored_memories = torch.stack(stored_memories, dim=0)

            # Remove unnecessary padding from TrXL memory, if applicable
            actual_max_episode_steps = (stored_memory_indices * stored_memory_masks).max().item() + 1
            if actual_max_episode_steps < args.trxl_memory_length:
                b_memory_indices = b_memory_indices[:, :actual_max_episode_steps]
                b_memory_mask = b_memory_mask[:, :actual_max_episode_steps]
                stored_memories = stored_memories[:, :actual_max_episode_steps]

            # Optimizing the policy and value network
            clipfracs = []
            for epoch in range(args.update_epochs):
                b_inds = torch.randperm(args.batch_size)
                for start in range(0, args.batch_size, args.minibatch_size):
                    end = start + args.minibatch_size
                    mb_inds = b_inds[start:end]
                    mb_memories = stored_memories[b_memory_index[mb_inds]]
                    mb_memory_windows = batched_index_select(mb_memories, 1, b_memory_indices[mb_inds])

                    _, newlogprob, entropy, newvalue, _ = agent.get_action_and_value(
                        b_obs[mb_inds], mb_memory_windows, b_memory_mask[mb_inds], b_memory_indices[mb_inds], b_actions[mb_inds]
                    )

                    # Policy loss
                    mb_advantages = b_advantages[mb_inds]
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                    mb_advantages = mb_advantages.unsqueeze(1).repeat(
                        1, len(action_space_shape)
                    )  # Repeat is necessary for multi-discrete action spaces
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = torch.exp(logratio)
                    pgloss1 = -mb_advantages * ratio
                    pgloss2 = -mb_advantages * torch.clamp(ratio, 1.0 - args.clip_coef, 1.0 + args.clip_coef)
                    pg_loss = torch.max(pgloss1, pgloss2).mean()

                    # Value loss
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    if args.clip_vloss:
                        v_loss_clipped = b_values[mb_inds] + (newvalue - b_values[mb_inds]).clamp(
                            min=-args.clip_coef, max=args.clip_coef
                        )
                        v_loss = torch.max(v_loss_unclipped, (v_loss_clipped - b_returns[mb_inds]) ** 2).mean()
                    else:
                        v_loss = v_loss_unclipped.mean()

                    # Entropy loss
                    entropy_loss = entropy.mean()

                    # Combined losses
                    loss = pg_loss - ent_coef * entropy_loss + v_loss * args.vf_coef

                    # Add reconstruction loss if used
                    r_loss = torch.tensor(0.0)
                    if args.reconstruction_coef > 0.0:
                        r_loss = bce_loss(agent.reconstruct_observation(), b_obs[mb_inds] / 255.0)
                        loss += args.reconstruction_coef * r_loss

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=args.max_grad_norm)
                    optimizer.step()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                if args.target_kl is not None and approx_kl > args.target_kl:
                    break

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            # Log and monitor training statistics
            episode_infos.extend(sampled_episode_infos)
            episode_result = {}
            if len(episode_infos) > 0:
                for key in episode_infos[0].keys():
                    episode_result[key + "_mean"] = np.mean([info[key] for info in episode_infos])

            print(
                "{:9} SPS={:4} return={:.2f} length={:.1f} pi_loss={:.3f} v_loss={:.3f} entropy={:.3f} r_loss={:.3f} value={:.3f} adv={:.3f}".format(
                    iteration,
                    int(global_step / (time.time() - start_time)),
                    episode_result["r_mean"],
                    episode_result["l_mean"],
                    pg_loss.item(),
                    v_loss.item(),
                    entropy_loss.item(),
                    r_loss.item(),
                    torch.mean(values),
                    torch.mean(advantages),
                )
            )
            
            # Perform policy evaluation
            # If our current gobal step is above the required steps for the n-th eval,
            # and we haven't evaluated yet, then eval
            if global_step > eval_steps[eval_counter] and not eval_counter_vec[eval_counter]:
                eval_counter += 1
                start_time_eval = time.time()
                episode_returns, episode_lengths = evaluate_policy(args, 
                                                                   agent,
                                                                   device=device, 
                                                                   envs=eval_envs, 
                                                                   max_episode_steps=max_episode_steps)
                score = np.mean(episode_returns)
                print("Evaluation {} r_mean={:.2f} l_mean={:.2f} time={:.2f}s".format(
                    eval_counter, np.mean(episode_returns), np.mean(episode_lengths), time.time() - start_time_eval))
                writer.add_scalar("eval/mean_return", np.mean(episode_returns), global_step)
                writer.add_scalar("eval/std_return", np.std(episode_returns), global_step)
                writer.add_scalar("eval/mean_length", np.mean(episode_lengths), global_step)
                writer.add_scalar("eval/min_return", np.min(episode_returns), global_step)
                writer.add_scalar("eval/max_return", np.max(episode_returns), global_step)

                eval_counter_vec[eval_counter-1] = True

                # Report intremediate value in study for this trial
                trial.report(score, step=global_step)

                # Handle pruning based on the intermediate value.
                if trial.should_prune():
                    raise optuna.TrialPruned()

            if episode_result:
                for key in episode_result:
                    writer.add_scalar("episode/" + key, episode_result[key], global_step)
            writer.add_scalar("episode/value_mean", torch.mean(values), global_step)
            writer.add_scalar("episode/advantage_mean", torch.mean(advantages), global_step)
            writer.add_scalar("charts/learning_rate", lr, global_step)
            writer.add_scalar("charts/entropy_coefficient", ent_coef, global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/loss", loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("losses/reconstruction_loss", r_loss.item(), global_step)
            writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            writer.add_scalar("losses/explained_variance", explained_var, global_step)
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        writer.close()
        envs.close()

        # Return the evaluation score achieved during the last evaluation
        return score

    study = optimize_hyperparameters(optimize_trial, args)

