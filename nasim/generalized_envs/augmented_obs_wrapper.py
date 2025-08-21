import numpy as np
import gymnasium as gym
from gymnasium import spaces

import nasim
from nasim.envs.observation import Observation


class AugmentedObsWrapper(gym.Wrapper):
    """In NASim's Partially Observable setting, the agent only observes the
    outcom of it's latest action. This significantly increases the difficulty,
    and requires some form of memory or recurrence to find the optimal policy.

    With this wrapper the add information to the observation, in the form of
    and explicit belief representation. It also follows the logic that once
    information about hosts has been discovered, it remains valid thoughout
    the episode.

    The augemented observations as we call them, are contructed in the following
    manner:

    Augmented Observation = [Current Observation | Accumulated Knowledge Vector]

    Args:
        gym (_type_): _description_
    """
    def __init__(self, env):
        super().__init__(env)

        self.env = env
        self.host_vec_len = self.env.host_vec_len
        
        # Determine new bounds for the observation dimensions.
        if self.env.flat_obs:
            obs_shape = self.last_obs.shape_flat()
            state_shape_flat = self.env.current_state.tensor.flatten().shape
            self.current_knowledge = np.zeros(state_shape_flat)
            obs_shape = (obs_shape[0] + state_shape_flat[0],)
        else:
            obs_shape = self.last_obs.shape()
            state_shape = self.env.current_state.tensor.shape
            self.current_knowledge = np.zeros(state_shape)
            obs_shape = tuple(obs_shape[0] + state_shape[0], obs_shape[1])

        obs_low, obs_high = Observation.get_space_bounds(self.env.scenario)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, shape=obs_shape)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        #print(obs[:-self.host_vec_len])
        #print(self.current_knowledge)
        #print("-----")
        
        self.current_knowledge = np.maximum(self.current_knowledge, obs[:-self.host_vec_len])
        augmented_obs = np.concatenate((obs, self.current_knowledge))

        return augmented_obs, reward, terminated, truncated, info
    
    def reset(self, *, seed = None, options = None):
        self.current_knowledge[:] = 0 # Zero out the knowledge
        obs, info = super().reset(seed=seed, options=options)
        augmented_obs = np.concatenate((obs, self.current_knowledge))

        return augmented_obs, info
    
    def render(self):
        return self.env.render()
    
    def render_obs(self, mode="human", obs=None):
        return self.env.render_obs(mode=mode, obs=obs)

    def render_state(self, mode="human", state=None):
        return self.env.render_state(mode=mode, state=state)

    def render_action(self, action):
        return self.env.render_action(action)
    
    def action_masks(self):
        return self.env.action_masks()
    
    def close(self):
        return super().close()


class StackedObsWrapper(gym.Wrapper):
    """In NASim's Partially Observable setting, the agent only observes the
    outcom of it's latest action. This significantly increases the difficulty,
    and requires some form of memory or recurrence to find the optimal policy.

    With this wrapper the add information to the observation, in the form of
    and explicit belief representation. It also follows the logic that once
    information about hosts has been discovered, it remains valid thoughout
    the episode.

    The stacked observations as we call them, contain the stacked information
    received over the course of the episode. This is a weaker representation
    than the AugmentedObsWrapper. We basically don't do a concatenation and
    just return the 'Accumulated Knowledge Vector'

    Args:
        gym (_type_): _description_
    """
    def __init__(self, env):
        super().__init__(env)

        self.env = env
        self.host_vec_len = self.env.host_vec_len
        
        # Create shape of current_knowledge tensor based the shape of the state.
        # The auxiliary information will be concatenated to it afterwards.
        if self.env.flat_obs:
            state_shape_flat = self.env.current_state.tensor.flatten().shape
            self.current_knowledge = np.zeros(state_shape_flat)
        else:
            state_shape = self.env.current_state.tensor.shape
            self.current_knowledge = np.zeros(state_shape)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        aux_info = obs[-self.host_vec_len:]
        # Update current knowledge, but keep aux_info out
        self.current_knowledge = np.maximum(self.current_knowledge, obs[:-self.host_vec_len])
        stacked_obs = np.concatenate((self.current_knowledge, aux_info))

        return stacked_obs, reward, terminated, truncated, info
    
    def reset(self, *, seed = None, options = None):
        self.current_knowledge[:] = 0 # Zero out the knowledge
        obs, info = super().reset(seed=seed, options=options)

        return obs, info
    
    def render(self):
        return self.env.render()
    
    def render_obs(self, mode="human", obs=None):
        return self.env.render_obs(mode=mode, obs=obs)

    def render_state(self, mode="human", state=None):
        return self.env.render_state(mode=mode, state=state)

    def render_action(self, action):
        return self.env.render_action(action)
    
    def action_masks(self):
        return self.env.action_masks()
    
    def close(self):
        return super().close()


if __name__ == '__main__':
    
    from generalization_env import NASimGenEnv

    env = NASimGenEnv()
    aug_obs_env = AugmentedObsWrapper(env)
    stacked_obs_env = StackedObsWrapper(env)

    obs, _ = aug_obs_env.reset()
    print("AugmentedObsWrapper obs shape:", obs.shape)
    obs, _ = stacked_obs_env.reset()
    print("StackedObsWrapper obs shape:", obs.shape)

    def random_actions(env_to_test, num_actions=10):
        for _ in range(num_actions):
            obs, reward, terminated, truncated, info = env_to_test.step(env_to_test.action_space.sample())
            print(obs, obs.shape)
            print(reward)
            print(terminated)
            print(truncated)
            print(info)
            print()
            if terminated or truncated:
                break

    print("="*30 + " Testing AugmentedObsWrapper " + "="*30)
    random_actions(aug_obs_env)
    print("="*30 + " Testing StackedObsWrapper " + "="*30)
    random_actions(stacked_obs_env)

