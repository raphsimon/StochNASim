import numpy as np
import gymnasium as gym
import nasim


class NASimWrapper:
    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self, env_name, seed=None, render_mode=None, min_num_hosts=5, max_num_hosts=8
    ):
        print("Environment seed:", seed)
        self.seed = seed
        self._env = gym.make(
            env_name,
            min_num_hosts=min_num_hosts,
            max_num_hosts=max_num_hosts,
            exploit_probs=0.9,
            privesc_probs=0.9,
            seed=seed,
            render_mode=render_mode,
        )
        self.max_episode_steps = self._env.unwrapped.scenario.step_limit
        # Whether to make CartPole partial observable by masking out the velocity.
        self.spec = self._env.spec
        self.current_num_hosts = self._env.current_num_hosts
        self.current_state = self._env.current_state.tensor
        self.action_success_reward_trajec = []
        self.action_success_reward_trajec_pre_reset = self.action_success_reward_trajec

    def _binary_array_to_int(self, arr):
        """We use this functino to obtain a unique integer representation of
        the starting state of the environment. We want to
        """
        # Ensure array is 1D
        arr = arr.flatten()
        # Convert to integer
        return np.packbits(arr).dot(2 ** np.arange(len(np.packbits(arr)))[::-1])

    @property
    def observation_space(self):
        return self._env.observation_space

    @property
    def action_space(self):
        return self._env.action_space

    def reset(self, seed=None, options=None):
        obs, info = self._env.reset(seed=self.seed, options=options)
        self.current_state = self._env.current_state.tensor
        self.current_num_hosts = self._env.current_num_hosts
        state_as_int = self._binary_array_to_int(
            np.array(self._env.current_state.tensor, dtype=int)
        )
        # For debugging, we print the integer representation of the state and the number of
        # hosts in the network. This can help us verify that the environment is resetting
        # correctly and that we are getting a unique state representation.
        # print("Env reset to state:", state_as_int, "Network size:", self.current_num_hosts)
        self.action_success_reward_trajec_pre_reset = self.action_success_reward_trajec
        self.action_success_reward_trajec = []
        return obs, info

    def step(self, action):
        obs, reward, done, step_limit_reached, info = self._env.step(action[0])
        self.action_success_reward_trajec.append(
            (int(action[0]), info["success"], float(reward))
        )
        info = {}
        return obs, reward, done, step_limit_reached, info

    def render(self):
        self._env.render()

    def close(self):
        self._env.close()
