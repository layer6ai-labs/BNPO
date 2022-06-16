import gym
import numpy as np

class AugmentedAtariEnv(gym.Wrapper):
    def __init__(self, env, model, gamma=0.99, max_steps=15):
        gym.Wrapper.__init__(self, env)
        self._model = model
        self._gamma = gamma
        self._max_steps = max_steps
        self._count_high_level_actions = 0
        self._count_total_actions = 0
        self._base_action_dim = self.env.action_space.n
        self.env.action_space.n = self.env.action_space.n + self._model.K

    def reset(self):
        if self.env.was_real_done:
            self._count_high_level_actions = 0
            self._count_total_actions = 0
        return self.env.reset()

    def step(self, action):
        self._count_total_actions += 1
        # frames = []
        if action < self._base_action_dim:
            steps_count = 1
            obs, reward, done, info = self.env.step(action)
            # frames.append(self.env.unwrapped.render(mode="rgb_array"))
        else:
            self._count_high_level_actions += 1
            done = False
            steps_count = 0
            reward = 0
            while not done and steps_count < self._max_steps:
                steps_count += 1
                sub_action, termination = self._model.play_from_observation(
                    option=action-self._base_action_dim, 
                    obs=np.unpackbits(self.unwrapped.ale.getRAM())
                )
                obs, sub_reward, done, info = self.env.step(sub_action)
                # frames.append(self.env.unwrapped.render(mode="rgb_array"))
                reward += self._gamma**(steps_count-1) * sub_reward
                # if np.random.random() < termination:
                #     break
        info['steps_count'] = steps_count
        # info['frames'] = frames
        if self.env.was_real_done:
            info['episode']['hlp'] = self._count_high_level_actions / self._count_total_actions
        return obs, reward, done, info
    
    def render(self):
        self.env.render()