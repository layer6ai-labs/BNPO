import gym
import numpy as np

class AtariEnv:
    def __init__(self, env_name, path=None):

        self.env_type = 'atari'
        self.env = gym.make(env_name)
        self.demo_file = path
        self.data = np.load(self.demo_file, allow_pickle=True) if path is not None else None
        self.n_traj = len(self.data) if self.data is not None else 0

        self.state_dim = 128*8
        self.action_dim = self.env.action_space.n
    
    def reset(self):
        obs = self.env.reset()
        return np.unpackbits(obs, axis=-1)

    def step(self, action, frame_skip=4):
        for i in range(frame_skip):
            obs, reward, done, info = self.env.step(action)
        return np.unpackbits(obs, axis=-1), reward, done, info
    
    def render(self):
        self.env.render()

    def get_expert_trajectories(self, max_steps=200):
        # For RAM states
        if self.data is None:
            return None, None, None
        print(f'Loading {self.n_traj} trajectories...')
        data_states = np.zeros([self.n_traj, max_steps, self.state_dim], dtype='float32')
        data_actions = np.zeros([self.n_traj, max_steps - 1, self.action_dim], dtype='float32')
        for i, trajectory in enumerate(self.data):
            for j in range(max_steps-1):
                if j >= len(trajectory):
                    break
                data_states[i, j] = np.unpackbits(trajectory[j][0])
                data_actions[i, j, int(trajectory[j][1])] = 1.
            data_states[i, max_steps-1] = np.unpackbits(trajectory[max_steps-1][0])
        return data_states, data_actions, None

