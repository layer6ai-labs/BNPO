import numpy as np


class RoomEnv:
    """
    Environment where the agent starts at the bottom of a 5 unit long room where they can only move vertically. There
    n_rooms+1 such rooms, and the agent always starts in room 0. The reward is randomly placed at the top of one of the
    non-starting rooms. The state is described by 3 coordinates, the first one indicating the y-axis, the second one
    the room the agent is in, and the third one is a dummy variable usually set to 0. At the beginning of the starting
    room, the dummy coordinate is used to tell the agent which room the reward is in. When the agent reaches the top
    of the starting room, the can teleport to the bottom of any room.
    """
    def __init__(self, rng, n_rooms=2, max_steps=5, room_size=5):
        self.env_type = 'room'
        self._rng = rng
        self.n_rooms = n_rooms
        self.state = None
        self.starting_room = None
        self.reward_room = None
        self.max_steps = max_steps
        self.room_size = room_size
        self.n_steps = None
        self.state_dim = 3
        self.action_dim = n_rooms+1
        self.one_hot_dict = {}
        for room in range(0, n_rooms+1):
            one_hot_vector = np.zeros(n_rooms+1, dtype='float32')
            one_hot_vector[room] = 1.
            self.one_hot_dict[room] = one_hot_vector

    def reset(self):
        self.starting_room = 0
        # self.starting_room = self._rng.randint(0, self.n_rooms+1)
        self.reward_room = self._rng.randint(1, self.n_rooms+1)
        if self.reward_room == self.starting_room:
            self.reward_room = 0
        self.n_steps = 0
        self.state = np.array([0, self.starting_room, self.reward_room], dtype=np.float32)
        return self.state

    def _is_at_reward(self):
        return self.state[1] == self.reward_room

    def _move(self, teleport_location):
        if self.state[0] == self.room_size-1 and self.state[1] == self.starting_room:  
            # if at the top of the first room, teleport instead of moving
            assert teleport_location in range(0, self.n_rooms + 1)
            self.state[0] = 0
            self.state[1] = teleport_location
        else:
            self.state[0] = self.state[0] + 1
        if self.state[0] == 0 and self.state[1] == self.starting_room:
            self.state[2] = self.reward_room
        else:
            self.state[2] = -1

    def step(self, action):
        assert self.state is not None, 'Cannot call env.step() before calling reset()'
        self._move(action)
        self.n_steps += 1
        reward = 0
        if self._is_at_reward():
            reward = 1
        done = self.n_steps >= self.max_steps
        return np.array(self.state), reward, done, {}

    def generate_expert_trajectories(self, n_traj, noise_level=0.1, max_steps=5, verbose=False, action_seed=12345):

        rng_actions = np.random.RandomState(action_seed)

        data_states = np.zeros([n_traj, max_steps, self.state_dim], dtype='float32')
        data_actions = np.zeros([n_traj, max_steps - 1, self.action_dim], dtype='float32')
        data_rewards = np.zeros([n_traj, max_steps - 1], dtype='float32')

        for i in range(n_traj):
            self.reset()
            remembered = None
            data_states[i, 0] = np.array(self.state)
            for t in range(max_steps - 1):
                u = rng_actions.uniform()
                action = 0
                if u < noise_level:
                    # Act randomly
                    action = rng_actions.choice(self.n_rooms + 1)
                else:
                    # Act optimally
                    # Store the 'room hint' if we're given it.
                    if self.state[2] != -1 and remembered is None:
                        remembered = int(self.state[2])
                    if remembered is not None:
                        # If we are at the top of the first room and have recorded a hint,
                        # teleport to the room that was hinted.
                        action = remembered
                    else:
                        # We are either not at the top of the room (in which case action[1] doesn't matter)
                        # or we didn't record the room hint, in which case all we can do is pick randomly.
                        action = rng_actions.choice(self.n_rooms + 1)
                data_actions[i, t] = self.one_hot_dict[action]
                _, reward, _, _ = self.step(action)
                data_states[i, t + 1] = np.array(self.state)
                data_rewards[i, t] = reward
        if verbose:
            av_rew = np.mean(np.max(data_rewards, axis=1))
            print(f'dataset of expert trajectories generated with reward reached {100*av_rew:.2f}% of trajectories')

        return data_states, data_actions, data_rewards